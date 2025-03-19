import os 
import argparse
import re, ast, json, random
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelBinarizer
from transformers import (
    AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig,
    Qwen2VLProcessor, ViTFeatureExtractor
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from qwen_vl_utils import process_vision_info
from data_utils import get_final_task_message, SYSTEM_MESSAGE, resize_image, format_data, SAVE_DIR, VLM_ANN_DIR
 
OBS_TASK_MESSAGE = (
    "Observe the image and describe key elements of the image. "
    "Example: The dark sky serves as a dramatic background, emphasizing people holding tools. "
    "Objects resembling agricultural tools appear threatening, creating tension. "
    "The synchronized movement of the crowd and strong silhouette effects contribute to a striking atmosphere. "
    "The high contrast highlights the shapes of the tools as primary visual elements."
)
INF_TASK_MESSAGE = (
    "Observe the image and describe the process of inferring the emotions conveyed in the image. "
    "Example: The crowd holding various tools suggests intense energy and the excitement of collective action. "
    "The silhouette contrasting with the dark sky evokes tension, potentially eliciting anxiety or fear. "
    "The threatening shapes of the tools and the collective behavior may hint at anger or hostility. "
    "The overall scene's organized yet uncertain atmosphere limits the possibility of comfort."
) 

def load_annotation_data(file_path):
    """Load annotation data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_datasets(data, cls_task_message):
    """
    Prepare training datasets (observation, inference, classification) from annotation data.
    Extract key elements and reasoning using regex patterns.
    """
    train_obs, train_inf, train_cls = [], [], []
    key_elements_pattern = r"1\.\s\*?\*?Key elements\*?\*?\s+(.*?)(?=\n2\.\s\*?\*?Reasoning\*?\*?)"
    reasoning_pattern = r"2\.\s\*?\*?Reasoning\*?\*?\s+(.*)"
    
    for item in data:
        text = item['gpt_response']
        key_match = re.search(key_elements_pattern, text, re.DOTALL | re.IGNORECASE)
        reason_match = re.search(reasoning_pattern, text, re.DOTALL | re.IGNORECASE)
        key_text = key_match.group(1).strip() if key_match else None
        reason_text = reason_match.group(1).strip() if reason_match else None
        label = item['label']
        image = Image.open(item['img_path'])
        file = item['img']
         
        train_obs.append(format_data(OBS_TASK_MESSAGE, image, key_text, file, vlm=True))  
        train_inf.append(format_data(INF_TASK_MESSAGE, image, reason_text, file, vlm=True)) 
        train_cls.append(format_data(cls_task_message, image, f"{{'emotion': '{label}'}}", file, vlm=True))
    
    dataset = train_obs + train_inf + train_cls
    random.shuffle(dataset)
    return dataset

def get_model_and_processor(model_id):
    """Load the model and processor with 4-bit quantization settings."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def get_training_config(data_name, lr, epoch):
    """Configure training settings using SFTConfig.""" 
    output_dir = f"{SAVE_DIR}/vlm/qwen7b_{data_name}"
    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epoch,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=100,
        disable_tqdm=False,
        save_strategy="epoch",
        learning_rate=lr,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    config.remove_unused_columns = False
    return config

def collate_fn(examples, processor):
    """Collate function to encode text and image pairs for training."""
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Mask image tokens
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652, 151653, 151655]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for token_id in image_tokens:
        labels[labels == token_id] = -100
    batch["labels"] = labels
    return batch

def create_trainer(model, training_config, train_dataset, peft_config, processor):
    """Initialize the SFTTrainer."""
    processor.tokenizer.padding_side = 'right'
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        data_collator=lambda examples: collate_fn(examples, processor),
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )
    return trainer 
 
def inference(messages, model, processor):
    """Perform inference using the given model and processor."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]
 

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str) 
    parser.add_argument("--gpu", type=int) 
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--epoch", type=int, default=5)  

    args = parser.parse_args()
    data_name = args.dataname 
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    
    # Prepare training data
    train_annotation_file = f'{VLM_ANN_DIR}/{data_name}_annotation_train.json'
    train_data = load_annotation_data(train_annotation_file)
    cls_task_message = get_final_task_message(data_name) 
    train_dataset = prepare_datasets(train_data, cls_task_message)
    
    # Load model and processor for training
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    model, processor = get_model_and_processor(model_id)
    
    # Configure training and LoRA settings
    training_config = get_training_config(data_name, args.lr, args.epoch)
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        task_type="CAUSAL_LM",
    )
     
    trainer = create_trainer(model, training_config, train_dataset, peft_config, processor)
    trainer.train()
    trainer.save_model(training_config.output_dir) 
    
if __name__ == "__main__":
    main()
