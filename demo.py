import os 
import argparse
import re, ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import gradio as gr
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from data_utils import get_emo_categories, SAVE_DIR, VLM_NAME
from distill_model import QwenDistillationModel
from transformers import AutoModelForVision2Seq, AutoProcessor
from train_gate import create_gate_model, load_qwen_model, load_distilled_model
from data_utils import format_data, get_final_task_message

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Default arguments and global setup
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="emotion6", help="Dataset name (default: emotion6)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (default: 0)")
    parser.add_argument("--gate", type=str, default="ConcatLinearGate", help="Gate model type (default: ConcatLinearGate)")
    parser.add_argument("--depth", type=int, default=1, help="Depth for distilled model (default: 1)")
    parser.add_argument("--save", type=str, default='', help="Suffix for saved model filename (default: '')")
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Load emotion categories and initialize LabelBinarizer
emo_categories = get_emo_categories(args.dataname)
lb = LabelBinarizer()
lb.fit(emo_categories)
num_labels = len(emo_categories)

# -----------------------------
# Model Loading Functions
# -----------------------------
# Load Qwen (VLM) model and its processor
qwen_model, processor = load_qwen_model(args.dataname, device)

# Load distilled model using the Qwen model
distil_model = load_distilled_model(qwen_model, args.dataname, num_labels, args.depth, device)

# Create the Gate model (saved weights can be loaded externally if needed)
gate_model = create_gate_model(args.gate, num_labels, device)
gate_model.eval()

# -----------------------------
# Helper Functions for Demo Prediction
# -----------------------------
def preprocess_input(batch, processor, qwen_model):
    """Convert messages in a batch into a text prompt and process image info into model input."""
    prompt = processor.apply_chat_template(batch["messages"], tokenize=False)
    # Assume process_vision_info is implemented elsewhere and returns processed images.
    from qwen_vl_utils import process_vision_info  # Import locally if needed.
    qwen_img = process_vision_info(batch["messages"])[0][0]
    input_batch = processor(
        text=prompt,
        images=qwen_img,
        return_tensors="pt",
        padding=True
    ).to(qwen_model.device)
    return input_batch

def extract_qwen_emotion(data, processor, qwen_model, lb):
    """Generate text from qwen_model and extract the emotion and its reason using regex.
    
    Expects the generated text to have a dictionary string with keys 'emotion' and 'reason'.
    
    Returns:
        tuple: (predicted emotion string, predicted reason string, one-hot encoded tensor)
    """
    input_batch = preprocess_input(data, processor, qwen_model)
    generated_ids = qwen_model.generate(
        **input_batch,
        max_new_tokens=256,
        top_p=1.0,
        do_sample=True,
        temperature=0.8
    )
    # Trim generated tokens from input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_batch.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    # Look for a string like: {'emotion': 'happy', 'reason': 'the image shows smiles'}
    pattern = r"\{'emotion':\s*'(.*?)',\s*'reason':\s*'(.*?)'\}"
    match = re.search(pattern, output_text[0])
    if match:
        pred_emo = match.group(1)
        pred_reason = match.group(2)
    else:
        pred_emo = 'none'
        pred_reason = 'No explanation provided.'
    # Create one-hot encoding for the predicted emotion
    if len(lb.classes_) == 2:
        qwen_prob = torch.Tensor(np.eye(2)[lb.transform([pred_emo]).reshape(-1)]).to(qwen_model.device)
    else:
        qwen_prob = torch.Tensor(lb.transform([pred_emo])).to(qwen_model.device)
    return pred_emo, pred_reason, qwen_prob

def prepare_image_input(data, processor, device):
    """Process image input from data using the processor and return image tensor and grid info."""
    # Extract image from the data structure
    img = data['messages'][1]['content'][1]['image']
    processed = processor.image_processor(img)
    images = torch.Tensor(processed['pixel_values']).to(device)
    grid_thw = torch.LongTensor(processed['image_grid_thw']).to(device)
    return images, grid_thw

# -----------------------------
# Demo Prediction Function
# -----------------------------

def get_prompt(dataname):
    emo_categories = get_emo_categories(dataname)
    return f"""Observe the image and select the emotion category that best matches this image from the following 8 categories {emo_categories}. Additionally, provide a brief explanation for your choice. Answer in dictionary form as follows:
    {{'emotion': 'anger', 'reason': 'the image shows aggressive facial expressions'}} or {{'emotion': 'amusement', 'reason': 'the image depicts a smiling face and playful context'}}."""

def demo_predict(image):
    """
    Given an input PIL image, returns:
      - Qwen (VLM) predicted emotion and reason,
      - Distilled model predicted emotion,
      - Gate model predicted emotion.
    """
    # Build dummy data dictionary using the provided helper functions.
    cls_task_message = get_prompt(args.dataname)
    data = format_data(cls_task_message, image, None, None, return_img=True)
    
    # Qwen (VLM) Prediction: Get emotion and explanation.
    vlm_pred, vlm_reason, qwen_prob = extract_qwen_emotion(data, processor, qwen_model, lb)
    
    # Distilled Model Prediction
    images, grid_thw = prepare_image_input(data, processor, device)
    distill_output = distil_model(images, grid_thw)
    distill_pred = lb.inverse_transform(distill_output.cpu())[0]
    
    # Gate Model Prediction (combining distilled output and Qwen probabilities)
    gate_output = gate_model(distill_output, qwen_prob)
    gate_pred = lb.inverse_transform(gate_output.cpu())[0]
    
    return f"Emotion: {vlm_pred}\nReason: {vlm_reason}", f"{distill_pred}", f"{gate_pred}"

# -----------------------------
# Gradio Interface
# -----------------------------
iface = gr.Interface(
    fn=demo_predict,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=[
        gr.Textbox(label="VLM"),
        gr.Textbox(label="Distill Module"),
        gr.Textbox(label="EmoVLM-KD")
    ], 
    title="EmoVLM-KD Demo",
    description="Upload an image to see emotion predictions from the Qwen (VLM) module (with reason), distilled, and gate models. (Dataset defaults to emotion6)"
)

if __name__ == "__main__":
    iface.launch()
