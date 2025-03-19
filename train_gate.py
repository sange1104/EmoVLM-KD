import argparse
import os  
import pandas as pd
from tqdm import tqdm
from PIL import Image
import seaborn
import torch
import torch.nn as nn
from torch.utils.data import Subset
import pickle
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
import random
import torchvision.transforms as transforms
import random
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
import json
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from trl import SFTConfig
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from sklearn.model_selection import train_test_split
from functools import partial
from torch.utils.data import default_collate
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math, re, ast
from tqdm import tqdm 
import random
import numpy as np
from collections import Counter 
from transformers import ViTForImageClassification, ViTFeatureExtractor
from data_utils import load_vlm_dataset, get_emo_categories, VIT_NAME, VLM_NAME, SAVE_DIR, save_model
from distill_model import QwenDistillationModel

def preprocess_input(batch, processor, qwen_model):
    """Convert messages in a batch into a text prompt and process image info into model input."""
    prompt = processor.apply_chat_template(batch["messages"], tokenize=False)
    # Assume process_vision_info is implemented elsewhere
    qwen_img = process_vision_info(batch["messages"])[0][0]
    input_batch = processor(
        text=prompt,
        images=qwen_img,
        return_tensors="pt",
        padding=True
    ).to(qwen_model.device)
    return input_batch

def extract_qwen_emotion(data, processor, qwen_model, lb):
    """Generate text from qwen_model and extract the emotion string using regex.
    
    Returns the predicted emotion and its one-hot encoded tensor.
    """
    input_batch = preprocess_input(data, processor, qwen_model)
    generated_ids = qwen_model.generate(
        **input_batch,
        max_new_tokens=256,
        top_p=1.0,
        do_sample=True,
        temperature=0.8
    )
    # Trim generated tokens from the input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_batch.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    match = re.search(r"{'emotion':\s*'.*?'}", output_text[0])
    try:
        pred_emo = ast.literal_eval(match.group(0))['emotion']
    except Exception:
        pred_emo = 'none'
    
    # Handle binary vs. multi-class encoding
    if len(lb.classes_) == 2:
        qwen_prob = torch.Tensor(np.eye(2)[lb.transform([pred_emo]).reshape(-1)]).to(qwen_model.device)
    else:
        qwen_prob = torch.Tensor(lb.transform([pred_emo])).to(qwen_model.device)
    return pred_emo, qwen_prob

def prepare_image_input(data, processor, device):
    """Process image input from data using the processor and return image tensor and grid info."""
    # Extract image from a fixed location in the data structure
    img = data['messages'][1]['content'][1]['image']
    processed = processor.image_processor(img)
    images = torch.Tensor(processed['pixel_values']).to(device)
    grid_thw = torch.LongTensor(processed['image_grid_thw']).to(device)
    return images, grid_thw
 
class ConcatLinearGate(nn.Module):
    def __init__(self, num_emotions):
        """Concatenation-based linear gating."""
        super().__init__()
        self.fc = nn.Linear(num_emotions * 2, num_emotions)

    def forward(self, v1, v2):
        """Concatenate inputs and apply a linear layer."""
        x = torch.cat([v1, v2], dim=1)
        return self.fc(x)

class BilinearGate(nn.Module):
    def __init__(self, num_emotions):
        """Bilinear gating combining two inputs."""
        super().__init__()
        self.bilinear = nn.Bilinear(num_emotions, num_emotions, num_emotions)
        self.fc = nn.Linear(num_emotions, num_emotions)

    def forward(self, v1, v2):
        """Apply bilinear layer followed by a linear transformation."""
        x = self.bilinear(v1, v2)
        return self.fc(x)

class MoEGate(nn.Module):
    def __init__(self, num_emotions, num_experts=3):
        """MoE gating with multiple experts.
        
        Args:
            num_emotions: Dimension of emotion features.
            num_experts: Number of experts.
        """
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(num_emotions * 2, num_emotions) for _ in range(num_experts)])
        self.gate = nn.Linear(num_emotions * 2, num_experts)
        
    def forward(self, v1, v2):
        """Combine two inputs via expert networks weighted by a gating mechanism."""
        x = torch.cat([v1, v2], dim=1)
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=1)
        expert_outputs = [expert(x) for expert in self.experts]
        outputs = torch.stack(expert_outputs, dim=0)
        gate_weights = gate_weights.unsqueeze(2).transpose(0, 1)
        weighted_expert = outputs * gate_weights
        output = weighted_expert.sum(dim=0)
        return output

class DynamicWeightingGate(nn.Module):
    def __init__(self, num_emotions):
        """Dynamic weighting gate using softmax-based weighting."""
        super().__init__()
        self.gate = nn.Linear(num_emotions * 2, 2)
        self.fc = nn.Linear(num_emotions, num_emotions)

    def forward(self, v1, v2):
        """Compute weighted sum of v1 and v2 and apply a final linear layer."""
        x = torch.cat([v1, v2], dim=1)
        weights = F.softmax(self.gate(x), dim=1)
        weighted_sum = weights[:, 0].unsqueeze(1) * v1 + weights[:, 1].unsqueeze(1) * v2
        return self.fc(weighted_sum)

class CrossGatingGate(nn.Module):
    def __init__(self, num_emotions):
        """Cross gating where each input gates the other."""
        super().__init__()
        self.gate1 = nn.Linear(num_emotions, num_emotions)
        self.gate2 = nn.Linear(num_emotions, num_emotions)
        self.fc = nn.Linear(num_emotions, num_emotions)

    def forward(self, v1, v2):
        """Apply cross gating and combine the gated inputs with a linear layer."""
        g1 = torch.sigmoid(self.gate1(v2))
        g2 = torch.sigmoid(self.gate2(v1))
        gated_v1 = g1 * v1
        gated_v2 = g2 * v2
        combined = gated_v1 + gated_v2
        return self.fc(combined)
 
def train_gate(model, distil_model, qwen_model, processor, dataset, lb, optimizer, device):
    """Training loop for the gate model."""
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for data in tqdm(dataset, desc="Gate Training"):
        # Process image input for distillation model
        images, grid_thw = prepare_image_input(data, processor, device)
        labels_emo = data['gt']
        # Convert label to tensor using LabelBinarizer and take argmax
        labels = torch.LongTensor(lb.transform([labels_emo])).argmax(-1).to(device)

        optimizer.zero_grad()
        distill_prob = distil_model(images, grid_thw)
        # Extract Qwen emotion probabilities
        _, qwen_prob = extract_qwen_emotion(data, processor, qwen_model, lb)
        gate_output = model(distill_prob, qwen_prob)
        loss = nn.CrossEntropyLoss()(gate_output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        gate_predicted = lb.inverse_transform(gate_output.cpu())[0]
        if gate_predicted == labels_emo:
            correct += 1
        total += 1 

    return total_loss / total, correct / total

def evaluate_gate(model, distil_model, qwen_model, processor, dataset, lb, device):
    """Evaluation loop for the gate model."""
    model.eval()
    total_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for data in tqdm(dataset, desc="Gate Evaluation"):
            images, grid_thw = prepare_image_input(data, processor, device)
            labels_emo = data['gt']
            labels = torch.LongTensor(lb.transform([labels_emo])).argmax(-1).to(device)

            distill_prob = distil_model(images, grid_thw)
            _, qwen_prob = extract_qwen_emotion(data, processor, qwen_model, lb)
            gate_output = model(distill_prob, qwen_prob)
            loss = nn.CrossEntropyLoss()(gate_output, labels)

            total_loss += loss.item()
            gate_predicted = lb.inverse_transform(gate_output.cpu())[0]
            if gate_predicted == labels_emo:
                correct += 1
            total += 1

    return total_loss / total, correct / total

def str2bool(value):
    """Convert a string to a boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1'}:
        return True
    elif value.lower() in {'false', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_gate_model(gate, num_labels, device):
    """Create and return the gate model based on the given type.

    Args:
        gate (str): The type of gate model to create.
        num_labels (int): The number of emotion categories.
        device (torch.device): The device to which the model should be moved.

    Returns:
        nn.Module: The gate model moved to the specified device.

    Raises:
        ValueError: If an unknown gate model type is provided.
    """
    if gate == 'ConcatLinearGate':
        return ConcatLinearGate(num_labels).to(device)
    elif gate == 'BilinearGate':
        return BilinearGate(num_labels).to(device)
    elif gate == 'MoEGate':
        return MoEGate(num_labels).to(device)
    elif gate == 'DynamicWeightingGate':
        return DynamicWeightingGate(num_labels).to(device)
    elif gate == 'CrossGatingGate':
        return CrossGatingGate(num_labels).to(device)
    else:
        raise ValueError("Unknown gate model type.")

def load_qwen_model(dataname, device):
    """Load the Qwen model with adapters enabled and return the model and its processor.
    
    Args:
        dataname (str): The dataset name used for constructing the adapter path.
        device (torch.device): The device to load the model on.
    
    Returns:
        tuple: (qwen_model, processor)
    """
    adapter_path = f"{SAVE_DIR}/vlm/qwen7b_{dataname}"
    qwen_model = AutoModelForVision2Seq.from_pretrained(
        VLM_NAME,
        load_in_4bit=False,
        device_map="cuda",
        torch_dtype=torch.float32
    )
    qwen_model.load_adapter(adapter_path)
    qwen_model.enable_adapters()
    processor = AutoProcessor.from_pretrained(VLM_NAME)
    
    # Freeze Qwen model parameters and set to evaluation mode
    for param in qwen_model.parameters():
        param.requires_grad = False
    qwen_model.eval()
    
    return qwen_model, processor

def load_distilled_model(qwen_model, dataname, num_labels, depth, device):
    """Initialize and load the distilled model using the provided Qwen model.
    
    Args:
        qwen_model (nn.Module): The loaded Qwen model.
        dataname (str): The dataset name used for constructing the distillation path.
        num_labels (int): Number of emotion categories.
        depth (int): Depth parameter for the distilled model.
        device (torch.device): The device to load the distilled model on.
    
    Returns:
        nn.Module: The distilled model in evaluation mode.
    """
    distil_model = QwenDistillationModel(qwen_model, num_labels, depth=depth).to(device)
    distill_path = f"{SAVE_DIR}/distill/distill_{dataname}.pth"
    distil_model.load_state_dict(torch.load(distill_path))
    
    # Freeze distilled model parameters and set to evaluation mode
    for param in distil_model.parameters():
        param.requires_grad = False
    distil_model.eval()
    
    return distil_model

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--save", type=str, default='')
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gate", type=str, default='ConcatLinearGate')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load emotion categories and initialize LabelBinarizer
    emo_categories = get_emo_categories(args.dataname)
    lb = LabelBinarizer()
    lb.fit(emo_categories)
    num_labels = len(emo_categories) 

    # Load dataset for distillation
    train_dataset, val_dataset, test_dataset = load_vlm_dataset(args.dataname)

    # Load Qwen model and enable adapters
    qwen_model, processor = load_qwen_model(args.dataname, device)
    distil_model = load_distilled_model(qwen_model, args.dataname, num_labels, args.depth, device)

    # Initialize the gate model based on argument
    gate_model = create_gate_model(args.gate, num_labels, device) 
    optimizer = optim.Adam(gate_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_epochs = args.epoch

    # Training loop for gate model
    for epoch in range(num_epochs):
        train_loss, train_acc = train_gate(
            gate_model, distil_model, qwen_model, processor, train_dataset, lb, optimizer, device
        )
        print(f"[Gate Train] Epoch {epoch+1}/{num_epochs}: Loss {train_loss:.4f}, Acc {train_acc:.4f}")
        val_loss, val_acc = evaluate_gate(
            gate_model, distil_model, qwen_model, processor, val_dataset, lb, device
        )
        print(f"[Gate Val] Epoch {epoch+1}/{num_epochs}: Loss {val_loss:.4f}, Acc {val_acc:.4f}")

    test_loss, test_acc = evaluate_gate(
        gate_model, distil_model, qwen_model, processor, test_dataset, lb, device
    )
    print(f"[Gate Test]: Loss {test_loss:.4f}, Acc {test_acc:.4f}")

    # Save the gate model if required
    save_model(gate_model, "gate", args.dataname, args.save, SAVE_DIR)

if __name__ == "__main__":
    main()