import argparse
import os    
import re, ast
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelBinarizer
from transformers import (
    AutoModelForVision2Seq, AutoProcessor,
    ViTForImageClassification
)
from data_utils import load_vlm_dataset, get_emo_categories, VIT_NAME, VLM_NAME, SAVE_DIR, save_model
from distill_model import QwenDistillationModel

def prepare_qwen_input(data, processor, device):
    """Prepare the Qwen input by processing the image from data and returning the tensor and grid info."""
    img_input = data['messages'][1]['content'][1]['image']
    processed = processor.image_processor(img_input)
    imgs = torch.Tensor(processed['pixel_values']).to(device)
    grid_thw = torch.LongTensor(processed['image_grid_thw']).to(device)
    return imgs, grid_thw

def train_distillation(model, vit_model, processor, dataset, lb, optimizer, device, emo_categories, alpha=0.5, temperature=2.0):
    """Perform one training epoch for distillation, computing KL divergence and cross-entropy losses."""
    model.train()
    total_loss, correct_qwen, correct_vit, total = 0, 0, 0, 0

    for data in tqdm(dataset, desc="Distillation Training"):
        # Process Qwen input
        qwen_imgs, grid_thw = prepare_qwen_input(data, processor, device)
        # Get ground truth label
        labels_emo = data['gt']
        label_index = emo_categories.index(labels_emo)
        labels = torch.tensor(label_index, device=device).unsqueeze(0)

        optimizer.zero_grad()
        # Obtain Qwen model output
        qwen_outputs = model(qwen_imgs, grid_thw)
        # Obtain ViT model output using image from data['img']
        vit_outputs = vit_model(data['img'].unsqueeze(0).to(device)).logits

        # Temperature scaling for distillation loss
        qwen_log_probs = F.log_softmax(qwen_outputs / temperature, dim=-1)
        vit_probs = F.softmax(vit_outputs / temperature, dim=-1).detach()
        loss_kl = F.kl_div(qwen_log_probs, vit_probs, reduction="batchmean") * (temperature ** 2)
        loss_ce = nn.CrossEntropyLoss()(qwen_outputs, labels)
        loss = alpha * loss_kl + (1 - alpha) * loss_ce

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)
        # Inverse transform predictions using LabelBinarizer
        qwen_predicted = lb.inverse_transform(qwen_outputs.cpu())[0]
        vit_predicted = lb.inverse_transform(vit_outputs.cpu())[0]
        if qwen_predicted == labels_emo:
            correct_qwen += 1
        if vit_predicted == labels_emo:
            correct_vit += 1

    return total_loss / total, correct_qwen / total, correct_vit / total

def evaluate_distillation(model, vit_model, processor, dataset, lb, device, emo_categories, alpha=0.5):
    """Evaluate the distillation model on the dataset and return average loss and accuracies."""
    model.eval()
    total_loss, correct_qwen, correct_vit, total = 0, 0, 0, 0

    with torch.no_grad():
        for data in tqdm(dataset, desc="Distillation Evaluation"):
            # Process Qwen input
            qwen_imgs, grid_thw = prepare_qwen_input(data, processor, device)
            labels_emo = data['gt']
            label_index = emo_categories.index(labels_emo)
            labels = torch.tensor(label_index, device=device).unsqueeze(0)

            qwen_outputs = model(qwen_imgs, grid_thw)
            vit_outputs = vit_model(data['img'].unsqueeze(0).to(device)).logits

            qwen_log_probs = F.log_softmax(qwen_outputs, dim=-1)
            vit_probs = F.softmax(vit_outputs, dim=-1).detach()
            loss_kl = F.kl_div(qwen_log_probs, vit_probs, reduction="batchmean")
            loss_ce = nn.CrossEntropyLoss()(qwen_outputs, labels)
            loss = alpha * loss_kl + (1 - alpha) * loss_ce

            total_loss += loss.item()
            total += labels.size(0)
            qwen_predicted = lb.inverse_transform(qwen_outputs.cpu())[0]
            vit_predicted = lb.inverse_transform(vit_outputs.cpu())[0]
            if qwen_predicted == labels_emo:
                correct_qwen += 1
            if vit_predicted == labels_emo:
                correct_vit += 1

    return total_loss / total, correct_qwen / total, correct_vit / total

def main():
    """Main function to run distillation training and evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--save", type=str, default='')
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=5)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load emotion categories and initialize LabelBinarizer
    emo_categories = get_emo_categories(args.dataname) 
    lb = LabelBinarizer()
    lb.fit(emo_categories)
    num_labels = len(emo_categories)

    # Initialize and load pre-trained ViT model
    vit_model = ViTForImageClassification.from_pretrained(VIT_NAME)
    vit_model.classifier = nn.Linear(768, num_labels)
    vit_path = f"{SAVE_DIR}/vit/vit_{args.dataname}.pth"
    vit_model.load_state_dict(torch.load(vit_path))
    vit_model = vit_model.to(device)

    # Load Qwen model and enable adapters
    qwen_model = AutoModelForVision2Seq.from_pretrained(
        VLM_NAME,
        load_in_4bit=False,
        device_map="cuda",
        torch_dtype=torch.float32
    )
    adapter_path = f"{SAVE_DIR}/vlm/qwen7b_{args.dataname}"
    qwen_model.load_adapter(adapter_path)
    qwen_model.enable_adapters()
    processor = AutoProcessor.from_pretrained(VLM_NAME)

    # Freeze Qwen model parameters
    for param in qwen_model.parameters():
        param.requires_grad = False

    # Load dataset for distillation
    train_dataset, val_dataset, test_dataset = load_vlm_dataset(args.dataname)

    # Initialize distillation model
    distil_model = QwenDistillationModel(qwen_model, num_labels, depth=args.depth).to(device)
    for param in qwen_model.parameters():
        param.requires_grad = False

    optimizer_distil = optim.Adam(distil_model.distillation_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_epochs = args.epoch

    # Distillation training loop
    for epoch in range(num_epochs):
        train_loss, train_acc_distill, train_acc_vit = train_distillation(
            distil_model, vit_model, processor, train_dataset, lb, optimizer_distil, device, emo_categories, alpha=args.alpha
        )
        print(f"[Distil Train] Epoch {epoch+1}/{num_epochs}: Loss {train_loss:.4f}, "
              f"Distil Acc {train_acc_distill:.4f}, ViT Acc {train_acc_vit:.4f}")
        
        val_loss, val_acc_distill, val_acc_vit = evaluate_distillation(
            distil_model, vit_model, processor, val_dataset, lb, device, emo_categories, alpha=args.alpha
        )
        print(f"[Distil Val] Epoch {epoch+1}/{num_epochs}: Loss {val_loss:.4f}, "
              f"Distil Acc {val_acc_distill:.4f}, ViT Acc {val_acc_vit:.4f}")

    # Final test evaluation
    test_loss, test_acc_distill, test_acc_vit = evaluate_distillation(
        distil_model, vit_model, processor, test_dataset, lb, device, emo_categories, alpha=args.alpha
    )
    print(f"[Distil Test]: Loss {test_loss:.4f}, Distil Acc {test_acc_distill:.4f}, ViT Acc {test_acc_vit:.4f}")

    # Save distilled model if required
    save_model(distil_model, "distill", args.dataname, args.save, SAVE_DIR) 
    
if __name__ == "__main__":
    main()