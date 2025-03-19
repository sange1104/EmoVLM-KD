import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import argparse
from data_utils import load_img_dataset, get_emo_categories, VIT_NAME, SAVE_DIR, save_model
from transformers import ViTForImageClassification

def get_model(modelname, num_labels):
    """Initialize the model based on the given model name."""
    if modelname == 'ViT': 
        model = ViTForImageClassification.from_pretrained(VIT_NAME)
        model.classifier = nn.Linear(768, num_labels)
    elif modelname == 'vgg':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_labels)
    elif modelname == 'resnet':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
    return model

def train(model, train_loader, criterion, optimizer, device):
    """Training loop for one epoch."""
    model.train()
    running_loss, correct, total = 0, 0, 0

    for data in tqdm(train_loader): 
        images, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        model_output = model(images)
        outputs = model_output.logits if hasattr(model_output, 'logits') else model_output 
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = outputs.argmax(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100 * correct / total

def evaluate(model, data_loader, criterion, device):
    """Evaluation loop for validation or test data."""
    model.eval()
    loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for data in tqdm(data_loader): 
            images, labels = data[0].to(device), data[1].to(device)

            model_output = model(images)
            outputs = model_output.logits if hasattr(model_output, 'logits') else model_output 

            loss += criterion(outputs, labels).item()
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return loss / len(data_loader), 100 * correct / total

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--model", type=str, default="ViT")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Set device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets and dataloaders
    train_dataset, val_dataset, test_dataset = load_img_dataset(args.dataname)    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    num_labels = len(get_emo_categories(args.dataname))
    model = get_model(args.model, num_labels).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_epochs = args.epoch
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"[ViT Train] Epoch {epoch+1}/{num_epochs}: Loss {train_loss:.4f}, Acc {train_acc:.4f}")
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"[ViT Val] Epoch {epoch+1}/{num_epochs}: Loss {val_loss:.4f}, Acc {val_acc:.4f}")
        
    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[ViT Test]: Loss {test_loss:.4f}, Acc {test_acc:.4f}")

    # Save trained model if required 
    save_model(model, "vit", args.dataname, args.save, SAVE_DIR) 

if __name__ == "__main__":
    main()
