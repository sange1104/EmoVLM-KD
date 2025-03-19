import argparse
import os 
import torch
from data_utils import load_vlm_dataset, get_emo_categories, VIT_NAME, VLM_NAME, SAVE_DIR
from distill_model import QwenDistillationModel
from transformers import AutoModelForVision2Seq, AutoProcessor
from sklearn.preprocessing import LabelBinarizer
from train_gate import (
    create_gate_model, load_qwen_model, load_distilled_model, evaluate_gate
) 

def load_saved_gate_model(gate_model, dataname, save_suffix, base_save_dir, model_type_prefix="gate"):
    """Load the saved gate model state dictionary from a file."""
    save_dir = os.path.join(base_save_dir, "gate")
    if save_suffix == '':
        save_path = os.path.join(save_dir, f"{model_type_prefix}_{dataname}.pth")
    else:
        save_path = os.path.join(save_dir, f"{model_type_prefix}_{dataname}_{save_suffix}.pth")
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Gate model file not found: {save_path}")
    gate_model.load_state_dict(torch.load(save_path))
    return gate_model

def test_main():
    """Load the saved Qwen, distilled, and gate models and evaluate on the test dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, required=True, help="Dataset name")
    parser.add_argument("--gpu", type=int, required=True, help="GPU id to use")
    parser.add_argument("--save", type=str, default='', help="Suffix for the saved model filename")
    parser.add_argument("--depth", type=int, default=1, help="Depth for distilled model")
    parser.add_argument("--gate", type=str, default='ConcatLinearGate', help="Gate model type")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load emotion categories and initialize label binarizer
    emo_categories = get_emo_categories(args.dataname)
    lb = LabelBinarizer()
    lb.fit(emo_categories)
    num_labels = len(emo_categories)

    # Load test dataset only
    _, _, test_dataset = load_vlm_dataset(args.dataname)

    # Load Qwen model and its processor
    qwen_model, processor = load_qwen_model(args.dataname, device)

    # Load distilled model using the Qwen model
    distil_model = load_distilled_model(qwen_model, args.dataname, num_labels, args.depth, device)

    # Create the gate model and load its saved weights
    gate_model = create_gate_model(args.gate, num_labels, device)
    gate_model = load_saved_gate_model(gate_model, args.dataname, args.save, SAVE_DIR)

    # Evaluate on the test dataset using the evaluate_gate helper function
    test_loss, test_acc = evaluate_gate(gate_model, distil_model, qwen_model, processor, test_dataset, lb, device)
    print(f"[Gate Test]: Loss {test_loss:.4f}, Acc {test_acc:.4f}")

if __name__ == "__main__":
    test_main()
