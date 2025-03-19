import torch
import random
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split 
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor
import torchvision.transforms as transforms
import json

######################################
# configuration
###################################### 
PROJECT_DIR = os.path.dirname(__file__) 
with open(os.path.join(PROJECT_DIR, 'config', 'config.json'), "r") as f:
    config = json.load(f)  
INPUT_SIZE = config['input_size']
VIT_NAME = config['vit_model']
VLM_NAME = config['vlm_model'] 
SAVE_DIR = config['output_dir']
for model in ['vlm', 'vit', 'gate', 'distill']:
    if not os.path.exists(os.path.join(SAVE_DIR, model)):
        os.makedirs(os.path.join(SAVE_DIR, model)) 
VLM_ANN_DIR = config['vlm_annotation_dir']
 

FEATURE_EXTRACTOR = ViTFeatureExtractor.from_pretrained(VIT_NAME) 
TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=FEATURE_EXTRACTOR.image_mean, std=FEATURE_EXTRACTOR.image_std)
    ]) 
SYSTEM_MESSAGE= """You are an AI visual assistant, and you are seeing a single image.""" 



def get_img_dir(dataname):
    if dataname == 'FI':
        return 'images/FI'
    if dataname == 'flickr':
        return 'images/flickr'
    if dataname == 'instagram':
        return 'images/instagram'
    if dataname == 'abstract':
        return 'images/abstract'
    if dataname == 'artphoto':
        return 'images/artphoto'
    if dataname == 'emoset':
        return 'images/EmoSet'
    if dataname == 'emotion6':
        return 'images/Emotion6'
    
def get_emo_categories(dataname):
    return os.listdir(os.path.join(get_img_dir(dataname), 'train'))     
    
class ImageDataset(Dataset):
    def __init__(self, img_dir, mode='train'):  
        self.img_dir = img_dir 
        
        # Data augmentation and normalization for training
        if mode == 'train': 
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(INPUT_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=FEATURE_EXTRACTOR.image_mean, std=FEATURE_EXTRACTOR.image_std)
            ])
        elif mode == 'val': 
            self.transform = transforms.Compose([
                transforms.Resize(INPUT_SIZE),
                transforms.CenterCrop(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=FEATURE_EXTRACTOR.image_mean, std=FEATURE_EXTRACTOR.image_std)
            ])
        elif mode == 'test':
            self.transform = transforms.Compose([
                transforms.Resize(INPUT_SIZE),
                transforms.CenterCrop(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=FEATURE_EXTRACTOR.image_mean, std=FEATURE_EXTRACTOR.image_std)
            ]) 
        self.target_dir = os.path.join(self.img_dir, mode)
        emo_categories = sorted(os.listdir(self.target_dir))
        self.emo_dict = {j:i for i,j in enumerate(emo_categories)}  

        self.img_list, self.label_list = [], [] 
        for emo in os.listdir(self.target_dir):   
            self.img_list += [os.path.join(self.target_dir, emo, i) for i in os.listdir(os.path.join(self.target_dir, emo))]
            self.label_list += [self.emo_dict[emo]] * len(os.listdir(os.path.join(self.target_dir, emo))) 
        
    def __len__(self):
        return len(self.img_list)
 
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        label = self.label_list[idx] 
        return image, label, img_path
    
    
def load_img_dataset(dataname):
    img_dir = get_img_dir(dataname)
    if dataname in ['abstract', 'artphoto', 'emotion6', 'emoset']:
        dataset = ImageDataset(img_dir, mode='train') 
        labels = np.array(dataset.label_list)   
        indices = np.arange(len(dataset)) 

        train_indices, val_indices = train_test_split(
            indices, 
            test_size=0.1, 
            stratify=labels, 
            random_state=42 
        )
 
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)  
        test_dataset = ImageDataset(img_dir, mode='test') 
    else:        
        train_dataset = ImageDataset(img_dir, mode='train') 
        val_dataset = ImageDataset(img_dir, mode='val') 
        test_dataset = ImageDataset(img_dir, mode='test') 
    return train_dataset, val_dataset, test_dataset
 

def get_final_task_message(dataset):
    if dataset in ['FI', 'emoset']:
        emo_categories = ['amusement', 'excitement', 'sadness', 'fear', 'contentment', 'anger', 'disgust', 'awe']
        cls_task_message = f"""Observe the image and select the emotion category that best matches this image from the following 8 categories {emo_categories}. Answer in dictionary form as follows
        {{'emotion':'anger'}} or {{'emotion':'amusement'}}
        """
    elif dataset == 'emotion6':
        emo_categories = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        cls_task_message = f"""Observe the image and select the emotion category that best matches this image from the following 6 categories {emo_categories}. Answer in dictionary form as follows
        {{'emotion':'anger'}} or {{'emotion':'disgust'}}
        """
    elif dataset in ['flickr', 'instagram']:
        emo_categories = ['positive', 'negative']
        cls_task_message = f"""Observe the image and select the emotion category that best matches this image from the following 2 categories {emo_categories}. Answer in dictionary form as follows
        {{'emotion':'positive'}} or {{'emotion':'negative'}}
        """

def resize_image(image):
    width, height = image.size

    max_size = 1024

    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image

def format_data(message, image, label, file, return_img=True, vlm=False):
    if vlm:
        return {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": [
                    {"type": "text", "text": message},
                    {"type": "image", "image": image}
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": label}]},
            ],
        }
    if return_img:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                        {"type": "image", "image": resize_image(image)},
                    ],
                },
            ],
            "gt": label,
            "img": TEST_TRANSFORM(image),
            "file": file
        }
    else:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                    ],
                },
            ],
            "gt": label,
            "img": TEST_TRANSFORM(image),
            "file": file
        }

def emo8_to_emo2(label):
    if label in ['content', 'awe', 'amusement', 'excitement']:
        return 'positive'
    elif label in ['sad', 'anger', 'fear', 'disgust']:
        return 'negative'
    else:
        print(label)
        
class VLMDataset(Dataset):
    def __init__(self, img_dir, cls_task_message, mode='train', cache=None):
        self.target_dir = os.path.join(img_dir, mode)
        self.dataset_list = [] 
        if cache is not None: 
            for emo in os.listdir(self.target_dir):
                emo_dir = os.path.join(self.target_dir, emo) 
                self.dataset_list += [
                    format_data(cls_task_message, Image.open(os.path.join(emo_dir, file)).convert("RGB"), emo, file, return_img=True)
                    for file in tqdm(os.listdir(emo_dir), desc=f"Processing {emo} in {mode}") if file in cache
                ]
                
        else: 
            for emo in os.listdir(self.target_dir): 
                emo_dir = os.path.join(self.target_dir, emo)  
                self.dataset_list += [
                    format_data(cls_task_message, Image.open(os.path.join(emo_dir, file)).convert("RGB"), emo, file, return_img=True)
                    for file in tqdm(os.listdir(emo_dir), desc=f"Processing {emo} in {mode}")
                ]
        if mode == 'train':
            random.shuffle(self.dataset_list) 
    
    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        return self.dataset_list[idx]
    

def load_vlm_dataset(dataname, test=False, cache=None):
    img_dir = get_img_dir(dataname)
    cls_task_message = get_final_task_message(dataname)
    if test:
        return None, None, VLMDataset(img_dir, cls_task_message, mode='test')
    
    if dataname in ['abstract', 'artphoto', 'emotion6', 'emoset']:
        print('data', dataname)
        dataset = VLMDataset(img_dir, cls_task_message, mode='train', cache=cache)
        labels = np.array([i['gt'] for i in dataset])   
        indices = np.arange(len(dataset)) 

        train_indices, val_indices = train_test_split(
            indices, 
            test_size=0.1, 
            stratify=labels, 
            random_state=42
        )

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)  
        test_dataset = VLMDataset(img_dir, cls_task_message, mode='test')
    else:        
        train_dataset = VLMDataset(img_dir, cls_task_message, mode='train', cache=cache)
        val_dataset = VLMDataset(img_dir, cls_task_message, mode='val')
        test_dataset = VLMDataset(img_dir, cls_task_message, mode='test') 
    
    return train_dataset, val_dataset, test_dataset

def save_model(model, model_type, dataname, save_suffix, base_save_dir):
    """Save the given model's state dictionary with a dynamic filename.

    Args:
        model (nn.Module): The model to be saved.
        model_type (str): The model type or prefix (e.g., "emovlm-kd", "distill", "vit", etc.).
        dataname (str): The dataset name to include in the filename.
        save_suffix (str): Additional suffix for the filename. If empty, the filename is formatted as
                           "{model_type}_{dataname}.pth"; otherwise as "{model_type}_{dataname}_{save_suffix}.pth".
        base_save_dir (str): The base directory where the model will be saved.

    Returns:
        None
    """
    if save_suffix != 'False':
        save_dir = os.path.join(base_save_dir, model_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_suffix == '':
            save_path = os.path.join(save_dir, f"{model_type}_{dataname}.pth")
        else:
            save_path = os.path.join(save_dir, f"{model_type}_{dataname}_{save_suffix}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"{model_type} model saved to {save_path}")
