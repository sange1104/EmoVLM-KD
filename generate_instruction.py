import os
import json
import base64
from tqdm import tqdm
from openai import OpenAI 
from data_utils import IMAGE_DIR, VLM_ANN_DIR

def load_prompt_template(prompt_file_path):
    """
    Load and return the prompt template as a string from the specified file.
    
    The prompt template file should include a placeholder {emotion} where the emotion label will be substituted.
    """
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_prompt(emo, prompt_template):
    """
    Replace the {emotion} placeholder in the prompt template with the actual emotion value.
    """
    return prompt_template.format(emotion=emo)

def encode_image(image_path):
    """
    Encode the image file to a base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_answer(client, user_prompt, encoded_img):
    """
    Call the OpenAI API with the image and text prompt, and return the generated response.
    """
    system_prompt = 'You are an expert in analyzing emotions expressed in images.'
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_img}"}}
            ]}
        ]
    )
    return response.choices[0].message.content

def process_images(api_key, dataname, prompt_file_path):
    """
    Process all images by iterating through directories, calling the OpenAI API for each image,
    and saving the results as a JSON file.
    
    Parameters:
        api_key (str): OpenAI API key.
        dataname (str): Dataset name (used to build paths based on image_dir and vlm_annotation_dir from data_utils).
        prompt_file_path (str): Path to the prompt template file.
    """
    client = OpenAI(api_key=api_key)
    prompt_template = load_prompt_template(prompt_file_path)
    annotation_list = []
    
    # Directory containing training images.
    target_dir = os.path.join(IMAGE_DIR, dataname, 'train')
    # Directory to save results (create if it does not exist). 
    os.makedirs(VLM_ANN_DIR, exist_ok=True)
    save_file_path = os.path.join(VLM_ANN_DIR, f'{dataname}_annotations_train.json')
    
    # Iterate through each emotion folder in target_dir.
    for emo_dir_name in tqdm(os.listdir(target_dir), desc="Processing emotion folders"):
        emo_dir = os.path.join(target_dir, emo_dir_name)
        if not os.path.isdir(emo_dir):
            continue
        for file_name in os.listdir(emo_dir):
            if not file_name.lower().endswith('jpg'):
                continue
            img_path = os.path.join(emo_dir, file_name)
            # Extract emotion label from the file name (e.g., "anger_001.jpg" -> "anger"). 
            user_prompt = get_prompt(emo_dir_name, prompt_template)
            encoded_img = encode_image(img_path)
            response = get_answer(client, user_prompt, encoded_img)
            annotation_dict = {
                'gpt_response': response,
                'img': file_name,
                'img_path': img_path,
                'label': emo_dir_name
            }
            annotation_list.append(annotation_dict)
    
    # Save the final results as a JSON file.
    with open(save_file_path, "w", encoding="utf-8") as json_file:
        json.dump(annotation_list, json_file, ensure_ascii=False, indent=4)
    
    return annotation_list

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process images and generate annotations using GPT-4o-mini.")
    parser.add_argument("--dataname", type=str, required=True, help="Dataset name (e.g., emotion6)")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--prompt_file", type=str, default="prompt_template.txt", help="Path to the prompt template file")
    args = parser.parse_args()
    
    process_images(args.api_key, args.dataname, args.prompt_file)
