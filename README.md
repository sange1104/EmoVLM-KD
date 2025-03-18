# EmoVLM-KD
Official code for the paper "EmoVLM-KD: Fusing Distilled Expertise with Vision-Language Models for Visual Emotion Analysis"
![Model architecture](architecture_v10.jpg)

## Setup 
**Environmental setup**
```
# Clone the repository
git clone 

# Install dependencies
pip install -r requirements.txt
```

**Dataset**

We assume that the image emotion dataset is located as follows. For some datasets, if a validation dataset is not available, they may consist of only train and test sets.
- images  
  - FI  
    - train  
      - amusement  
      - anger  
      - sadness  
      - fear
      - ...
    - val  
    - test
  - ...

## Emotion Instruction Data Generation 
```
python generate_instructions.py --dataname emoset --api_key <api_key>
```



## Vision-language model training
```
python train_vlm.py --dataname emoset --gpu 1
```



## Knowledge distillation

```
python train_distillation.py --dataname emoset --gpu 1
```

## Training a gate module

```
python train_gate.py --dataname emoset --gpu 1
```

## Inference EmoVLM-KD
