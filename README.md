# EmoVLM-KD
Official code for the paper "EmoVLM-KD: Fusing Distilled Expertise with Vision-Language Models for Visual Emotion Analysis"


## Abstract 
Visual emotion analysis, which has gained considerable attention in the field of affective computing, aims to predict the dominant emotions conveyed by an image. Despite advancements in visual emotion analysis with the emergence of vision-language models, we observed that instruction-tuned vision-language models and conventional vision models exhibit complementary strengths in visual emotion analysis, as vision-language models excel in certain cases, whereas vision models perform better in others. This finding highlights the need to integrate these capabilities to enhance the performance of visual emotion analysis. To bridge this gap, we propose EmoVLM-KD, an instruction-tuned vision-language model augmented with a lightweight module distilled from conventional vision models. Instead of deploying both models simultaneously, which incurs high computational costs, we transfer the predictive patterns of a conventional vision model into the vision-language model using a knowledge distillation framework. Our approach first fine-tunes a vision-language model on emotion-specific instruction data and then attaches a distilled module to its visual encoder while keeping the vision-language model frozen. Predictions from the vision language model and distilled modules are effectively balanced by the gate module, which subsequently generate the final outcome. 

<div align="center">
  <img src="figure/architecture_v10.jpg" alt="Model architecture" width="700">
</div>


## Setup 
**Environmental setup**
```  
conda env create -f environment.yml
```

**Dataset Structure**

We assume that the image emotion dataset is organized as follows.
For some datasets, if a validation dataset is not available, they may consist of only train and test sets.
```
images/
│── FI/
│   ├── train/
│   │   ├── amusement/
│   │   ├── anger/
│   │   ├── sadness/
│   │   ├── fear/
│   │   ├── ... (other categories)
│   │
│   ├── val/   (Optional: May not be available in some datasets)
│   │   ├── amusement/
│   │   ├── anger/
│   │   ├── sadness/
│   │   ├── fear/
│   │   ├── ...
│   │
│   ├── test/
│   │   ├── amusement/
│   │   ├── anger/
│   │   ├── sadness/
│   │   ├── fear/
│   │   ├── ...
```

Each category (e.g., amusement, anger, sadness, fear) contains images corresponding to that emotion. 


## 0. Emotion-specific Instruction Generation using GPT4

```
python generate_instruction.py --dataname emoset --api_key <your_api_key>
```
Using the vlm_annotation_dir and image_dir from the config file, the training data of the desired dataset (located in image_dir) is processed through GPT-4 to generate an appropriate instruction dataset, which is then saved in vlm_annotation_dir.


## 1. Instruction tuning VLM
```
python train_vlm.py --dataname emoset --gpu 1
```
Arguments are as follows.  
| args | type | default |
|:---------:|:---------:|:---------:|
| dataname   | str   | -   |
| gpu   | int   | -   |
| lr   | float   | 1e-4   |
| epoch   | int   | 5   |


## 2. Knowledge distillation

```
python train_distillation.py --dataname emoset --gpu 1
```

Arguments are as follows.  
| args | type | default |
|:---------:|:---------:|:---------:|
| dataname   | str   | -   |
| gpu   | int   | -   |
| lr   | float   | 1e-5   |
| weight_decay   | float   | 1e-4   |
| epoch   | int   | 5   |
| save   | str   | ''   |
| depth   | int   |  1  |
| alpha   | float   | 0.5  |



## 3. Training a gate module

```
python train_gate.py --dataname emoset --gpu 1
```

Arguments are as follows.  
| args | type | default |
|:---------:|:---------:|:---------:|
| dataname   | str   | -   |
| gpu   | int   | -   |
| lr   | float   | 1e-4   |
| weight_decay   | float   | 1e-4   |
| epoch   | int   | 10   |
| save   | str   | ''   |
| depth   | int   |  1  | 


## Simple demo
You can run demo.py to test the demo through Gradio.

```
python demo.py
```


<div align="center">
  <img src="figure/demo.png" width="1000">
</div>

