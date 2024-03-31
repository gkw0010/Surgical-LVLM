# Surgical-LVLM: Learning to Adapt Large Vision-Language Model for Grounded VQA in Robotic Surgery


## Environment

- PyTorch==2.1.0+cu121
- numpy
- pandas
- scipy
- scikit-learn
- timm
- transformers
- h5py
- accelerate
- tiktoken
- einops
- transformers_stream_generator
- torchvision
- pillow
- tensorboard
- matplotlib

---

## Dataset

### Surgical VQLA
Refer to [Surgical VQLA](https://github.com/longbai1006/Surgical-VQLA) for EndoVis-17-VQLA and EndoVis-18-VQLA Dataset.

### EndoVis Conversations Dataset
Our own EndoVis Conversations Dataset can be downloaded [here] (https://drive.google.com/drive/folders/1UtR75us6xTosuZXoUUmtb4Hbh0TRUmck?usp=sharing).

---


## Training
### Run training of Qwen-VL
- Train on original Qwen-VL
    ```bash
    sh finetune/finetune_lora_ds.sh
    ```
    
- Train on Qwen-VL with Visual Perception LoRA
    ```bash
    sh finetune/finetune_lora_ds_ga.sh
    ```
### Run training of Grounding Model
- Train for Visual Grounding with Multimodal Alignment 
    ```bash
    python cat-vil/train.py --model cat --validate True --checkpoint_dir checkpoints/cat_o
    ```
---

    





