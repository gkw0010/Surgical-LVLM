

<div align="center">

<samp>

<h2> CAT-ViL: Co-Attention Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery </h1>

<h4> Long Bai*, Mobarakol Islam*, and Hongliang Ren </h3>

</samp>   

</div>     
    
---

If you find our code or paper useful, please cite as

```bibtex

```

---
## Abstract
Purpose: Medical students and junior surgeons often rely on senior surgeons and specialists to answer their questions when learning surgery. However, experts are often busy with clinical and academic work, and have little time to give guidance. Meanwhile, existing artificial intelligence-based surgical Visual Question Answering (VQA) systems can only give simple answers without the location of the answers. In addition, vision-language (ViL) embedding is still a less explored research in these kinds of tasks.  We developed a surgical Visual Question Localized-Answering (VQLA) system to help medical students and junior surgeons learn and understand from recorded surgical videos.

Methods: We develop an end-to-end Transformer with Co-Attention gaTed Vision-Language (CAT-ViL) embedding for VQLA in surgical scenarios, which does not require feature extraction through detection models. The CAT-ViL embedding module is carefully designed to fuse heterogeneous features from visual and textual sources. The fused feature will feed a standard Data-Efficient Image Transformer (DeiT) module, before the parallel classifier and detector for joint prediction.

Results: We conduct the experimental validation on public surgical videos from MICCAI EndoVis Challenge 2017 and 2018. The experimental results highlight the superior performance and robustness of our proposed model compared to the state-of-the-art approaches. Ablation studies further prove the outstanding performance of all the proposed components.

Conclusion: The proposed Transformer with CAT-ViL embedding provides a promising solution for surgical VQLA and surgical scene understanding. Specifically, the CAT-ViL embedding module provides efficient information fusion from heterogeneous sources. The proposed model can serve as an effective tool for surgical training.


---
## Environment

- PyTorch
- numpy
- pandas
- scipy
- scikit-learn
- timm
- transformers
- h5py

## Directory Setup
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch library, the structure is as follows: 

- `checkpoints/`: Contains trained weights.
- `dataset/`
    - `bertvocab/`
        - `v2` : bert tokernizer
    - `EndoVis-18-VQLA/` : seq_{1,2,3,4,5,6,7,9,10,11,12,14,15,16}. Each sequence folder follows the same folder structure. 
        - `seq_1`: 
            - `left_frames`: Image frames (left_frames) for each sequence can be downloaded from EndoVIS18 challange.
            - `vqla`
                - `label`: Q&A pairs and bounding box label.
                - `img_features`: Contains img_features extracted from each frame with different patch size.
                    - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
                    - `frcnn`: img_features extracted by Fast-RCNN and ResNet101.
        - `....`
        - `seq_16`
    - `EndoVis-17-VQLA/` : selected 97 frames from EndoVIS17 challange for external validation. 
        - `left_frames`
        - `vqla`
            - `label`: Q&A pairs and bounding box label.
            - `img_features`: Contains img_features extracted from each frame with different patch size.
                - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
                - `frcnn`: img_features extracted by Fast-RCNN and ResNet101.
    - `featre_extraction/`:
        - `feature_extraction_EndoVis18-VQLA-frcnn.py`: Used to extract features with Fast-RCNN and ResNet101.
        - `feature_extraction_EndoVis18-VQLA-resnet`: Used to extract features with ResNet18 (based on patch size).
- `models/`: 
    - GatedLanguageVisualEmbedding.py : GLVE module for visual and word embeddings and fusion.
    - LViTPrediction.py : our proposed LViT model for VQLA task.
    - VisualBertResMLP.py : VisualBERT ResMLP encoder from Surgical-VQA.
    - visualBertPrediction.py : VisualBert encoder-based model for VQLA task.
    - VisualBertResMLPPrediction.py : VisualBert ResMLP encoder-based model for VQLA task.
- dataloader.py
- train.py
- utils.py

---
## Dataset (will release after acceptance)
1. EndoVis-18-VQA
    - Images (Images can be downloaded directly from EndoVis Challenge Website, we cannot release the data in our repository)
    - VQLA
2. EndoVis-17-VLQA
    - Images
    - VQLA  

---

### Run training
- Train on EndoVis-18-VLQA 
    ```bash
    python train.py --checkpoint_dir /mnt/data-hdd3/wgk/CAT-ViL/checkpoint/robustness/cat/ --transformer_ver cat --batch_size 64 --epochs 80
    ```

---
## Evaluation
- Evaluate on both EndoVis-18-VLQA & EndoVis-17-VLQA
    ```bash
    python train.py --validate True --checkpoint_dir checkpoint/robustness/gvle/ --model gvle --batch_size 64 --robustpath /left_frames_c/brightness/1/vqla/img_features/
    ```