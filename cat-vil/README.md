

<div align="center">

<samp>

<h2> CAT-ViL: Co-Attention Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery </h1>

<h4> Long Bai*, Mobarakol Islam*, and Hongliang Ren </h3>

</samp>   

</div>     
    

## Abstract
Purpose: Medical students and junior surgeons often rely on senior surgeons and specialists to answer their questions when learning surgery. However, experts are often busy with clinical and academic work, and have little time to give guidance. Meanwhile, existing artificial intelligence-based surgical Visual Question Answering (VQA) systems can only give simple answers without the location of the answers. In addition, vision-language (ViL) embedding is still a less explored research in these kinds of tasks.  We developed a surgical Visual Question Localized-Answering (VQLA) system to help medical students and junior surgeons learn and understand from recorded surgical videos.

Methods: We develop an end-to-end Transformer with Co-Attention gaTed Vision-Language (CAT-ViL) embedding for VQLA in surgical scenarios, which does not require feature extraction through detection models. The CAT-ViL embedding module is carefully designed to fuse heterogeneous features from visual and textual sources. The fused feature will feed a standard Data-Efficient Image Transformer (DeiT) module, before the parallel classifier and detector for joint prediction.

Results: We conduct the experimental validation on public surgical videos from MICCAI EndoVis Challenge 2017 and 2018. The experimental results highlight the superior performance and robustness of our proposed model compared to the state-of-the-art approaches. Ablation studies further prove the outstanding performance of all the proposed components.

Conclusion: The proposed Transformer with CAT-ViL embedding provides a promising solution for surgical VQLA and surgical scene understanding. Specifically, the CAT-ViL embedding module provides efficient information fusion from heterogeneous sources. The proposed model can serve as an effective tool for surgical training.

