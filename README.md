# Edge-Optimized Cutaneous Carcinoma Detection via Polyphase Lightweight Integration

### Official code implementation of Volumetric Axial Disentanglement

### [Project page](https://github.com/IMOP-lab/PinNet) | [Our laboratory home page](https://github.com/IMOP-lab)

<div align=left>
  <img src="https://github.com/IMOP-lab/PinNet/blob/main/figures/overall.png"width=100% height=100%>
</div>
<p align=left>
  Figure 1: The overall structure of PinNet (left) and the detailed structure of DBAM (right).
</p>

<div align=left>
  <img src="https://github.com/IMOP-lab/PinNet/blob/main/figures/mshl.png"width=100% height=100%>
</div>
<p align=left>
  Figure 2: The structure of the Morpho Spectral Harmonic Layer (MSHL).
</p>


The proposed PinNet is an advanced polyphase integration network for for skin cancer segmentation. This network combines Morpho-Spectral Harmonic Layer(MSHL) to extract multi-scale bi-domain features and the Dual-perspective attention module(DBAM) for the model’s lesion detail capture. Empirical evaluations conducted on two public datasets ISIC2017 and ISIC2018 demonstrate that PinNet surpassing multiple existing State-of-the-Art (SoTA) methodologies while substantially reducing parameter size to 50KB and computational complexity to 0.068 GFLOPs.

We will first introduce our method and underlying principles, explaining how PinNet uses multi-scale bi-domain features and attention mechanisms to improve feature extraction from skin cancer images. Next, we provide details on the experimental setup, performance metrics, and GitHub links to previous methods used for comparison. Finally, we present the experimental results, showing how PinNet achieves high performance across multiple datasets.

## Installation
We run PinNet and previous methods on a system running Ubuntu 20.04, with Python 3.8, PyTorch 1.11.0, and CUDA 11.3.

## Experiment

### Compare with others on the ISIC2017 and ISIC2018 dataset

<div align=left>
  <img src="https://github.com/IMOP-lab/PinNet/blob/main/tables/ISIC2017_compare.png">
</div>
<p align=left>
  Figure 3: Comparison of skin cancer segmentation performance between PinNet and other methods on the ISIC2017 dataset.
</p>

<div align=left>
  <img src="https://github.com/IMOP-lab/PinNet/blob/main/tables/ISIC2018_compare.png">
</div>
<p align=left>
  Figure 4: Comparison of skin cancer segmentation performance between PinNet and other methods on the ISIC2018 dataset.
</p>

Our method achieves a best-in-class performance-to-size ratio, surpassing existing lightweight models on both the ISIC2017 and ISIC2018 datasets with minimal computational overhead.

<div align=left>
  <img src="https://github.com/IMOP-lab/PinNet/blob/main/figures/compare.png">
</div>
<p align=left>
  Figure 5: Prediction results of different models: (a) Original image, (b) PinNet, (c) EGEUNet, (d) ULite, (e) MALUNet, (f) UNeXt, (g)
CPFNet, (h) ResUNet++, (i) ResUNet, (j) Att-UNet, and (k) UNet. The red line marks the boundary of the ground truth labels, while the
green regions represent the predictions of the models. As shown in the figure, our model achieves prediction results closer to the ground
truth labels in the context of different challenges in skin cancer lesion areas compared to other models.
</p>


### Ablation study

#### Modelwise Ablation
<div align=left>
  <img src="https://github.com/IMOP-lab/PinNet/blob/main/tables/Modelwise ablation.png">
</div>
<p align=left>
  Figure 6: Albution on single module assessing the impact of individual module enhancements on network performance on the ISIC2018 dataset.
</p>

#### Ablation on MSHL
<div align=left>
  <img src="https://github.com/IMOP-lab/PinNet/blob/main/tables/Ablation on MSHL.png">
</div>
<p align=left>
  Figure 7: Albution on MSHL, using PinNet with MSHL replaced by
3x3 convolution as the baseline: A represents Morpho Conv, B
represents parallel structure, C represents multiscale fusion, D
represents Spectral Attention.
</p>

#### Ablation on DBAM
<div align=left>
  <img src="https://github.com/IMOP-lab/PinNet/blob/main/tables/Ablation on DBAM.png">
</div>
<p align=left>
  Figure 8: Albution on DBAM, using PinNet without DBAM as the baseline.
A represents CAB, B represents SAB, C represents FEB, and D
represents Channel Shuffle.

### Model Visualization

<div align=left>
  <img src="https://github.com/IMOP-lab/PinNet/blob/main/figures/ablation.png">
</div>
<p align=left>
  Figure 9: Heatmaps validate the structural soundness of the proposed component.
</p>

(a) Original images displaying various types of skin lesions. (b) Results from the implementation of PinNet, showing enhanced detail
and contrast in the segmentation outputs. (c) Outputs from Baseline combined with MorphoSpectralHarmonicLayer, indicating improved
edge delineation and morphological consistency. (d) Outputs from Baseline with an AttentionBridge, demonstrating focused enhancements
in lesion boundary precision and internal structure differentiation. (e) Baseline results, which provide a reference point showcasing the
basic level of segmentation without additional enhancements. Each column represents different cases processed by the respective models
across two computational layers, highlighting the incremental improvements each architectural modification offers in handling complex
skin textures and lesion types.

# Question
if you have any questions, please contact 'mingzhi.chen@hdu.edu.cn'
