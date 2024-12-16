# PinNet: Polyphase Integration Network for skin cancer image segmentation

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
