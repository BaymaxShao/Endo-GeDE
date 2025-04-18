<p align="center">
<img src="https://capsule-render.vercel.app/api?color=0000FF&type=waving&fontAlignY=35&text=GV-EMoDE&fontColor=FFFFFF&desc=Generalizable%20Self-supervised%20Monocular%20Depth%20Estimation%20for%20Various%20Endoscopic%20Scenes&height=250" />
</p>

![](assets/teaser.png)

Here is the pre-released implementation for "**G**eneralizable Self-supervised <ins>Mo</ins>nocular <ins>D</ins>epth <ins>E</ins>stimation with Block-wise Mixture of Low-Rank Experts for **V**arious <ins>E</ins>ndoscopic Scenes". 

In this repository, **the evaluation code and instruction** have been released. <ins>The whole code will be published upon acceptance</ins>.

## Installation
**Information of Our Platform**: Ubuntu 22.04 + NVIDIA RTX 4090 *1 + CUDA 11.8

**Anaconda Enviroment**: `requirements.txt`

## Data Preparation

- **SCARED**: Following [Challenge Rules](https://endovissub2019-scared.grand-challenge.org/) --- Train and Eval, Realistic
- **SERV-CT**: Download from [Here](https://www.ucl.ac.uk/interventional-surgical-sciences/weiss-open-research/weiss-open-data-server/serv-ct) (From [This Ref.](https://www.sciencedirect.com/science/article/pii/S1361841521003479)) --- Zero-shot Eval, Realistic
- **Hamlyn**: Download Rectified data from [Here](https://unizares-my.sharepoint.com/personal/recasens_unizar_es/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Frecasens%5Funizar%5Fes%2FDocuments%2FDoctorado%2FEndo%2DDepth%2Dand%2DMotion%2FOpen%20access%20files%2Fhamlyn%5Fdata&ga=1) (From [This Ref.](https://ieeexplore.ieee.org/abstract/document/9478277)) --- Zero-shot Eval, Realistic
- **SimCol**: Download from [Here](https://rdr.ucl.ac.uk/articles/dataset/Simcol3D_-_3D_Reconstruction_during_Colonoscopy_Challenge_Dataset/24077763) (From [This Ref.](https://arxiv.org/abs/2307.11261)) --- Train and Eval, Simulated
- **C3VD**: Download Evaluation split from [Here](https://drive.google.com/drive/folders/1QfacGUjaD1-ByC1XvukUzu84HGdwKXhF) (From [This Ref.](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_27)) --- Zero-shot Eval, Simulated
- **EndoMapper**: Download from [Here with request](https://www.synapse.org/#!Synapse:syn52137895) (From [This Ref.](https://www.nature.com/articles/s41597-023-02564-7)) --- Qualitative Sim-to-Real

_Sincerely thanks for the remarkable contribution from above datasets to the community!!!_

## Evaluation of Depth Estimation


## Evaluation of Ego-motion Estimation
