[![Website Badge](https://raw.githubusercontent.com/referit3d/referit3d/eccv/images/project_website_badge.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-2303.10437-b31b1b.svg?style=plastic)]()
# 3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians (ACM MM 2025)
PyTorch implementation of "3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians". This repository contains PyTorch training, evaluation, inference code, the pretrained model and the 3DAffordSplat dataset.

## 📋 Table of content
 1. [💡 Abstract](#1)
 2. [📖 Method](#2)
 3. [📂 Dataset](#3)
 4. [📃 Requirements](#4)
 5. [✏️ Usage](#5)
    1. [Demo](#51)
    2. [Train](#52)
    3. [Evaluate](#53)
 6. [🍎 Potential Applications](#6)
 7.  [✉️ Statement](#7)
 8.  [🔍 Citation](#8)

## News: .

## 💡 Abstract <a name="1"></a> 
3D affordance reasoning plays a critical role in associating human instructions with the functional regions of 3D objects, facilitating precise, task-oriented manipulations in embodied AI. However, current methods, which predominantly depend on sparse 3D point clouds, exhibit limited generalizability and robustness due to their sensitivity to coordinate variations and the inherent sparsity of the data. By contrast, 3D Gaussian Splatting (3DGS) delivers high-fidelity, real-time rendering with minimal computational overhead by representing scenes as dense, continuous distributions. This positions 3DGS as a highly effective approach for capturing fine-grained affordance details and improving recognition accuracy. Nevertheless, its full potential remains largely untapped due to the absence of large-scale, 3DGS-specific affordance datasets. To overcome these limitations, we present \textbf{3DAffordSplat}, the first large-scale, multi-modal dataset tailored for 3DGS-based affordance reasoning. This dataset includes 23,677 Gaussian instances, 8,354 point cloud instances, and 6,631 manually annotated affordance labels, encompassing 21 object categories and 18 affordance types. Building upon this dataset, we introduce \textbf{AffordSplatNet}, a novel model specifically designed for affordance reasoning using 3DGS representations. AffordSplatNet features an innovative cross-modal structure alignment module that exploits structural consistency priors to align 3D point cloud and 3DGS representations, resulting in enhanced affordance recognition accuracy. Extensive experiments demonstrate that the 3DAffordSplat dataset significantly advances affordance learning within the 3DGS domain, while AffordSplatNet consistently outperforms existing methods across both seen and unseen settings, highlighting its robust generalization capabilities. 

<p align="center">
    <img src="./img/Dataset Overview.pdf" width="500"/> <br />
    <em> 
    </em>
</p>

**Grounding Affordance from Interactions.** We propose to ground 3D object affordance through 2D interactions. Inputting an object point cloud with an interactive image, grounding the corresponding affordance on the 3D object.

## 📖 Method <a name="3"></a> 
### IAG-Net <a name="31"></a> 
<p align="center">
    <img src="./img/pipeline.png" width="750"/> <br />
    <em> 
    </em>
</p>

**Our Interaction-driven 3D Affordance Grounding Network.** It firstly extracts localized features $F_{i}$, $F_{p}$ respectively, then takes the Joint Region Alignment Module to align them and get the joint feature $F_{j}$. Next, Affordance Revealed Module utilizes $F_{j}$ to reveal affordance $F_{\alpha}$ with $F_{s}$, $F_{e}$ by cross-attention. Eventually, $F_{j}$ and $F_{\alpha}$ are sent to the decoder to obtain the final results $\hat{\phi}$ and $\hat{y}$.

## 📂 Dataset <a name="4"></a> 
<p align="center">
    <img src="./img/PIAD.png" width="750"/> <br />
    <em> 
    </em>
</p>

**Properties of the PIAD dataset.** **(a)** Data pairs in the PIAD, the red region in point clouds is the affordance annotation. **(b)** Distribution of the image data. The horizontal axis represents the category of affordance, the vertical axis represents quantity, and different colors represent different objects. **(c)** Distribution of the point cloud data. **(d)** The ratio of images and point clouds in each affordance class. It shows that images and point clouds are not fixed one-to-one pairing, they can form multiple pairs.

<p align="center">
    <img src="./img/data_sample.png" width="750"/> <br />
    <em> 
    </em>
</p>

**Examples of PIAD.** Some paired images and point clouds in PIAD. The ''yellow'' box in the image is the bounding box of the interactive subject, the ''red'' box is the bounding box of the interactive object.


## 📃 Requirements <a name="5"></a> 
  - python-3.9 
  - pytorch-1.13.1
  - torchvision-0.14.1
  - open3d-0.16.0
  - scipy-1.10.0
  - matplotlib-3.6.3
  - numpy-1.24.1
  - OpenEXR-1.3.9
  - scikit-learn-1.2.0
  - mitsuba-3.0.1

## ✏️ Usage <a name="6"></a> 

```bash  
git clone https://github.com/yyvhang/IAGNet.git
```

### Download PIAD and the model checkpoint <a name="41"></a>
- The pretrained model could download at [Google Drive](https://drive.google.com/drive/folders/1X768E6Oy4fBvMlsjV3WjJsq7yppX5mIh?usp=sharing), put the `.pt` file in the `ckpts\` folder
- The PIAD could download at [Google Drive](https://drive.google.com/drive/folders/1F242TsdXjRZkKQotiBsiN2u6rJAGRZ2W?usp=sharing).

### Run a Demo <a name="61"></a> 
To inference the results with IAG-Net model, run `inference.py` to get the `.ply` file
```bash  
python inference.py --model_path ckpts/IAG_Seen.pt
```

### Train <a name="62"></a> 
To train the IAG-Net model, you can modify the training parameter in `config/config_seen.yaml` and then run the following command:
```bash  
python train.py --name IAG --yaml config/config_seen.yaml
```

### Evaluate <a name="63"></a> 
To evaluate the trained IAG-Net model, run `evalization.py`:
```bash  
python evalization.py --model_path ckpts/IAG_Seen.pt --yaml config/config_seen.yaml
```

### Render the result <a name="64"></a> 
To render the `.ply` file, we provide the script `rend_point.py`, please read this script carefully. Put all `.ply` file path in one `.txt` file and run this command to get `.xml` files:
```bash  
python rend_point.py
```
Once you get the `.xml` files, just rend them with `mitsuba`, you will get `.exr` results.:
```bash  
mitsuba Chair.xml
```
If your device could not visualize `.exr` file, you can use function `ConvertEXRToJPG` in `rend_point.py` to covert it to `.jpg` file.


## 🍎 Potential Applications <a name="8"></a> 

<p align="center">
    <img src="./img/application.png" width="650"/> <br />
    <em> 
    </em>
</p>

**Potential Applications of IAG affordance system.** This work has the potential to bridge the gap between perception and operation, serving areas like demonstration learning, robot manipulation, and may be a part of human-assistant agent system e.g. Tesla Bot, Boston Dynamics Atlas.

## ✉️ Statement <a name="9"></a> 
This project is for research purpose only, please contact us for the licence of commercial use. For any other questions please contact [yyuhang@mail.ustc.edu.cn](yyuhang@mail.ustc.edu.cn).

## 🔍 Citation <a name="10"></a> 

```
@InProceedings{Yang_2023_ICCV,
    author    = {Yang, Yuhang and Zhai, Wei and Luo, Hongchen and Cao, Yang and Luo, Jiebo and Zha, Zheng-Jun},
    title     = {Grounding 3D Object Affordance from 2D Interactions in Images},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {10905-10915}
}
```






# 3DAffordSplat
3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians


# [3DAffordSplat](https://arxiv.org)      

### Abstract
3D affordance reasoning plays a critical role in associating human instructions with the functional regions of 3D objects, facilitating precise, task-oriented manipulations in embodied AI. However, current methods, which predominantly depend on sparse 3D point clouds, exhibit limited generalizability and robustness due to their sensitivity to coordinate variations and the inherent sparsity of the data. By contrast, 3D Gaussian Splatting (3DGS) delivers high-fidelity, real-time rendering with minimal computational overhead by representing scenes as dense, continuous distributions. This positions 3DGS as a highly effective approach for capturing fine-grained affordance details and improving recognition accuracy. Nevertheless, its full potential remains largely untapped due to the absence of large-scale, 3DGS-specific affordance datasets.
To overcome these limitations, we present \textbf{3DAffordSplat}, the first large-scale, multi-modal dataset tailored for 3DGS-based affordance reasoning. This dataset includes 23,677 Gaussian instances, 8,354 point cloud instances, and 6,631 manually annotated affordance labels, encompassing 21 object categories and 18 affordance types. Building upon this dataset, we introduce \textbf{AffordSplatNet}, a novel model specifically designed for affordance reasoning using 3DGS representations. AffordSplatNet features an innovative cross-modal structure alignment module that exploits structural consistency priors to align 3D point cloud and 3DGS representations, resulting in enhanced affordance recognition accuracy. Extensive experiments demonstrate that the 3DAffordSplat dataset significantly advances affordance learning within the 3DGS domain, while AffordSplatNet consistently outperforms existing methods across both seen and unseen settings, highlighting its robust generalization capabilities.

### Installation

```
```

### EXPRESS-Bench

EXPRESS-Bench comprises 777 exploration trajectories and 2,044 question-trajectory pairs. The corresponding question-answer pairs are stored in [express-bench.json](https://github.com/kxxxxxxxxxx/EXPRESS-Bench/tree/main/data/express-bench.json), while the full set of episodes for EXPRESS-Bench can be accessed from [[Google Drive](https://drive.google.com/file/d/1_FyeWi62d7NcB2VtBQPwkHSpsiWAQaL3/view?usp=sharing)], [[Baidu](https://pan.baidu.com/s/1s_q_QedXMFQzgvY4Ty6Unw?pwd=mj3f)] and [[ModelScope](https://www.modelscope.cn/datasets/kxxxxxxx/EXPRESS-Bench)]. 

To obtain the train and val splits of the HM3D dataset, please download them [here](https://github.com/matterport/habitat-matterport-3dresearch). Note that semantic annotations are required, and access must be requested in advance.

Afterward, your [data](https://github.com/kxxxxxxxxxx/EXPRESS-Bench/tree/main/data) directory structure should be:

```
|→ 3DAffordSplat
	|→ episode
		|→ 0000-00006-HkseAnWCgqk
		|→ ...
	|→ hm3d
		|→ train
			|→ 00000-kfPV7w3FaU5
			|→ ...
		|→ val
			|→ 00800-TEEsavR23oF
			|→ ...
	|→ Affordance_Question.json
	|→ 
```

### Fine-EQA model

We will release it as soon as possible!

### Citation
If you use this code for your research, please cite our paper.      
```
@article{EXPRESSBench,
  title={Beyond the Destination: A Novel Benchmark for Exploration-Aware Embodied Question Answering},
  author={Jiang, Kaixuan and Liu, Yang and Chen, Weixing and Luo, Jingzhou and Chen, Ziliang and Pan, Ling and Li, Guanbin and Lin, Liang},
  year={2025}
  journal={arXiv preprint arXiv:2503.11117}
}

``` 
If you have any question about this code, feel free to reach (jiangkx3@mail2.sysu.edu.cn or liuy856@mail.sysu.edu.cn). 
