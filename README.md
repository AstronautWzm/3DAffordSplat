# 3DAffordSplat
3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians


# [3DAffordSplat](https://arxiv.org)      

### Abstract
3D affordance reasoning plays a critical role in associating human instructions with the functional regions of 3D objects, facilitating precise, task-oriented manipulations in embodied AI. However, current methods, which predominantly depend on sparse 3D point clouds, exhibit limited generalizability and robustness due to their sensitivity to coordinate variations and the inherent sparsity of the data. By contrast, 3D Gaussian Splatting (3DGS) delivers high-fidelity, real-time rendering with minimal computational overhead by representing scenes as dense, continuous distributions. This positions 3DGS as a highly effective approach for capturing fine-grained affordance details and improving recognition accuracy. Nevertheless, its full potential remains largely untapped due to the absence of large-scale, 3DGS-specific affordance datasets.
To overcome these limitations, we present \textbf{3DAffordSplat}, the first large-scale, multi-modal dataset tailored for 3DGS-based affordance reasoning. This dataset includes 23,677 Gaussian instances, 8,354 point cloud instances, and 6,631 manually annotated affordance labels, encompassing 21 object categories and 18 affordance types. Building upon this dataset, we introduce \textbf{AffordSplatNet}, a novel model specifically designed for affordance reasoning using 3DGS representations. AffordSplatNet features an innovative cross-modal structure alignment module that exploits structural consistency priors to align 3D point cloud and 3DGS representations, resulting in enhanced affordance recognition accuracy. Extensive experiments demonstrate that the 3DAffordSplat dataset significantly advances affordance learning within the 3DGS domain, while AffordSplatNet consistently outperforms existing methods across both seen and unseen settings, highlighting its robust generalization capabilities.

### Installation

Install the latest version of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) on headless machines:

```
conda install habitat-sim headless -c conda-forge -c aihabitat
```

### EXPRESS-Bench

EXPRESS-Bench comprises 777 exploration trajectories and 2,044 question-trajectory pairs. The corresponding question-answer pairs are stored in [express-bench.json](https://github.com/kxxxxxxxxxx/EXPRESS-Bench/tree/main/data/express-bench.json), while the full set of episodes for EXPRESS-Bench can be accessed from [[Google Drive](https://drive.google.com/file/d/1_FyeWi62d7NcB2VtBQPwkHSpsiWAQaL3/view?usp=sharing)], [[Baidu](https://pan.baidu.com/s/1s_q_QedXMFQzgvY4Ty6Unw?pwd=mj3f)] and [[ModelScope](https://www.modelscope.cn/datasets/kxxxxxxx/EXPRESS-Bench)]. 

To obtain the train and val splits of the HM3D dataset, please download them [here](https://github.com/matterport/habitat-matterport-3dresearch). Note that semantic annotations are required, and access must be requested in advance.

Afterward, your [data](https://github.com/kxxxxxxxxxx/EXPRESS-Bench/tree/main/data) directory structure should be:

```
|→ data
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
	|→ express-bench.json
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
