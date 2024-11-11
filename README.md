# InfoFusion Controller: Informed TRRT Star with Mutual Information based on Fusion of Pure Pursuit and MPC for Enhanced Path Planning

This is the official implementation of our ICCE 2024 (Oral) paper `InfoFusion Controller: Informed TRRT Star with Mutual Information based on Fusion of Pure Pursuit and MPC for Enhanced Path Planning`.

### [Project Page](https://drawingprocess.github.io/InfoFusionController) <!-- | [Paper](https://arxiv.org/abs/2112.01759) -->

Abstract: *In this paper, we propose the InfoFusion Controller, an advanced path planning algorithm that integrates both global and local planning strategies to enhance autonomous driving in complex urban environments. The global planner utilizes the informed Theta-Rapidly-exploring Random Tree Star (Informed-TRRT\*) algorithm to generate an optimal reference path, while the local planner combines Model Predictive Control (MPC) and Pure Pursuit algorithms. Mutual Information (MI) is employed to fuse the outputs of the MPC and Pure Pursuit controllers, effectively balancing their strengths and compensating for their weaknesses.* </br>
*The proposed method addresses the challenges of navigating in dynamic environments with unpredictable obstacles by reducing uncertainty in local path planning and improving dynamic obstacle avoidance capabilities. Experimental results demonstrate that the InfoFusion Controller outperforms traditional methods in terms of safety, stability, and efficiency across various scenarios, including complex maps generated using SLAM techniques.* </br>

## Env Settings

```shell
conda env create -f planning.yaml
conda activate planning
```

```shell
export QT_QPA_PLATFORM=xcb
export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Compare Planning Algorithms
Please check the configuration in the scripts. You can always modify it to your desired model config (especially the dataset path and input/output resolutions).

### Compare Route Planning Algorithms
```bash
python test/test_route_planner.py
```

### Compare Controller Algorithms
```bash
python test/test_controller.py
```

## Citation
If you consider our paper or code useful, please cite our paper:
```
@inproceedings{lee2025codebooknerf,
  title={InfoFusion Controller: Informed TRRT Star with Mutual Information based on Fusion of Pure Pursuit and MPC for Enhanced Path Planning},
  author={Seongjun Choi, Youngbum Kim, Nam Woo Kim, Mansun Shin, Byunggi Chae, Sungjin Lee},
  booktitle={ICCE},
  year={2025}
}
```