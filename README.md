# Adapt, Then Refine: Test-Time Geometry Adaptation for Stereo Matching

## Network architecture
![image](figure/Figure2.jpg)
AdRe-Stereo introduces a lightweight three-branch Test-Time Geometry Adapter (TTGA). It adapts the geometry encoding volume to each stereo pair before dense disparity initialization and recurrent refinement.

## Create a virtual environment and activate it.

```Shell
conda create -n AdRe-Stereo python=3.8
conda activate AdRe-Stereo 
pip install -r requirements.txt
```


## Datasets

- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [KITTI 2012/2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- [Middlebury](https://vision.middlebury.edu/stereo/submit3/)
- [ETH3D](https://www.eth3d.net/datasets)

Default directory layout:

```text
datasets/
├── SceneFlow/
│   ├── FlyingThings3D/
│   ├── Monkaa/
│   └── Driving/
├── KITTI/
│   ├── KITTI2012/
│   └── KITTI2015/
├── Middlebury/
└── ETH3D/
```

Custom paths can be passed with `--sceneflow_root`, `--kitti_root`,
`--middlebury_root`, and `--eth3d_root`.

## Training

Scene Flow:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_adre_stereo.py \
  --name adre_stereo_sceneflow \
  --logdir checkpoints/adre_stereo_sceneflow \
  --train_datasets sceneflow
```

KITTI 2012 and 2015:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_adre_stereo.py \
  --name adre_stereo_kitti \
  --logdir checkpoints/adre_stereo_kitti \
  --restore_ckpt checkpoints/adre_stereo_sceneflow/best.pth \
  --train_datasets kitti
```

## Evaluation

```bash
python evaluate_adre_stereo.py \
  --restore_ckpt checkpoints/adre_stereo_sceneflow/best.pth \
  --dataset sceneflow
```

Supported datasets: `sceneflow`, `kitti2012`, `kitti2015`, `middlebury_H`,
`middlebury_Q`, and `eth3d`.

Use `--adaptive_refinement` to enable early exit:

```bash
python evaluate_adre_stereo.py \
  --restore_ckpt checkpoints/adre_stereo_sceneflow/best.pth \
  --dataset eth3d \
  --adaptive_refinement
```

## KITTI submission

```bash
python submit_adre_stereo.py \
  --restore_ckpt checkpoints/adre_stereo_kitti/best.pth \
  --dataset kitti2015
```

Results and the submission archive are saved under `submissions/`.

## Tests

```bash
python -m pytest -q
python -m compileall -q core train_adre_stereo.py evaluate_adre_stereo.py \
  submit_adre_stereo.py
```
