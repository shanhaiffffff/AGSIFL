# AGSIFL

This is the official Pytorch implementation of our paper: "Adaptive-fusion of Geometric and Semantic Information for Floorplan Localization "

## Setup

We conducted experiments based on F<sup>3</sup>Loc and OneFormer. You can download the dataset from F<sup>3</sup>Loc and use its depth estimation method. You need to download relevant checkpoints in OneFormer to support semantic segmentation for subsequent tasks.

[F<sup>3</sup>Loc](https://github.com/felix-ch/f3loc) 

[OneFormer](https://github.com/SHI-Labs/OneFormer)  

After configuring OneFormer, you can use the following commands to achieve semantic segmentation, save door masks, and furniture masks.You need to modify the relevant file paths according to your needs.

```
export task=semantic
python demo/demo.py --config-file configs/ade20k/swin/oneformer_swin_large_bs16_160k_1280x1280.yaml \
  --input test/ \
  --output test_output/ \
  --task $task \
  --opts MODEL.WEIGHTS checkpoint/896x896_250_16_swin_l_oneformer_ade20k_160k.pth
```
You can download the semantic_floorplan data from [here](https://pan.baidu.com/s/1QOUc_Z_cs9pWpQqyHPewUQ?pwd=f5iq).\
Extracted code: f5iq.

You can use the following command to generate semantic_DESDF for floorplans.You need to modify the relevant file paths according to your needs.

```
python semantic_desdf.py
```
Place dataset under the data folder:
```
├── AGSIFL
│   ├── data
│       ├── Gibson Floorplan Localization Dataset
│           ├── README.md
│           ├── gibson_f
│               ├── Spencerville
│                   ├──door_mask
│                   ├──furniture_mask
│                   ├── ... 
│               ├── ...
│           ├── gibson_g
│               ├── ...
│           ├── gibson_t
│               ├── ...
│           ├── desdf
│               ├── ...
│           ├── semantic_desdf
│               ├── ...
```

## Usage
### Evaluate the Single-Frame Localization
```
python eval_observation.py --dataset <dataset>
```

### Evaluate the Sequential Localization
```
python eval_filtering.py --traj_len <traj-len> --evol_path ./visualization
```
### Train mlp
You need to modify the checkpoint's saving path according to your needs.
```
python train_mlp.py --dataset <dataset>
```

## Acknowledgement

We thank the authors of [F<sup>3</sup>Loc](https://github.com/felix-ch/f3loc) and [OneFormer](https://github.com/SHI-Labs/OneFormer) for releasing their helpful codebases.
