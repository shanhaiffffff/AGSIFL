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
├── f3loc
│   ├── data
│       ├── Gibson Floorplan Localization Dataset
│           ├── README.md
│           ├── gibson_f
│               ├── ...
│           ├── gibson_g
│               ├── ...
│           ├── gibson_t
│               ├── ...
│           ├── desdf
│               ├── ...
```
