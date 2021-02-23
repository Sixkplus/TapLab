# TapLab
This project aims at providing a fast, modular reference implementation for 
semantic segmentation inference models.

## Environment
We conducted experiments in the following environment:
 - Linux
 - Python 3.6
 - FFmpeg
 - pytorch 1.2


## 0. Build the data loader to extract MV and Res
First, build the dataloader according to [GETTING_STARTED](./data_loader/GETTING_STARTED.md)

## 1. Demo
Take the demo video in Cityscapes `stuttgart_00.avi` as an example.

### 1.0 Download the checkpoint 
We use [BiSeNet(pytorch version)](https://drive.google.com/file/d/1hFF-J9qoXlbVRRUr29aWeQpL4Lwn4) as the per-frame model. Download the checkpoint and put it in the root dir `./`

You can choose other models as listed in [TorchSeg](https://github.com/ycszen/TorchSeg).

### 1.1. Offline inference (recommend)
Extract and store the motion vectors, then load them in the inference phase.
 - Decode the video to extract the motion vectors, residuals and RGB frames. `python decode.py`. The files are generated in `./stuttgart_00`
 - `python demo_offline.py --rgc --rgfs`
 - Results are listed in `./stuttgart_00/output/`

### 1.2. Online inferecne
Extract motion vectors and residuals online and then infer.
 - `python demo_online.py --rgc --rgfs`
 - The per-frame results are generated in `./stuttgart_00/output/`


## 2. For evaluation
We also offer an inference script for evalutation on the cityscapes dataset.

### 2.0. Preperation
 - Download `leftImg8bit_sequence_trainvaltest.zip` from the [official website](https://www.cityscapes-dataset.com/downloads/).
 - Generate video from the sequences. `python image2video_12.py`
 - Extract motion vectors and residuals using `decode.py`
 - We provide an example(`lindau_000000_000019_leftImg8bit`) of how the extracted files should be put in `./val_sequence`.

```bash
.
├── l
│   └── lindau_000000_000019_leftImg8bit
│       ├── mv_cont
│       │   ├── frame0.png
│       │   ├── frame1.png
│       │   ├── frame2.png
│       │   ├── frame3.png
│       │   ├── frame4.png
│       │   ├── frame5.png
│       │   ├── frame6.png
│       │   ├── frame7.png
│       │   ├── frame8.png
│       │   ├── frame9.png
│       │   ├── frame10.png
│       │   └── frame11.png
│       └── res_cont
│           ├── frame0.png
│           ├── frame1.png
│           ├── frame2.png
│           ├── frame3.png
│           ├── frame4.png
│           ├── frame5.png
│           ├── frame6.png
│           ├── frame7.png
│           ├── frame8.png
│           ├── frame9.png
│           ├── frame10.png
│           └── frame11.png
└── lindau
    ├── lindau_000000_000000_leftImg8bit.png
    ├── lindau_000000_000001_leftImg8bit.png
    ├── lindau_000000_000002_leftImg8bit.png
    ├── lindau_000000_000003_leftImg8bit.png
    ├── lindau_000000_000004_leftImg8bit.png
    ├── lindau_000000_000005_leftImg8bit.png
    ├── lindau_000000_000006_leftImg8bit.png
    ├── lindau_000000_000007_leftImg8bit.png
    ├── lindau_000000_000008_leftImg8bit.png
    ├── lindau_000000_000009_leftImg8bit.png
    ├── lindau_000000_000010_leftImg8bit.png
    ├── lindau_000000_000011_leftImg8bit.png
    ├── lindau_000000_000012_leftImg8bit.png
    ├── lindau_000000_000013_leftImg8bit.png
    ├── lindau_000000_000014_leftImg8bit.png
    ├── lindau_000000_000015_leftImg8bit.png
    ├── lindau_000000_000016_leftImg8bit.png
    ├── lindau_000000_000017_leftImg8bit.png
    ├── lindau_000000_000018_leftImg8bit.png
    ├── lindau_000000_000019_leftImg8bit.png
    ├── lindau_000000_000020_leftImg8bit.png
    ├── lindau_000000_000021_leftImg8bit.png
    ├── lindau_000000_000022_leftImg8bit.png
    ├── lindau_000000_000023_leftImg8bit.png
    ├── lindau_000000_000024_leftImg8bit.png
    ├── lindau_000000_000025_leftImg8bit.png
    ├── lindau_000000_000026_leftImg8bit.png
    ├── lindau_000000_000027_leftImg8bit.png
    ├── lindau_000000_000028_leftImg8bit.png
    └── lindau_000000_000029_leftImg8bit.png
```


### 2.1. Inference (offline)
 - `python infer_ffw_rgc_rgfs.py --rgc --rgfs --gop 12`
 - The results are generated in `./Test_id`, you can use them to evaluate the accuracy(mIoU).

## Citation
The following are BibTeX references. The BibTeX entry requires the url LaTeX package.

Please consider citing the paper and the related projects in your publications if it helps your research. 

```
@article{feng2020taplab,
  title={TapLab: A Fast Framework for Semantic Video Segmentation Tapping into Compressed-Domain Knowledge},
  author={Feng, Junyi and Li, Songyuan and Li, Xi and Wu, Fei and Tian, Qi and Yang, Ming-Hsuan and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}
```

```
@misc{TapLab-torch,
  author =       {Feng, Junyi},
  title =        {TapLab},
  howpublished = {\url{https://github.com/Sixkplus/TapLab}},
  year =         {2020}
}
```

```
@misc{torchseg2019,
  author =       {Yu, Changqian},
  title =        {TorchSeg},
  howpublished = {\url{https://github.com/ycszen/TorchSeg}},
  year =         {2019}
}
```

```
@misc{coviar-torch,
  author =       {Wu, Chao-Yuan},
  title =        {pytorch-coviar},
  howpublished = {\url{https://github.com/chaoyuaw/pytorch-coviar}},
  year =         {2018}
}
```


