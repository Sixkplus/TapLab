# TapLab
This project aims at providing a fast, modular reference implementation for 
semantic segmentation inference models.

## Environment
We conducted experiments in the following environment:
 - Linux
 - Python 3.6
 - FFmpeg
 - pytorch 1.2


# 0. Build the data loader to extract MV and Res
First, build the dataloader according to [GETTING_STARTED](./data_loader/GETTING_STARTED.md)

# 1. Demo

## 1.1. Offline inference
Extract and store the motion vectors, then load them in the inference phase.
`decode.py`
`infer_ffw_rgc_rgfs.py`

## 1.2. Online inferecne
Extract and Inference
`demo.py`

# 2. For evaluation
We also offer an inference script for evalutation on the cityscapes dataset.
`decode.py`
`infer_ffw_rgc_rgfs.py`

# Citation
The following are BibTeX references. The BibTeX entry requires the url LaTeX package.

Please consider citing this project in your publications if it helps your research. 

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


