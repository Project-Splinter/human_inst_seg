# Human Instance Segmentation

A Single Human Instance Segmentor runs at **XX FPS** on GV100.

## Install

```
pip install git+https://github.com/liruilong940607/human_inst_seg --upgrade

```

## Train
First Download dataset from [here](https://github.com/liruilong940607/ATR_RemoveBG)

```
git clone https://github.com/liruilong940607/human_inst_seg; cd human_inst_seg;
mkdir ./data # put all dataset zip under here and unzip them. It should contain two folders: `ATR_RemoveBG` and `alignment`
python human_inst_seg/train.py
```

## Usage



## API

