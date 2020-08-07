# Human Instance Segmentation

A Single Human Instance Segmentor runs at **50 FPS** on GV100. Both training and inference is included in this repo.

## Install

```
# via pip
pip install git+https://github.com/Project-Splinter/human_inst_seg --upgrade

# via git clone
git clone https://github.com/Project-Splinter/human_inst_seg
cd human_inst_seg
python setup.py develop
```

Note to run `demo.py`, you also need to install [streamer_pytorch](https://github.com/Project-Splinter/streamer_pytorch) through:
```
pip install git+https://github.com/Project-Splinter/streamer_pytorch --upgrade
```

## Train
First Download dataset from [here](https://github.com/Project-Splinter/ATR_RemoveBG)

```
git clone https://github.com/Project-Splinter/human_inst_seg; cd human_inst_seg;
mkdir ./data # put all dataset zip under here and unzip them. It should contain two folders: `ATR_RemoveBG` and `alignment`
python human_inst_seg/train.py
```

## Usage

```
# images
python demo.py --images <IMAGE_PATH> <IMAGE_PATH> <IMAGE_PATH> --loop --vis
# videos
python demo.py --videos <VIDEO_PATH> <VIDEO_PATH> <VIDEO_PATH> --vis
# capture device
python demo.py --camera --vis
```

## API
```
seg_engine = Segmentation(ckpt=None, device="cuda:0", init=True):
seg_engine.init(pretrained="")
seg_engine.forward(input)  
```
**Note**: `Segmentation` **is** an instance of `nn.Module`, so you need to be carefull when you want to integrate this to other trainable model.
