# Human-Video-Tracking

## Installation

```
pip3 install -v -e .  # or  python3 setup.py develop
```

## Usage

```
sh eval.sh
```

modify --path in `eval.sh` to video path

for ensemble pre-trained model, run

```
sh eval_en.sh
```

we use [weighted box fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) to weighted combine detection results (bbox and confidence)

## Pre-trained Model
you can find pre-trained weight in [this link](https://github.com/ifzhang/ByteTrack)

## Demo

![](/video_output/video.gif)