# SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection

# Structure
![Image text](https://github.com/xdFai/SCTransNet/blob/main/Fig/picture02.png)

![Image text](https://github.com/xdFai/SCTransNet/blob/main/Fig/picture03.png)


# Introduction
SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection, Shuai Yuan, Hanlin Qin, Xiang Yan, Naveed Akhtar, Aimal Main, arxiv 2024 [[Paper]](https://arxiv.org/pdf/2401.15583.pdf)

We present a Spatial-channel Cross Transformer Network (SCTransNet) to IRSTD task. Experiments on both public (e.g., NUAA-SIRST, NUDT-SIRST, IRSTD-1K) demonstrate the effectiveness of our method. Our main contributions are as follows:

1. We propose SCTransNet, leveraging spatial-channel cross transformer blocks (SCTB)  to predict the context of targets and backgrounds in the deeper network layers.

2. A spatial-embedded single-head channel-cross attention (SSCA) module is utilized to foster semantic interactions across all feature levels and learn the long-range context.

3. We devise a novel complementary feed-forward network (CFN) by crossing spatial-channel information to enhance the semantic difference between the target and background.

## Usage

##### 1. Train.
```bash
python train.py
```

#### 2. Test.
```bash
python test.py
```
#### 2. demo.

```bash
python demo.py
```

## Results and Trained Models

#### Quantitative Results on NUAA-SIRST, NUDT-SIRST, and IRSTD-1K

| Model         | mIoU (x10(-2)) | nIoU (x10(-2)) | F-measure (x10(-2))| Pd (x10(-2))|  Fa (x10(-6))|
| ------------- |:-------------:|:-----:|:-----:|:-----:|:-----:|
| NUAA-SIRST    | 77.50  |  81.08 | 87.32 | 96.95 | 13.92 |
| NUDT-SIRST    | 94.09  |  94.38 | 96.95 | 98.62 | 4.29  | 
| IRSTD-1K      | 68.03  |  68.15 | 80.96 | 93.27 | 10.74 |
| [[Weights]](https://drive.google.com/file/d/1Kxs2wKG2uq2YiGJOBGWoVz7B1-8DJoz3/view?usp=sharing)||||









