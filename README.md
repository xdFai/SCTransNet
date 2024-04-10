# SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection

# Chanlleges and inspiration   
![Image text](https://github.com/xdFai/SCTransNet/blob/main/Fig/picture01.png)

# Structure
![Image text](https://github.com/xdFai/SCTransNet/blob/main/Fig/picture2.png)

![Image text](https://github.com/xdFai/SCTransNet/blob/main/Fig/picture03.png)


# Introduction
SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection, Shuai Yuan, Hanlin Qin, Xiang Yan, Naveed Akhtar, Aimal Main, IEEE Transactions on Geoscience and Remote Sensing 2024 [[Paper]]([https://arxiv.org/pdf/2401.15583.pdf](https://ieeexplore.ieee.org/document/10486932))
[[Weight]](https://drive.google.com/file/d/1Kxs2wKG2uq2YiGJOBGWoVz7B1-8DJoz3/view?usp=sharing)

We present a Spatial-channel Cross Transformer Network (SCTransNet) to the IRSTD task. Experiments on both public (e.g., NUAA-SIRST, NUDT-SIRST, IRSTD-1K) demonstrate the effectiveness of our method. Our main contributions are as follows:

1. We propose SCTransNet, leveraging spatial-channel cross transformer blocks (SCTB)  to predict the context of targets and backgrounds in the deeper network layers.

2. A spatial-embedded single-head channel-cross attention (SSCA) module is utilized to foster semantic interactions across all feature levels and learn the long-range context.

3. We devise a novel complementary feed-forward network (CFN) by crossing spatial-channel information to enhance the semantic difference between the target and background.

## Citation

If you find the code useful, please consider citing our paper using the following BibTeX entry.

```
@ARTICLE{10486932,
  author={Yuan, Shuai and Qin, Hanlin and Yan, Xiang and Akhtar, Naveed and Mian, Ajmal},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Semantics;Transformers;Decoding;Feature extraction;Task analysis;Object detection;Visualization;Infrared small target detection;transformer;cross attention;CNN;deep learning},
  doi={10.1109/TGRS.2024.3383649}}
```

## Usage

#### 1. Data

* **Our project has the following structure:**
  ```
  ├──./datasets/
  │    ├── IRSTD-1K
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_IRSTD-1K.txt
  │    │    │    ├── test_IRSTD-1K.txt
  │    ├── NUDT-SIRST
  │    │    ├── images
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUDT-SIRST.txt
  │    │    │    ├── test_NUDT-SIRST.txt
  │    ├── ...
  │    ├── ...
  │    ├── SIRST3
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_SIRST3.txt
  │    │    │    ├── test_SIRST3.txt
  
  ```


##### 2. Train.
```bash
python train.py
```

#### 3. Test and demo.
```bash
python test.py
```

## Results and Trained Models

#### Qualitative Results
![Image text](https://github.com/xdFai/SCTransNet/blob/main/Fig/picture06.png)




#### Quantitative Results on NUAA-SIRST, NUDT-SIRST, and IRSTD-1K

| Model         | mIoU (x10(-2)) | nIoU (x10(-2)) | F-measure (x10(-2))| Pd (x10(-2))|  Fa (x10(-6))|
| ------------- |:-------------:|:-----:|:-----:|:-----:|:-----:|
| NUAA-SIRST    | 77.50  |  81.08 | 87.32 | 96.95 | 13.92 |
| NUDT-SIRST    | 94.09  |  94.38 | 96.95 | 98.62 | 4.29  | 
| IRSTD-1K      | 68.03  |  68.15 | 80.96 | 93.27 | 10.74 |
| [[Weights]](https://drive.google.com/file/d/1Kxs2wKG2uq2YiGJOBGWoVz7B1-8DJoz3/view?usp=sharing)|


*This code is highly borrowed from [IRSTD-Toolbox](https://github.com/XinyiYing/BasicIRSTD). Thanks to Xinyi Ying.

*This code is highly borrowed from [UCTransNet](https://github.com/McGregorWwww/UCTransNet). Thanks to Haonan Wang.

*The overall repository style is highly borrowed from [DNA-Net](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li.

## Contact
**Welcome to raise issues or email to [yuansy@stu.xidian.edu.cn](yuansy@stu.xidian.edu.cn) or [yuansy2@student.unimelb.edu.au](yuansy2@student.unimelb.edu.au) for any question regarding our SCTransNet.**









