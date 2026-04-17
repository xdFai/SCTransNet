# SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection [[Paper]](https://ieeexplore.ieee.org/document/10486932) [[Weight]](https://drive.google.com/file/d/1Kxs2wKG2uq2YiGJOBGWoVz7B1-8DJoz3/view?usp=sharing) 

Shuai Yuan, Hanlin Qin, Xiang Yan, Naveed Akhtar, Aimal Main, IEEE Transactions on Geoscience and Remote Sensing 2024.

# SCTransNet 是PRCV 2024、ICPR 2024 Track 1、ICPR 2024 Track 2 三项比赛冠军方案的 Baseline, 同时也是多个优胜算法的Baselines. [[Paper]](https://arxiv.org/abs/2408.09615)

# Bilibili 视频分享
https://www.bilibili.com/video/BV1kr421M7wx/

# 极市平台 推文分享
https://mp.weixin.qq.com/s/H7KLmtFX7j09f-Xc6X1FRw

# If the implementation of this repo is helpful to you, just star it！⭐⭐⭐

# Challenges and inspiration   
![Image text](https://github.com/xdFai/SCTransNet/blob/main/Fig/picture01.png)

# Structure
![Image text](https://github.com/xdFai/SCTransNet/blob/main/Fig/picture2.png)

![Image text](https://github.com/xdFai/SCTransNet/blob/main/Fig/picture03.png)


# Introduction

We present a Spatial-channel Cross Transformer Network (SCTransNet) to the IRSTD task. Experiments on both public (e.g., SIRST, NUDT-SIRST, IRSTD-1K) demonstrate the effectiveness of our method. Our main contributions are as follows:

1. We propose SCTransNet, leveraging spatial-channel cross transformer blocks (SCTB)  to predict the context of targets and backgrounds in the deeper network layers.

2. A spatial-embedded single-head channel-cross attention (SSCA) module is utilized to foster semantic interactions across all feature levels and learn the long-range context.

3. We devise a novel complementary feed-forward network (CFN) by crossing spatial-channel information to enhance the semantic difference between the target and background.


## Usage

#### 1. Data

The **SIRST3** dataset, which combines **IRSTD-1K**, **NUDT-SIRST**, and **SIRST-v1**, is used to train SCTransNet.
* **SIRST-v1** &nbsp; [[download]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)
* **NUDT-SIRST** &nbsp; [[download]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/abstract/document/9864119)
* **IRSTD-1K** &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)

* Apologies for misnaming the **SIRST-v1** dataset as **NUAA-SIRST** in both the article and code. We will follow the original authors’ naming convention in future work.

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
  │    ├── SIRSTv1 (~which is misnamed as NUAA-SIRST~)
  │    │    ├── images
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUAA-SIRST.txt
  │    │    │    ├── test_NUAA-SIRST.txt
  │    ├── SIRST3 (~The sum of SIRSTv1, NUDT-SIRST and IRSTD-1K~)
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
权重文件的百度网盘链接：[https://pan.baidu.com/s/1_hlEaqnJI246GWN4N8k8wA?pwd=t28j](https://pan.baidu.com/s/1B0mANHXSfJaQjHr00XIwgQ?pwd=s7nh)

权重文件的谷歌云盘链接：https://drive.google.com/file/d/1Kxs2wKG2uq2YiGJOBGWoVz7B1-8DJoz3/view?usp=sharing
```bash
python test.py
```

## Results and Trained Models

#### Qualitative Results
![Image text](https://github.com/xdFai/SCTransNet/blob/main/Fig/picture06.png)




#### Quantitative Results on Mixed SIRSTv1, NUDT-SIRST, and IRSTD-1K. i.e, one weight for three Datasets.

| Model         | mIoU (x10(-2)) | nIoU (x10(-2)) | F-measure (x10(-2))| Pd (x10(-2))|  Fa (x10(-6))|
| ------------- |:-------------:|:-----:|:-----:|:-----:|:-----:|
| SIRSTv1    | 77.50  |  81.08 | 87.32 | 96.95 | 13.92 |
| NUDT-SIRST    | 94.09  |  94.38 | 96.95 | 98.62 | 4.29  | 
| IRSTD-1K      | 68.03  |  68.15 | 80.96 | 93.27 | 10.74 |
| [[Weights]](https://drive.google.com/file/d/1Kxs2wKG2uq2YiGJOBGWoVz7B1-8DJoz3/view?usp=sharing)|


*This code is highly borrowed from [IRSTD-Toolbox](https://github.com/XinyiYing/BasicIRSTD). Thanks to Xinyi Ying.

*This code is highly borrowed from [UCTransNet](https://github.com/McGregorWwww/UCTransNet). Thanks to Haonan Wang.

*The overall repository style is highly borrowed from [DNA-Net](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li.

## Citation

If you find the code useful, please consider citing our paper using the following BibTeX entry.

```
@ARTICLE{SCTransNet,
  author={Yuan, Shuai and Qin, Hanlin and Yan, Xiang and Akhtar, Naveed and Mian, Ajmal},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SCTransNet: Spatial-Channel Cross Transformer Network for Infrared Small Target Detection}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  keywords={Semantics;Transformers;Decoding;Feature extraction;Task analysis;Object detection;Visualization;Convolutional neural network (CNN);cross-attention;deep learning;infrared small target detection (IRSTD);transformer},
  doi={10.1109/TGRS.2024.3383649}}


@article{SP-KAN,
title = {SP-KAN: Sparse-sine perception Kolmogorov–Arnold networks for infrared small target detection},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {234},
pages = {1-19},
year = {2026},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2026.02.019},
url = {https://www.sciencedirect.com/science/article/pii/S0924271626000705},
author = {Shuai Yuan and Yu Liu and Xiaopei Zhang and Xiang Yan and Hanlin Qin and Naveed Akhtar},
}


```


## Contact
**Welcome to raise issues or email to [yuansy@stu.xidian.edu.cn](yuansy@stu.xidian.edu.cn) for any question regarding our SCTransNet.**









