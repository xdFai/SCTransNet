# SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection

# Structure
![Image text](https://github.com/xdFai/SCTransNet/blob/main/Fig/picture02.png)

![Image text](https://github.com/xdFai/SCTransNet/blob/main/Fig/picture03.png)


# Introduction
SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection, Shuai Yuan, Hanlin Qin, Xiang Yan, Naveed Akhtar, Aimal Main, arxiv 2024 [[Paper]](https://arxiv.org/pdf/2401.15583.pdf)

We present a Spatial-channel Cross Transformer Network (SCTransNet) to IRSTD task. Experiments on both public (e.g., NUAA-SIRST, NUDT-SIRST, IRSTD-1K) demonstrate the effectiveness of our method. Our main contributions are as follows:

1. We propose SCTransNet, leveraging spatial-channel cross transformer blocks (SCTB)  to predict the context of targets and backgrounds in the deeper network layers.

2. A spatial-embedded single-head channel-cross attention (SSCA) module is utilized to foster semantic interactions across all feature levels and learn the long-range context.

3. We devise a novel complementary feed-forward network (CFN) by crossing spatial-channel information to enhance the semantic difference between the target and background.

