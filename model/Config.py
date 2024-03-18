# -*- coding: utf-8 -*-
# @Author  : Shuai Yuan
# @File    : Config.py
# @Software: PyCharm
# coding=utf-8
import os
import torch
import time
import ml_collections




##########################################################################
# SCTrans configs
##########################################################################
def get_SCTrans_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net
    config.n_classes = 1

    # ********** unused **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config
