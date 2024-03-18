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
    config.KV_size = 480  # KV channel dimension size = C1 + C2 + C3 + C4
    config.transformer.num_layers = 4  # the number of SCTB
    config.expand_ratio = 2.66  # CFN channel dimension expand ratio
    config.base_channel = 32  # base channel of SCTransNet
    config.patch_sizes = [16, 8, 4, 2]
    config.n_classes = 1
    return config
