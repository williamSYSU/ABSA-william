# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : config.py
# @Time         : Created at 2018/9/17
# @Blog         : http://zhiweil.ml/
# @Description  : Global parameters, including model parameters, training parameters and so on.
# Copyrights (C) 2018. All Rights Reserved.

import torch
import os

# Automatically choose GPU or CPU
if torch.cuda.is_available():
    os.system('nvidia-smi -q -d Utilization | grep Gpu > log/gpu')
    util_gpu = [int(line.strip().split()[2]) for line in open('log/gpu', 'r')]

    device = util_gpu.index(min(util_gpu))
else:
    device = -1

# training parameters
learning_rate = 0.001
epoch_num = 50
train_batch_size = 16
val_batch_size = 128
test_batch_size = 128
batch_size_tuple = (train_batch_size, val_batch_size, test_batch_size)

# model parameters
embed_size = 300
hidden_size = 300

# dataset parameters
train_val_ratio = 0.7
max_sen_length = 80
max_asp_length = 20


def init_parameters(opt):
    global learning_rate, epoch_num, train_batch_size, val_batch_size, test_batch_size
    global embed_size, hidden_size
    global train_val_ratio, max_sen_length, max_asp_length
