# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : config.py
# @Time         : Created at 2018/9/17
# @Blog         : http://zhiweil.ml/
# @Description  : Global parameters, including model parameters, training parameters and so on.
# Copyrights (C) 2018. All Rights Reserved.

import torch

# Automatically choose GPU or CPU
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    capabilitys = []
    for idx in range(gpu_count):
        capabilitys.append(torch.cuda.get_device_capability(idx)[0])

    device = capabilitys.index(max(capabilitys))
else:
    device = -1

learning_rate = 0.001
train_val_ratio = 0.7
