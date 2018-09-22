# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : config.py
# @Time         : Created at 2018/9/17
# @Blog         : http://zhiweil.ml/
# @Description  : Global parameters, including model parameters, training parameters and so on.
# Copyrights (C) 2018. All Rights Reserved.

import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.atae_lstm import ATAE_LSTM
from models.bi_lstm import Bi_LSTM

# 模型种类
model_classes = {
    'bi_lstm': Bi_LSTM,
    'atae_lstm': ATAE_LSTM,
}

# 优化器种类
optimizers = {
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'asgd': optim.ASGD,
    'sgd': optim.SGD
}

# 损失函数种类
criterions = {
    'bce': nn.BCELoss,
    'mse': nn.MSELoss,
    'nll': nn.NLLLoss,
    'cross_entropy': nn.CrossEntropyLoss,
    'cos_embed': nn.CosineEmbeddingLoss
}

"""
训练过程的可调参数
"""
learning_rate = 0.01
epoch_num = 100
train_batch_size = 25
val_batch_size = 64
test_batch_size = 64
batch_size_tuple = (train_batch_size, val_batch_size, test_batch_size)
model_name = 'atae_lstm'
optim_name = 'adagrad'
loss_name = 'cross_entropy'
model = model_classes[model_name]
optimizer = optimizers[optim_name]
criterion = criterions[loss_name]

if_step_verify = 0  # 是否在训练中验证
early_stop = 0.001  # 早停策略的阈值，loss低于这个阈值则停止训练
shuffle = 0  # 是否打乱每一轮的batch
"""
模型结构的可调参数
"""
embed_size = 300
hidden_size = 300
target_size = 3
dropout_rate = 0.3
uniform_rate = 0.01

if_embed_trainable = 1  # 设置词向量是否可训练
"""
数据集的可调参数
"""
train_val_ratio = 0.99  # 训练集和测试集的比例
max_sen_len = 80  # 最大句子长度
max_asp_len = 20  # 最大词向量长度
# max_asp_len = 20     # 最大词向量长度
"""
其它可调参数
"""
log_dir = 'log'  # tensorboard路径
log_step = 20  # 记录loss的步长
pretrain = 0  # 设置是否使用预训练模型
pretrain_path = ''  # 设置预训练模型路径

# Automatically choose GPU or CPU
if torch.cuda.is_available():
    os.system('nvidia-smi -q -d Utilization | grep Gpu > log/gpu')
    util_gpu = [int(line.strip().split()[2]) for line in open('log/gpu', 'r')]

    gpu_count = torch.cuda.device_count()
    device_choose = [i for i in range(gpu_count)]
    device = util_gpu.index(min(util_gpu))
else:
    device_choose = []
    device = -1
device_choose.append(-1)

"""
需要由其它函数初始化的变量
"""
text_vocab = None
aspect_vocab = None
text_vocab_size = 0
aspect_vocab_size = 0


def init_parameters(opt):
    global learning_rate, epoch_num, train_batch_size, val_batch_size, test_batch_size, \
        batch_size_tuple, model_name, optim_name, if_step_verify, early_stop, shuffle, \
        loss_name, model, optimizer, criterion
    global embed_size, hidden_size, target_size, dropout_rate
    global train_val_ratio, max_sen_len, max_asp_len
    global log_dir, log_step, device, pretrain, pretrain_path

    learning_rate = opt.learning_rate
    epoch_num = opt.epoch_num

    train_batch_size = opt.train_batch_size
    val_batch_size = opt.val_batch_size
    test_batch_size = opt.test_batch_size
    batch_size_tuple = (train_batch_size, val_batch_size, test_batch_size)

    model_name = opt.model
    optim_name = opt.optim
    loss_name = opt.loss_name
    model = model_classes[model_name]
    optimizer = optimizers[optim_name]
    criterion = criterions[loss_name]

    if_step_verify = opt.if_step_verify
    early_stop = opt.early_stop
    shuffle = opt.shuffle

    hidden_size = opt.hidden_size
    target_size = opt.target_size

    dropout_rate = opt.dropout_rate
    train_val_ratio = opt.train_val_ratio
    max_sen_len = opt.max_sen_len
    max_asp_len = opt.max_asp_len

    log_dir = opt.log_dir
    log_step = opt.log_step
    device = opt.device
    pretrain = opt.pretrain
    pretrain_path = opt.pretrain_path
