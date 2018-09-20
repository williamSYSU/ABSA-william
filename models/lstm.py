# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : lstm.py
# @Time         : Created at 2018/9/20
# @Blog         : http://zhiweil.ml/
# @Description  : LSTM Model
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn

import config


# 单向LSTM
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(config.embed_size, config.hidden_size, batch_first=True)
        self.text_embed = nn.Embedding.from_pretrained(
            config.text_vocab.vectors,
            freeze=True if config.if_embed_trainable else False)
        self.aspect_embed = nn.Embedding.from_pretrained(
            config.aspect_vocab.vectors,
            freeze=True if config.if_embed_trainable else False)

        self.fc1 = nn.Linear(config.hidden_size, 256)
        self.fc2 = nn.Linear(256, 50)
        self.fc3 = nn.Linear(50, config.target_size)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text, aspect):
        # 处理text
        text_out = self.text_embed(text)

        # 处理aspect
        aspect_len = torch.sum(aspect != 1, dim=1).unsqueeze(dim=1).float()  # 统计每个aspect实际有多少个
        aspect_out = self.aspect_embed(aspect)
        aspect_pool = aspect_out.sum(dim=1)
        aspect_pool = aspect_pool.div(aspect_len)   # 求均值后的aspect词向量

        text_out, _ = self.lstm(text_out)
        text_out = text_out[:, config.max_sen_length - 1, :]  # 切片处理，只要最后一个output

        text_out = self.fc1(text_out)
        text_out = self.fc2(text_out)
        text_out = self.fc3(text_out)
        text_out = self.dropout(text_out)
        text_out = self.softmax(text_out)
        return text_out
