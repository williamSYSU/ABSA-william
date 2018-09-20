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
        self.lstm1 = nn.LSTM(config.embed_size, config.hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(config.embed_size, config.hidden_size, batch_first=True)
        self.dense1 = nn.Linear(2 * config.hidden_size, 256)
        self.dense2 = nn.Linear(256, 50)
        self.dense3 = nn.Linear(50, config.target_size)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.stm = nn.Softmax(dim=1)
        # self.stm = nn.Sigmoid()

    def forward(self, input1, input2):
        out1, hidden1 = self.lstm1(input1)
        out2, hidden2 = self.lstm2(input2)
        # 当batch_size > 1时，需要根据batch_size手动合并
        all_merge = []
        for idx in range(len(out1)):
            merge = torch.cat((out1[idx][-1], out2[idx][-1]), dim=0)
            if idx is 0:
                all_merge = merge.unsqueeze(0)
            else:
                all_merge = torch.cat((all_merge, merge.unsqueeze(0)), dim=0)

        # merge = torch.cat((out1[0][-1], out2[0][-1]), dim=0)
        out = self.dense1(all_merge)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.dropout(out)
        out = self.stm(out)
        return out
