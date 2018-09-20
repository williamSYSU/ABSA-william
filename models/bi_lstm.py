# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : bi_lstm.py
# @Time         : Created at 2018/9/20
# @Blog         : http://zhiweil.ml/
# @Description  : Bi-LSTM Model
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn

import config

class Bi_LSTM(nn.Module):
    def __init__(self):
        super(Bi_LSTM, self).__init__()
        self.bi_lstm_context1 = nn.LSTM(config.embed_size, config.hidden_size, bidirectional=True, batch_first=True)
        self.bi_lstm_context2 = nn.LSTM(config.embed_size, config.hidden_size, bidirectional=True, batch_first=True)
        self.dense1 = nn.Linear(8 * config.hidden_size, 400)
        self.dense2 = nn.Linear(400, 100)
        self.dense3 = nn.Linear(100, config.target_size)

        self.dropout = nn.Dropout(config.dropout_rate)

        self.stm = nn.Softmax(dim=1)
        # self.stm = nn.Sigmoid()

    def forward(self, input1, input2):
        out1, (_, _) = self.bi_lstm_context1(input1)
        out2, (_, _) = self.bi_lstm_context2(input2)

        # 当batch_size > 1时，需要根据batch_size手动合并
        all_merge = []
        for idx in range(len(out1)):
            merge = torch.cat((out1[idx][0], out1[idx][-1], out2[idx][0], out2[idx][-1]), dim=0)
            if idx is 0:
                all_merge = merge.unsqueeze(0)
            else:
                all_merge = torch.cat((all_merge, merge.unsqueeze(0)), dim=0)

        out = self.dropout(all_merge)
        out = self.dense1(out)
        out = self.dropout(out)

        out = self.dense2(out)
        out = self.dense3(out)
        out = self.stm(out)
        return out