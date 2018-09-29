# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : attention.py
# @Time         : Created at 2018/9/21
# @Blog         : http://zhiweil.ml/
# @Description  : Attention Mechanism
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn

import config

"""
Attention Mechanism applied for ATAE-LSTM
Reference: Y. Wang, M. Huang, L. Zhao, and X. Zhu,
            "Attention-based LSTM for Aspect-level Sentiment Classification,"
            Proc. 2016 Conf. Empir. Methods Nat. Lang. Process., pp. 606–615, 2016.
"""
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.w_text = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.w_aspect = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.w_combine = nn.Linear(2 * config.hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.reset_parameters()

    def forward(self, text, aspect):
        """
        Attention forward
        :param text: size: [batch_size, max_sen_len, hidden_size]
        :param aspect: size: [batch_size, max_sen_len, embed_size]
        :return:
            "weight" of shape (batch_size, 1, max_sen_len)
            "out" of shape (batch_size, 1, hidden_size)
        """
        m_text = self.w_text(text)
        m_aspect = self.w_text(aspect)

        commbine = self.tanh(torch.cat((m_text, m_aspect), dim=2))
        combine_out = self.w_combine(commbine)
        combine_out = self.dropout(combine_out)
        weight = self.softmax(combine_out).permute(0, 2, 1)
        out = torch.bmm(weight, text)
        return weight, out

    def reset_parameters(self):
        self.w_text.weight.data.uniform_(-config.uniform_rate, config.uniform_rate)
        self.w_aspect.weight.data.uniform_(-config.uniform_rate, config.uniform_rate)
        self.w_combine.weight.data.uniform_(-config.uniform_rate, config.uniform_rate)
