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


class Attention(nn.Module):
    """
    Attention Mechanism applied for ATAE-LSTM
    Reference: Y. Wang, M. Huang, L. Zhao, and X. Zhu,
                "Attention-based LSTM for Aspect-level Sentiment Classification,"
                Proc. 2016 Conf. Empir. Methods Nat. Lang. Process., pp. 606–615, 2016.
    """

    def __init__(self, batch_size, embed_size, hidden_size, uniform_rate):
        super(Attention, self).__init__()
        self.uniform_rate = uniform_rate
        self.w_text = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_aspect = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_combine = nn.Linear(2 * hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # self.reset_parameters()

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
        weight = self.w_combine(commbine)
        weight = self.softmax(weight).permute(0, 2, 1)
        out = torch.bmm(weight, text)
        return weight, out

    def reset_parameters(self):
        self.w_text.weight.data.uniform_(-self.uniform_rate, self.uniform_rate)
        self.w_aspect.weight.data.uniform_(-self.uniform_rate, self.uniform_rate)
        self.w_combine.weight.data.uniform_(-self.uniform_rate, self.uniform_rate)
