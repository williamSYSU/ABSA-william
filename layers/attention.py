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
        # TODO: 把Parameter改为Linear
        super(Attention, self).__init__()
        self.uniform_rate = uniform_rate
        self.w_text = nn.Parameter(torch.Tensor(batch_size, hidden_size, hidden_size))
        self.w_aspect = nn.Parameter(torch.Tensor(batch_size, embed_size, embed_size))
        self.w_combine = nn.Parameter(torch.Tensor(batch_size, 1, hidden_size + embed_size))
        self.tanh = nn.Tanh()
        self.com_softmax = nn.Softmax(dim=2)
        self.final_softmax = nn.Softmax(dim=1)

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
        text_out = text.permute(0, 2, 1)
        aspect_out = aspect.permute(0, 2, 1)
        m_text = torch.bmm(self.w_text, text_out)
        m_aspect = torch.bmm(self.w_aspect, aspect_out)

        commbine = self.tanh(torch.cat((m_text, m_aspect), dim=1))
        tmp = torch.bmm(self.w_combine, commbine)
        weight = self.com_softmax(torch.bmm(self.w_combine, commbine))
        out = torch.bmm(text_out, weight.permute(0, 2, 1)).permute(0, 2, 1)
        return weight, out

    def reset_parameters(self):
        self.w_text.data.uniform_(-self.uniform_rate, self.uniform_rate)
        self.w_aspect.data.uniform_(-self.uniform_rate, self.uniform_rate)
        self.w_combine.data.uniform_(-self.uniform_rate, self.uniform_rate)
