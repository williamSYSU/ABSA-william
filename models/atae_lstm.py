# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : lstm.py
# @Time         : Created at 2018/9/20
# @Blog         : http://zhiweil.ml/
# @Description  : ATAT-LSTM implementation
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn

import config
from layers.aspect_mean import AspectMean
from layers.attention import Attention

"""
Attention-based LSTM with Aspect Embedding (ATAE-LSTM) implementation
Reference: Y. Wang, M. Huang, L. Zhao, and X. Zhu,
            "Attention-based LSTM for Aspect-level Sentiment Classification,"
            Proc. 2016 Conf. Empir. Methods Nat. Lang. Process., pp. 606–615, 2016.
"""


class ATAE_LSTM(nn.Module):
    def __init__(self):
        super(ATAE_LSTM, self).__init__()
        self.uniform_rate = config.uniform_rate

        self.lstm = nn.LSTM(2 * config.embed_size, config.hidden_size, batch_first=True)
        self.text_embed = nn.Embedding.from_pretrained(
            config.text_vocab.vectors,
            freeze=False if config.if_embed_trainable else True)
        self.aspect_embed = nn.Embedding.from_pretrained(
            config.aspect_vocab.vectors,
            freeze=False if config.if_embed_trainable else True)
        self.aspect_mean = AspectMean(config.max_sen_len)
        self.attention = Attention(config.hidden_size, config.uniform_rate)

        self.proj1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.proj2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.fc = nn.Linear(config.hidden_size, config.target_size)
        self.tanh = nn.Tanh()
        self.so1ftmax = nn.Softmax(dim=1)

        # reset parameters
        self.reset_param()

    def forward(self, text, aspect):
        """
        ATAE-LSTM forward
        :param text: size: [batch_size, max_sen_len]
        :param aspect: size: [batch_size, max_asp_len]
        :return: out: size: [batch_size, target_size]
        """
        '''get embedding'''
        text_out = self.text_embed(text)
        aspect_out = self.aspect_embed(aspect)

        '''averaging aspect embedding'''
        aspect_mean = self.aspect_mean(aspect_out)

        '''cat text and aspect embedding'''
        combine = torch.cat((text_out, aspect_mean), dim=2)

        '''LSTM -> Attention -> weight, at_out'''
        lstm_out, _ = self.lstm(combine)
        weight, at_out = self.attention(lstm_out, aspect_mean)

        '''projection'''
        r_out = self.proj1(at_out.squeeze(dim=1))
        hn_out = self.proj2(lstm_out[:, config.max_sen_len - 1, :].squeeze(dim=1))
        h_out = self.tanh(r_out + hn_out)
        out = self.fc(h_out)
        return out, weight, at_out

    def reset_param(self):
        self.proj1.weight.data.uniform_(-self.uniform_rate, self.uniform_rate)
        self.proj2.weight.data.uniform_(-self.uniform_rate, self.uniform_rate)
        self.fc.weight.data.uniform_(-self.uniform_rate, self.uniform_rate)
        for param in self.lstm._parameters.values():
            param.data.uniform_(-self.uniform_rate, self.uniform_rate)
