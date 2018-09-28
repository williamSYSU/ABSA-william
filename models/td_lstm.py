# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : td_lstm.py
# @Time         : Created at 2018/9/28
# @Blog         : http://zhiweil.ml/
# @Description  : TD-LSTM implementation
# Copyrights (C) 2018. All Rights Reserved.

import torch.nn as nn

import config
from layers.aspect_mean import AspectMean

"""
Target-Dependent LSTM implementation
Reference: D. Tang, B. Qin, X. Feng, and T. Liu,
            "Effective LSTMs for Target-Dependent Sentiment Classification," 2015.
"""


class TD_LSTM(nn.Module):
    def __init__(self):
        super(TD_LSTM, self).__init__()

        self.uniform_rate = config.uniform_rate
        self.l_lstm = nn.LSTM(config.embed_size, config.hidden_size, batch_first=True)
        self.r_lstm = nn.LSTM(config.embed_size, config.hidden_size, batch_first=True)
        self.embed = nn.Embedding.from_pretrained(
            config.text_vocab.vectors,
            freeze=False if config.if_embed_trainable else True)
        self.tar_mean = AspectMean(config.max_sen_len)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, text, tar_w):
        """
        TD-LSTM forward
        :param text: size: [batch_size, max_sen_len]
        :param tar_w: target word. size: [batch_size, max_asp_len]
        :return: out: size: [batch_size, target_size]
        """
        '''get target word index and embedding'''
        l_idx, r_idx = self.get_tar_index(text, tar_w)
        text_out = self.embed(text)
        tar_out = self.embed(tar_w)

        '''averaging target word embedding'''
        tar_mean = self.tar_mean(tar_out)

        '''processed by L-LSTM and R-LSTM'''
        l_out, _ = self.l_lstm(text_out)
        r_out, _ = self.r_lstm(reversed(text_out))



    def reset_param(self):
        for param in self.l_lstm._parameters.values():
            param.data.uniform_(-self.uniform_rate, self.uniform_rate)
        for param in self.r_lstm._parameters.values():
            param.data.uniform_(-self.uniform_rate, self.uniform_rate)

    def get_tar_index(self, text, tar_w):
        tex_str = [''.join(str(x) for x in sen.tolist()) for sen in text]
        tar_str = [''.join(str(x) for x in sen.tolist()) for sen in tar_w]
        l_idx = [[tex.find(tar) for tex, tar in zip(tex_str, tar_str)]]
        r_idx = [idx + len(tar) - 1 for idx, tar in zip(l_idx, tar_str)]
        return l_idx, r_idx
