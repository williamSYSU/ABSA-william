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
from layers.aspect_mean import AspectMean
from layers.attention import Attention


# Attentioned-based LSTM with Aspect Embedding (ATAE-LSTM)
class ATAE_LSTM(nn.Module):
    def __init__(self):
        super(ATAE_LSTM, self).__init__()
        self.lstm = nn.LSTM(2 * config.embed_size, config.hidden_size, batch_first=True)
        self.text_embed = nn.Embedding.from_pretrained(
            config.text_vocab.vectors,
            freeze=False if config.if_embed_trainable else True)
        self.aspect_embed = nn.Embedding.from_pretrained(
            config.aspect_vocab.vectors,
            freeze=False if config.if_embed_trainable else True)
        self.aspect_mean = AspectMean(config.max_sen_len)
        self.attention = Attention(
            config.train_batch_size, config.embed_size, config.hidden_size, config.uniform_rate)

        self.proj1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.proj2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.fc = nn.Linear(config.hidden_size, config.target_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text, aspect):
        """
        ATAE-LSTM forward
        :param text: size: [batch_size, max_sen_len]
        :param aspect: size: [batch_size, max_asp_len]
        :return: out: size: [batch_size, target_size]
        """
        # 处理text
        text_out = self.text_embed(text)

        # 处理aspect，aspect做均值处理
        aspect_out = self.aspect_embed(aspect)
        aspect_out = self.aspect_mean(aspect_out)

        # 拼接text和aspect
        combine = torch.cat((text_out, aspect_out), dim=2)

        # lstm > attention
        lstm_out, _ = self.lstm(combine)
        weight, at_out = self.attention(lstm_out, aspect_out)

        # projection
        r_out = self.proj1(at_out.squeeze(dim=1))
        hn_out = self.proj2(lstm_out[:, config.max_sen_len - 1, :].squeeze(dim=1))
        h_out = self.tanh(r_out + hn_out)
        out = self.softmax(self.fc(h_out))
        return out
