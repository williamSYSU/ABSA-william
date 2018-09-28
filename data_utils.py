# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : data_utils.py
# @Time         : Created at 2018/9/17
# @Blog         : http://zhiweil.ml/
# @Description  : 加载并预处理数据集
# Copyrights (C) 2018. All Rights Reserved.

import spacy
import torch
import torch.nn as nn
from torchtext import data

import config

spacy_en = spacy.load('en')


class ABSAData:
    def __init__(self):
        TEXT = data.Field(
            sequential=True, lower=True, batch_first=True,
            tokenize=self.tokenizer, fix_length=config.max_sen_len
        )
        ASPECT = data.Field(
            sequential=True, lower=True, batch_first=True,
            fix_length=config.max_asp_len
        )
        LABEL = data.Field(
            sequential=False, use_vocab=False, batch_first=True
        )

        # Get data from .tsv file
        train_and_val, test = data.TabularDataset.splits(
            path='dataset/', format='tsv',
            train=config.train_file, test=config.test_file,
            fields=[
                ('Text', TEXT),
                ('Aspect', ASPECT),
                ('Label', LABEL)
            ]
        )

        # Split data into train set and validation set
        train, val = train_and_val.split(
            split_ratio=config.train_val_ratio
        )

        # Build vocab for Field TEXT, ASPECT and LABEL
        TEXT.build_vocab(train, val, test, vectors='glove.840B.300d', unk_init=self.unk_init_uniform)
        ASPECT.build_vocab(train, val, test, vectors='glove.840B.300d', unk_init=self.unk_init_uniform)
        LABEL.build_vocab(train, val, test)

        # Get Iterator for train, validation and test data
        self.train_iter, self.val_iter, self.test_iter = data.Iterator.splits(
            (train, val, test),
            shuffle=True if config.shuffle else False,
            repeat=False,
            sort_key=lambda x: len(x.Text),
            batch_sizes=config.batch_size_tuple,
            device=config.device
        )

        # Get vocab of text, aspect and label
        self.text_vocab = TEXT.vocab
        self.aspect_vocab = ASPECT.vocab
        self.label_vocab = LABEL.vocab

        # initialize vocab with special vectors
        self.ini_vocob()

        # Set vocab size in config
        config.text_vocab = self.text_vocab
        config.aspect_vocab = self.aspect_vocab
        config.text_vocab_size = len(self.text_vocab)
        config.aspect_vocab_size = len(self.aspect_vocab)

    def tokenizer(self, text):
        """
        构建分词函数
        :param text: 输入
        :return: 分词后的列表
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def clean_symbol(self, text):
        """
        清洗标点符号
        :param text: 句子分词后的一个列表
        :return: 清洗后的列表
        """
        ex_symbol = [',', '.', '!', '?', '"', '\'']
        return [i for i in text if i not in ex_symbol]

    def unk_init_uniform(self, text):
        """
        词典里面没有的词用均匀分布来初始化。
        :return: uniform vectors
        """
        vec = torch.empty(1, config.embed_size)
        return nn.init.uniform_(vec, a=-config.uniform_rate, b=config.uniform_rate)

    def ini_vocob(self):
        tx_pad_idx = self.text_vocab.stoi['<pad>']
        as_pad_idx = self.aspect_vocab.stoi['<pad>']
        zero_vec = torch.zeros(1, config.embed_size)
        self.text_vocab.vectors[tx_pad_idx] = zero_vec
        self.aspect_vocab.vectors[as_pad_idx] = zero_vec
