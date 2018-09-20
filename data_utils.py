# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : data_utils.py
# @Time         : Created at 2018/9/17
# @Blog         : http://zhiweil.ml/
# @Description  : 加载并预处理数据集
# Copyrights (C) 2018. All Rights Reserved.

from torchtext import data, datasets
import spacy

import config

spacy_en = spacy.load('en')


class ABSAData():
    def __init__(self):
        TEXT = data.Field(
            sequential=True, lower=True, batch_first=True,
            tokenize=self.tokenizer, preprocessing=self.clean_symbol,
            fix_length=config.max_sen_length
        )
        ASPECT = data.Field(
            sequential=True, lower=True, batch_first=True,
            fix_length=config.max_asp_length
        )
        LABEL = data.Field(
            sequential=False, use_vocab=False, batch_first=True
        )

        # Get data from .tsv file
        train_and_val, test = data.TabularDataset.splits(
            path='dataset/', format='tsv',
            train='laptops_train.tsv', test='laptops_test.tsv',
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
        TEXT.build_vocab(train, vectors='glove.840B.300d')
        ASPECT.build_vocab(train, vectors='glove.840B.300d')
        LABEL.build_vocab(train)

        # Get Iterator for train, validation and test data
        self.train_iter, self.val_iter, self.test_iter = data.Iterator.splits(
            (train, val, test),
            shuffle=False,
            sort_key=lambda x: len(x.Text),
            batch_sizes=config.batch_size_tuple,
            device=config.device
        )

        # Get vocab of text, aspect and label
        self.text_vocab = TEXT.vocab
        self.aspect_vocab = ASPECT.vocab
        self.label_vocab = LABEL.vocab

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
