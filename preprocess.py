# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : preprocess.py
# @Time         : Created at 2018/9/17
# @Blog         : http://zhiweil.ml/
# @Description  : 测试数据集，预处理数据集
# Copyrights (C) 2018. All Rights Reserved.

from xml.dom.minidom import parse

import spacy

import config
from data_utils import ABSAData

spacy_en = spacy.load('en')


def write_file(filename, sentences):
    polarity_dict = {
        'positive': '2',
        'neutral': '1',
        'negative': '0',
        'conflict': '-1'
    }
    with open(filename, encoding='utf-8', mode='wt') as file:
        for sentence in sentences:
            if len(sentence.getElementsByTagName('aspectTerms')) > 0:
                aspect_list = sentence.getElementsByTagName('aspectTerms')[0]
                aspect_list = aspect_list.getElementsByTagName('aspectTerm')
                for aspect in aspect_list:
                    polarity = aspect.getAttribute('polarity')
                    if polarity != 'conflict':  # 去除conflict标签
                        file.write(sentence.getElementsByTagName('text')[0].childNodes[0].data + '\n')
                        file.write(aspect.getAttribute('term') + '\n')
                        file.write(polarity_dict[polarity] + '\n')
                        # file.write(aspect.getAttribute('from') + '\n')
                        # file.write(aspect.getAttribute('to') + '\n')
            else:  # 没有Aspect信息的
                pass
                # file.write(sentence.getAttribute('id') + '\n')
                # file.write(sentence.getElementsByTagName('text')[0].childNodes[0].data + '\n')


def xml_to_pre(type):
    laptop_data_file = 'dataset/laptops_{}.xml'.format(type)
    restaurant_data_file = 'dataset/restaurants_{}.xml'.format(type)

    save_laptop_train_file = 'dataset/laptops_{}.pre'.format(type)
    save_restaurant_train_file = 'dataset/restaurant_{}.pre'.format(type)

    laptop_data_dom = parse(laptop_data_file)
    sentences = laptop_data_dom.getElementsByTagName('sentence')
    write_file(save_laptop_train_file, sentences)

    restaurant_data_dom = parse(restaurant_data_file)
    sentences = restaurant_data_dom.getElementsByTagName('sentence')
    write_file(save_restaurant_train_file, sentences)


def pre_to_tsv(type):
    laptop_train_file = 'dataset/laptops_{}.pre'.format(type)
    restaurant_train_file = 'dataset/restaurant_{}.pre'.format(type)

    save_laptop_train_file = 'dataset/laptops_{}.tsv'.format(type)
    save_restaurant_train_file = 'dataset/restaurant_{}.tsv'.format(type)

    with open(laptop_train_file, encoding='utf-8', mode='r') as src_file:
        with open(save_laptop_train_file, encoding='utf=8', mode='w') as tar_file:
            idx = 0
            for line in src_file:
                tar_file.write(line.strip())
                if (idx + 1) % 3 != 0:
                    tar_file.write('\t')
                else:
                    tar_file.write('\n')
                idx += 1

    with open(restaurant_train_file, encoding='utf-8', mode='r') as src_file:
        with open(save_restaurant_train_file, encoding='utf=8', mode='w') as tar_file:
            idx = 0
            for line in src_file:
                tar_file.write(line.strip())
                if (idx + 1) % 3 != 0:
                    tar_file.write('\t')
                else:
                    tar_file.write('\n')
                idx += 1


def count_max_length(index):
    max_length = 0
    with open('dataset/laptops_test.tsv', mode='r') as file:
        for line in file:
            sentence = line.strip().split('\t')[index]
            max_length = max(max_length, len(sentence.split()))
    with open('dataset/laptops_train.tsv', mode='r') as file:
        for line in file:
            sentence = line.strip().split('\t')[index]
            max_length = max(max_length, len(sentence.split()))
    with open('dataset/restaurant_test.tsv', mode='r') as file:
        for line in file:
            sentence = line.strip().split('\t')[index]
            max_length = max(max_length, len(sentence.split()))
    with open('dataset/restaurant_train.tsv', mode='r') as file:
        for line in file:
            sentence = line.strip().split('\t')[index]
            max_length = max(max_length, len(sentence.split()))

    print(max_length)


if __name__ == '__main__':
    xml_to_pre('train')
    pre_to_tsv('train')
    xml_to_pre('test')
    pre_to_tsv('test')
