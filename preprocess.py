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
    lap_data_file = 'dataset/lap_{}.xml'.format(type)
    rest_data_file = 'dataset/rest_{}.xml'.format(type)

    save_lap_train_file = 'dataset/lap_{}.pre'.format(type)
    save_rest_train_file = 'dataset/rest_{}.pre'.format(type)

    lap_data_dom = parse(lap_data_file)
    sentences = lap_data_dom.getElementsByTagName('sentence')
    write_file(save_lap_train_file, sentences)

    rest_data_dom = parse(rest_data_file)
    sentences = rest_data_dom.getElementsByTagName('sentence')
    write_file(save_rest_train_file, sentences)


def pre_to_tsv(type):
    lap_train_file = 'dataset/lap_{}.pre'.format(type)
    rest_train_file = 'dataset/rest_{}.pre'.format(type)

    save_lap_train_file = 'dataset/lap_{}.tsv'.format(type)
    save_rest_train_file = 'dataset/rest_{}.tsv'.format(type)

    with open(lap_train_file, encoding='utf-8', mode='r') as src_file:
        with open(save_lap_train_file, encoding='utf=8', mode='w') as tar_file:
            idx = 0
            for line in src_file:
                tar_file.write(line.strip())
                if (idx + 1) % 3 != 0:
                    tar_file.write('\t')
                else:
                    tar_file.write('\n')
                idx += 1

    with open(rest_train_file, encoding='utf-8', mode='r') as src_file:
        with open(save_rest_train_file, encoding='utf=8', mode='w') as tar_file:
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
    with open('dataset/lap_val.tsv', mode='r') as file:
        for line in file:
            sentence = line.strip().split('\t')[index]
            max_length = max(max_length, len(sentence.split()))
    with open('dataset/lap_train.tsv', mode='r') as file:
        for line in file:
            sentence = line.strip().split('\t')[index]
            max_length = max(max_length, len(sentence.split()))
    with open('dataset/rest_val.tsv', mode='r') as file:
        for line in file:
            sentence = line.strip().split('\t')[index]
            max_length = max(max_length, len(sentence.split()))
    with open('dataset/rest_train.tsv', mode='r') as file:
        for line in file:
            sentence = line.strip().split('\t')[index]
            max_length = max(max_length, len(sentence.split()))

    print(max_length)


def seg_to_tsv():
    lap_file = 'dataset/lap_test.seg'
    rest_file = 'dataset/rest_test.seg'

    save_lap = 'dataset/lap_test.tsv'
    save_rest = 'dataset/rest_test.tsv'

    idx = 1
    with open(save_lap, encoding='utf-8', mode='w') as lap:
        with open(lap_file, encoding='utf-8', mode='r') as file:
            for line in file:
                lap.write(line.strip())
                if idx % 3 is 0:
                    lap.write('\n')
                else:
                    lap.write('\t')
                idx += 1
    idx = 1
    with open(save_rest, encoding='utf-8', mode='w') as rest:
        with open(rest_file, encoding='utf-8', mode='r') as file:
            for line in file:
                rest.write(line.strip())
                if idx % 3 is 0:
                    rest.write('\n')
                else:
                    rest.write('\t')
                idx += 1

    print('seg to tsv done!')


def fill_aspect():
    lap_file = 'dataset/lap_test_seg.tsv'
    rest_file = 'dataset/rest_test_seg.tsv'

    tar_lap = 'dataset/tar_lap.tsv'
    tar_rest = 'dataset/tar_rest.tsv'

    with open(tar_lap, encoding='utf-8', mode='w') as lap:
        with open(lap_file, encoding='utf-8', mode='r') as file:
            for line in file:
                items = line.strip().split('\t')
                items[0] = items[0].replace('$T$', items[1])
                items[2] = str(int(items[2]) + 1)
                for idx, item in enumerate(items):
                    lap.write(item)
                    if idx is not 2:
                        lap.write('\t')
                        continue
                    lap.write('\n')
    with open(tar_rest, encoding='utf-8', mode='w') as rest:
        with open(rest_file, encoding='utf-8', mode='r') as file:
            for line in file:
                items = line.strip().split('\t')
                items[0] = items[0].replace('$T$', items[1])
                items[2] = str(int(items[2]) + 1)
                for idx, item in enumerate(items):
                    rest.write(item)
                    if idx is not 2:
                        rest.write('\t')
                        continue
                    rest.write('\n')

    print('fill done!')


if __name__ == '__main__':
    # xml_to_pre('train')
    # pre_to_tsv('train')
    # xml_to_pre('test')
    # pre_to_tsv('test')
    fill_aspect()
