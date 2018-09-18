# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : preprocess.py
# @Time         : Created at 2018/9/17
# @Blog         : http://zhiweil.ml/
# Copyrights (C) 2018. All Rights Reserved.

from xml.dom.minidom import parse
from torchtext import data, datasets
import spacy
import nltk
import json
import xmltodict

spacy_en = spacy.load('en')


def write_file(filename, sentences):
    polarity_dict = {
        'positive': '1',
        'negative': '-1',
        'neutral': '0',
        'conflict': '2'
    }
    with open(filename, encoding='utf-8', mode='wt') as file:
        for sentence in sentences:
            if len(sentence.getElementsByTagName('aspectTerms')) > 0:
                aspect_list = sentence.getElementsByTagName('aspectTerms')[0]
                aspect_list = aspect_list.getElementsByTagName('aspectTerm')
                for aspect in aspect_list:
                    file.write(sentence.getElementsByTagName('text')[0].childNodes[0].data + '\n')
                    file.write(aspect.getAttribute('term') + '\n')
                    polarity = aspect.getAttribute('polarity')
                    file.write(polarity_dict[polarity] + '\n')
                    # file.write(aspect.getAttribute('from') + '\n')
                    # file.write(aspect.getAttribute('to') + '\n')
            else:  # 没有Aspect信息的
                pass
                # file.write(sentence.getAttribute('id') + '\n')
                # file.write(sentence.getElementsByTagName('text')[0].childNodes[0].data + '\n')


def xml_to_pre():
    laptop_data_file = 'dataset/Laptops_Train_v2.xml'
    restaurant_data_file = 'dataset/Restaurants_Train_v2.xml'

    save_laptop_train_file = 'dataset/laptops_train_v2.pre'
    save_restaurant_train_file = 'dataset/restaurant_train_v2.pre'

    laptop_data_dom = parse(laptop_data_file)
    sentences = laptop_data_dom.getElementsByTagName('sentence')
    write_file(save_laptop_train_file, sentences)

    restaurant_data_dom = parse(restaurant_data_file)
    sentences = restaurant_data_dom.getElementsByTagName('sentence')
    write_file(save_restaurant_train_file, sentences)


def xml_to_csv():
    laptop_train_file = 'dataset/laptops_train_v2.pre'
    restaurant_train_file = 'dataset/restaurant_train_v2.pre'

    save_laptop_train_file = 'dataset/laptops_train_v2.csv'
    save_restaurant_train_file = 'dataset/restaurant_train_v2.csv'

    with open(laptop_train_file, encoding='utf-8', mode='r') as src_file:
        with open(save_laptop_train_file, encoding='utf=8', mode='w') as tar_file:
            idx = 0
            for line in src_file:
                tar_file.write(line.strip())
                if (idx + 1) % 3 != 0:
                    tar_file.write(',')
                else:
                    tar_file.write('\n')
                idx += 1


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def test_torchtext():
    TEXT = data.Field(
        sequential=True,
        tokenize=tokenizer,
        lower=True,
        fix_length=100
    )
    ASPECT = data.Field(
        # sequential=False,
        # lower=True
    )
    LABEL = data.Field(
        # sequential=False,
        # use_vocab=False
    )

    train, val, test = data.TabularDataset.splits(
        path='dataset/',
        train='laptops_train_v2.pre',
        validation='laptops_train_v2.pre',
        test='laptops_train_v2.pre',
        format='csv',
        fields=[
            ('Text', TEXT),
            ('Aspect', ASPECT),
            ('Label', LABEL)
        ]
    )

    TEXT.build_vocab(train, vectors='glove.6B.100d')
    ASPECT.build_vocab(train)
    LABEL.build_vocab(train)
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test),
        sort_key=lambda x: len(x.Text),
        batch_sizes=(8, 16, 16),
        device=-1
    )
    vocab = TEXT.vocab
    print(train_iter)
    for idx, item in enumerate(train_iter):
        if idx is 1:
            print(item.Text)


if __name__ == '__main__':
    test_torchtext()
