# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : preprocess.py
# @Time         : Created at 2018/9/17
# @Blog         : http://zhiweil.ml/
# Copyrights (C) 2018. All Rights Reserved.

import xml.dom.minidom
from xml.dom.minidom import parse


def write_file(filename, sentences):
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


polarity_dict = {
    'positive': '1',
    'negative': '-1',
    'neutral': '0',
    'conflict': '2'
}

laptop_data_file = 'dataset/Laptops_Train.xml'
restaurant_data_file = 'dataset/Restaurants_Train.xml'

save_laptop_train_file = 'dataset/laptops_train.pre'
save_restaurant_train_file = 'dataset/restaurant_train.pre'

laptop_data_dom = parse(laptop_data_file).documentElemenxt
sentences = laptop_data_dom.getElementsByTagName('sentence')
write_file(save_laptop_train_file, sentences)

restaurant_data_dom = parse(restaurant_data_file).documentElement
sentences = restaurant_data_dom.getElementsByTagName('sentence')
write_file(save_restaurant_train_file, sentences)
