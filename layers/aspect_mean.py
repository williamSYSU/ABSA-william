# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : aspect_sum.py
# @Time         : Created at 2018/9/20
# @Blog         : http://zhiweil.ml/
# @Description  : 对几个Aspect词向量求均值
# Copyrights (C) 2018. All Rights Reserved.

import config

def aspect_mean(aspect):
    """
    对一个batch里面每个input的aspect词向量求均值。
    :param aspect: size: [batch_size, max_asp_len, embed_size]
    :return: 返回求均值后的aspect，size: [batch_size, embed_size]
    """
    aspect_pool = aspect.sum(dim=1)
    aspect_pool = aspect_pool.div(config.embed_size)
    return aspect_pool
