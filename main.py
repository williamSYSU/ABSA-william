# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : main.py
# @Time         : Created at 2018/9/17
# @Blog         : http://zhiweil.ml/
# @Description  : 获取命令行参数，启动模型
# Copyrights (C) 2018. All Rights Reserved.

import argparse
import time

import os

import config
from train import Instructor


def create_path():
    print('=' * 100)
    current = time.strftime('%m-%d %H:%M', time.localtime())
    path = os.path.join('pretrained_model', '{}_{}_lr{}_lrde{}_b{}_d{}_{}_{}').format(
        current, 'clean' if config.if_clean_symbol else 'symb',
        config.learning_rate, config.lr_decay, config.train_batch_size, config.dropout_rate,
        config.train_file.split('.')[0], config.test_file.split('_')[0])
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('>>> New folder:', path)
    else:
        print('>>> Folder {} already exists!'.format(path))

    return path


def init_program(param):
    """
    initialize config with special parameters
    :param param: [train_file, test_file, target_size, lr_decay]
    :return:
    """
    config.train_file = param[0]
    config.test_file = param[1]
    config.target_size = param[2]
    config.lr_decay = param[3]


if __name__ == '__main__':
    '''可调超参'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=config.model_name, type=str)
    parser.add_argument('-l', '--learning_rate', default=config.learning_rate, type=float)
    parser.add_argument('-e', '--epoch_num', default=config.epoch_num, type=int)
    parser.add_argument('-b', '--train_batch_size', default=config.train_batch_size, type=int)
    parser.add_argument('--val_batch_size', default=config.val_batch_size, type=int)
    parser.add_argument('--test_batch_size', default=config.test_batch_size, type=int)
    parser.add_argument('-d', '--dropout_rate', default=config.dropout_rate, type=float)
    parser.add_argument('-c', '--device', default=config.device, type=int)
    parser.add_argument('-o', '--optim', default=config.optim_name, type=str)
    parser.add_argument('--loss_name', default=config.loss_name, type=str)
    parser.add_argument('--hidden_size', default=config.hidden_size, type=int)
    parser.add_argument('--target_size', default=config.target_size, type=int)
    parser.add_argument('--train_val_ratio', default=config.train_val_ratio, type=float)
    parser.add_argument('--max_sen_len', default=config.max_sen_len, type=int)
    parser.add_argument('--max_asp_len', default=config.max_asp_len, type=int)
    parser.add_argument('--log_dir', default=config.log_dir, type=str)
    parser.add_argument('--log_step', default=config.log_step, type=int)
    parser.add_argument('--if_step_verify', default=config.if_step_verify, type=int)
    parser.add_argument('--early_stop', default=config.early_stop, type=float)
    parser.add_argument('--shuffle', default=config.shuffle, type=int)
    parser.add_argument('--pretrain', default=config.pretrain, type=int)
    parser.add_argument('--pretrain_path', default=config.pretrain_path, type=str)
    opt = parser.parse_args()

    '''初始化config'''
    config.init_parameters(opt)

    '''输出模型参数'''
    print('=' * 100)
    print('> training arguments:')
    for arg in vars(opt):
        print('>>> {}: {}'.format(arg, getattr(opt, arg)))

    '''准备训练模型数据'''
    # instructor = Instructor()

    '''开始训练模型'''
    '''program_run_param: in each item means:
    [train_file, test_file, target_size, lr_decay]'''
    program_run_param = [
        # three categoried model
        ['all_train.tsv', 'all_test.tsv', 3, 0],
        ['all_train.tsv', 'lap_test.tsv', 3, 0],
        ['all_train.tsv', 'rest_test.tsv', 3, 0],
        ['lap_train.tsv', 'lap_test.tsv', 3, 0],
        ['rest_train.tsv', 'rest_test.tsv', 3, 0],

        # two categories model
        ['all_train_2.tsv', 'all_test_2.tsv', 2, 0],
        ['all_train_2.tsv', 'lap_test_2.tsv', 2, 0],
        ['all_train_2.tsv', 'rest_test_2.tsv', 2, 0],
        ['lap_train_2.tsv', 'lap_test_2.tsv', 2, 0],
        ['rest_train_2.tsv', 'rest_test_2.tsv', 2, 0],
    ]

    # for param in program_run_param:
    #     print('=' * 100)
    #     print('>>> Current program param: {}'.format(param))

    # init_program(param)
    pre_dir = create_path()
    if not config.pretrain:
        for i in range(config.save_model_num):
            print('=' * 100)
            print('>>> Current run times {} of {}'.format(i + 1, config.save_model_num))
            instructor = Instructor(pre_dir)
            instructor.begin_train()
            instructor.test_model()
        avg_ac = instructor.load_model_and_test(pre_dir)
        os.rename(pre_dir, pre_dir + '_{:6f}'.format(avg_ac))
    else:
        instructor = Instructor(pre_dir)
        instructor.begin_train()

        '''测试模型'''
        # instructor.test_model()
