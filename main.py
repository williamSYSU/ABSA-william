# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : main.py
# @Time         : Created at 2018/9/17
# @Blog         : http://zhiweil.ml/
# @Description  : 获取命令行参数，启动模型
# Copyrights (C) 2018. All Rights Reserved.

import argparse

import config
from train import Instructor

if __name__ == '__main__':
    # 可调超参
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=config.model_name, type=str,
                        help='>>> choose: [{}], default: {}'.format(
                            ', '.join(config.model_classes.keys(), ),
                            config.model_name
                        ))
    parser.add_argument('-l', '--learning_rate', default=config.learning_rate, type=float,
                        help='>>> default: {}'.format(config.learning_rate))
    parser.add_argument('-e', '--epoch_num', default=config.epoch_num, type=int,
                        help='>>> default: {}'.format(config.epoch_num))
    parser.add_argument('-b', '--train_batch_size', default=config.train_batch_size, type=int,
                        help='>>> default:{}'.format(config.train_batch_size))
    parser.add_argument('--val_batch_size', default=config.val_batch_size, type=int,
                        help='>>> default:{}'.format(config.val_batch_size))
    parser.add_argument('--test_batch_size', default=config.test_batch_size, type=int,
                        help='>>> default:{}'.format(config.test_batch_size))
    parser.add_argument('-d', '--dropout_rate', default=config.dropout_rate, type=float,
                        help='>>> default: {}'.format(config.dropout_rate))
    parser.add_argument('-c', '--device', default=config.device, type=int,
                        help='>>> choose: {} default: {}'.format(
                            config.device_choose, config.device
                        ))
    parser.add_argument('-o', '--optim', default=config.optim_name, type=str,
                        help='> choose: [{}], default: {}'.format(
                            ', '.join(config.optimizers.keys()),
                            config.optim_name
                        ))
    parser.add_argument('--loss_name', default=config.loss_name, type=str,
                        help='> choose: [{}], default: {}'.format(
                            ', '.join(config.criterions.keys()),
                            config.loss_name
                        ))
    parser.add_argument('--hidden_size', default=config.hidden_size, type=int,
                        help='>>> default: {}'.format(config.hidden_size))
    parser.add_argument('--target_size', default=config.target_size, type=int,
                        help='>>> default: {}'.format(config.target_size))
    parser.add_argument('--train_val_ratio', default=config.train_val_ratio, type=float,
                        help='>>> default: {}'.format(config.train_val_ratio))
    parser.add_argument('--max_sen_len', default=config.max_sen_len, type=int,
                        help='>>> default: {}'.format(config.max_sen_len))
    parser.add_argument('--max_asp_len', default=config.max_asp_len, type=int,
                        help='>>> default: {}'.format(config.max_asp_len))
    parser.add_argument('--log_dir', default=config.log_dir, type=str,
                        help='>>> Path to log loss data. default: {}'.format(config.log_dir))
    parser.add_argument('--log_step', default=config.log_step, type=int,
                        help='>>> default: {}'.format(config.log_step))
    parser.add_argument('--if_step_verify', default=config.if_step_verify, type=int,
                        help='>>> If verify per log_step. default: {}'.format(config.if_step_verify))
    parser.add_argument('--early_stop', default=config.early_stop, type=float,
                        help='>>> Early stop threshold. default: {}'.format(config.early_stop))
    parser.add_argument('--shuffle', default=config.shuffle, type=int,
                        help='>>> default: {}'.format(config.shuffle))
    parser.add_argument('--pretrain', default=config.pretrain, type=int,
                        help='>>> default: {}'.format(config.pretrain))
    parser.add_argument('--pretrain_path', default=config.pretrain_path, type=str,
                        help='>>> default: {}'.format(config.pretrain_path))
    opt = parser.parse_args()

    # 初始化config
    config.init_parameters(opt)

    # 输出模型参数
    print('=' * 100)
    print('> training arguments:')
    for arg in vars(opt):
        print('>>> {}: {}'.format(arg, getattr(opt, arg)))

    # 准备训练模型数据
    instructor = Instructor()

    # 开始训练模型
    instructor.begin_train()

    # 测试模型
    instructor.test_model()
