# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : train.py
# @Time         : Created at 2018/9/17
# @Blog         : http://zhiweil.ml/
# @Description  : 训练模型
# Copyrights (C) 2018. All Rights Reserved.

import glob
import time

import os
import torch
from tensorboardX import SummaryWriter

import config
from data_utils import ABSAData

restore_loss = []
pretrained_dict = {}


class Instructor:
    def __init__(self, pre_path=''):
        self.all_data = ABSAData()
        self.train_iter = self.all_data.train_iter
        self.val_iter = self.all_data.val_iter
        self.test_iter = self.all_data.test_iter

        self.text_vocab = self.all_data.text_vocab
        self.aspect_vocab = self.all_data.aspect_vocab
        self.label_vocab = self.all_data.label_vocab

        self.device_dict = {
            -1: 'cpu',
            0: 'cuda:0',
            1: 'cuda:1',
            2: 'cuda:2',
        }
        self.model = config.model().to(self.device_dict[config.device])
        if config.pretrain:
            self.load_model(config.pretrain_path)

        self.criterion = config.criterion()
        # TODO: Set momentum for optimizer, momentum=0.9
        self.optimizer = config.optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
            lr_decay=config.lr_decay,
            weight_decay=0.001)
        if config.if_log:
            self.writer = SummaryWriter(log_dir=config.log_dir)

        # Create pretrained model folder
        if not config.pretrain:
            if pre_path != '':
                self.pre_dir = pre_path
            # else:
            #     self.pre_dir = self.create_pretrain_dir()

    def begin_train(self):
        print('=' * 100)
        print('> Begin learning......')

        global_step = 0
        min_loss = 999
        max_ac_rate = 0
        # 设置当前训练的预训练模型路径
        if not config.pretrain:
            pre_path = os.path.join(self.pre_dir,
                                    '{}.pkl'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

        for epoch in range(config.epoch_num):
            print('>' * 100)
            loss = torch.Tensor([0.])
            ac_rate = torch.Tensor([0.])
            for idx, sample_batch in enumerate(self.train_iter):
                global_step += 1

                self.model.train()  # 切换模型至训练模式l
                self.model.zero_grad()  # 清空积累的梯度

                # 取训练数据和标签
                text = sample_batch.Text
                aspect = sample_batch.Aspect
                label = sample_batch.Label

                # 计算模型的输出
                outputs, _, _ = self.model(text, aspect)

                # 显示某个batch的值
                if idx is 5:
                    # print('output:', outputs)
                    # print('label:', label)
                    pass

                # 计算loss，并更新参数
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()

                # 查看模型在验证集上的验证效果
                if config.if_step_verify and global_step % config.log_step is 0:
                    ac_rate = self.cal_ac_rate(self.val_iter)
                    if config.if_log:
                        self.writer.add_scalar('Accurate_rate', ac_rate, global_step)
                    if epoch > 3:
                        max_ac_rate = max(max_ac_rate, ac_rate)

                    # 保存验证集准确率最高的模型
                    if not config.pretrain:
                        if ac_rate == max_ac_rate:
                            print('> save ac_rate model:', ac_rate)
                            torch.save(self.model.state_dict(), pre_path)

                    # 保存预训练模型参数
                    # state_dict = copy.deepcopy(self.model.state_dict())
                    # global pretrained_dict
                    # pretrained_dict[ac_rate] = state_dict

            print('>> epoch {} of {}, -loss: {} -min loss: {} -ac_rate: {} -max ac_rate: {}'.format(
                epoch + 1, config.epoch_num, loss.item(), min_loss, ac_rate, max_ac_rate))

            min_loss = min(min_loss, loss.item())  # 计算训练过程的最小Loss
            restore_loss.append(loss.item())  # 保存每轮loss
            if config.if_log:
                self.writer.add_scalar('Train_Loss', loss, epoch)  # 画loss曲线

            # "早停"策略，loss低于设定值时，停止训练
            if loss.item() <= config.early_stop:
                print('> !!!Training is forced to stop!!!')
                print('> Current loss: {}, threshold loss: {}'.format(loss.item(), config.early_stop))
                break

        if config.if_log:
            self.writer.close()

    def cal_ac_rate(self, data_iter):
        self.model.eval()  # 设置模型为验证模式
        with torch.no_grad():
            total_count = 0
            ac_count = 0
            for idx, sample_batch in enumerate(data_iter):
                self.model.zero_grad()

                text = sample_batch.Text
                aspect = sample_batch.Aspect
                label = sample_batch.Label

                outputs, _, _ = self.model(text, aspect)
                out_label = torch.topk(outputs, 1, dim=1)[1].squeeze(dim=1)
                ac_count += torch.sum(out_label == label, dim=0).item()
                total_count += len(out_label)

            return float(ac_count / total_count)

    def begin_verify(self):
        print('=' * 100)
        print('> Begin verify......')
        print('>>> Accurate rate: ', self.cal_ac_rate(self.val_iter))

    # 在测试集上测试并保存模型参数和模型
    def test_model(self):
        # 在测试集上测试并保存结果
        print('=' * 100)
        print('> Begin test model......')
        print('> Final avg_rate:', self.cal_ac_rate(self.test_iter))

    # 保存预训练模型（保存参数）
    def save_model(self, loss, ac_rate):
        # 保存模型参数以及Loss
        print('=' * 100)
        print('> Saving model...')
        save_path = 'pretrained_model/{model_name}_loss{loss:.6f}_ac{ac_rate:.6f}.pkl'.format(
            model_name=config.model_name,
            loss=loss,
            ac_rate=ac_rate
        )
        torch.save(self.model.state_dict(), save_path)
        print('Done save: {} !'.format(save_path))

    # 加载预训练模型（加载参数）
    def load_model(self, path):
        print('=' * 100)
        print('> Loading model...')
        pre_trained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.model.state_dict()
        pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
        model_dict.update(pre_trained_dict)
        self.model.load_state_dict(model_dict)
        print('> Done load: {} !'.format(path))

    # 创建预训练模型的文件夹
    def create_pretrain_dir(self):
        print('=' * 100)
        current = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        path = os.path.join('pretrained_model', '{}_{}_lr{}_ep{}').format(
            current, config.model_name, config.learning_rate, config.epoch_num)
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
            print('> New folder:', path)
        else:
            print('> Folder {} already exists!'.format(path))

        return path

    # 保存验证集最高的几个模型参数
    def save_some_model(self):
        print('=' * 100)
        print('> Saving {} top ac_rate models...'.format(config.save_model_num))
        # 选出验证集准确率最好的几个模型
        ac_list = sorted(pretrained_dict.keys(), reverse=True)[:config.save_model_num]
        # 分别保存模型
        for ac in ac_list:
            save_path = os.path.join(self.pre_path, '{}.pkl'.format(ac))
            torch.save(pretrained_dict[ac], save_path)
        print('> Saving done!')

    # 加载在验证集上效果最好的模型，用于测试集
    def load_model_and_test(self, path=config.model_list_path):
        print('=' * 100)
        print('> Loading {} models and calculate avg_rate...'.format(config.save_model_num))
        model_list = glob.glob(os.path.join(path, '*.pkl'))
        total_ac = 0
        for path in model_list:
            self.load_model(path)
            ac_rate = self.cal_ac_rate(self.test_iter)
            print(ac_rate)
            total_ac += ac_rate

        avg_ac = total_ac / config.save_model_num
        print('> In {}, avg_rate: {}'.format(config.model_list_path, avg_ac))

        return avg_ac
