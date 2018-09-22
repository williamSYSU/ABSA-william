# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : ABSA-william
# @FileName     : train.py
# @Time         : Created at 2018/9/17
# @Blog         : http://zhiweil.ml/
# @Description  : 训练模型
# Copyrights (C) 2018. All Rights Reserved.

import torch
from tensorboardX import SummaryWriter

import config
from data_utils import ABSAData

restore_loss = []


class Instructor:
    def __init__(self):
        self.all_data = ABSAData()
        self.train_iter = self.all_data.train_iter
        self.val_iter = self.all_data.val_iter
        self.test_iter = self.all_data.test_iter

        self.text_vocab = self.all_data.text_vocab
        self.aspect_vocab = self.all_data.aspect_vocab
        self.label_vocab = self.all_data.label_vocab

        device_dict = {
            -1: 'cpu',
            0: 'cuda:0',
            1: 'cuda:1',
            2: 'cuda:2',
        }
        self.model = config.model().to(device_dict[config.device])
        self.criterion = config.criterion()
        self.optimizer = config.optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
            weight_decay=0.001)
        self.writer = SummaryWriter(log_dir=config.log_dir)

    def begin_train(self):
        print('=' * 100)
        print('> Begin learning......')

        global_step = 0
        min_train_loss = 999
        min_verify_loss = 999

        for epoch in range(config.epoch_num):
            print('>' * 100)
            loss = torch.Tensor([0])
            for idx, sample_batch in enumerate(self.train_iter):
                global_step += 1

                self.model.train()  # 切换模型至训练模式
                self.model.zero_grad()  # 清空积累的梯度

                # 取训练数据和标签
                text = sample_batch.Text
                aspect = sample_batch.Aspect
                label = sample_batch.Label

                # 计算模型的输出
                outputs = self.model(text, aspect)

                # 指定一个batch查看其在每轮的优化效果如何
                # if idx is 5:
                #     print('output: {}\nlabel: {}'.format(outputs, label))

                # 计算loss，并更新参数
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()

                # 查看模型在验证集上的验证效果
                if config.if_step_verify and global_step % config.log_step is 0:
                    verify_loss = self.verify_model()
                    self.writer.add_scalar('Verify Loss', verify_loss, global_step)
                    min_verify_loss = min(min_verify_loss, verify_loss)

            print('>> epoch {} of {}, -loss: {} -min train loss: {} -min verify loss: {}'.format(
                epoch + 1, config.epoch_num, loss.item(), min_train_loss, min_verify_loss))

            min_train_loss = min(min_train_loss, loss.item())  # 计算训练过程的最小Loss
            restore_loss.append(loss.item())  # 保存每轮loss
            self.writer.add_scalar('Train_Loss', loss, epoch)  # 画loss曲线

            # "早停"策略，loss低于设定值时，停止训练
            if loss.item() <= config.early_stop:
                print('> !!!Training is forced to stop!!!')
                print('> Current loss: {}, threshold loss: {}'.format(loss.item(), config.early_stop))
                break

        self.writer.close()

    def begin_verify(self):
        print('=' * 100)
        print('> Begin verify......')
        print('>>> Verify loss: ', self.verify_model())

    def verify_model(self):
        self.model.eval()  # 设置模型为验证模式
        sum_loss = 0

        with torch.no_grad():
            for idx, sample_batch in enumerate(self.val_iter):
                self.model.zero_grad()

                text = sample_batch.Text
                aspect = sample_batch.Aspect
                label = sample_batch.Label

                outputs = self.model(text, aspect)

                loss = self.criterion(outputs, label)
                sum_loss += loss

            return float(sum_loss / len(self.val_iter))

    # 在测试集上测试并保存模型参数和模型
    def test_model(self):
        # 在测试集上测试并保存结果
        print('=' * 100)
        print('> Begin test and save model......')

        for idx, sample_batch in enumerate(self.test_iter):
            text = sample_batch.Text
            aspect = sample_batch.Aspect

            outputs = self.model(text, aspect)[:, 1].view(-1)

            # TODO: 计算准确率

    # 保存预训练模型
    def save_model(self, loss, ac_rate):
        # 保存模型参数以及Loss
        save_path = 'pretrained_model/{model_name}_{loss}_{ac_rate}'.format(
            model_name=config.model_name,
            loss=loss,
            ac_rate=ac_rate
        )
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, path):
        if config.pretrain:
            pre_trained_dict = torch.load(path, map_location=lambda storage, loc: storage)
            model_dict = self.model.state_dict()
            pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
            model_dict.update(pre_trained_dict)
            self.model.load_state_dict(model_dict)
