# TODO in ATAE-LSTM
---

1. 将laptops和restaurant数据集分开，检验分别对两个数据集的效果
2. 可视化attention中对句子中的那个词的权重较高
3. 在论文中，训练数据和测试数据如何分配？测试数据是否是从训练数据中划出一部分出来？
4. 把restaurant的aspect categories分离出来作为训练数据，测试模型结果。
5. 画流程图
6. 数据去停词
7. 检查计算正确率的函数
8. 修改损失函数，Pytorch的交叉熵函数包含了LogSoftmax的过程。