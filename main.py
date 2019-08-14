from dataallo import data_allocator
from tester import test_unit as tu
from trainer import binary_real_trainer as brt

"""
样例，数据集摘自西瓜书P89，判断西瓜的种类
训练方法为线性回归，采用8折交叉验证法
"""

c = 8  # 折数
k = 8  # 重复产生次数

path = "watermelon.txt"
handler = data_allocator.DataHandler(path, dtype=data_allocator.real)
folds_list = handler.split_folds(c, k)
test = tu.TestUnit()
for folds in folds_list:
    for i in range(c):
        train_set = []
        test_set = folds[i]
        for j in range(c):
            if j != i:
                train_set.extend(folds[j])
        applier = brt.train(train_set, algorithm=brt.linear_regression, debug_mode=True)
        test.test(applier, test_set)
