import tester
from dataallo import data_allocator
from trainer import binary_real_trainer as brt
from trainer import decision_tree_trainer

"""
样例1，数据集摘自西瓜书P89，判断西瓜的种类
训练方法为线性回归，采用8折交叉验证法

样例2，数据集摘自西瓜书P76，判断西瓜的种类
训练方法为决策树分析，在原数据集上验证
"""


def sample1():
    c = 8  # 折数
    k = 8  # 重复产生次数
    path = "watermelon3.txt"
    handler = data_allocator.DataHandler(path, dtype=data_allocator.real)
    folds_list = handler.split_folds(c, k)
    tu = tester.TestUnit()
    for folds in folds_list:
        for i in range(c):
            train_set = []
            test_set = folds[i]
            for j in range(c):
                if j != i:
                    train_set.extend(folds[j])
            applier = brt.train(train_set, algorithm=brt.linear_regression, debug_mode=True)
            tu.test(applier, test_set)


def sample2():
    data_path = "watermelon3.txt"
    dh = data_allocator.DataHandler(path=data_path, split_token=", ", dtype=data_allocator.categorical)
    data_set = dh.get_data_set()
    applier = decision_tree_trainer.train(train_set=data_set)
    tu = tester.TestUnit()
    tu.test(applier, data_set)


# if __name__ == "__main__":
sample1()
sample2()
