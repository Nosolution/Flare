import tester
from dataallo import data_allocator
from trainer import binary_real_trainer as brt
from trainer import decision_tree_trainer
from plot.lenses_tree_plot import *

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
    data_path = "watermelon2.txt"
    handler = data_allocator.DataHandler(path=data_path, split_token=", ", dtype=data_allocator.any_type)
    data_set = handler.get_data_set()
    applier = decision_tree_trainer.train(train_set=data_set)
    tu = tester.TestUnit()
    tu.test(applier, data_set)


def sample3():
    data_path = "watermelon2.txt"
    handler = data_allocator.DataHandler(path=data_path, split_token=", ", dtype=data_allocator.any_type, del_col=[0])
    train_set, test_set = handler.split_tt_sets(0.7)
    applier = decision_tree_trainer.train(train_set=train_set, preprune_test_set=test_set)
    applier.root.print()
    tu = tester.TestUnit()
    tu.test(applier, handler.get_data_set())


def sample4():
    data_path = "lenses.txt"
    handler = data_allocator.DataHandler(path=data_path, split_token="\t", dtype=data_allocator.any_type)
    # train_set, test_set = handler.split_tt_sets(0.7)
    # applier = decision_tree_trainer.train(train_set=train_set, preprune_test_set=test_set)
    data_set = handler.get_data_set()
    applier = decision_tree_trainer.train(train_set=data_set)
    # applier.root.print()
    # print(applier.root.serialize())
    create_plot(applier.root)
    tu = tester.TestUnit()
    tu.test(applier, handler.get_data_set())


# if __name__ == "__main__":
# sample1()
# sample2()
sample4()
