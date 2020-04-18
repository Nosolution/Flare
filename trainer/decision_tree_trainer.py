from collections import Counter

from algorithm.classify import *


def train(train_set: list, attrs: dict = None, strategy: int = inf_ent_gain,
          preprune_test_set: list = None, max_depth: int = 0):
    if not attrs:
        attrs = get_all_attrs(train_set)
    return generate_tree(train_set, attrs, strategy,
                         Prepruner(preprune_test_set) if preprune_test_set else None, max_depth)


def get_all_attrs(data_set: list) -> dict:
    assert data_set
    v = len(data_set[0]) - 1
    attrs = {}
    for i in range(v):
        values = list(map(lambda x: x[i], data_set))  # 取出该列
        if any(list(map(lambda a: '.' in a, values))):  # 以包含小数点作为实数标志
            atype = real
            # 转化为浮点数用于之后判断
            for j in range(len(data_set)):
                data_set[j][i] = float(data_set[j][i])
            values = list(map(float, values))
            attrs[i] = [atype, [min(values), max(values) + 1]]  # max+1是为了向前兼容之后的判断
        else:
            atype = discrete
            values = Counter(values).keys()
            attrs[i] = [atype, values]
    return attrs
