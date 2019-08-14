from collections import Counter

from algorithm.classify import decisiontree


def train(train_set: list, attrs: dict = None, strategy: int = decisiontree.inf_ent_gain):
    if not attrs:
        attrs = get_all_attrs(train_set)
    return decisiontree.generate_int_tree(train_set, attrs, strategy)


def get_all_attrs(data_set: list) -> dict:
    assert data_set
    v = len(data_set[0]) - 1
    attrs = {}
    for i in range(v):
        attrs[i] = Counter(list(map(lambda x: x[i], data_set))).keys()
    return attrs
