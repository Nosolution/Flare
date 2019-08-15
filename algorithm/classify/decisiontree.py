from collections import Counter
from typing import Tuple, Any
from copy import deepcopy
import numpy as np

from applier import MultiClassApplier

"""
实数决策树实现模块
"""

inf_ent_gain = 21
info_gain_ratio = 22
gini_inx = 23

discrete = 101
real = 102

__all__ = ['inf_ent_gain', 'info_gain_ratio', 'gini_inx', 'real', 'discrete', 'generate_tree', 'TreeNode']


def __split(data_set: list, i: int, value: float) -> Tuple[list, list]:
    """
    根据属性值相对指定值的大小分割数据集
    :param data_set: 待分割数据集
    :param i: 属性下标
    :param value: 用于比较大小的值
    :return: 两个数据集，分别包含属性取值小于等于指定值与大于指定值的数据
    """
    s1 = list(filter(lambda x: x[i] <= value, data_set))
    s2 = list(filter(lambda x: x[i] > value, data_set))
    return s1, s2


def __inf_entropy(data_set: list) -> float:
    """
    根据数据集的标记分布情况计算信息熵. 算式: Ent(D) = -sum_{k=1}^abs(m)(p_k*log_2{p_k})
    :param data_set: 待使用数据集
    :return: 数据集的类别的信息熵
    """
    s = len(data_set)
    c = Counter(list(map(lambda x: x[-1], data_set))).values()
    c = list(map(lambda x: x / s, c))
    return -sum(list(map(lambda x: x * np.log2(x), c)))


def __info_ent_gain(data_set: list, i: int, atype: int, mid_value: float = 0) -> float:
    """
    计算通过属性a_i和中间值mid_value分割数据集获得的信息增益
    :param data_set: 待分割计算的数据集
    :param i: 属性a_i的下标
    :param atype: 属性类型, 取值real或者discrete
    :param mid_value: 用作分割数据集的中间值
    :return: 通过分割数据集获得的信息增益
    """
    assert 0 <= i < len(data_set[0])
    m = len(data_set)
    before = __inf_entropy(data_set)
    after = 0
    if atype == real:
        s1, s2 = __split(data_set, i, mid_value)
        after = len(s1) / m * __inf_entropy(s1) + len(s2) / m * __inf_entropy(s2)
    else:
        values = Counter(list(map(lambda x: x[i], data_set))).keys()
        for value in values:
            sub_set = list(filter(lambda x: x[i] == value, data_set))
            after += len(sub_set) / m * __inf_entropy(sub_set)
    return before - after


# 连续值不存在增益率
def __gain_ratio(data_set: list, i: int, atype: int, **kwargs) -> float:
    """
    计算通过属性a_i分割数据集获得的信息增益率
    :param data_set: 待分割计算的数据集
    :param i: 属性a_i的下标
    :param atype: 属性类型, 取值real或者discrete
    :return: 通过分割数据集获得的信息增益率
    """
    gain = __info_ent_gain(data_set, i, atype)
    value_counts = Counter(list(map(lambda x: x[i], data_set))).values()
    return gain / sum(map(lambda x: -x * np.log2(x), value_counts))


def __gini(data_set: list) -> float:
    """
    对数据集进行基尼值计算
    :param data_set: 待计算的数据集
    :return: 该数据集的基尼值
    """
    m = len(data_set)
    d = sorted(data_set, key=lambda x: x[-1])
    i, j = 0, 0
    res = 1
    while j < m:
        if d[j][-1] != d[i][-1]:
            res -= ((j - i) / m) ** 2
            i = j
        j += 1
    return res


def __gini_index(data_set: list, i: int, atype: int, mid_value: float = 0) -> float:
    """
    计算通过属性a_i和中间值mid_value分割数据集的基尼指数
    :param data_set: 待计算的数据集
    :param i: 属性a_i的下标
    :param atype: 属性类型, 取值real或者discrete
    :param mid_value: 用作分割数据集的中间值
    :return: 该数据集的基尼指数
    """
    m = len(data_set)
    res = 0
    if atype == real:
        s1, s2 = __split(data_set, i, mid_value)
        res = len(s1) / m * __gini(s1) + len(s2) / m * __gini(s2)
    else:
        values = Counter(list(map(lambda x: x[i], data_set))).keys()
        for value in values:
            sub_set = list(filter(lambda x: x[i] == value, data_set))
            res += len(sub_set) / m * __gini(sub_set)
    return res


class TreeNode(object):
    def __init__(self):
        self.label = -1  # 叶节点标记
        self.ai = 0  # 对应的属性下标
        self.atype = 0  # 属性类型, 为real或者discrete
        self.standard = 0  # real类型需要的对比值
        self.children = {}  # 子节点

    def decide(self, x: list) -> Any:
        """
        判断输入样例的类型
        :param x: 待决定类型的输入样例
        :return: 判断的类型
        """
        if not self.children:
            return self.label
        else:
            if self.atype == real:
                if x[self.ai] > self.standard:
                    return self.children[1].decide(x)
                else:
                    return self.children[0].decide(x)
            else:
                return self.children[x[self.ai]].decide(x)

    def print(self, layer: int = 1) -> None:
        """
        输出节点信息
        :param layer: 节点所在层数
        """
        if not self.children:
            print("  " * layer + "label: {}".format(self.label))
        else:
            if self.atype == real:
                print("when leq than standard: {}".format(self.standard))
                self.children[0].print(layer + 1)
                print("when greater than  standard: {}".format(self.standard))
                self.children[0].print(layer + 1)
            else:
                for item in self.children.items():
                    print("  " * layer + "on ai:{}={}".format(self.ai, item[0]))
                    item[1].print(layer + 1)


def generate_tree(train_set: list, attrs: dict, strat: int) -> MultiClassApplier:
    return TreeApplier(__generate_tree(train_set, attrs, strat))


def __generate_tree(train_set: list, attrs: dict, strat: int) -> TreeNode:
    """
    递归生成决策树，返回根节点
    :param train_set: 训练集, 要求属性值都为整数, 格式: [(x_1,x_2,x_3,...,label),...,(x_1,x_2,x_3,...,label)]
    :param attrs: 参与决策的属性及相关信息, 格式: {attr1:[atype, [value_1, value_2,...]], attr2:[atype, [value_1, value_2,...]],...}
    如果atype==real, 则value元组为attr取值范围[min, max], 否则为离散属性的所有取值[value_1, value_2,...]
    :param strat: 采用的学习策略
    :return: 生成的决策树的根节点
    """

    assert train_set
    node = TreeNode()

    # 如果训练集中的样例属于同一类，标记社为该类，返回根节点
    if len(Counter(list(map(lambda x: x[-1], train_set))).keys()) == 1:
        node.label = train_set[0][-1]
        return node
    else:
        same_flag = True
        # 判断是不是在参与决策的所有属性上，训练集的类别都一致，虽然严格来说实数值都相等不太可能
        for k in attrs.keys():
            if not len(Counter(list(map(lambda x: x[k], train_set))).keys()) == 1:
                same_flag = False
                break
        # 或者是否属性已使用完
        if not attrs or same_flag:
            labels = sorted(Counter(list(map(lambda x: x[-1], train_set))).items(), key=lambda x: x[1], reverse=True)
            node.label = labels[0][0]
            return node

    # 偷懒写法
    def measure(x, i, atype, mid_value=0):
        return 0

    if strat == inf_ent_gain:
        measure = __info_ent_gain
    elif strat == gini_inx:
        measure = __gini_index
    elif strat == info_gain_ratio:
        assert len(list(filter(lambda x: x[0] == real, attrs.values()))) == 0
        measure = __info_ent_gain

    final_i = 0
    final_mid = 0
    max_measure = 0
    # 判断收益最大的属性
    for item in attrs.items():
        # 属性为实数时
        if item[1][0] == real:
            final_mid = 0
            max_mid_measure = 0
            values = list(map(lambda x: x[item[0]], train_set))
            values = sorted(list(filter(lambda v: item[1][1][0] <= v < item[1][1][1], values)))  # attrs规定范围内在数据集中的所有取值
            mids = list(map(lambda i: (values[i] + values[i + 1]) / 2, range(len(values) - 1)))  # 所有中间值
            # 找到使判断数值最大的中间值
            for mid in mids:
                ms = measure(train_set, i=item[0], atype=real, mid_value=mid)
                if ms > max_mid_measure:
                    max_mid_measure = ms
                    final_mid = mid
                    final_i = item[0]
        # 属性为离散值时
        else:
            ms = measure(train_set, i=item[0], atype=discrete)
            if ms > max_measure:
                max_measure = ms
                final_i = item[0]

    # 按该属性上的取值对数据集进行分类
    node.ai = final_i
    if attrs[final_i][0] == real:
        attrs1 = deepcopy(attrs)
        attrs1[final_i][1][1] = final_mid  # 取值缩为(min, final_mid)
        attrs2 = deepcopy(attrs)
        attrs2[final_i][1][0] = final_mid  # 取值缩为(final_mid, max)
        node.standard = final_mid
        node.atype = real
        # 判断分割后不是不是有一端没有符合的数据
        labels = sorted(Counter(list(map(lambda x: x[-1], train_set))).items(), key=lambda x: x[1],
                        reverse=True)
        max_label = labels[0][0]
        s1, s2 = __split(train_set, final_i, final_mid)
        if len(s1) == 0:
            child = TreeNode()
            child.label = max_label
            node.children[0] = child
        else:
            node.children[0] = __generate_tree(s1, attrs1, strat)
        if len(s2) == 0:
            child = TreeNode()
            child.label = max_label
            node.children[1] = child
        else:
            node.children[1] = __generate_tree(s2, attrs2, strat)
    else:
        # 获得训练集在该属性上的所有取值
        value_counts = {}
        for value in attrs[final_i][1]:
            value_counts[value] = list(filter(lambda x: x[final_i] == value, train_set))
        attrs.pop(final_i)  # 移除该属性
        # 对每一种取值进行判断
        for value in value_counts:
            child = TreeNode()
            # 如果在该取值上无样例
            if not value_counts[value]:
                # 赋予原数据集中比例最大的标签
                labels = sorted(Counter(list(map(lambda x: x[-1], train_set))).items(), key=lambda x: x[1],
                                reverse=True)
                child.label = labels[0][0]
            else:
                sub_set = value_counts[value]
                child = __generate_tree(sub_set, attrs.copy(), strat)
            node.children[value] = child
        node.atype = discrete
    return node


class TreeApplier(MultiClassApplier):
    """
    实数决策树的模型应用器
    """

    def __init__(self, root: TreeNode, model: list = None):
        super().__init__(model)
        self.root = root

    def predict(self, x: list) -> Any:
        return self.root.decide(x)
