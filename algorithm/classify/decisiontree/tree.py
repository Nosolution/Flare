from collections import Counter
from typing import Tuple, Any, Union
from copy import deepcopy
import numpy as np
from .pruning import Prepruner
from applier import MultiClassApplier
from queue import Queue
from typing import Callable

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
        self.depth = 0
        self.height = 0
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

    def serialize(self):
        if not self.children:
            return self.label
        else:
            res = {}
            children_dict = {}
            for item in self.children.items():
                children_dict[item[0]] = item[1].serialize()
            res[self.ai] = children_dict
            return res


def __max_label(data_set: list) -> Any:
    return sorted(Counter(list(map(lambda x: x[-1], data_set))).items(), key=lambda x: x[1], reverse=True)[0][0]


def __same_label(data_set: list) -> bool:
    return __same_attr_value(data_set, -1)


def __same_attr_value(data_set: list, i: int) -> bool:
    return len(set(map(lambda x: x[i], data_set))) == 1


def __get_measure_func(strat: int) -> Callable:
    def measure(x, i, atype, mid_value=0):
        return 0

    if strat == inf_ent_gain:
        measure = __info_ent_gain
    elif strat == gini_inx:
        measure = __gini_index
    elif strat == info_gain_ratio:
        measure = __info_ent_gain

    return measure


def __can_distinguish(data_set: list, attrs: dict) -> bool:
    if not attrs:
        return False
    if __same_label(data_set):
        return True
    else:
        same_flag = True
        # 判断是不是在参与决策的所有属性上，训练集的取值都一致，虽然严格来说实数值都相等不太可能
        for k in attrs.keys():
            if not __same_attr_value(data_set, k):
                same_flag = False
                break
        # 或者是否属性已使用完
        return not same_flag


def __judge_max_measure_i(data_set: list, attrs: dict, measure: Callable) -> Tuple[int, float]:
    final_i = 0
    if attrs:
        final_i = list(attrs.items())[0][0]
    final_mid = 0
    max_measure = 0
    # 判断收益最大的属性
    for item in attrs.items():
        # 属性为实数时
        if item[1][0] == real:
            final_mid = 0
            max_mid_measure = 0
            values = list(map(lambda x: x[item[0]], data_set))
            values = sorted(list(filter(lambda v: item[1][1][0] <= v < item[1][1][1], values)))  # attrs规定范围内在数据集中的所有取值
            mids = list(map(lambda i: (values[i] + values[i + 1]) / 2, range(len(values) - 1)))  # 所有中间值
            # 找到使判断数值最大的中间值
            for mid in mids:
                ms = measure(data_set, i=item[0], atype=real, mid_value=mid)
                if ms > max_mid_measure:
                    max_mid_measure = ms
                    final_mid = mid
                    final_i = item[0]
        # 属性为离散值时
        else:
            ms = measure(data_set, i=item[0], atype=discrete)
            if ms > max_measure:
                max_measure = ms
                final_i = item[0]
    return final_i, final_mid


def __need_preprune(data_set: list, node: TreeNode, i: int, prepruner: Prepruner, **kwargs) -> bool:
    max_label = __max_label(data_set)
    tr = prepruner.test(node)
    if 'mid' in kwargs:
        leq_func = (lambda x: x[i] <= kwargs['mid'])
        gr_func = (lambda x: x[i] > kwargs['mid'])
        s1, s2 = __split(data_set, i, kwargs['mid'])
        node.label = max_label
        left_child = TreeNode()
        left_child.label = __max_label(s1)
        right_child = TreeNode()
        right_child.label = __max_label(s2)

        t1 = prepruner.filtered(leq_func).test(left_child)
        t2 = prepruner.filtered(gr_func).test(right_child)
        return tr >= t1 + t2
    else:
        values = kwargs['values']
        value_counts = {}
        for value in values:
            value_counts[value] = list(filter(lambda x: x[i] == value, data_set))
        tc = 0
        for value in value_counts:
            child = TreeNode()
            eq_func = (lambda x: x[i] == value)
            if not value_counts[value]:
                child.label = max_label
            else:
                child.label = __max_label(list(filter(eq_func, data_set)))
            tc += prepruner.filtered(eq_func).test(child)
        return tr >= tc


def generate_tree(train_set: list, attrs: dict, strat: int, prepruner: Prepruner = None,
                  map_depth: int = 0) -> MultiClassApplier:
    if map_depth > 0:
        return TreeApplier(__control_generate_tree(train_set, attrs, strat, prepruner, map_depth))
    else:
        return TreeApplier(__generate_tree(train_set, attrs, strat, prepruner))


def __generate_tree(train_set: list, attrs: dict, strat: int, prepruner: Prepruner) -> TreeNode:
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
    cur_max_label = __max_label(train_set)

    # 如果训练集中的样例属于同一类，标记社为该类，返回根节点
    if not __can_distinguish(train_set, attrs):
        node.label = cur_max_label
        return node

    measure = __get_measure_func(strat)  # 偷懒写法

    final_i, final_mid = __judge_max_measure_i(train_set, attrs, measure)

    # 按该属性上的取值对数据集进行分类
    node.atype = attrs[final_i][0]
    node.ai = final_i
    if attrs[final_i][0] == real:
        s1, s2 = __split(train_set, final_i, final_mid)
        leq_func = (lambda x: x[final_i] <= final_mid)
        gr_func = (lambda x: x[final_i] > final_mid)

        if prepruner and __need_preprune(data_set=train_set, node=node, i=final_i, prepruner=prepruner,
                                         mid=final_mid):  # 如果需要测试是否预剪枝
            node.label = cur_max_label

        else:
            attrs1 = deepcopy(attrs)
            attrs1[final_i][1][1] = final_mid  # 取值缩为(min, final_mid)
            attrs2 = deepcopy(attrs)
            attrs2[final_i][1][0] = final_mid  # 取值缩为(final_mid, max)
            node.standard = final_mid
            # 判断分割后不是不是有一端没有符合的数据
            if len(s1) == 0:
                child = TreeNode()
                child.label = cur_max_label
                node.children[0] = child
            else:
                node.children[0] = __generate_tree(s1, attrs1, strat,
                                                   prepruner.filtered(leq_func) if prepruner else None)
            if len(s2) == 0:
                child = TreeNode()
                child.label = cur_max_label
                node.children[1] = child
            else:
                node.children[1] = __generate_tree(s2, attrs2, strat,
                                                   prepruner.filtered(gr_func) if prepruner else None)
    else:
        values = attrs[final_i][1]
        attrs.pop(final_i)  # 移除该属性

        if prepruner and __need_preprune(data_set=train_set, node=node, i=final_i, prepruner=prepruner,
                                         values=values):
            # if (prepruner is not None) and False:
            node.label = cur_max_label
        else:
            value_counts = {}
            # 获得训练集在该属性上的所有取值
            for value in values:
                # 对每一种取值进行判断
                value_counts[value] = list(filter(lambda x: x[final_i] == value, train_set))
            for value in value_counts:
                child = TreeNode()
                eq_func = (lambda x: x[final_i] == value)
                # 如果在该取值上无样例
                if not value_counts[value]:
                    # 赋予原数据集中比例最大的标签
                    child.label = __max_label(train_set)
                else:
                    sub_set = value_counts[value]
                    child = __generate_tree(sub_set, attrs.copy(), strat,
                                            prepruner.filtered(eq_func) if prepruner else None)
                node.children[value] = child
    return node


def __control_generate_tree(train_set: list, attrs: dict, strat: int, prepruner: Prepruner,
                            max_depth: int) -> TreeNode:
    """
    使用队列控制生成决策树的最大深度
    :param train_set: 训练集, 要求属性值都为整数, 格式: [(x_1,x_2,x_3,...,label),...,(x_1,x_2,x_3,...,label)]
    :param attrs: 参与决策的属性及相关信息, 格式: {attr1:[atype, [value_1, value_2,...]], attr2:[atype, [value_1, value_2,...]],...}
    如果atype==real, 则value元组为attr取值范围[min, max], 否则为离散属性的所有取值[value_1, value_2,...]
    :param strat: 采用的学习策略
    :param max_depth: 生成决策树的最大深度
    :return: 生成的决策树的根节点
    """
    assert train_set
    node_queue = Queue()
    root = TreeNode()
    node_queue.put((root, train_set, attrs, prepruner))
    while not node_queue.empty():
        node, train_set, attrs, prepruner = node_queue.get()
        cur_max_label = __max_label(train_set)

        if node.depth == max_depth:  # 如果已经到达最大深度
            node.label = cur_max_label
            continue
        else:
            # 如果训练集中的样例属于同一类，标记社为该类，返回根节点
            if not __can_distinguish(train_set, attrs):
                node.label = cur_max_label
                continue

            measure = __get_measure_func(strat)  # 偷懒写法
            # 找到增益最大的属性下标
            final_i, final_mid = __judge_max_measure_i(train_set, attrs, measure)
            # 按该属性上的取值对数据集进行分类
            node.atype = attrs[final_i][0]
            node.ai = final_i

            if attrs[final_i][0] == real:
                s1, s2 = __split(train_set, final_i, final_mid)
                leq_func = (lambda x: x[final_i] <= final_mid)
                gr_func = (lambda x: x[final_i] > final_mid)

                if prepruner and __need_preprune(data_set=train_set, node=node, i=final_i, prepruner=prepruner,
                                                 mid=final_mid):  # 如果需要测试是否预剪枝
                    node.label = cur_max_label

                else:
                    attrs1 = deepcopy(attrs)
                    attrs1[final_i][1][1] = final_mid  # 取值缩为(min, final_mid)
                    attrs2 = deepcopy(attrs)
                    attrs2[final_i][1][0] = final_mid  # 取值缩为(final_mid, max)
                    node.standard = final_mid
                    # 判断分割后不是不是有一端没有符合的数据
                    left_child = TreeNode()
                    node.children[0] = left_child
                    right_child = TreeNode()
                    node.children[1] = right_child

                    if len(s1) == 0:
                        left_child.label = cur_max_label
                    else:
                        node_queue.put((left_child, s1, attrs1, prepruner.filtered(leq_func)) if prepruner else None)
                    if len(s2) == 0:
                        right_child.label = cur_max_label
                    else:
                        node_queue.put((right_child, s2, attrs2, prepruner.filtered(gr_func)) if prepruner else None)
            else:
                values = attrs[final_i][1]
                attrs.pop(final_i)  # 移除该属性

                if prepruner and __need_preprune(data_set=train_set, node=node, i=final_i, prepruner=prepruner,
                                                 values=values):
                    node.label = cur_max_label
                else:
                    value_counts = {}
                    # 获得训练集在该属性上的所有取值
                    for value in values:
                        # 对每一种取值进行判断
                        value_counts[value] = list(filter(lambda x: x[final_i] == value, train_set))
                    for value in value_counts:
                        child = TreeNode()
                        eq_func = (lambda x: x[final_i] == value)
                        node.children[value] = child
                        # 如果在该取值上无样例
                        if not value_counts[value]:
                            # 赋予原数据集中比例最大的标签
                            child.label = __max_label(train_set)
                        else:
                            sub_set = value_counts[value]
                            node_queue.put((sub_set, attrs.copy(), prepruner.filtered(eq_func)) if prepruner else None)

    return root


class TreeApplier(MultiClassApplier):
    """
    实数决策树的模型应用器
    """

    def __init__(self, root: TreeNode, model: list = None):
        super().__init__(model)
        self.root = root

    def predict(self, x: list) -> Any:
        return self.root.decide(x)
