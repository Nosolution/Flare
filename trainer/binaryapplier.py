import numpy as np
from collections import Counter
import random


# 没有查python有没有抽象类的概念
class BinaryApplier(object):
    """
    二分类实数模型应用器类，封装不同算法训练出的模型的具体判断过程, 完成对测试的解耦
    """

    def __init__(self, w: list) -> None:
        # 目前只有线性模型的算法, 因此固定参数只有w
        self.__w = np.array(w)

    @property
    def w(self) -> np.ndarray:
        return self.__w

    @w.setter
    def w(self, value) -> None:
        assert (isinstance(value, list) or isinstance(value, np.ndarray))
        self.__w = np.array(value)

    def predict(self, x: list) -> int:
        # 基本方法，各子类实现自己的决策过程
        return 0


class LinearApplier(BinaryApplier):
    """
    线性回归模型的应用器
    """

    def predict(self, x: list) -> int:
        x = np.array([*x, 1])
        r = self.__w.dot(x)
        return 1 if r > 0.5 else 0


class LogisticApplier(BinaryApplier):
    """
    对率回归模型的应用器
    """

    def predict(self, x: list) -> int:
        x = np.array([*x, 1])
        r = self.__w.dot(x)
        return 1 if r > 1 else 0


class LDAApplier(BinaryApplier):
    """
    线性判别分析模型的应用器
    """

    def predict(self, x: list) -> int:
        x = np.array(x)
        u0 = self.w[-2]
        u1 = self.w[-1]
        r = self.w.dot(x)
        return 1 if abs(self.w.dot(u1) - r) < abs(self.w.dot(u0) - r) else 0


class MultiClassApplier(object):
    """
    多分类实数分类模型的应用器类, 因其判断逻辑不能很好地对二分类实数分类模型应用器对进行向上兼容, 因此选择不继承BinaryApplier
    """

    def __init__(self, models: list) -> None:
        self.models = models

    def predict(self, x: list) -> int:
        return 0


class OvOApplier(MultiClassApplier):
    """
    一对一多分类师叔分类模型应用器
    """

    def predict(self, x: list) -> int:
        predictions = []
        for model in self.models:
            predictions.append(model[-1] if model[0].predict(x) == 0 else model[-2])
        # 所有预测结果按重复次数从高到低排序
        c = sorted(list(Counter(predictions)), key=(lambda y: y[-1]), reverse=True)
        i = 1
        # 找出最高的几组
        while c[i][1] == c[i - 1][1]:
            i += 1
        # 随机选取一种结果
        return c[random.randint[0, i - 1]][0]


class OvRApplier(MultiClassApplier):
    """
    一对多多分类师叔分类模型应用器
    """

    def predict(self, x: list) -> int:
        predictions = []
        for model in self.models:
            predictions.append(model[-1] if model[0].predict(x) == 0 else -1 * model[-1])
        c = list(filter(lambda y: y > 0, predictions))
        # TODO OvR需要实现置信度
        if len(c) == 0:
            return 0
        elif len(c) == 1:
            return c[0]
        else:
            return c[random[0, len(c) - 1]]


class MvMApplier(MultiClassApplier):
    """
    多对多多分类师叔分类模型应用器
    """

    def predict(self, x: list) -> int:
        code = list(map(lambda model: model(x), self.models))
        # 找出海明距离最小的类
        index = 0
        min_d = 0
        # 自加属性: codewords = [[codeword1, class1], [codeword2, class2],...,[codewordn, classn]]
        codeword = self.codewords[0][0]
        for i in range(len(code)):
            if code[i] != codeword[i]:
                min_d += 1
        for k in range(len(self.codewords)):
            codeword = self.codewords[k][0]
            d = 0
            for i in range(len(code)):
                if code[i] != codeword[i]:
                    d += 1
            if d < min_d:
                min_d = d
                index = k
        return self.codewords[index][1]
