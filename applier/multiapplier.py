import random
from collections import Counter

__all__ = ['MultiClassApplier', 'OvOApplier', 'OvRApplier', 'MvMApplier']


class MultiClassApplier(object):
    """
    多分类实数分类模型的应用器类, 因其判断逻辑不能很好地对二分类实数分类模型应用器对进行向上兼容, 因此选择不继承BinaryApplier
    """

    def __init__(self, model: list) -> None:
        self.model = model

    def predict(self, x: list) -> int:
        return 0


class OvOApplier(MultiClassApplier):
    """
    一对一多分类师叔分类模型应用器
    """

    def predict(self, x: list) -> int:
        predictions = []
        for model in self.model:
            predictions.append(model[-1] if model[0].predict(x) == 0 else model[-2])
        # 所有预测结果按重复次数从高到低排序
        c = sorted(list(Counter(predictions)), key=(lambda y: y[-1]), reverse=True)
        i = 1
        # 找出最高的几组
        while c[i][1] == c[i - 1][1]:
            i += 1
        # 随机选取一种结果
        return c[random.randint(0, i - 1)][0]


class OvRApplier(MultiClassApplier):
    """
    一对多多分类实数分类模型应用器
    """

    def predict(self, x: list) -> int:
        predictions = []
        for model in self.model:
            predictions.append((model[0].confidence(x), model[-1]))
        # 根据置信度排序选取最高类别
        return sorted(predictions, key=lambda y: y[0], reverse=True)[0][1]


class MvMApplier(MultiClassApplier):
    """
    多对多多分类师叔分类模型应用器
    """

    def predict(self, x: list) -> int:
        code = list(map(lambda model: model(x), self.model))
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
