import numpy as np


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

    def confidence(self, x: list):
        """
        返回样例被分为正例的置信度
        :param x: 被测试的样例
        :return: 样例被分类为正类的置信度
        """
        return 0.5

    def predict(self, x: list) -> int:
        """
        预测样例的分类
        :param x: 被测试的样例
        :return: 返回1，如果样例被预测为1的置信度更大，反之返回0
        """
        return 1


class LinearApplier(BinaryApplier):
    """
    线性回归模型的应用器
    """

    def confidence(self, x: list):
        x = np.array([*x, 1])
        r = self.w.dot(x)
        return r

    def predict(self, x: list) -> int:
        r = self.confidence(x)
        return 1 if r > 0.5 else 0


class LogisticApplier(BinaryApplier):
    """
    对率回归模型的应用器
    """

    def confidence(self, x: list):
        x = np.array([*x, 1])
        r = self.w.dot(x)
        return r

    def predict(self, x: list) -> int:
        r = self.confidence()
        return 1 if r > 1 else 0


class LDAApplier(BinaryApplier):
    """
    线性判别分析模型的应用器
    """

    def confidence(self, x: list):
        x = np.array(x)
        u0 = self.w[-1]
        u1 = self.w[-2]
        r = self.w[0].dot(x)
        return abs(self.w[0].dot(u1) - r) / abs(self.w[0].dot(u0) - r)

    def predict(self, x: list) -> int:
        x = np.array(x)
        u0 = self.w[-1]
        u1 = self.w[-2]
        r = self.w[0].dot(x)
        return 1 if abs(self.w[0].dot(u1) - r) < abs(self.w[0].dot(u0) - r) else 0
