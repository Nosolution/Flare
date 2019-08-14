from typing import Union

from applier.binaryapplier import BinaryApplier
from applier.multiapplier import MultiClassApplier

"""
测试单元，完成测试工作
"""


class TestUnit(object):
    """
    测试类
    """

    def __init__(self):
        self.__refresh()
        self.binary_flag = False  # 是否为二分类任务
        self.f1_flag = False  # 是否输出F1指标(仅在开启二分类后有效)
        self.__f1_preference = 1  # 查准率/查全率的偏好比率

    @property
    def f1_preference(self) -> float:
        return self.__f1_preference

    @f1_preference.setter
    def f1_preference(self, value: float) -> None:
        assert value > 0
        self.__f1_preference = value

    def __refresh(self) -> None:
        self.t = 0  # 预测正确实例数量
        self.f = 0  # 预测错误实例数量

    def test(self, applier: Union[BinaryApplier, MultiClassApplier], test_set: list) -> None:
        """
        进行测试, 输出测试结果
        :param applier: 模型应用器
        :param test_set: 测试集, 要求格式[(x_1, x_2, x_3,..., label),..., [x_1, x_2, x_3,..., label]]
        """
        self.__refresh()
        # 如果开启二分类模式，记录tp, tn, fp, fn
        if self.binary_flag:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
        for data in test_set:
            x = data[:-1]
            y = data[-1]
            pred = applier.predict(x)
            if pred == y:
                self.t += 1
            else:
                self.f += 1
            if self.binary_flag:
                if pred == y:
                    if pred == 0:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if pred == 0:
                        fp += 1
                    else:
                        fn += 1
        params = {}
        if self.binary_flag:
            params['tp'] = tp
            params['tn'] = tn
            params['fp'] = fp
            params['fn'] = fn
        self.__print_test_res(params)

    def __print_test_res(self, params: dict = None) -> None:
        """
        输出测试结果, 如果开启了二分类模式会要求额外参数
        :param params: {'tp':<tp>, 'tn':<tn>, 'fp':<fp>, 'fn':<fn>} (目前)
        """
        print("The accuracy is: {}".format(self.t / (self.t + self.f)))
        if self.binary_flag:
            print("TP is: {}, FP is: {}, TN is: {}, FN is: {}.".format(params['tp'], params['fp'],
                                                                       params['tn'], params['fn']))
            p = params['tp'] / (params['tp'] + params['fp'])
            r = params['tp'] / (params['tp'] + params['fn'])
            print("The precise rate is: {}, and the recall rate is: {}".format(p, r))
            if self.f1_flag:
                beta = self.__f1_preference
                f1 = ((1 + beta) ** 2 * p * r) / ((beta ** 2 * p) + r)
                print("F1 is: {}".format(f1))
