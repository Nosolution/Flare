import copy

from trainer.applier.binaryapplier import *

"""
最基本的二分类实数分类器
因为不用对属性进行处理而最先实现
"""

# 设定为有warning时抛出
np.seterr(over='raise')


def linear_regression(train_set: list, debug_mode: bool = False, **kwargs) -> BinaryApplier:
    """
    线性回归，模型: y = Wx/Y = WX.
    其中x = (x_1,x_2,...x_n, 1), x_i为在属性a_i上的取值. W = (w_1, w_2, w_3,...,w_n, b), b为偏置项.
    :param train_set: 训练集, 要求格式[(x_1,x_2,x_3,...,label),...,(x_1,x_2,x_3,...,label)]
    :param debug_mode: 产生debug用输出
    :return: 训练完毕的模型
    """

    assert len(train_set) > 0
    X = np.array(list(map(lambda x: np.array([*(x[:-1]), 1]), train_set)))
    Y = np.array(list(map(lambda x: x[-1], train_set))).reshape(len(train_set), 1)
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    w = w.flatten()
    if debug_mode:
        print('W is: {}'.format(w))
    return LinearApplier(w)


def likelihood(w: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    对数似然的估计函数, 算式: l(<omega>) = sum_{i=1}^{m}(-y_i<omega>x_i) + ln(1+e^{<omega>x_i}), 其中x_i, y_i为第i个实例的数据与标记.
    :param w: <omega>
    :param X: 属性集， 要求格式: [(x_1,x_2,x_3,..., x_m, 1),...,(x_1,x_2,x_3,..., x_m, 1)]
    :param Y: 标记集, 要求格式
    :return: 对应的对数似然值
    """
    return np.sum(-Y * np.dot(X, w) + np.log(1 + np.exp(np.dot(X, w))))


def nega_llh_gradient(w: list, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    对数似然的梯度负值, 算式:-grad(l(<omega>)) = sum_{i=1}^{m}(x_i(y_i-p_1(x_i;<omega>))),
    其中p_1(x_i;<omega>) = 1-1/(1+e^{<omega>x_i})
    :param w: <omega>
    :param Y: 标记集, 要求格式
    :param X: 属性集， 要求格式: [(x_1,x_2,x_3,..., x_m, 1),...,(x_1,x_2,x_3,..., x_m, 1)]
    :return: 对应的对数似然梯度负值
    """
    Z = 1 / (1 + np.exp(-np.dot(X, w)))
    return np.tensordot(X.T, (Y - Z), (1, 0))


def logistic_regression(train_set: list,
                        step: float = 0.0001,
                        min_step: int = 0.0,
                        max_round: int = 1000,
                        max_detect: int = 1000,
                        debug_mode: bool = False,
                        **kwargs) -> BinaryApplier:
    """
    对数几率回归, 模型: ln(y/(1-y)) = Wx.
    其中x = (x_1,x_2,...x_n, 1), x_i为在属性a_i上的取值. W = (w_1, w_2, w_3,...,w_n, b), b为偏置项.
    使用梯度下降法训练
    :param train_set: 训练集, 要求格式[(x_1,x_2,x_3,...,label),...,(x_1,x_2,x_3,...,label)]
    :param step: 下降步长
    :param min_step: 最小步长，小于该值则退出训练
    :param max_round: 最多训练轮数(下降次数)
    :param max_detect: 一次下降过程中的最多探测次数，
    :param debug_mode: 产生debug用输出
    :return: 训练完毕的模型
    """
    assert len(train_set) > 0

    w = np.zeros(len(train_set[0]))
    round_count = 0
    X = np.array(train_set)[:, 0:-1]
    X = np.array([np.append(i, 1) for i in X])
    Y = np.array(train_set)[:, -1]

    # 开始训练
    while round_count < max_round and step > min_step:
        round_count += 1
        origin_llh = likelihood(w, X, Y)
        origin_w = w
        detect_count = 0

        delta = nega_llh_gradient(w, X, Y)

        if debug_mode:
            print('current round_count is: {}, llh is:{},\nw is:{}.'.format(round_count, origin_llh, origin_w))

        # 找出使llh减少的最大步长
        while detect_count < max_detect and step > min_step:
            detect_count += 1
            w = origin_w + step * delta
            if likelihood(w, X, Y) <= origin_llh:
                break
            step /= 2
        if debug_mode:
            print('total detect_count is: {}.'.format(detect_count))
    w = w.flatten()
    if debug_mode:
        print('final w is: {}'.format(w))

    return LogisticApplier(w)


def lda(train_set: list, debug_mode: bool = False, **kwargs) -> BinaryApplier:
    """
    线性判别分析, 模型: y = Wx.
    其中x = (x_1,x_2,...x_n), x_i为在属性a_i上的取值. W = (w_1, w_2, w_3,...,w_n).
    :param train_set: 训练集, 要求格式[(x_1,x_2,x_3,...,label),...,(x_1,x_2,x_3,...,label)]
    :param debug_mode: 产生debug用输出
    :return: 训练完毕的模型
    """
    param_len = len(train_set[0]) - 1
    posi_set = []
    nega_set = []
    for i in range(len(train_set)):
        if train_set[i][-1] == 1:
            posi_set.append(copy.copy(train_set[i][:-1]))
        else:
            nega_set.append(copy.copy(train_set[i][:-1]))
    # 获取反例与正例的类中心
    u1 = np.zeros(param_len)
    u0 = np.zeros(param_len)
    for data in posi_set:
        u1 += data
    u1 /= len(posi_set)
    for data in nega_set:
        u0 += data
    u0 /= len(nega_set)

    # 类间散度矩阵
    sw0 = np.zeros((param_len, param_len))
    sw1 = np.zeros((param_len, param_len))
    for data in posi_set:
        r = data - u1
        sw1 += r.reshape(param_len, 1).dot(r.reshape(1, param_len))
    for data in nega_set:
        r = data - u0
        sw0 += r.reshape(param_len, 1).dot(r.reshape(1, param_len))
    sw = sw1 + sw0

    w = np.linalg.inv(sw).dot(u0.reshape(param_len, 1) - u1.reshape(param_len, 1))
    w = w.flatten()
    if debug_mode:
        print('u0 is:{}, u1 is: {}.\nsw is: {}, w is: {}'.format(u0, u1, sw, w))
    return LDAApplier([w.reshape(param_len), u0, u1])
