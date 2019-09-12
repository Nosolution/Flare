import numpy as np
from auxiliary.helper import count_class_num
import math
from applier.multiapplier import NNApplier, MultiClassApplier
from auxiliary.helper import sigmoid


def g(pred: float, y: float) -> float:
    return pred * (1 - pred) * (y - pred)


def e(G: np.ndarray, W: list, b: float) -> float:
    return b * (1 - b) * sum(map(lambda j: W[j] * G[j], range(len(W))))


def to_binary(y, labels) -> list:
    res = np.zeros(len(labels))
    for j in range(len(labels)):
        if y == labels[j]:
            res[j] = 1
            return list(res)


def standard_bp(train_set: list, eta: float = 0.6, expect_error=0.05, max_iter=1000) -> MultiClassApplier:
    assert len(train_set) > 0
    l = count_class_num(train_set)
    d = len(train_set[0]) - 1
    train_set = np.array(train_set)
    q = math.ceil(math.sqrt(d * l))
    output_layer = np.random.random(l)
    oh_weight = np.random.random((l, q))
    hidden_layer = np.random.random(q)
    hi_weight = np.random.random((q, d))
    Y = train_set[:, -1]
    labels = sorted(list(set(Y)))
    applier = NNApplier([hi_weight, hidden_layer, oh_weight, output_layer])
    while max_iter > 0:
        max_iter -= 1
        for i in range(len(train_set)):
            x = train_set[i][:-1]
            y = to_binary(train_set[i][-1], labels)
            hidden_out = []
            for h in range(len(hidden_layer)):
                hidden_out.append(
                    sigmoid(sum(map(lambda i: (x[i] * hi_weight[h][i] - hidden_layer[h]), range(len(x))))))
            hidden_out = np.array(hidden_out)
            pred = applier.confidence(x)
            error = sum(map(lambda i: (y[i] - pred[i]) ** 2, range(l))) / 2
            if error < expect_error:
                return applier

            G = np.array(map(lambda j: g(pred[j], y[j]), range(l)))
            E = np.array(map(lambda h: e(G, oh_weight.T[h], hidden_out[h]), range(q)))

            delta_w = np.tensordot(G.reshape((l, 1)), hidden_out) * eta
            delta_theta = -eta * G
            delta_v = np.tensordot(E.reshape((q, 1)), x) * eta
            delta_gamma = -eta * E

            output_layer += delta_theta
            hidden_layer += delta_gamma
            oh_weight += delta_w
            hi_weight += delta_v

        return applier
