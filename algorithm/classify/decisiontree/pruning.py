from typing import Callable
from copy import deepcopy

__all__ = ['Prepruner']


class Prepruner(object):
    def __init__(self, test_set: list):
        self.test_set = test_set

    def filtered(self, judge_func: Callable[[list], bool]):
        res = Prepruner(list(filter(lambda x: judge_func(x), deepcopy(self.test_set))))
        return res

    def data_size(self) -> int:
        return len(self.test_set)

    def test(self, node) -> int:
        return len(list(filter(lambda x: node.decide(x[:-1]) == x[-1], self.test_set)))
