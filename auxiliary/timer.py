import time
from typing import Callable

"""
可能用于计时
"""


def time_method(runnable: Callable, *args, **kwargs) -> None:
    """
    计算一个method的运行时间, 并输出
    :param runnable: 可运行的函数
    :param args: 函数所需的参数
    """
    start = time.time()
    runnable(*args, **kwargs)
    end = time.time()
    print("Spent time is: {}".format(end - start))
