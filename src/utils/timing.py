import time
import functools
from contextlib import contextmanager


def timing(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start_time = time.clock()
        func(*args, **kwargs)
        elapsed_time = time.clock() - start_time
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsed_time * 1000)))
    return newfunc


@contextmanager
def timeit_context(name):
    start_time = time.clock()
    yield
    elapsed_time = time.clock() - start_time
    print('[{}] finished in {} ms'.format(name, int(elapsed_time * 1000)))
