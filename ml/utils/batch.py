import numpy as np
import itertools

def blank(*args):
    return args


def randomiser(iterable):
    np.random.shuffle(iterable)
    return iterable


def repeater(iterable):
    return itertools.cycle(iterable)


def shuffle_repeater(iterable):
    while True:
        yield next(repeater(randomiser(iterable)))
    


