import numpy as np


def calc_C(r):
    def calc_next(r):
        result = np.concatenate((r, np.zeros((r.shape[0], 1))), axis=1)
        result2 = np.concatenate(
            (r + np.ones((r.shape[0], 1)) @ r[-1:], np.ones((r.shape[0], 1))), axis=1
        )
        return np.concatenate((result, result2), axis=0)

    cur = np.array([[0], [1]])
    for i in range(r - 1):
        cur = calc_next(cur)
    return cur


def calc_K(r):
    def calc_next(r):
        result = np.concatenate((r, np.zeros((r.shape[0], 1))), axis=1)
        result2 = np.concatenate((np.flipud(r), np.ones((r.shape[0], 1))), axis=1)
        return np.concatenate((result, result2), axis=0)

    cur = np.array([[0], [1]])
    for i in range(r - 1):
        cur = calc_next(cur)
    return cur


def calc_Z(r):
    def calc_next(r):
        result = np.concatenate((r, np.zeros((r.shape[0], 1))), axis=1)
        result2 = np.concatenate((r, np.ones((r.shape[0], 1))), axis=1)
        return np.concatenate((result, result2), axis=0)

    cur = np.array([[0], [1]])
    for i in range(r - 1):
        cur = calc_next(cur)
    return cur
