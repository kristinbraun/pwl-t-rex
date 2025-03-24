from dataclasses import dataclass
import numpy as np
from scipy.special import gamma
import math
import pyomo.environ as pyo
from typing import Tuple


@dataclass
class Operation:
    symbol: str
    dim: Tuple[int, int]
    num_param: None

    def __repr__(self):
        return self.symbol + ("" if self.num_param is None else " (" + self.num_param + ")")

    def __init__(self, symbol, dim, num_param=None):
        self.symbol = symbol
        self.dim = dim
        self.num_param = num_param


@dataclass
class LinearOperation(Operation):
    def __init__(self, symbol, dim, num_param=None):
        super().__init__(symbol, dim, num_param)


@dataclass
class Reformulation:
    pass


@dataclass
class NonlinearOperation(Operation):
    def __init__(self, symbol, dim, num_param=None):
        super().__init__(symbol, dim, num_param)


def get_line(level):
    return ((level - 1) * " " + "|" + "-") if level > 0 else ""


@dataclass
class Nonlinear_ExpTree:
    num_children: int
    children: list
    operation: Operation
    nl_idx: int
    root_idx: int

    def __init__(self, num_children, children, operation, nl_idx=-1, root_idx=-1):
        self.num_children = num_children
        self.children = children
        self.operation = operation
        self.nl_idx = nl_idx
        self.root_idx = root_idx

    def __repr__(self):
        res = str(self.operation.symbol) + "("
        for i in range(self.num_children):
            res += self.children[i].__repr__()
            if i < self.num_children - 1:
                res += ", "
        res += ")"
        return res

    def get_tree(self, level="", last=False):
        res = ""
        res += level + str(self.operation.symbol)
        res += " (" + str(self.nl_idx) + ', NL-Root ' + str(self.root_idx) + ")"
        res += "\n"
        for i in range(self.num_children):
            if last:
                leveln = level.replace("+", " ")
            else:
                leveln = level.replace("+", "|")
            leveln = leveln.replace("-", " ")
            res += self.children[i].get_tree(
                leveln + " +--", last=(i == self.num_children - 1)
            )
        return res


@dataclass
class Variable(Nonlinear_ExpTree):
    coef: float
    idx: int
    lb: float
    ub: float

    def __repr__(self):
        return (
            "["
            + str(self.coef)
            + "*var:"
            + str(self.idx)
            + " | ("
            + str(self.lb)
            + ","
            + str(self.ub)
            + ")"
            + "]"
        )

    def get_tree(self, level="", last=False):
        return level + "[" + str(self.coef) + "*var:" + str(self.idx) + "] (" + str(self.nl_idx) + ', NL-Root ' + str(self.root_idx) + ")\n"

    def __init__(self, idx, coef=1, lb=-np.inf, ub=np.inf, nl_idx=-1, root_idx=-1):
        self.coef = coef
        self.idx = idx
        self.lb = lb
        self.ub = ub
        self.num_children = 0
        self.children = []
        self.operation = None
        self.nl_idx = nl_idx
        self.root_idx = root_idx


@dataclass
class Number(Nonlinear_ExpTree):
    value: float

    def __init__(self, value, nl_idx=-1, root_idx=-1):
        self.value = value
        self.num_children = 0
        self.children = []
        self.operation = None
        self.nl_idx = nl_idx
        self.root_idx = root_idx

    def __repr__(self):
        return str(self.value)

    def get_tree(self, level="", last=False):
        return level + str(self.value) + " (" + str(self.nl_idx) + ', NL-Root ' + str(self.root_idx) + ")\n"


# This is a list of all possible operators.
nonlinearExpressions_inp = [
    ("divide", (2, 2)),
    ("exp", (1, 1)),
    ("product", (2, np.inf)),
    ("ln", (1, 1)),
    ("square", (1, 1)),
    ("sqrt", (1, 1)),
    ("sin", (1, 1)),
    ("cos", (1, 1)),
    ("log10", (1, 1)),
    ("tanh", (1, 1)),
    ("min", (1, np.inf)),
    ("inverse", (1, 1)),
    ("power_ax", (1, 1)),
    ("power_xa", (1, 1)),
    ("power_xy", (2, 2)),
    ("power", (2, 2)),
    ("xabsx", (1, 1))
]
linearExpressions_inp = [("sum", (2, np.inf)), ("negate", (1, 1)), ("abs", (1, 1))]

expressions = {}
for l in linearExpressions_inp:
    expressions[l[0]] = LinearOperation(symbol=l[0], dim=l[1])
for nl in nonlinearExpressions_inp:
    expressions[nl[0]] = NonlinearOperation(symbol=nl[0], dim=nl[1])


# bounds[i][0]: lower bound of xi  (also for numbers)
# bounds[i][1]: upper bound of xi  (also for numbers)
def get_lower_and_upper_bound(op, bounds):
    lb = None
    ub = None
    if op.symbol == "sum":
        lb = 0
        ub = 0
        for l, u in bounds:
            lb += l
            ub += u
    elif op.symbol == "divide":
        assert (bounds[1][0] < 0 and bounds[1][1] < 0) or (
            bounds[1][0] > 0 and bounds[1][1] > 0
        )
        lb = min(
            [
                bounds[0][0] / bounds[1][0],
                bounds[0][0] / bounds[1][1],
                bounds[0][1] / bounds[1][0],
                bounds[0][1] / bounds[1][1],
            ]
        )
        ub = max(
            [
                bounds[0][0] / bounds[1][0],
                bounds[0][0] / bounds[1][1],
                bounds[0][1] / bounds[1][0],
                bounds[0][1] / bounds[1][1],
            ]
        )
    elif op.symbol == "exp":
        lb = np.exp(bounds[0][0])
        ub = np.exp(bounds[0][1])
    elif op.symbol == "product":
        lb = min(
            [
                bounds[0][0] * bounds[1][0],
                bounds[0][0] * bounds[1][1],
                bounds[0][1] * bounds[1][0],
                bounds[0][1] * bounds[1][1],
            ]
        )
        ub = max(
            [
                bounds[0][0] * bounds[1][0],
                bounds[0][0] * bounds[1][1],
                bounds[0][1] * bounds[1][0],
                bounds[0][1] * bounds[1][1],
            ]
        )
    elif op.symbol == "ln":
        assert bounds[0][0] > 0
        lb = np.log(bounds[0][0])
        ub = np.log(bounds[0][1])
    elif op.symbol == "log10":
        assert bounds[0][0] > 0
        lb = np.log10(bounds[0][0])
        ub = np.log10(bounds[0][1])
    elif op.symbol == "power_xa":
        a = op.num_param
        if a % 2 == 0:
            lb = (
                0
                if (bounds[0][0] < 0 and bounds[0][1] > 0)
                else min(bounds[0][0] ** a, bounds[0][1] ** a)
            )
            ub = max(bounds[0][0] ** a, bounds[0][1] ** a)
        else:
            lb = min(bounds[0][0] ** a, bounds[0][1] ** a)
            ub = max(bounds[0][0] ** a, bounds[0][1] ** a)

    elif op.symbol == "negate":
        lb = -1 * bounds[0][1]
        ub = -1 * bounds[0][0]
    elif op.symbol == "abs":
        lb = 0
        ub = max(np.abs(bounds[0][0]), np.abs(bounds[0][1]))
    elif op.symbol == "square":
        lb = (
            0
            if (bounds[0][0] < 0 and bounds[0][1] > 0)
            else min(bounds[0][0] ** 2, bounds[0][1] ** 2)
        )
        ub = max(bounds[0][0] ** 2, bounds[0][1] ** 2)
    elif op.symbol == "sqrt":
        assert bounds[0][0] >= 0
        lb = np.sqrt(bounds[0][0])
        ub = np.sqrt(bounds[0][1])
    elif op.symbol == "cos" or op.symbol == "sin":
        if op.symbol == "cos":
            l = bounds[0][0]
            u = bounds[0][1]
        else:
            l = bounds[0][0] - np.pi / 2
            u = bounds[0][1] - np.pi / 2
        l, u = l % (2 * np.pi), l % (2 * np.pi) + (u - l)
        if l < np.pi:
            if u < np.pi:
                ub = np.cos(l)
                lb = np.cos(u)
            elif u < 2 * np.pi:
                lb = -1
                ub = max(np.cos(l), np.cos(u))
            else:
                lb = -1
                ub = 1
        elif l < 2 * np.pi:
            if u < 2 * np.pi:
                lb = np.cos(l)
                ub = np.cos(u)
            elif u < 3 * np.pi:
                lb = min(np.cos(l), np.cos(u))
                ub = 1
            else:
                lb = -1
                ub = 1
    elif op.symbol == "tanh":
        lb = np.tanh(bounds[0][0])
        ub = np.tanh(bounds[0][1])
    # elif op.symbol == 'signpower':
    #    return lambda x: np.power(x[0], x[1]) if x[1] > 0 else -1*np.power(np.abs(x[0], x[1]))
    elif op.symbol == "min":
        lb = np.inf
        ub = np.inf
        for l, u in bounds:
            lb = min(lb, l)
            ub = min(ub, u)
    elif op.symbol == "inverse":
        assert (bounds[0][0] < 0 and bounds[0][1] < 0) or (
            bounds[0][0] > 0 and bounds[0][1] > 0
        )
        lb = min(1 / bounds[0][0], 1 / bounds[0][1])
        ub = max(1 / bounds[0][0], 1 / bounds[0][1])
    elif op.symbol == "xabsx":
        lb = bounds[0][0] * abs(bounds[0][0])
        ub = bounds[0][1] * abs(bounds[0][1])
    else:
        exit(op.symbol + " not implemented for lower/upper")
    return lb, ub


def get_np_expression(op):
    if op.symbol == "exp":
        return np.exp
    elif op.symbol == "ln":
        return np.log
    elif op.symbol == "log10":
        return np.log10
    elif op.symbol == "abs":
        return np.abs
    elif op.symbol == "square":
        return lambda x: x * x
    elif op.symbol == "power_xa":
        return lambda x: x**op.num_param
    elif op.symbol == "sqrt":
        return np.sqrt
    elif op.symbol == "sin":
        return np.sin
    elif op.symbol == "cos":
        return np.cos
    elif op.symbol == "tanh":
        return np.tanh
    elif op.symbol == "inverse":
        return lambda x: 1 / x
    elif op.symbol == "xabsx":
        return lambda x: x * np.abs(x)
    else:
        exit(op.symbol + " not implemented for np expression")


def get_pyomo_expression(op):
    if op.symbol == "divide":
        return lambda x: x[0] / x[1]
    elif op.symbol == "exp":
        return lambda x: pyo.exp(x[0])
    elif op.symbol == "product":

        def mullt(x):
            res = 1
            for s in x:
                res *= s
            return res

        return mullt
    elif op.symbol == "ln":
        return lambda x: pyo.log(x[0])
    elif op.symbol == "log10":
        return lambda x: pyo.log10(x[0])
    elif op.symbol == "power_xa":
        return lambda x: x[0] ** op.num_param
    elif op.symbol == "negate":
        return lambda x: -x[0]
    elif op.symbol == "abs":
        return lambda x: np.abs(x[0])
    elif op.symbol == "sum":

        def summ(x):
            res = 0
            for s in x:
                res += s
            return res

        return summ
    elif op.symbol == "square":
        return lambda x: x[0] ** 2
    elif op.symbol == "sqrt":
        return lambda x: pyo.sqrt(x[0])
    elif op.symbol == "sin":
        return lambda x: pyo.sin(x[0])
    elif op.symbol == "cos":
        return lambda x: pyo.cos(x[0])
    elif op.symbol == "tanh":
        return lambda x: pyo.tanh(x[0])
    # elif op.symbol == 'signpower':
    #    return lambda x: np.power(x[0], x[1]) if x[1] > 0 else -1*np.power(np.abs(x[0], x[1]))
    elif op.symbol == "min":
        return lambda x: np.min(x)
    elif op.symbol == "inverse":
        return lambda x: 1 / x[0]
    elif op.symbol == "xabsx":
        return lambda x: x[0] * np.abs(x[0])
    else:
        exit(op.symbol + " not implemented for pyomo")


# return derivation and maximum of f(x) - mx + t. It holds that f'(r) = m for all r in roots
def get_der_andX(op, x_low, x_up, m, t):
    der = None
    roots = None
    if op.symbol == "exp":
        der = lambda x: np.exp(x)
        roots = [np.log(m)]
    elif op.symbol == "ln":
        der = lambda x: 1 / x
        roots = [1 / m]
    elif op.symbol == "log10":
        der = lambda x: 1 / (np.log(10) * x)
        roots = [1 / (np.log(10) * m)]
    elif op.symbol == "square":
        der = lambda x: 2 * x
        roots = [m / 2]
    elif op.symbol == "sqrt":
        der = lambda x: 1 / np.sqrt(x)
        roots = [1 / (4 * m * m)]
    elif op.symbol == "sin":
        der = lambda x: np.cos(x)

        def sin_roots(m_s):
            am = np.arccos(m_s)
            roots = []
            for k in range(
                int(np.ceil((x_low - am) / (2 * np.pi))),
                int(np.floor((x_up - am) / (2 * np.pi))) + 1,
            ):
                roots += [am + 2 * k * np.pi]
            for k in range(
                int(np.ceil((x_low + am) / (2 * np.pi))),
                int(np.floor((x_up + am) / (2 * np.pi))) + 1,
            ):
                roots += [-am + 2 * k * np.pi]
            return roots

        roots = sin_roots(m)

    elif op.symbol == "cos":
        der = lambda x: np.sin(x)

        def cos_roots(m_c):
            am = np.arcsin(-m_c)
            roots = []
            for k in range(
                int(np.ceil((x_low - am) / (2 * np.pi))),
                int(np.floor((x_up - am) / (2 * np.pi))) + 1,
            ):
                roots += [am + 2 * k * np.pi]
            for k in range(
                int(np.ceil((-np.pi + x_low + am) / (2 * np.pi))),
                int(np.floor((-np.pi + x_up + am) / (2 * np.pi))) + 1,
            ):
                roots += [np.pi - am + 2 * k * np.pi]
            return roots

        roots = cos_roots(m)

    elif op.symbol == "tanh":
        der = lambda x: 1 - np.tanh(x) ** 2
        roots = [np.arctanh(np.sqrt(1 - m)), -np.arctanh(np.sqrt(1 - m))]
    elif op.symbol == "inverse":
        der = lambda x: 1 / (x * x)
        roots = [np.sqrt(-1 / m), -np.sqrt(-1 / m)]
    elif op.symbol == "power_xa":
        a = op.num_param
        der = lambda x: a * (x ** (a - 1))
        roots = [np.abs(m / a) ** (1 / (a - 1)), -np.abs(m / a) ** (1 / (a - 1))]
    elif op.symbol == "xabsx":
        der = lambda x: 2*np.abs(x)
        roots = [m/2, -m/2]
    else:
        exit(op.symbol + " not implemented for der and X")
    ret_roots = []
    for r in roots:
        if x_low <= r <= x_up:
            ret_roots += [r]
    return der, ret_roots
