import argparse

import numpy as np
import datastructure_nonlinearTree as nltree
import MIPRef_osilToOnedim as oto


# create relaxation for a given function and interval, f: nltree.Operation, x_low, x_up: float
def create_single_relaxation(f_nl, x_low, x_up):
    f = nltree.get_np_expression(f_nl)
    m = (f(x_up) - f(x_low)) / (x_up - x_low)
    t = f(x_low) - m * x_low
    der, roots = nltree.get_der_andX(f_nl, x_low, x_up, m, t)

    left = (m * x_low + t) - f(x_low)
    right = (m * x_up + t) - f(x_up)
    vals = [left, right]
    for x in roots:
        vals += [(m * x + t) - f(x)]

    error_low = -1 * min(vals)
    error_up = max(vals)

    return m, t, error_low, error_up



def find_breakpoints_abs(f_nl, x_low, x_up):
    errors_low = [None, 0]
    errors_up = [None, 0]
    m_vals = [None]
    t_vals = [None]
    f = nltree.get_np_expression(f_nl)

    if x_low >= 0 or x_up <= 0:
        breakpoints = [x_low, x_up]
        m = 1 if x_low >= 0 else 0
        t = 0
        m_vals += [m]
        t_vals += [t]
    else:
        breakpoints = [x_low, 0, x_up]
        errors_low += [0]
        errors_up += [0]
        m1 = -1
        t1 = 0
        m2 = 1
        t2 = 0
        m_vals += [m1, m2]
        t_vals += [t1, t2]

    y = []
    for b in breakpoints:
        y += [f(b)]

    return breakpoints, y, errors_low, errors_up, m_vals, t_vals


def find_breakpoints(f, x_low, x_up, max_error):
    if f.symbol == "abs":
        return find_breakpoints_abs(f, x_low, x_up)

    eps = 1e-6  # how far away from error
    last_bp = x_low
    breakpoints = [x_low]
    errors_low = [None]
    errors_up = [None]
    m_vals = [None]
    t_vals = [None]
    while np.abs(last_bp - x_up) > eps:
        lower = last_bp
        upper = x_up
        found = False
        while not found:
            m, t, e_low, e_up = create_single_relaxation(
                f, last_bp, (upper + lower) / 2
            )
            if np.abs(lower - upper) <= eps:
                last_bp = (upper + lower) / 2
                breakpoints += [last_bp]
                errors_low += [e_low]
                errors_up += [e_up]
                m_vals += [m]
                t_vals += [t]
                found = True
            elif e_low <= max_error and e_up <= max_error:  # e_low + e_up <= max_error:
                lower = (upper + lower) / 2
            else:
                upper = (upper + lower) / 2
    breakpoints[-1] = x_up
    f_exp = nltree.get_np_expression(f)
    y = []
    for b in breakpoints:
        y += [f_exp(b)]
    return breakpoints, y, errors_low, errors_up, m_vals, t_vals


def test_minlplibfile():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--file", action="store", type=str, default="alan", help="Filename"
    )
    parser.add_argument(
        "--hpc",
        action="store",
        type=int,
        default=1,
        help="Is programm called from Cluster?",
    )
    args = parser.parse_args()
    TESTNAME = args.file
    oto.HPC = args.hpc
    TESTFILE = "../instances/minlplib/osil/" + TESTNAME
    if not ".osil" in TESTFILE:
        TESTFILE = TESTFILE + ".osil"
    print("Running", TESTFILE)

    eps = 0.0001

    rep = oto.obtain_1d_representation(TESTFILE)

    for n in rep.nonlinearexprs:
        f = n["expression"]
        if isinstance(f, nltree.Variable) or isinstance(f, nltree.Number):
            continue
        if f.operation.symbol == "sum":  # we can keep a sum
            continue
        if f.operation.symbol == "product":
            assert not (
                isinstance(f.children[0], nltree.Variable)
                and isinstance(f.children[1], nltree.Variable)
            )
            assert len(f.children) <= 2
            continue
        assert f.children[0].coef == 1  # TODO is that true?
        x_low, x_up = f.children[0].lb, f.children[0].ub
        breakpoints, e_low, e_up, m, t = find_breakpoints(f.operation, x_low, x_up, eps)
        # plot_relaxation(f.operation, breakpoints, m, t, e_low, e_up)
