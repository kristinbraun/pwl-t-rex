import numpy as np
import MIPRef_graycode as gray

# the following functions create different MIP representations for PWL approximations of nonlinear 1d functions
# input: nonlinear expression
# output: additional variables and constraints


def disaggregated_convex_combination_model(
    expression,
    breakpoints,
    f_breakpoints,
    add_name,
    cur_idx,
    bias_varidx,
    bias_considx,
    errors_low,
    errors_up,
    relax=True,
):
    x = expression.children[0]
    n = len(breakpoints) - 1
    lambda1 = [
        {"name": add_name + "_lambda_left_" + str(i), "lb": 0, "ub": 1, "type": "C"}
        for i in range(n)
    ]
    lambda2 = [
        {"name": add_name + "_lambda_right_" + str(i), "lb": 0, "ub": 1, "type": "C"}
        for i in range(n)
    ]
    y = [
        {"name": add_name + "_y_" + str(i), "lb": 0, "ub": 1, "type": "B"}
        for i in range(n)
    ]
    n_lambda_left = bias_varidx
    n_lambda_right = bias_varidx + n
    n_y = bias_varidx + 2 * n

    new_vars = lambda1 + lambda2 + y
    new_cons = []
    new_lins = []

    # \sum_{i = 1}^n \left(\lambda_i^1\bar x_{i-1} + \lambda_i^2 \bar x_i\right) &= x
    new_cons += [{"name": add_name + "_x_cons", "lb": 0, "ub": 0}]
    for i in range(0, n):
        new_lins += [(bias_considx, n_lambda_left + i, breakpoints[i])]
        new_lins += [(bias_considx, n_lambda_right + i, breakpoints[i + 1])]
    new_lins += [(bias_considx, x.idx, -1)]

    # \lambda_i^1 + \lambda_i^2 &= y_i &\forall i \in \{1, \dots, n\}
    for i in range(n):
        new_cons += [{"name": add_name + "_lambda_sum_" + str(i), "lb": 0, "ub": 0}]
        new_lins += [(bias_considx + 1 + i, n_lambda_left + i, 1)]
        new_lins += [(bias_considx + 1 + i, n_lambda_right + i, 1)]
        new_lins += [(bias_considx + 1 + i, n_y + i, -1)]

    # \sum_{i=1}^n y_i &= 1
    new_cons += [{"name": add_name + "_y_sum", "lb": 1, "ub": 1}]
    for i in range(n):
        new_lins += [(bias_considx + n + 1, n_y + i, 1)]

    if relax == 1:
        n_epsup = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_up",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]
        n_epslow = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_low",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]

        for i in range(n):
            new_lins += [(cur_idx, n_epsup + i, -1)]
            new_lins += [(cur_idx, n_epslow + i, 1)]

        for i in range(n):
            # \epsilon_{low} \leq e\_low[i] \cdot y_i
            # \epsilon_{up} \leq e\_up[i] \cdot y_i
            new_cons += [
                {"name": add_name + "_eps_low_sum_" + str(i), "lb": 0, "ub": np.inf}
            ]
            new_cons += [
                {"name": add_name + "_eps_up_sum_" + str(i), "lb": 0, "ub": np.inf}
            ]
            new_lins += [(bias_considx + len(new_cons) - 2, n_y + i, errors_low[i + 1])]
            new_lins += [(bias_considx + len(new_cons) - 1, n_y + i, errors_up[i + 1])]
            new_lins += [(bias_considx + len(new_cons) - 2, n_epslow + i, -1)]
            new_lins += [(bias_considx + len(new_cons) - 1, n_epsup + i, -1)]

    # \sum_{i = 1}^n \left( \lambda_i^1 f(\bar x_{i-1}) + \lambda_i^2 f(\bar x_i) \right) &\leq z
    for i in range(n):
        new_lins += [(cur_idx, n_lambda_left + i, f_breakpoints[i])]
        new_lins += [(cur_idx, n_lambda_right + i, f_breakpoints[i + 1])]

    return new_vars, new_cons, new_lins


def get_binary_representation(i, digits):
    assert 0 <= i <= 2**digits - 1
    res = [1] if i >= 2 ** (digits - 1) else [0]
    if digits <= 1:
        return res
    else:
        return res + get_binary_representation(i % 2 ** (digits - 1), digits - 1)


def logarithmic_disaggregated_convex_combination_model(
    expression,
    breakpoints,
    f_breakpoints,
    add_name,
    cur_idx,
    bias_varidx,
    bias_considx,
    errors_low,
    errors_up,
    relax=True,
):
    x = expression.children[0]
    n = len(breakpoints) - 1
    n_L = int(np.ceil(np.log2(n)))
    lambda1 = [
        {"name": add_name + "_lambda_left_" + str(i), "lb": 0, "ub": 1, "type": "C"}
        for i in range(n)
    ]
    lambda2 = [
        {"name": add_name + "_lambda_right_" + str(i), "lb": 0, "ub": 1, "type": "C"}
        for i in range(n)
    ]
    y = [
        {"name": add_name + "_y_" + str(i), "lb": 0, "ub": 1, "type": "B"}
        for i in range(n_L)
    ]
    n_lambda_left = bias_varidx
    n_lambda_right = bias_varidx + n
    n_y = bias_varidx + 2 * n

    new_vars = lambda1 + lambda2 + y
    new_cons = []
    new_lins = []

    # \sum_{i = 1}^n \left(\lambda_i^1\bar x_{i-1} + \lambda_i^2 \bar x_i\right) &= x
    new_cons += [{"name": add_name + "_x_cons", "lb": 0, "ub": 0}]
    for i in range(0, n):
        new_lins += [(bias_considx, n_lambda_left + i, breakpoints[i])]
        new_lins += [(bias_considx, n_lambda_right + i, breakpoints[i + 1])]
    new_lins += [(bias_considx, x.idx, -1)]

    # \sum_{i=1}^n \lambda_i^1 + \lambda_i^2 &= 1
    new_cons += [{"name": add_name + "_lambda_sum", "lb": 1, "ub": 1}]
    for i in range(n):
        new_lins += [(bias_considx + 1, n_lambda_left + i, 1)]
        new_lins += [(bias_considx + 1, n_lambda_right + i, 1)]

    B = []
    for i in range(n):
        B += [get_binary_representation(i, n_L)]

    # \sum_{i \in \mathscr P^+(B, l)} \lambda_i^1 + \lambda_i^2 &\leq y_l &\forall l \in \{1, \dots, \lceil \log_2 n \rceil \}
    for l in range(n_L):
        new_cons += [
            {"name": add_name + "_lambda_y_cons_(+)" + str(l), "lb": 0, "ub": np.inf}
        ]
        for i in range(n):
            if B[i][l] == 1:
                new_lins += [(bias_considx + len(new_cons) - 1, n_lambda_left + i, -1)]
                new_lins += [(bias_considx + len(new_cons) - 1, n_lambda_right + i, -1)]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + l, 1)]

    # \sum_{i \in \mathscr P^0(B, l)} \lambda_i^1 + \lambda_i^2 &\leq 1 - y_l &\forall l \in \{1, \dots, \lceil \log_2 n \rceil \}
    for l in range(n_L):
        new_cons += [
            {"name": add_name + "_lambda_y_cons_(0)" + str(l), "lb": -1, "ub": np.inf}
        ]
        for i in range(n):
            if B[i][l] == 0:
                new_lins += [(bias_considx + len(new_cons) - 1, n_lambda_left + i, -1)]
                new_lins += [(bias_considx + len(new_cons) - 1, n_lambda_right + i, -1)]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + l, -1)]

    # \sum_{i = 1}^n \left( \lambda_i^1 f(\bar x_{i-1}) + \lambda_i^2 f(\bar x_i) \right) &\leq z
    for i in range(n):
        new_lins += [(cur_idx, n_lambda_left + i, f_breakpoints[i])]
        new_lins += [(cur_idx, n_lambda_right + i, f_breakpoints[i + 1])]

    if relax == 1:
        n_epsup = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_up",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]
        n_epslow = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_low",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]

        for i in range(n):
            new_lins += [(cur_idx, n_epsup + i, -1)]
            new_lins += [(cur_idx, n_epslow + i, 1)]

        for i in range(n):
            # \epsilon_{low} \leq e\_low[i] \cdot y_i
            # \epsilon_{up} \leq e\_up[i] \cdot y_i
            new_cons += [
                {"name": add_name + "_eps_low_sum_" + str(i), "lb": 0, "ub": np.inf}
            ]
            new_cons += [
                {"name": add_name + "_eps_up_sum_" + str(i), "lb": 0, "ub": np.inf}
            ]
            new_lins += [
                (bias_considx + len(new_cons) - 2, n_lambda_left + i, errors_low[i + 1])
            ]
            new_lins += [
                (
                    bias_considx + len(new_cons) - 2,
                    n_lambda_right + i,
                    errors_low[i + 1],
                )
            ]
            new_lins += [
                (bias_considx + len(new_cons) - 1, n_lambda_left + i, errors_up[i + 1])
            ]
            new_lins += [
                (bias_considx + len(new_cons) - 1, n_lambda_right + i, errors_up[i + 1])
            ]
            new_lins += [(bias_considx + len(new_cons) - 2, n_epslow + i, -1)]
            new_lins += [(bias_considx + len(new_cons) - 1, n_epsup + i, -1)]

    return new_vars, new_cons, new_lins


def aggregated_convex_combination_model(
    expression,
    breakpoints,
    f_breakpoints,
    add_name,
    cur_idx,
    bias_varidx,
    bias_considx,
    errors_low,
    errors_up,
    relax=True,
):
    x = expression.children[0]
    n = len(breakpoints) - 1
    lambdas = [
        {"name": add_name + "_lambda_" + str(i), "lb": 0, "ub": 1, "type": "C"}
        for i in range(n + 1)
    ]
    n_lambda = bias_varidx
    y = [
        {"name": add_name + "_y_" + str(i), "lb": 0, "ub": 1, "type": "B"}
        for i in range(n)
    ]
    n_y = bias_varidx + n + 1

    new_vars = lambdas + y
    new_cons = []
    new_lins = []

    # \sum_{i = 0}^n \lambda_i \bar x_i &= x
    new_cons += [{"name": add_name + "_x_cons", "lb": 0, "ub": 0}]
    for i in range(0, n + 1):
        new_lins += [(bias_considx, n_lambda + i, breakpoints[i])]
    new_lins += [(bias_considx, x.idx, -1)]

    # \sum_{i=0}^n \lambda_i &= 1
    new_cons += [{"name": add_name + "_lambda_sum", "lb": 1, "ub": 1}]
    for i in range(n + 1):
        new_lins += [(bias_considx + 1, n_lambda + i, 1)]

    # \sum_{i=1}^n y_i &= 1
    new_cons += [{"name": add_name + "_y_sum", "lb": 1, "ub": 1}]
    for i in range(n):
        new_lins += [(bias_considx + 2, n_y + i, 1)]

    # \lambda_0 &\leq y_1
    new_cons += [{"name": add_name + "_lambda_y_0", "lb": 0, "ub": np.inf}]
    new_lins += [(bias_considx + len(new_cons) - 1, n_lambda, -1)]
    new_lins += [(bias_considx + len(new_cons) - 1, n_y, 1)]

    # \lambda_i &\leq y_i + y_{i+1} & \forall i \in \{1, \dots, n-1\}
    for i in range(1, n):
        new_cons += [{"name": add_name + "_lambda_y_" + str(i), "lb": 0, "ub": np.inf}]
        new_lins += [(bias_considx + len(new_cons) - 1, n_lambda + i, -1)]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + i, 1)]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + i - 1, 1)]

    # \lambda_n &\leq y_n
    new_cons += [{"name": add_name + "_lambda_y_n", "lb": 0, "ub": np.inf}]
    new_lins += [(bias_considx + len(new_cons) - 1, n_lambda + n, -1)]
    new_lins += [(bias_considx + len(new_cons) - 1, n_y + n - 1, 1)]

    if relax == 1:
        n_epsup = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_up",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]
        n_epslow = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_low",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]

        for i in range(n):
            new_lins += [(cur_idx, n_epsup + i, -1)]
            new_lins += [(cur_idx, n_epslow + i, 1)]

        for i in range(n):
            # \epsilon_{low} \leq e\_low[i] \cdot y_i
            # \epsilon_{up} \leq e\_up[i] \cdot y_i
            new_cons += [
                {"name": add_name + "_eps_low_sum_" + str(i), "lb": 0, "ub": np.inf}
            ]
            new_cons += [
                {"name": add_name + "_eps_up_sum_" + str(i), "lb": 0, "ub": np.inf}
            ]
            new_lins += [(bias_considx + len(new_cons) - 2, n_y + i, errors_low[i + 1])]
            new_lins += [(bias_considx + len(new_cons) - 1, n_y + i, errors_up[i + 1])]
            new_lins += [(bias_considx + len(new_cons) - 2, n_epslow + i, -1)]
            new_lins += [(bias_considx + len(new_cons) - 1, n_epsup + i, -1)]

    # \sum_{i = 0}^n \lambda_i f(\bar x_i) &\leq z
    for i in range(n + 1):
        new_lins += [(cur_idx, bias_varidx + i, f_breakpoints[i])]

    return new_vars, new_cons, new_lins


def get_branching_scheme(n):
    logn = max(1, int(np.ceil(np.log2(n))))
    n_use = 2**logn
    L = [[] for _ in range(logn)]
    R = [[] for _ in range(logn)]
    if n_use <= 2:
        L[0] = [0]
        R[0] = [2]
    else:
        L_prev, R_prev = get_branching_scheme(n_use / 2)
        for i in range(logn - 1):
            L[i] = L_prev[i] + [2**logn - j for j in L_prev[i]]
            R[i] = R_prev[i] + [2**logn - j for j in R_prev[i]]
        L[logn - 1] = list(range(0, 2 ** (logn - 1)))
        R[logn - 1] = list(range(2 ** (logn - 1) + 1, n_use + 1))

    if n_use > n:
        for i in range(logn):
            L[i] = [x for x in L[i] if x <= n]
            R[i] = [x for x in R[i] if x <= n]
    return L, R


def logarithmic_aggregated_convex_combination_model(
    expression,
    breakpoints,
    f_breakpoints,
    add_name,
    cur_idx,
    bias_varidx,
    bias_considx,
    errors_low,
    errors_up,
    relax=True,
):
    x = expression.children[0]
    n = len(breakpoints) - 1
    n_L = max(1, int(np.ceil(np.log2(n))))
    lambdas = [
        {"name": add_name + "_lambda_" + str(i), "lb": 0, "ub": 1, "type": "C"}
        for i in range(n + 1)
    ]
    y = [
        {"name": add_name + "_y_" + str(i), "lb": 0, "ub": 1, "type": "B"}
        for i in range(n_L)
    ]
    n_lambda = bias_varidx
    n_y = bias_varidx + n + 1

    new_vars = lambdas + y
    new_cons = []
    new_lins = []

    L, R = get_branching_scheme(n)

    # \sum_{i = 0}^n \lambda_i \bar x_i &= x
    new_cons += [{"name": add_name + "_x_cons", "lb": 0, "ub": 0}]
    for i in range(0, n + 1):
        new_lins += [(bias_considx, n_lambda + i, breakpoints[i])]
    new_lins += [(bias_considx, x.idx, -1)]

    # \sum_{i=0}^n \lambda_i &= 1
    new_cons += [{"name": add_name + "_lambda_sum", "lb": 1, "ub": 1}]
    for i in range(n + 1):
        new_lins += [(bias_considx + 1, n_lambda + i, 1)]

    # \sum_{i \in L_s} \lambda_i &\leq y_s &\forall s \in S
    for l in range(n_L):
        new_cons += [
            {"name": add_name + "_lambda_y_cons_(L)" + str(l), "lb": 0, "ub": np.inf}
        ]
        for i in range(n + 1):
            if i in L[l]:
                new_lins += [(bias_considx + len(new_cons) - 1, n_lambda + i, -1)]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + l, 1)]

    # \sum_{i \in R_s} \lambda_i &\leq 1 - y_s &\forall s \in S
    for l in range(n_L):
        new_cons += [
            {"name": add_name + "_lambda_y_cons_(R)" + str(l), "lb": -1, "ub": np.inf}
        ]
        for i in range(n + 1):
            if i in R[l]:
                new_lins += [(bias_considx + len(new_cons) - 1, n_lambda + i, -1)]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + l, -1)]

    if relax == 1:
        n_epsup = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_up",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]
        n_epslow = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_low",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]

        for i in range(n):
            new_lins += [(cur_idx, n_epsup + i, -1)]
            new_lins += [(cur_idx, n_epslow + i, 1)]

        # if is necessary instead of elif due to the behaviour at the breakpoints (i.e. the intersection of y_i and y_{i+1}
        for l in range(n_L):
            for i in range(n + 1):
                if i in L[l]:
                    if i < n:
                        new_cons += [
                            {
                                "name": add_name
                                + "_eps_low_sum_"
                                + str(i)
                                + "_"
                                + str(l),
                                "lb": 0,
                                "ub": np.inf,
                            }
                        ]
                        new_cons += [
                            {
                                "name": add_name
                                + "_eps_up_sum_"
                                + str(i)
                                + "_"
                                + str(l),
                                "lb": 0,
                                "ub": np.inf,
                            }
                        ]

                        new_lins += [
                            (bias_considx + len(new_cons) - 2, n_epslow + i, -1)
                        ]
                        new_lins += [
                            (
                                bias_considx + len(new_cons) - 2,
                                n_y + l,
                                errors_low[i + 1],
                            )
                        ]

                        new_lins += [
                            (bias_considx + len(new_cons) - 1, n_epsup + i, -1)
                        ]
                        new_lins += [
                            (
                                bias_considx + len(new_cons) - 1,
                                n_y + l,
                                errors_up[i + 1],
                            )
                        ]
                    if i > 0:
                        new_cons += [
                            {
                                "name": add_name
                                + "_eps_low_sum-_"
                                + str(i)
                                + "_"
                                + str(l),
                                "lb": 0,
                                "ub": np.inf,
                            }
                        ]
                        new_cons += [
                            {
                                "name": add_name
                                + "_eps_up_sum-_"
                                + str(i)
                                + "_"
                                + str(l),
                                "lb": 0,
                                "ub": np.inf,
                            }
                        ]

                        new_lins += [
                            (bias_considx + len(new_cons) - 2, n_epslow + i - 1, -1)
                        ]
                        new_lins += [
                            (bias_considx + len(new_cons) - 2, n_y + l, errors_low[i])
                        ]

                        new_lins += [
                            (bias_considx + len(new_cons) - 1, n_epsup + i - 1, -1)
                        ]
                        new_lins += [
                            (bias_considx + len(new_cons) - 1, n_y + l, errors_up[i])
                        ]

                if i in R[l]:
                    if i < n:
                        new_cons += [
                            {
                                "name": add_name
                                + "_eps_low_sum_"
                                + str(i)
                                + "_"
                                + str(l),
                                "lb": -1 * errors_low[i + 1],
                                "ub": np.inf,
                            }
                        ]
                        new_cons += [
                            {
                                "name": add_name
                                + "_eps_up_sum_"
                                + str(i)
                                + "_"
                                + str(l),
                                "lb": -1 * errors_up[i + 1],
                                "ub": np.inf,
                            }
                        ]

                        new_lins += [
                            (bias_considx + len(new_cons) - 2, n_epslow + i, -1)
                        ]
                        new_lins += [
                            (
                                bias_considx + len(new_cons) - 2,
                                n_y + l,
                                -1 * errors_low[i + 1],
                            )
                        ]

                        new_lins += [
                            (bias_considx + len(new_cons) - 1, n_epsup + i, -1)
                        ]
                        new_lins += [
                            (
                                bias_considx + len(new_cons) - 1,
                                n_y + l,
                                -1 * errors_up[i + 1],
                            )
                        ]
                    if i > 0:
                        new_cons += [
                            {
                                "name": add_name
                                + "_eps_low_sum-_"
                                + str(i)
                                + "_"
                                + str(l),
                                "lb": -1 * errors_low[i],
                                "ub": np.inf,
                            }
                        ]
                        new_cons += [
                            {
                                "name": add_name
                                + "_eps_up_sum-_"
                                + str(i)
                                + "_"
                                + str(l),
                                "lb": -1 * errors_up[i],
                                "ub": np.inf,
                            }
                        ]

                        new_lins += [
                            (bias_considx + len(new_cons) - 2, n_epslow + i - 1, -1)
                        ]
                        new_lins += [
                            (
                                bias_considx + len(new_cons) - 2,
                                n_y + l,
                                -1 * errors_low[i],
                            )
                        ]

                        new_lins += [
                            (bias_considx + len(new_cons) - 1, n_epsup + i - 1, -1)
                        ]
                        new_lins += [
                            (
                                bias_considx + len(new_cons) - 1,
                                n_y + l,
                                -1 * errors_up[i],
                            )
                        ]

    # \sum_{i = 0}^n \lambda_i f(\bar x_i) &\leq z
    for i in range(n + 1):
        new_lins += [(cur_idx, n_lambda + i, f_breakpoints[i])]

    return new_vars, new_cons, new_lins


def classical_incremental_method(
    expression,
    breakpoints,
    f_breakpoints,
    add_name,
    cur_idx,
    bias_varidx,
    bias_considx,
    errors_low,
    errors_up,
    relax=True,
):
    x = expression.children[0]
    n = len(breakpoints) - 1
    deltas = [
        {"name": add_name + "_delta_" + str(i), "lb": 0, "ub": 1, "type": "C"}
        for i in range(n)
    ]
    n_delta = bias_varidx
    y = [
        {"name": add_name + "_y_" + str(i), "lb": 0, "ub": 1, "type": "B"}
        for i in range(n - 1)
    ]
    n_y = bias_varidx + n

    new_vars = deltas + y
    new_cons = []
    new_lins = []

    # \bar x_0 + \sum_{k=1}^K \delta_k (\bar x _k - \bar x _{k-1}) &= x
    new_cons += [
        {
            "name": add_name + "_x_cons",
            "lb": -1 * breakpoints[0],
            "ub": -1 * breakpoints[0],
        }
    ]
    for i in range(0, n):
        new_lins += [(bias_considx, n_delta + i, breakpoints[i + 1] - breakpoints[i])]
    new_lins += [(bias_considx, x.idx, -1)]

    # f(\bar x _0) + \sum_{k=1}^K \delta_k(f(\bar x _k) - f(\bar x _{k-1})) &\leq z
    # \bar x _ 0 is done after the function call
    for i in range(0, n):
        new_lins += [(cur_idx, n_delta + i, f_breakpoints[i + 1] - f_breakpoints[i])]

    for i in range(n - 1):
        new_cons += [
            {
                "name": add_name + "_delta_" + str(i) + "_y_" + str(i),
                "lb": 0,
                "ub": np.inf,
            }
        ]
        new_lins += [(bias_considx + len(new_cons) - 1, n_delta + i, 1)]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + i, -1)]
        new_cons += [
            {
                "name": add_name + "_delta_" + str(i + 1) + "_y_" + str(i),
                "lb": 0,
                "ub": np.inf,
            }
        ]
        new_lins += [(bias_considx + len(new_cons) - 1, n_delta + i + 1, -1)]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + i, 1)]

    if relax == 1:
        n_epsup = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_up",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]
        n_epslow = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_low",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]

        for i in range(n):
            new_lins += [(cur_idx, n_epsup + i, -1)]
            new_lins += [(cur_idx, n_epslow + i, 1)]

        new_cons += [
            {
                "name": add_name + "_eps_low_sum_0",
                "lb": -1 * errors_low[1],
                "ub": np.inf,
            }
        ]
        new_cons += [
            {"name": add_name + "_eps_up_sum_0", "lb": -1 * errors_up[1], "ub": np.inf}
        ]
        new_lins += [(bias_considx + len(new_cons) - 2, n_y, -1 * errors_low[1])]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y, -1 * errors_up[1])]
        new_lins += [(bias_considx + len(new_cons) - 2, n_epslow, -1)]
        new_lins += [(bias_considx + len(new_cons) - 1, n_epsup, -1)]
        for i in range(1, n - 1):
            new_cons += [
                {"name": add_name + "_eps_low_sum_" + str(i), "lb": 0, "ub": np.inf}
            ]
            new_cons += [
                {"name": add_name + "_eps_up_sum_" + str(i), "lb": 0, "ub": np.inf}
            ]
            new_lins += [
                (bias_considx + len(new_cons) - 2, n_y + i - 1, errors_low[i + 1])
            ]
            new_lins += [
                (bias_considx + len(new_cons) - 1, n_y + i - 1, errors_up[i + 1])
            ]
            new_lins += [
                (bias_considx + len(new_cons) - 2, n_y + i, -1 * errors_low[i + 1])
            ]
            new_lins += [
                (bias_considx + len(new_cons) - 1, n_y + i, -1 * errors_up[i + 1])
            ]
            new_lins += [(bias_considx + len(new_cons) - 2, n_epslow + i, -1)]
            new_lins += [(bias_considx + len(new_cons) - 1, n_epsup + i, -1)]
        new_cons += [
            {"name": add_name + "_eps_low_sum_" + str(n), "lb": 0, "ub": np.inf}
        ]
        new_cons += [
            {"name": add_name + "_eps_up_sum_" + str(n), "lb": 0, "ub": np.inf}
        ]
        new_lins += [(bias_considx + len(new_cons) - 2, n_y + n - 2, errors_low[n])]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + n - 2, errors_up[n])]
        new_lins += [(bias_considx + len(new_cons) - 2, n_epslow + n - 1, -1)]
        new_lins += [(bias_considx + len(new_cons) - 1, n_epsup + n - 1, -1)]

    return new_vars, new_cons, new_lins


def multiple_choice_model(
    expression,
    breakpoints,
    f_breakpoints,
    m,
    t,
    add_name,
    cur_idx,
    bias_varidx,
    bias_considx,
    errors_low,
    errors_up,
    relax=True,
):
    x = expression.children[0]
    n = len(breakpoints) - 1

    x_vars = [
        {"name": add_name + "_x_" + str(i), "lb": -np.inf, "ub": np.inf, "type": "C"}
        for i in range(n)
    ]
    n_x = bias_varidx
    y_vars = [
        {"name": add_name + "_y_" + str(i), "lb": 0, "ub": 1, "type": "B"}
        for i in range(n)
    ]
    n_y = bias_varidx + n

    new_vars = x_vars + y_vars
    new_cons = []
    new_lins = []

    # \sum_{i=1}^n x_i &= x
    new_cons += [{"name": add_name + "_x_cons", "lb": 0, "ub": 0}]
    for i in range(n):
        new_lins += [(bias_considx, n_x + i, 1)]
    new_lins += [(bias_considx, x.idx, -1)]

    # \sum_{i=1}^n (m_i x_i + t_i y_i) &\leq z
    for i in range(n):
        new_lins += [(cur_idx, n_x + i, m[i + 1])]
        new_lins += [(cur_idx, n_y + i, t[i + 1])]

    # y_i \bar x_{i-1} &\leq x_i &\forall i \in \{1, \dots, n\}
    # x_i &\leq y_i \bar x_i &\forall i \in \{1, \dots, n\}
    for i in range(n):
        new_cons += [
            {"name": add_name + "_cons_x_y_1_" + str(i), "lb": 0, "ub": np.inf}
        ]
        new_lins += [(bias_considx + len(new_cons) - 1, n_x + i, 1)]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + i, -1 * breakpoints[i])]

        new_cons += [
            {"name": add_name + "_cons_x_y_2_" + str(i), "lb": 0, "ub": np.inf}
        ]
        new_lins += [(bias_considx + len(new_cons) - 1, n_x + i, -1)]
        new_lins += [
            (bias_considx + len(new_cons) - 1, n_y + i, 1 * breakpoints[i + 1])
        ]

    # \sum_{i=1}^n y_i &= 1
    new_cons += [{"name": add_name + "_y_sum", "lb": 1, "ub": 1}]
    for i in range(n):
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + i, 1)]

    if relax == 1:
        n_epsup = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_up",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]
        n_epslow = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_low",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]

        for i in range(n):
            new_lins += [(cur_idx, n_epsup + i, -1)]
            new_lins += [(cur_idx, n_epslow + i, 1)]

        for i in range(n):
            # \epsilon_{low} \leq \sum_{i=1}^n e\_low[i] \cdot y_i
            # \epsilon_{up} \leq \sum_{i=1}^n e\_up[i] \cdot y_i
            new_cons += [
                {"name": add_name + "_eps_low_sum_" + str(i), "lb": 0, "ub": np.inf}
            ]
            new_cons += [
                {"name": add_name + "_eps_up_sum_" + str(i), "lb": 0, "ub": np.inf}
            ]
            new_lins += [(bias_considx + len(new_cons) - 2, n_y + i, errors_low[i + 1])]
            new_lins += [(bias_considx + len(new_cons) - 1, n_y + i, errors_up[i + 1])]
            new_lins += [(bias_considx + len(new_cons) - 2, n_epslow + i, -1)]
            new_lins += [(bias_considx + len(new_cons) - 1, n_epsup + i, -1)]

    return new_vars, new_cons, new_lins


def binary_zig_zag_model(
    expression,
    breakpoints,
    f_breakpoints,
    m,
    t,
    add_name,
    cur_idx,
    bias_varidx,
    bias_considx,
    errors_low,
    errors_up,
    relax=True,
):
    x = expression.children[0]
    n = len(breakpoints) - 1
    r = 1 if n <= 1 else max(1, int(np.ceil(np.log2(n))))
    lambdas = [
        {"name": add_name + "_lambda_" + str(i), "lb": 0, "ub": 1, "type": "C"}
        for i in range(n + 1)
    ]
    y = [
        {"name": add_name + "_y_" + str(i), "lb": 0, "ub": 1, "type": "B"}
        for i in range(r)
    ]
    n_lambda = bias_varidx
    n_y = bias_varidx + n + 1

    new_vars = lambdas + y
    new_cons = []
    new_lins = []

    # \sum_{i = 0}^n \lambda_i \bar x_i &= x
    new_cons += [{"name": add_name + "_x_cons", "lb": 0, "ub": 0}]
    for i in range(0, n + 1):
        new_lins += [(bias_considx, n_lambda + i, breakpoints[i])]
    new_lins += [(bias_considx, x.idx, -1)]

    # \sum_{i=0}^n \lambda_i &= 1
    new_cons += [{"name": add_name + "_lambda_sum", "lb": 1, "ub": 1}]
    for i in range(n + 1):
        new_lins += [(bias_considx + 1, n_lambda + i, 1)]

    # \sum_{v=1}^{n} C_{v-1, k}^r \lambda_v  &\leq y_k + \sum_{l=k+1}^r 2^{l-k-1} y_l &\forall k \in \{1, \dots, r\}
    C = gray.calc_C(r)
    C = C[:n]
    C = np.concatenate(([C[0]], C, [C[-1]]), axis=0)

    for k in range(r):
        new_cons += [
            {"name": add_name + "_lambda_y_cons_left" + str(k), "lb": 0, "ub": np.inf}
        ]
        for l in range(k + 1, r):
            new_lins += [(bias_considx + len(new_cons) - 1, n_y + l, 2 ** (l - k - 1))]
        for v in range(n + 1):
            new_lins += [(bias_considx + len(new_cons) - 1, n_lambda + v, -C[v, k])]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + k, 1)]

    # y_k + \sum_{l=k+1}^r 2^{l-k-1} y_l &\leq \sum_{v=0}^{n} C_{v+1, k}^r \lambda_v &\forall k \in \{1, \dots, r\}
    for k in range(r):
        new_cons += [
            {"name": add_name + "_lambda_y_cons_right" + str(k), "lb": 0, "ub": np.inf}
        ]
        for l in range(k + 1, r):
            new_lins += [
                (bias_considx + len(new_cons) - 1, n_y + l, -(2 ** (l - k - 1)))
            ]
        for v in range(n + 1):
            new_lins += [(bias_considx + len(new_cons) - 1, n_lambda + v, C[v + 1, k])]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + k, -1)]

    if relax == 1:
        Z = gray.calc_Z(r)
        n_epsup = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_up",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]
        n_epslow = bias_varidx + len(new_vars)
        for i in range(n):
            new_vars += [
                {
                    "name": add_name + "_epsilon_" + str(i) + "_low",
                    "lb": 0,
                    "ub": np.inf,
                    "type": "C",
                }
            ]

        for i in range(n):
            new_lins += [(cur_idx, n_epsup + i, -1)]
            new_lins += [(cur_idx, n_epslow + i, 1)]

        for v in range(n):
            for k in range(r):
                if Z[v, k] == 1:
                    new_cons += [
                        {
                            "name": add_name + "_eps_low_sum1_" + str(v) + "_" + str(k),
                            "lb": 0,
                            "ub": np.inf,
                        }
                    ]
                    new_cons += [
                        {
                            "name": add_name + "_eps_up_sum1_" + str(v) + "_" + str(k),
                            "lb": 0,
                            "ub": np.inf,
                        }
                    ]
                    new_lins += [
                        (bias_considx + len(new_cons) - 2, n_y + k, errors_low[v + 1])
                    ]
                    new_lins += [
                        (bias_considx + len(new_cons) - 1, n_y + k, errors_up[v + 1])
                    ]
                    new_lins += [(bias_considx + len(new_cons) - 2, n_epslow + v, -1)]
                    new_lins += [(bias_considx + len(new_cons) - 1, n_epsup + v, -1)]
                else:
                    new_cons += [
                        {
                            "name": add_name + "_eps_low_sum1_" + str(v) + "_" + str(k),
                            "lb": -errors_low[v + 1],
                            "ub": np.inf,
                        }
                    ]
                    new_cons += [
                        {
                            "name": add_name + "_eps_up_sum1_" + str(v) + "_" + str(k),
                            "lb": -errors_up[v + 1],
                            "ub": np.inf,
                        }
                    ]
                    new_lins += [
                        (bias_considx + len(new_cons) - 2, n_y + k, -errors_low[v + 1])
                    ]
                    new_lins += [
                        (bias_considx + len(new_cons) - 1, n_y + k, -errors_up[v + 1])
                    ]
                    new_lins += [(bias_considx + len(new_cons) - 2, n_epslow + v, -1)]
                    new_lins += [(bias_considx + len(new_cons) - 1, n_epsup + v, -1)]

    # \sum_{i = 0}^n \lambda_i f(\bar x_i) &\leq z
    for i in range(n + 1):
        new_lins += [(cur_idx, n_lambda + i, f_breakpoints[i])]

    return new_vars, new_cons, new_lins


def general_integer_zig_zag_model(
    expression,
    breakpoints,
    f_breakpoints,
    m,
    t,
    add_name,
    cur_idx,
    bias_varidx,
    bias_considx,
    errors_low,
    errors_up,
    relax=True,
):
    x = expression.children[0]
    n = len(breakpoints) - 1
    r = 1 if n <= 1 else max(1, int(np.ceil(np.log2(n))))
    lambdas = [
        {"name": add_name + "_lambda_" + str(i), "lb": 0, "ub": 1, "type": "C"}
        for i in range(n + 1)
    ]
    y = [
        {"name": add_name + "_y_" + str(i), "lb": 0, "ub": np.inf, "type": "I"}
        for i in range(r)
    ]
    n_lambda = bias_varidx
    n_y = bias_varidx + n + 1

    new_vars = lambdas + y
    new_cons = []
    new_lins = []

    # \sum_{i = 0}^n \lambda_i \bar x_i &= x
    new_cons += [{"name": add_name + "_x_cons", "lb": 0, "ub": 0}]
    for i in range(0, n + 1):
        new_lins += [(bias_considx, n_lambda + i, breakpoints[i])]
    new_lins += [(bias_considx, x.idx, -1)]

    # \sum_{i=0}^n \lambda_i &= 1
    new_cons += [{"name": add_name + "_lambda_sum", "lb": 1, "ub": 1}]
    for i in range(n + 1):
        new_lins += [(bias_considx + 1, n_lambda + i, 1)]

    # \sum_{v=0}^{n} C_{v, k}^r \lambda_v &\leq y_k &\forall k  \in \{1, \dots, r\}
    C = gray.calc_C(r)
    C = C[:n]
    C = np.concatenate(([C[0]], C, [C[-1]]), axis=0)

    for k in range(r):
        new_cons += [
            {"name": add_name + "_lambda_y_cons_left" + str(k), "lb": 0, "ub": np.inf}
        ]
        for v in range(n + 1):
            new_lins += [(bias_considx + len(new_cons) - 1, n_lambda + v, -C[v, k])]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + k, 1)]

    # y_k &\leq \sum_{v=0}^{n} C_{v+1, k}^r \lambda_v &\forall k  \in \{1, \dots, r\}
    for k in range(r):
        new_cons += [
            {"name": add_name + "_lambda_y_cons_right" + str(k), "lb": 0, "ub": np.inf}
        ]
        for v in range(n + 1):
            new_lins += [(bias_considx + len(new_cons) - 1, n_lambda + v, C[v + 1, k])]
        new_lins += [(bias_considx + len(new_cons) - 1, n_y + k, -1)]

    if relax == 1:
        # TODO this relaxation is not working
        if False:
            n_epsup = bias_varidx + len(new_vars)
            for i in range(n):
                new_vars += [
                    {
                        "name": add_name + "_epsilon_" + str(i) + "_up",
                        "lb": 0,
                        "ub": np.inf,
                        "type": "C",
                    }
                ]
            n_epslow = bias_varidx + len(new_vars)
            for i in range(n):
                new_vars += [
                    {
                        "name": add_name + "_epsilon_" + str(i) + "_low",
                        "lb": 0,
                        "ub": np.inf,
                        "type": "C",
                    }
                ]

            for i in range(n):
                new_lins += [(cur_idx, n_epsup + i, -1)]
                new_lins += [(cur_idx, n_epslow + i, 1)]

            for v in range(n):
                for k in range(r):
                    if C[v, k] == 0:
                        continue
                    new_cons += [
                        {
                            "name": add_name + "_eps_low_sum1_" + str(v) + "_" + str(k),
                            "lb": 0,
                            "ub": np.inf,
                        }
                    ]
                    new_cons += [
                        {
                            "name": add_name + "_eps_up_sum1_" + str(v) + "_" + str(k),
                            "lb": 0,
                            "ub": np.inf,
                        }
                    ]

                    new_lins += [
                        (bias_considx + len(new_cons) - 2, n_epslow + v, -C[v, k])
                    ]
                    new_lins += [
                        (bias_considx + len(new_cons) - 1, n_epsup + v, -C[v, k])
                    ]
                    new_lins += [
                        (bias_considx + len(new_cons) - 2, n_y + k, errors_low[v + 1])
                    ]
                    new_lins += [
                        (bias_considx + len(new_cons) - 1, n_y + k, errors_up[v + 1])
                    ]

            # y_k + \sum_{l=k+1}^r 2^{l-k-1} y_l &\leq \sum_{v=0}^{n} C_{v+1, k}^r \lambda_v &\forall k \in \{1, \dots, r\}
            for v in range(0):
                for k in range(r):
                    if C[v + 1, k] == 0:
                        continue
                    new_cons += [
                        {
                            "name": add_name + "_eps_low_sum2_" + str(v) + "_" + str(k),
                            "lb": 0,
                            "ub": np.inf,
                        }
                    ]
                    new_cons += [
                        {
                            "name": add_name + "_eps_up_sum2_" + str(v) + "_" + str(k),
                            "lb": 0,
                            "ub": np.inf,
                        }
                    ]

                    new_lins += [
                        (bias_considx + len(new_cons) - 2, n_epslow + v, C[v + 1, k])
                    ]
                    new_lins += [
                        (bias_considx + len(new_cons) - 1, n_epsup + v, C[v + 1, k])
                    ]
                    new_lins += [
                        (bias_considx + len(new_cons) - 2, n_y + k, -errors_low[v + 1])
                    ]
                    new_lins += [
                        (bias_considx + len(new_cons) - 1, n_y + k, -errors_up[v + 1])
                    ]

    # \sum_{i = 0}^n \lambda_i f(\bar x_i) &\leq z
    for i in range(n + 1):
        new_lins += [(cur_idx, n_lambda + i, f_breakpoints[i])]

    return new_vars, new_cons, new_lins
