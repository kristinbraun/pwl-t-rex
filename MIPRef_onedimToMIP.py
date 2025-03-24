import copy

import MIPRef_mipRepresentations as miprep
import datastructure_nonlinearTree as nltree
import MIPRef_osilToOnedim as oto
import MIPRef_linearRelaxation as linrelax
import numpy as np


# Methods:
# 1: Disaggregated convex combination model
# 2: Logarithmic disaggregated convex combination model
# 3: Convex combination model
# 4: Logarithmic branching convex combination model
# 5: Classical Incremental Method
# 6: Multiple Choice Model
# 7: Binary Zig-Zag
# 8: Integer Zig-Zag


def obtain_combined_breakpoints(in_model, epsilon=1):
    breakpoints = {}  # var -> list(breakpoints)
    nonlinearities = {}
    breakpoints_list = []

    for j in range(len(in_model.nonlinearexprs)):
        nl = in_model.nonlinearexprs[j]
        # find breakpoints
        f = nltree.get_pyomo_expression(nl["expression"].operation)
        f = nl["expression"].operation
        childvar = nl["expression"].children[0]
        if childvar.idx not in breakpoints:
            breakpoints[childvar.idx] = []

        x_low = childvar.lb
        x_up = childvar.ub

        breakpoints[childvar.idx] += [(nl, x_low, x_up)]

    return breakpoints_list


def ease_model(in_model):
    old_cons = []
    for c in in_model.cons:
        old_cons += [copy.copy(c)]
    additional_lins = []
    add_name = "ref"
    bias_varidx = len(in_model.vars)
    bias_considx = len(in_model.cons)

    test_nls = []

    for j in range(len(in_model.nonlinearexprs)):
        nl = in_model.nonlinearexprs[j]
        coef = 1
        if "coef" in nl:
            coef = nl["coef"]
        if nl["expression"].operation is None:
            if isinstance(nl["expression"], nltree.Variable):
                additional_lins += [
                    (nl["idx"], nl["expression"].idx, nl["expression"].coef * coef)
                ]
            elif isinstance(nl["expression"], nltree.Number):
                old_cons[nl["idx"]]["lb"] -= nl["expression"].value * coef
                old_cons[nl["idx"]]["ub"] -= nl["expression"].value * coef
        elif nl["expression"].operation.symbol == "sum":
            for c in nl["expression"].children:
                if isinstance(c, nltree.Variable):
                    additional_lins += [(nl["idx"], c.idx, c.coef * coef)]
                elif isinstance(c, nltree.Number):
                    old_cons[nl["idx"]]["lb"] -= c.value * coef
                    old_cons[nl["idx"]]["ub"] -= c.value * coef
                else:
                    assert False  # This should never happen!
        else:
            test_nls += [nl]

    ret_model = oto.OSILData(
        in_model.vars,
        in_model.objs,
        old_cons,
        in_model.lincons + additional_lins,
        in_model.quadcons,
        test_nls,
    )
    return ret_model


def obtainMIPfrom1d(in_model, epsilon=1, method=1, relax=0):
    breakpoints_list = []
    breakpoint_info = []
    additional_cons = []
    additional_vars = []
    old_cons = []
    for c in in_model.cons:
        old_cons += [copy.copy(c)]
    additional_lins = []
    add_name = "ref"
    bias_varidx = len(in_model.vars)
    bias_considx = len(in_model.cons)

    test_nls = []

    n = len(in_model.nonlinearexprs)
    for j in range(len(in_model.nonlinearexprs)):
        progress = round(j/n * 100)
        progress_bar = '[' + '=' * (progress // 2) + ' ' * (50 - progress // 2) + ']'
        print(f'\rReformulating: {progress_bar} {progress}%', end='')
        nl = in_model.nonlinearexprs[j]
        coef = 1
        add_name_nl = (add_name + "_NL_IDX_" + str(nl['expression'].nl_idx) + '_COUNT_' + str(j)) if nl['expression'].nl_idx != -1 else (add_name + "_NL_IDX_X_COUNT_" + str(j))
        if "coef" in nl:
            coef = nl["coef"]
        if nl["expression"].operation is None:
            if isinstance(nl["expression"], nltree.Variable):
                additional_lins += [
                    (nl["idx"], nl["expression"].idx, nl["expression"].coef * coef)
                ]
            elif isinstance(nl["expression"], nltree.Number):
                old_cons[nl["idx"]]["lb"] -= nl["expression"].value * coef
                old_cons[nl["idx"]]["ub"] -= nl["expression"].value * coef
        elif nl["expression"].operation.symbol == "sum":
            for c in nl["expression"].children:
                if isinstance(c, nltree.Variable):
                    additional_lins += [(nl["idx"], c.idx, c.coef * coef)]
                elif isinstance(c, nltree.Number):
                    old_cons[nl["idx"]]["lb"] -= c.value * coef
                    old_cons[nl["idx"]]["ub"] -= c.value * coef
                else:
                    assert False  # This should never happen!
        else:
            # find breakpoints
            f = nltree.get_pyomo_expression(nl["expression"].operation)
            f = nl["expression"].operation
            childvar = nl["expression"].children[0]
            x_low = childvar.lb
            x_up = childvar.ub
            (
                breakpoints,
                y,
                errors_low,
                errors_up,
                m_vals,
                t_vals,
            ) = linrelax.find_breakpoints(
                f, x_low, x_up, epsilon
            )
            #TODO: return interesting stuff about breakpoints
            breakpoint_info += [{'breakpoints': breakpoints, 'nl': nl['expression'], 'var': in_model.vars[childvar.idx]}]
            breakpoints_list += [(len(breakpoints), x_up - x_low)]

            if 1 <= method <= 8:
                # call methods with cur_idx = idx of new constraint
                cur_idx = bias_considx
                ref_lb = 0
                ref_ub = 0

                if method == 1:
                    (
                        new_vars,
                        new_cons,
                        new_lins,
                    ) = miprep.disaggregated_convex_combination_model(
                        nl["expression"],
                        breakpoints,
                        y,
                        add_name_nl,
                        cur_idx,
                        bias_varidx,
                        bias_considx + 1,
                        errors_low,
                        errors_up,
                        relax=relax,
                    )
                elif method == 2:
                    (
                        new_vars,
                        new_cons,
                        new_lins,
                    ) = miprep.logarithmic_disaggregated_convex_combination_model(
                        nl["expression"],
                        breakpoints,
                        y,
                        add_name_nl,
                        cur_idx,
                        bias_varidx,
                        bias_considx + 1,
                        errors_low,
                        errors_up,
                        relax=relax,
                    )
                elif method == 3:
                    (
                        new_vars,
                        new_cons,
                        new_lins,
                    ) = miprep.aggregated_convex_combination_model(
                        nl["expression"],
                        breakpoints,
                        y,
                        add_name_nl,
                        cur_idx,
                        bias_varidx,
                        bias_considx + 1,
                        errors_low,
                        errors_up,
                        relax=relax,
                    )
                elif method == 4:
                    (
                        new_vars,
                        new_cons,
                        new_lins,
                    ) = miprep.logarithmic_aggregated_convex_combination_model(
                        nl["expression"],
                        breakpoints,
                        y,
                        add_name_nl,
                        cur_idx,
                        bias_varidx,
                        bias_considx + 1,
                        errors_low,
                        errors_up,
                        relax=relax,
                    )
                elif method == 5:
                    new_vars, new_cons, new_lins = miprep.classical_incremental_method(
                        nl["expression"],
                        breakpoints,
                        y,
                        add_name_nl,
                        cur_idx,
                        bias_varidx,
                        bias_considx + 1,
                        errors_low,
                        errors_up,
                        relax=relax,
                    )
                    ref_lb -= y[0]
                    ref_ub -= y[0]

                elif method == 6:
                    new_vars, new_cons, new_lins = miprep.multiple_choice_model(
                        nl["expression"],
                        breakpoints,
                        y,
                        m_vals,
                        t_vals,
                        add_name_nl,
                        cur_idx,
                        bias_varidx,
                        bias_considx + 1,
                        errors_low,
                        errors_up,
                        relax=relax,
                    )

                elif method == 7:
                    new_vars, new_cons, new_lins = miprep.binary_zig_zag_model(
                        nl["expression"],
                        breakpoints,
                        y,
                        m_vals,
                        t_vals,
                        add_name_nl,
                        cur_idx,
                        bias_varidx,
                        bias_considx + 1,
                        errors_low,
                        errors_up,
                        relax=relax,
                    )
                elif method == 8:
                    new_vars, new_cons, new_lins = miprep.general_integer_zig_zag_model(
                        nl["expression"],
                        breakpoints,
                        y,
                        m_vals,
                        t_vals,
                        add_name_nl,
                        cur_idx,
                        bias_varidx,
                        bias_considx + 1,
                        errors_low,
                        errors_up,
                        relax=relax,
                    )

                # create new variable z
                n_z = bias_varidx + len(new_vars)
                new_vars += [
                    {
                        "name": add_name_nl + "_z_" + str(j),
                        "lb": -np.inf,
                        "ub": np.inf,
                        "type": "C",
                    }
                ]
                # add z to cur_idx (nl['idx'])
                new_lins += [(nl["idx"], n_z, 1)]
                # create new constraint for REF(nl) [ + eps ] - z = 0
                new_cons = [
                    {"name": add_name_nl + "_cons_z_" + str(j), "lb": ref_lb, "ub": ref_ub}
                ] + new_cons
                new_lins += [(cur_idx, n_z, -1)]

                if relax == 2:
                    n_epsup = bias_varidx + len(new_vars)
                    new_vars += [
                        {
                            "name": add_name_nl + "_epsilon_up",
                            "lb": 0,
                            "ub": epsilon,
                            "type": "C",
                        }
                    ]
                    n_epslow = bias_varidx + len(new_vars)
                    new_vars += [
                        {
                            "name": add_name_nl + "_epsilon_down",
                            "lb": 0,
                            "ub": epsilon,
                            "type": "C",
                        }
                    ]
                    new_lins += [(cur_idx, n_epsup, -1)]
                    new_lins += [(cur_idx, n_epslow, 1)]

                additional_vars += new_vars
                additional_cons += new_cons
                additional_lins += new_lins

                bias_varidx += len(new_vars)
                bias_considx += len(new_cons)
            else:
                test_nls += [nl]

    print()
    ret_model = oto.OSILData(
        in_model.vars + additional_vars,
        in_model.objs,
        old_cons + additional_cons,
        in_model.lincons + additional_lins,
        in_model.quadcons,
        test_nls,
    )
    return ret_model, breakpoints_list, breakpoint_info
