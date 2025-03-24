import copy

import pyomo.environ as pyo
from bs4 import BeautifulSoup as bs
import numpy as np
import datastructure_nonlinearTree as nltree
from dataclasses import dataclass

@dataclass
class OSILData:
    vars: list
    objs: list
    cons: list
    lincons: list
    quadcons: list
    nonlinearexprs: list

    def __repr__(self):
        res = ""
        res += "Variables         : " + str(self.vars)
        res += "\n"
        res += "Objectives        : " + str(self.objs)
        res += "\n"
        res += "Constraints       : " + str(self.cons)
        res += "\n"
        res += "Lin. constraints  : " + str(self.lincons)
        res += "\n"
        res += "Quad. constraints : " + str(self.quadcons)
        res += "\n"
        res += "Nonlinearities    : " + str(self.nonlinearexprs)
        res += "\n"
        res += "Expressions       : \n"
        for n in self.nonlinearexprs:
            res += str(n["expression"])
            res += "\n"
            res += n["expression"].get_tree()
            res += "\n"

        return res


# read osil file and return string
def read_osil(filename):
    with open(filename, "r") as f:
        data = f.read()
    data = bs(data, "xml")
    return data


def get_variables(osil_data):
    vars = []
    vartag = osil_data.find("variables")
    if not vartag:
        return vars
    var_tags = vartag.find_all("var")
    for v in var_tags:
        v_info = {}
        for attr in ["name", "lb", "ub", "type"]:
            v_info[attr] = v.get(attr)
        if v_info["lb"] is None:
            v_info["lb"] = 0
        elif v_info["lb"] == "-INF":
            v_info["lb"] = -np.inf
        if v_info["ub"] is None:
            v_info["ub"] = np.inf
        elif v_info["ub"] == "INF":
            v_info["ub"] = np.inf
        v_info["lb"] = float(v_info["lb"])
        v_info["ub"] = float(v_info["ub"])
        if v_info["type"] is None:
            v_info["type"] = "C"
        vars += [v_info]
    return vars


def get_objectives(osil_data):
    objs = []
    objtag = osil_data.find("objectives")
    if not objtag:
        return [
            {
                "name": "no_objective",
                "idx": -1,
                "maxOrMin": "min",
                "numberOfObjCoef": 0,
                "contents": [],
            }
        ]
        return objs
    obj_tags = objtag.find_all("obj")
    for o in obj_tags:
        o_info = {}
        for attr in ["name", "maxOrMin", "numberOfObjCoef", "constant"]:
            o_info[attr] = o.get(attr)
        o_info["contents"] = []
        coef_tags = o.find_all("coef")
        for c in coef_tags:
            c_info = {}
            c_info["idx"] = int(c.get("idx"))
            c_info["value"] = float(c.string)
            o_info["contents"] += [c_info]
    objs += [o_info]
    return objs


def get_constraints(osil_data):
    cons = []
    constag = osil_data.find("constraints")
    if not constag:
        return cons
    cons_tags = constag.find_all("con")
    for c in cons_tags:
        c_info = {}
        for attr in ["name", "lb", "ub"]:
            c_info[attr] = c.get(attr)
        if c_info["lb"] is None:
            c_info["lb"] = -np.inf
        elif c_info["lb"] == "-INF":
            c_info["lb"] = -np.inf
        if c_info["ub"] is None:
            c_info["ub"] = np.inf
        elif c_info["ub"] == "INF":
            c_info["ub"] = np.inf
        c_info["lb"] = float(c_info["lb"])
        c_info["ub"] = float(c_info["ub"])
        cons += [c_info]
    return cons


def use_mult_incr(el):
    if el.get("mult"):
        mult = int(el.get("mult"))
    else:
        mult = 1
    if el.get("incr"):
        incr = float(el.get("incr"))
    else:
        incr = 0
    res = []
    for i in range(0, mult):
        res += [float(el.string) + i * incr]
    return np.array(res)


def get_linearConstraintCoefficients(osil_data):
    coeff = []
    lincontag = osil_data.find("linearConstraintCoefficients")
    if not lincontag:
        return coeff
    start_osil = lincontag.find("start")
    rowIdx_osil = lincontag.find("rowIdx")
    colIdx_osil = lincontag.find("colIdx")
    value_osil = lincontag.find("value")
    start = np.array([])
    for el in list(start_osil.children):
        start = np.append(start, use_mult_incr(el))
    rowIdx = np.array([])
    colIdx = np.array([])
    if rowIdx_osil:
        for el in list(rowIdx_osil.children):
            rowIdx += use_mult_incr(el)
        colIdx = np.array([int(-1)] * len(rowIdx))
        for s in start:
            colIdx[int(s) :] += 1

    if colIdx_osil:
        for el in list(colIdx_osil.children):
            colIdx = np.append(colIdx, use_mult_incr(el))
        rowIdx = np.array([int(-1)] * len(colIdx))
        for s in start:
            rowIdx[int(s) :] += 1
    value = np.array([])
    for el in list(value_osil.children):
        value = np.append(value, use_mult_incr(el))

    for i in range(len(value)):
        coeff += [(int(rowIdx[i]), int(colIdx[i]), value[i])]

    return coeff


def get_quadraticCoefficients(osil_data):
    quads = []
    quadtag = osil_data.find("quadraticCoefficients")
    if not quadtag:
        return quads
    quad_tags = quadtag.find_all("qTerm")
    for q in quad_tags:
        q_info = {}
        for attr in ["idx", "idxOne", "idxTwo", "coef"]:
            q_info[attr] = q.get(attr)
        q_info["idx"] = int(q_info["idx"])
        q_info["idxOne"] = int(q_info["idxOne"])
        q_info["idxTwo"] = int(q_info["idxTwo"])
        q_info["coef"] = float(q_info["coef"])
        quads += [q_info]
    return quads


def create_nlTree(root, vars):
    children = list(root.children)
    num_children = len(children)
    if num_children == 0:
        if root.name == "number":
            return nltree.Number(
                value=float(root.get("value")),
            )
        elif root.name == "variable":
            coef = root.get("coef")
            if not coef:
                coef = 1
            else:
                coef = float(coef)
            idx = int(root.get("idx"))

            if coef != 1:
                v = nltree.Variable(
                    coef=1, idx=idx, lb=vars[idx]["lb"], ub=vars[idx]["ub"]
                )
                f = nltree.Number(
                    value=coef
                )
                return nltree.Nonlinear_ExpTree(
                    num_children=2,
                    children=[f, v],
                    operation=nltree.expressions["product"],
                )

            return nltree.Variable(
                coef=float(coef), idx=idx, lb=vars[idx]["lb"], ub=vars[idx]["ub"]
            )
        else:
            exit(root + "Not implemented")
    operation = root.name
    children_trees = [create_nlTree(child, vars) for child in children]
    if operation in nltree.expressions:
        assert (
            nltree.expressions[operation].dim[0]
            <= num_children
            <= nltree.expressions[operation].dim[1]
        ), ("Dimension of " + operation + " failed. Got " + str(num_children))
        if operation == "negate" and num_children == 1:
            return nltree.Nonlinear_ExpTree(
                num_children=2,
                children=[
                    nltree.Number(
                        value=-1
                    )
                ]
                + children_trees,
                operation=nltree.expressions["product"],
            )
        if operation == "power":
            c1 = children_trees[0]
            c2 = children_trees[1]
            if isinstance(c1, nltree.Number):
                op = nltree.NonlinearOperation(
                    symbol="power_ax", dim=(1, 1), num_param=c1.value
                )
                return nltree.Nonlinear_ExpTree(
                    num_children=1, children=[c2], operation=op
                )
            if isinstance(c2, nltree.Number):
                op = nltree.NonlinearOperation(
                    symbol="power_xa", dim=(1, 1), num_param=c2.value
                )
                return nltree.Nonlinear_ExpTree(
                    num_children=1, children=[c1], operation=op
                )
            else:
                return nltree.Nonlinear_ExpTree(
                    num_children=2,
                    children=children_trees,
                    operation=nltree.expressions["power_xy"],
                )
        return nltree.Nonlinear_ExpTree(
            num_children=num_children,
            children=children_trees,
            operation=nltree.expressions[operation]
        )
    else:
        exit(operation + ": Not implemented (children: " + str(num_children) + ")")


def get_nonlinearExpressions(osil_data, vars):
    nons = []
    nontag = osil_data.find("nonlinearExpressions")
    if not nontag:
        return nons
    non_tags = nontag.find_all("nl")
    for n in non_tags:
        n_info = {}
        n_tree = create_nlTree(list(n.children)[0], vars)
        for attr in ["idx", "coef"]:
            n_info[attr] = n.get(attr)
        n_info["idx"] = int(n_info["idx"])
        n_info["coef"] = 1 if n_info["coef"] is None else float(n_info["coef"])
        tree = n_tree
        n_info["expression"] = tree
        nons += [n_info]
    return nons


# returns datastructures created from osil file
def create_datastructures_from_osil(filename):
    data = read_osil(filename)

    vars = get_variables(data)
    objs = get_objectives(data)
    cons = get_constraints(data)
    lincons = get_linearConstraintCoefficients(data)
    quadcons = get_quadraticCoefficients(data)
    nonlinearexprs = get_nonlinearExpressions(data, vars)

    return OSILData(vars, objs, cons, lincons, quadcons, nonlinearexprs)


def split_product(node):
    # TODO is it possible to sort this somehow?
    if node.num_children <= 2:
        return node
    left = node.children[0]
    right = split_product(
        nltree.Nonlinear_ExpTree(
            node.num_children - 1, node.children[1:], node.operation, nl_idx=node.nl_idx, root_idx=node.root_idx
        )
    )
    node.num_children = 2
    node.children = [left, right]
    return node


def remove_products3_from_nonlinearities(exptree):
    if exptree.num_children == 0:
        return exptree

    if exptree.operation.symbol == "product" and exptree.num_children > 2:
        exptree = split_product(exptree)

    for i in range(exptree.num_children):
        exptree.children[i] = remove_products3_from_nonlinearities(exptree.children[i])
    return exptree


def reformulate_xabsx(exptree):
    if exptree.num_children == 0:
        return exptree
    if exptree.operation.symbol == "product":
        abs_children = []
        var_children = []
        loc_abs = []
        loc_var = []
        for i in range(exptree.num_children):
            if isinstance(exptree.children[i], nltree.Number):
                continue
            if isinstance(exptree.children[i], nltree.Variable):
                var_children.append(exptree.children[i].idx)
                loc_var.append(i)
            elif isinstance(exptree.children[i], nltree.Nonlinear_ExpTree) and exptree.children[i].operation.symbol == "abs":
                if isinstance(exptree.children[i].children[0], nltree.Variable):
                    abs_children.append(exptree.children[i].children[0].idx)
                    loc_abs.append(i)

        for i, c in enumerate(abs_children):
            if c in var_children:
                coef = np.abs(exptree.children[loc_abs[i]].children[0].coef)*exptree.children[loc_var[i]].coef
                exptree.children[loc_abs[i]] = nltree.Number(value=coef)
                newvarchild = nltree.Variable(coef=1, idx=exptree.children[loc_var[i]].idx, lb=exptree.children[loc_var[i]].lb, ub=exptree.children[loc_var[i]].ub)
                exptree.children[loc_var[i]] = nltree.Nonlinear_ExpTree(num_children=1, children=[newvarchild], operation=nltree.expressions["xabsx"], nl_idx=exptree.nl_idx, root_idx=exptree.root_idx)
                break

    for i in range(exptree.num_children):
        exptree.children[i] = reformulate_xabsx(exptree.children[i])
    return exptree

def reformulate_floats_to_coef_from_nonlinearity(exptree):
    if exptree.num_children == 0:
        return exptree
    if exptree.operation.symbol == "product" and exptree.num_children >= 2:
        if isinstance(exptree.children[0], nltree.Number):
            if isinstance(exptree.children[1], nltree.Variable):
                coef = exptree.children[0].value
                exptree.children[1].coef *= coef
                exptree.num_children -= 1
                exptree.children = exptree.children[1:]
                if exptree.num_children == 1:
                    exptree = exptree.children[0]
    for i in range(exptree.num_children):
        exptree.children[i] = reformulate_floats_to_coef_from_nonlinearity(exptree.children[i])
    return exptree

def reformulate_floats_to_coef(in_model):
    for i in range(len(in_model.nonlinearexprs)):
        in_model.nonlinearexprs[i]["expression"] = reformulate_floats_to_coef_from_nonlinearity(in_model.nonlinearexprs[i]["expression"])
    return in_model


# reformulates all products with three or more multiplicands
# for example: prod(x1, x2, x3) -> prod(x1, prod(x2, x3))
def reformulate_products3(in_model):
    for i in range(len(in_model.nonlinearexprs)):
        in_model.nonlinearexprs[i]["expression"] = reformulate_xabsx(in_model.nonlinearexprs[i]["expression"])
        in_model.nonlinearexprs[i]["expression"] = remove_products3_from_nonlinearities(
            in_model.nonlinearexprs[i]["expression"]
        )
    return in_model


def remove_div(node):
    left = node.children[0]
    right = node.children[1]
    right = nltree.Nonlinear_ExpTree(1, [right], nltree.expressions["inverse"], nl_idx=node.nl_idx, root_idx=node.root_idx)
    return nltree.Nonlinear_ExpTree(2, [left, right], nltree.expressions["product"], nl_idx=node.nl_idx, root_idx=node.root_idx)


def remove_division_from_nonlinearities(exptree):
    if exptree.num_children == 0:
        return exptree

    if exptree.operation.symbol == "divide":
        exptree = remove_div(exptree)

    for i in range(exptree.num_children):
        exptree.children[i] = remove_division_from_nonlinearities(exptree.children[i])

    return exptree


def reformulate_division(in_model):
    for i in range(len(in_model.nonlinearexprs)):
        in_model.nonlinearexprs[i]["expression"] = remove_division_from_nonlinearities(
            in_model.nonlinearexprs[i]["expression"]
        )
    return in_model


# reformulate each nonlinearity that is not a product
# for this, we introduce new variables
# for example nl(x1) -> z1 = nl(x1) and use z1 from now on
def reformulate_nonlinearities(in_model, debug=False):
    artificial_vars = []
    artificial_cons = []
    artificial_lins = []
    artificial_nlins = []
    art_name = "art"
    bias_varidx = len(in_model.vars)
    bias_considx = len(in_model.cons)
    cur_idx = 0

    def reformulate_nonlinearity(exptree):
        nonlocal artificial_vars
        nonlocal artificial_nlins
        nonlocal artificial_cons
        nonlocal artificial_lins
        nonlocal bias_varidx
        nonlocal bias_considx
        nonlocal cur_idx

        art_name_nl = art_name + "_NL_IDX_" + str(exptree.nl_idx) + '_COUNT_' + str(cur_idx) if exptree.nl_idx != -1 else art_name + "_NL_IDX_X_COUNT_" + str(cur_idx)

        if debug:
            print(exptree.nl_idx, exptree.root_idx, exptree)
        if exptree.num_children == 0:
            return exptree

        for i in range(exptree.num_children):
            exptree.children[i] = reformulate_nonlinearity(exptree.children[i])

        bounds = []
        for i in range(exptree.num_children):
            ch = exptree.children[i]
            if isinstance(ch, nltree.Variable):
                bounds += [
                    (
                        min(ch.lb * ch.coef, ch.ub * ch.coef),
                        max(ch.lb * ch.coef, ch.ub * ch.coef),
                    )
                ]
            elif isinstance(exptree.children[i], nltree.Number):
                bounds += [(ch.value, ch.value)]
        lb, ub = nltree.get_lower_and_upper_bound(exptree.operation, bounds)

        var = {"name": art_name_nl + "_var" + str(cur_idx), "lb": lb, "ub": ub, "type": "C"}
        con = {"name": art_name_nl + "_con" + str(cur_idx), "lb": 0, "ub": 0}
        artificial_vars += [var]
        artificial_cons += [con]
        artificial_lins += [(cur_idx + bias_considx, cur_idx + bias_varidx, -1)]
        new_tree = nltree.Nonlinear_ExpTree(
            num_children=exptree.num_children,
            children=exptree.children,
            operation=exptree.operation,
            nl_idx=exptree.nl_idx,
            root_idx=exptree.root_idx
        )
        artificial_nlins += [{"idx": cur_idx + bias_considx, "expression": new_tree}]
        cur_idx += 1

        return nltree.Variable(idx=bias_varidx + cur_idx - 1, coef=1, lb=lb, ub=ub, nl_idx=-5, root_idx=exptree.root_idx)

    for i in range(len(in_model.nonlinearexprs)):
        in_model.nonlinearexprs[i]["expression"] = reformulate_nonlinearity(
            in_model.nonlinearexprs[i]["expression"]
        )
        artificial_nlins += [in_model.nonlinearexprs[i]]

    ret_model = OSILData(
        in_model.vars + artificial_vars,
        in_model.objs,
        in_model.cons + artificial_cons,
        in_model.lincons + artificial_lins,
        in_model.quadcons,
        artificial_nlins,
    )
    return ret_model


# reformulate xy to 1/2(p^2 - x^2 - y^2) and p = x+y
def remove_products2(in_model):
    additional_cons = []
    additional_vars = []
    additional_lins = []
    additional_nlins = []
    sq_dict = {}  # x -> x^2
    bilinear_dict = {}
    add_name = "art_sq"
    bias_varidx = len(in_model.vars)
    bias_considx = len(in_model.cons)
    cur_varidx = 0
    cur_considx = 0

    def remove_products2_from_nonlinearities(exptree, coef_prod=1):
        nonlocal additional_cons
        nonlocal additional_vars
        nonlocal additional_lins
        nonlocal additional_nlins
        nonlocal sq_dict
        nonlocal bilinear_dict
        nonlocal add_name
        nonlocal bias_varidx
        nonlocal bias_considx
        nonlocal cur_varidx
        nonlocal cur_considx

        add_name_nl = add_name + "_NL_IDX_" + str(exptree.nl_idx) + '_COUNT_' if exptree.nl_idx != -1 else add_name + "_X_" 

        if exptree.num_children == 0:
            return exptree

        if exptree.operation.symbol == "product":
            c1 = exptree.children[0]
            c2 = exptree.children[1]
            if isinstance(c1, nltree.Variable) and isinstance(c2, nltree.Variable):
                if c1.idx == c2.idx:
                    if c1.idx in sq_dict:
                        xs_idx, xs = sq_dict[c1.idx]
                    else:
                        xs_lb = (
                            0
                            if c1.lb <= 0 and c1.ub >= 0
                            else min(c1.lb**2, c1.ub**2)
                        )
                        xs_ub = max(c1.lb**2, c1.ub**2)
                        xs = {
                            "name": add_name_nl + "_xs_var" + str(cur_varidx),
                            "lb": xs_lb,
                            "ub": xs_ub,
                            "type": "C",
                        }
                        xs_con = {
                            "name": add_name_nl + "_xs_con" + str(cur_considx),
                            "lb": 0,
                            "ub": 0,
                        }
                        xs_idx = cur_varidx + bias_varidx
                        additional_lins += [(cur_considx + bias_considx, xs_idx, -1)]
                        new_tree = nltree.Nonlinear_ExpTree(
                            num_children=1,
                            children=[
                                nltree.Variable(coef=1, idx=c1.idx, lb=c1.lb, ub=c1.ub, root_idx=exptree.root_idx)
                            ],
                            operation=nltree.expressions["square"],
                            nl_idx=exptree.nl_idx,
                            root_idx=exptree.root_idx
                        )
                        additional_nlins += [
                            {"idx": cur_considx + bias_considx, "expression": new_tree}
                        ]
                        additional_vars += [xs]
                        additional_cons += [xs_con]
                        cur_considx += 1
                        cur_varidx += 1
                        sq_dict[c1.idx] = (xs_idx, xs)

                    newtreetest = nltree.Variable(
                        idx=xs_idx,
                        coef=coef_prod * c1.coef * c2.coef,
                        lb=xs["lb"],
                        ub=xs["ub"],
                        root_idx=exptree.root_idx
                    )
                    return newtreetest

                # TODO check if we already replaced c1*c2
                if (min(c1.idx, c2.idx), max(c1.idx, c2.idx)) in bilinear_dict:
                    z_idx, z_lb, z_ub = bilinear_dict[
                        (min(c1.idx, c2.idx), max(c1.idx, c2.idx))
                    ]
                    return_var = nltree.Variable(
                        idx=z_idx, coef=c1.coef * c2.coef * coef_prod, lb=z_lb, ub=z_ub, root_idx=exptree.root_idx
                    )
                    return return_var

                child_list = []
                # coeff = c1.coef * c2.coef * 0.5 * coef_prod
                coef_mccormick = 0.5

                if c1.idx in sq_dict:
                    xs_idx, xs = sq_dict[c1.idx]
                else:
                    xs_lb = (
                        0 if c1.lb <= 0 and c1.ub >= 0 else min(c1.lb**2, c1.ub**2)
                    )
                    xs_ub = max(c1.lb**2, c1.ub**2)
                    xs = {
                        "name": add_name_nl + "_xs_var" + str(cur_varidx),
                        "lb": xs_lb,
                        "ub": xs_ub,
                        "type": "C",
                    }
                    xs_con = {
                        "name": add_name_nl + "_xs_con" + str(cur_considx),
                        "lb": 0,
                        "ub": 0,
                    }
                    xs_idx = cur_varidx + bias_varidx
                    additional_lins += [(cur_considx + bias_considx, xs_idx, -1)]
                    new_tree = nltree.Nonlinear_ExpTree(
                        num_children=1,
                        children=[
                            nltree.Variable(coef=1, idx=c1.idx, lb=c1.lb, ub=c1.ub, root_idx=exptree.root_idx)
                        ],
                        operation=nltree.expressions["square"],
                        nl_idx=exptree.nl_idx,
                        root_idx=exptree.root_idx
                    )
                    additional_nlins += [
                        {"idx": cur_considx + bias_considx, "expression": new_tree}
                    ]
                    additional_vars += [xs]
                    additional_cons += [xs_con]
                    cur_considx += 1
                    cur_varidx += 1
                    sq_dict[c1.idx] = (xs_idx, xs)
                child_list += [
                    nltree.Variable(
                        coef=-1 * coef_mccormick, idx=xs_idx, lb=xs["lb"], ub=xs["ub"], root_idx=exptree.root_idx
                    )
                ]

                if c2.idx in sq_dict:
                    ys_idx, ys = sq_dict[c2.idx]
                else:
                    ys_lb = (
                        0 if c2.lb <= 0 and c2.ub >= 0 else min(c2.lb**2, c2.ub**2)
                    )
                    ys_ub = max(c2.lb**2, c2.ub**2)
                    ys = {
                        "name": add_name_nl + "_ys_var" + str(cur_varidx),
                        "lb": ys_lb,
                        "ub": ys_ub,
                        "type": "C",
                    }
                    ys_con = {
                        "name": add_name_nl + "_ys_con" + str(cur_considx),
                        "lb": 0,
                        "ub": 0,
                    }
                    ys_idx = cur_varidx + bias_varidx
                    additional_lins += [(cur_considx + bias_considx, ys_idx, -1)]
                    new_tree = nltree.Nonlinear_ExpTree(
                        num_children=1,
                        children=[
                            nltree.Variable(coef=1, idx=c2.idx, lb=c2.lb, ub=c2.ub, root_idx=exptree.root_idx)
                        ],
                        operation=nltree.expressions["square"],
                        nl_idx=exptree.nl_idx,
                        root_idx=exptree.root_idx
                    )
                    additional_nlins += [
                        {"idx": cur_considx + bias_considx, "expression": new_tree}
                    ]
                    additional_vars += [ys]
                    additional_cons += [ys_con]
                    cur_considx += 1
                    cur_varidx += 1
                    sq_dict[c2.idx] = (ys_idx, ys)
                child_list += [
                    nltree.Variable(
                        coef=-1 * coef_mccormick, idx=ys_idx, lb=ys["lb"], ub=ys["ub"], root_idx=exptree.root_idx
                    )
                ]

                p = {
                    "name": add_name_nl + "_p_var" + str(cur_varidx),
                    "lb": c1.lb + c2.lb,
                    "ub": c1.ub + c2.ub,
                    "type": "C",
                }
                p_con = {
                    "name": add_name_nl + "_p_con" + str(cur_considx),
                    "lb": 0,
                    "ub": 0,
                }
                p_idx = cur_varidx + bias_varidx
                additional_lins += [(cur_considx + bias_considx, p_idx, -1)]
                additional_lins += [(cur_considx + bias_considx, c1.idx, 1)]
                additional_lins += [(cur_considx + bias_considx, c2.idx, 1)]
                cur_considx += 1
                cur_varidx += 1

                ps_lb = (
                    0
                    if p["lb"] <= 0 and p["ub"] >= 0
                    else min(p["lb"] ** 2, p["ub"] ** 2)
                )
                ps_ub = max(p["lb"] ** 2, p["ub"] ** 2)
                ps = {
                    "name": add_name_nl + "_ps_var" + str(cur_varidx),
                    "lb": ps_lb,
                    "ub": ps_ub,
                    "type": "C",
                }
                ps_con = {
                    "name": add_name_nl + "_ps_con" + str(cur_considx),
                    "lb": 0,
                    "ub": 0,
                }
                ps_idx = cur_varidx + bias_varidx
                additional_lins += [(cur_considx + bias_considx, ps_idx, -1)]
                new_tree = nltree.Nonlinear_ExpTree(
                    num_children=1,
                    children=[
                        nltree.Variable(coef=1, idx=p_idx, lb=p["lb"], ub=p["ub"], root_idx=exptree.root_idx)
                    ],
                    operation=nltree.expressions["square"],
                    nl_idx=exptree.nl_idx,
                    root_idx=exptree.root_idx
                )
                additional_nlins += [
                    {"idx": cur_considx + bias_considx, "expression": new_tree}
                ]

                child_list += [
                    nltree.Variable(
                        coef=coef_mccormick, idx=ps_idx, lb=ps["lb"], ub=ps["ub"], root_idx=exptree.root_idx
                    )
                ]
                cur_considx += 1
                cur_varidx += 1
                additional_vars += [p, ps]
                additional_cons += [ps_con, p_con]

                # 1/2(p^2 - x^2 - y^2) >= x.lb * y + y.lb * x - x.lb * y.lb
                mcc_1_con = {
                    "name": add_name_nl + "_mcc1_con" + str(cur_considx),
                    "lb": -np.inf,
                    "ub": c1.lb * c2.lb,
                }
                additional_lins += [
                    (cur_considx + bias_considx, c2.idx, c1.lb)
                ]  # x.lb * y
                additional_lins += [
                    (cur_considx + bias_considx, c1.idx, c2.lb)
                ]  # y.lb * x
                additional_lins += [
                    (cur_considx + bias_considx, ps_idx, -0.5)
                ]  # -1/2(p^2)
                additional_lins += [
                    (cur_considx + bias_considx, xs_idx, 0.5)
                ]  # 1/2(p^2)
                additional_lins += [
                    (cur_considx + bias_considx, ys_idx, 0.5)
                ]  # 1/2(p^2)
                cur_considx += 1

                # 1/2(p^2 - x^2 - y^2) >= x.ub * y + y.ub * x - x.ub * y.ub
                mcc_2_con = {
                    "name": add_name_nl + "_mcc2_con" + str(cur_considx),
                    "lb": -np.inf,
                    "ub": c1.ub * c2.ub,
                }
                additional_lins += [
                    (cur_considx + bias_considx, c2.idx, c1.ub)
                ]  # x.ub * y
                additional_lins += [
                    (cur_considx + bias_considx, c1.idx, c2.ub)
                ]  # y.ub * x
                additional_lins += [
                    (cur_considx + bias_considx, ps_idx, -0.5)
                ]  # -1/2(p^2)
                additional_lins += [
                    (cur_considx + bias_considx, xs_idx, 0.5)
                ]  # 1/2(p^2)
                additional_lins += [
                    (cur_considx + bias_considx, ys_idx, 0.5)
                ]  # 1/2(p^2)
                cur_considx += 1

                # 1/2(p^2 - x^2 - y^2) <= x.ub * y + y.lb * x - x.ub * y.lb
                mcc_3_con = {
                    "name": add_name_nl + "_mcc3_con" + str(cur_considx),
                    "lb": c1.ub * c2.lb,
                    "ub": np.inf,
                }
                additional_lins += [
                    (cur_considx + bias_considx, c2.idx, c1.ub)
                ]  # x.ub * y
                additional_lins += [
                    (cur_considx + bias_considx, c1.idx, c2.lb)
                ]  # y.lb * x
                additional_lins += [
                    (cur_considx + bias_considx, ps_idx, -0.5)
                ]  # -1/2(p^2)
                additional_lins += [
                    (cur_considx + bias_considx, xs_idx, 0.5)
                ]  # 1/2(p^2)
                additional_lins += [
                    (cur_considx + bias_considx, ys_idx, 0.5)
                ]  # 1/2(p^2)
                cur_considx += 1

                # 1/2(p^2 - x^2 - y^2) >= x.lb * y + y.ub * x - x.lb * y.ub
                mcc_4_con = {
                    "name": add_name_nl + "_mcc4_con" + str(cur_considx),
                    "lb": c1.lb * c2.ub,
                    "ub": np.inf,
                }
                additional_lins += [
                    (cur_considx + bias_considx, c2.idx, c1.lb)
                ]  # x.lb * y
                additional_lins += [
                    (cur_considx + bias_considx, c1.idx, c2.ub)
                ]  # y.ub * x
                additional_lins += [
                    (cur_considx + bias_considx, ps_idx, -0.5)
                ]  # -1/2(p^2)
                additional_lins += [
                    (cur_considx + bias_considx, xs_idx, 0.5)
                ]  # 1/2(p^2)
                additional_lins += [
                    (cur_considx + bias_considx, ys_idx, 0.5)
                ]  # 1/2(p^2)
                cur_considx += 1

                additional_cons += [mcc_1_con, mcc_2_con, mcc_3_con, mcc_4_con]

                z_lb = min([c1.lb * c2.lb, c1.lb * c2.ub, c1.ub * c2.lb, c1.ub * c2.ub])
                z_ub = max([c1.lb * c2.lb, c1.lb * c2.ub, c1.ub * c2.lb, c1.ub * c2.ub])
                z = {
                    "name": add_name_nl + "_xy_var" + str(cur_varidx),
                    "lb": z_lb,
                    "ub": z_ub,
                    "type": "C",
                }

                # Constraint z = new_tree
                z_con = {
                    "name": add_name_nl + "_xy_con" + str(cur_considx),
                    "lb": 0,
                    "ub": 0,
                }
                z_idx = cur_varidx + bias_varidx
                additional_lins += [(cur_considx + bias_considx, z_idx, -1)]
                new_tree = nltree.Nonlinear_ExpTree(
                    num_children=3,
                    children=child_list,
                    operation=nltree.expressions["sum"],
                    nl_idx=exptree.nl_idx,
                    root_idx=exptree.root_idx
                )
                additional_nlins += [
                    {"idx": cur_considx + bias_considx, "expression": new_tree}
                ]
                additional_vars += [z]
                additional_cons += [z_con]
                cur_considx += 1
                cur_varidx += 1
                bilinear_dict[(min(c1.idx, c2.idx), max(c1.idx, c2.idx))] = (
                    z_idx,
                    z_lb,
                    z_ub,
                )
                return_var = nltree.Variable(
                    idx=z_idx, coef=c1.coef * c2.coef * coef_prod, lb=z_lb, ub=z_ub, root_idx=exptree.root_idx
                )
                return return_var
            else:
                assert exptree.num_children == 2
                if isinstance(exptree.children[0], nltree.Number):
                    assert isinstance(exptree.children[1], nltree.Variable)
                    exptree.children[1].coef *= exptree.children[0].value
                    return exptree.children[1]
                elif isinstance(exptree.children[1], nltree.Number):
                    assert isinstance(exptree.children[0], nltree.Variable)
                    exptree.children[0].coef *= exptree.children[1].value
                    return exptree.children[0]
        else:
            for i in range(exptree.num_children):
                exptree.children[i] = remove_products2_from_nonlinearities(
                    exptree.children[i]
                )
        return exptree

    new_linex = []
    for i in range(len(in_model.quadcons)):
        q = in_model.quadcons[i]
        v1 = q["idxOne"]
        v2 = q["idxTwo"]
        new_linex += [
            {
                "idx": q["idx"],
                "coef": q["coef"],
                "expression": nltree.Nonlinear_ExpTree(
                    num_children=2,
                    children=[
                        nltree.Variable(
                            idx=v1,
                            lb=in_model.vars[v1]["lb"],
                            ub=in_model.vars[v1]["ub"],
                            root_idx=len(new_linex) + len(in_model.nonlinearexprs)
                        ),
                        nltree.Variable(
                            idx=v2,
                            lb=in_model.vars[v2]["lb"],
                            ub=in_model.vars[v2]["ub"],
                            root_idx=len(new_linex) + len(in_model.nonlinearexprs)
                        ),
                    ],
                    operation=nltree.expressions["product"],
                    nl_idx = len(new_linex) + len(in_model.nonlinearexprs),
                    root_idx = len(new_linex) + len(in_model.nonlinearexprs)
                ),
            }
        ]

    for i in range(len(in_model.nonlinearexprs)):
        in_model.nonlinearexprs[i]["expression"] = remove_products2_from_nonlinearities(
            in_model.nonlinearexprs[i]["expression"]
        )

    for i in range(len(new_linex)):
        new_linex[i]["expression"] = remove_products2_from_nonlinearities(
            new_linex[i]["expression"], coef_prod=new_linex[i]["coef"]
        )
        new_linex[i]["coef"] = 1

    ret_model = OSILData(
        in_model.vars + additional_vars,
        in_model.objs,
        in_model.cons + additional_cons,
        in_model.lincons + additional_lins,
        [],
        in_model.nonlinearexprs + additional_nlins + new_linex,
    )
    return ret_model


def print_nonzero_vars(rep, pyo_model, name=None):
    for v in rep.vars:
        val = pyo.value(pyo_model.component(v["name"]))
        if val and (name is None or name in v["name"]):
            print(v["name"] + ": " + str(val) + "\n (" + str(v) + ")")


def create_pyomomodel_from_OSILdata(rep):
    type_dict = {"C": pyo.Reals, "B": pyo.Binary, "I": pyo.Integers}
    model = pyo.ConcreteModel()
    for v in rep.vars:
        model.add_component(
            v["name"], pyo.Var(within=type_dict[v["type"]], bounds=(v["lb"], v["ub"]))
        )
    pyomo_objectives = []
    for obj in rep.objs:
        res = 0
        for coeff in obj["contents"]:
            res += coeff["value"] * model.component(rep.vars[coeff["idx"]]["name"])
        if "constant" in obj and obj["constant"] is not None:
            res += float(obj["constant"])
        pyomo_objectives += [res]
    pyomo_constraints = []
    for _ in rep.cons:
        pyomo_constraints += [0]

    for lins in rep.lincons:
        if lins[0] >= 0:
            pyomo_constraints[lins[0]] += lins[2] * model.component(
                rep.vars[lins[1]]["name"]
            )
        elif lins[0] == -1:
            pyomo_objectives[0] += lins[2] * model.component(rep.vars[lins[1]]["name"])
        else:
            exit("Not implemented multiple objectives")

    for quads in rep.quadcons:
        if quads["idx"] < 0:
            if quads["idx"] < -1:
                exit("Not implemented multiple objectives")
            else:
                pyomo_objectives[0] += (
                    quads["coef"]
                    * model.component(rep.vars[quads["idxOne"]]["name"])
                    * model.component(rep.vars[quads["idxTwo"]]["name"])
                )
        else:
            pyomo_constraints[quads["idx"]] += (
                quads["coef"]
                * model.component(rep.vars[quads["idxOne"]]["name"])
                * model.component(rep.vars[quads["idxTwo"]]["name"])
            )

    def nonlinear_pyomo(nt):
        if isinstance(nt, nltree.Variable):
            py_use = nt.coef * model.component(rep.vars[nt.idx]["name"])
        elif isinstance(nt, nltree.Number):
            py_use = nt.value
        else:
            inp = [nonlinear_pyomo(c) for c in nt.children]
            f = nltree.get_pyomo_expression(nt.operation)
            py_use = f(inp)
        return py_use

    for nlins in rep.nonlinearexprs:
        py_use = nonlinear_pyomo(nlins["expression"])
        if nlins["idx"] < 0:
            if nlins["idx"] < -1:
                exit("Not implemented multiple objectives")
            else:
                pyomo_objectives[0] += py_use
        else:
            pyomo_constraints[nlins["idx"]] += py_use

    for i in range(len(rep.objs)):
        obj = rep.objs[i]
        model.add_component(
            "obj_" + obj["name"],
            pyo.Objective(
                rule=pyomo_objectives[i],
                sense=pyo.minimize if obj["maxOrMin"] == "min" else pyo.maximize,
            ),
        )
    for i in range(len(rep.cons)):
        con = rep.cons[i]
        if con["lb"] == con["ub"]:
            model.add_component(
                con["name"], pyo.Constraint(expr=con["lb"] == pyomo_constraints[i])
            )
        elif con["lb"] > -np.inf and con["ub"] == np.inf:
            model.add_component(
                con["name"], pyo.Constraint(expr=con["lb"] <= pyomo_constraints[i])
            )
        elif con["ub"] < np.inf and con["lb"] == -np.inf:
            model.add_component(
                con["name"], pyo.Constraint(expr=pyomo_constraints[i] <= con["ub"])
            )
        elif con["lb"] > -np.inf and con["ub"] < np.inf:
            model.add_component(
                con["name"] + "_l",
                pyo.Constraint(expr=con["lb"] <= pyomo_constraints[i]),
            )
            model.add_component(
                con["name"] + "_u",
                pyo.Constraint(expr=pyomo_constraints[i] <= con["ub"]),
            )
    return model


def obtain_init_representation(filename):
    initial_rep = create_datastructures_from_osil(filename)
    return initial_rep


def obtain_1d_and_prod_representation(filename):
    initial_rep = create_datastructures_from_osil(filename)
    removedproducts_rep = reformulate_products3(initial_rep)
    removeddivision_rep = reformulate_division(removedproducts_rep)
    reform_rep = reformulate_nonlinearities(removeddivision_rep, debug=False)
    return reform_rep


def obtain_1d_representation(filename, ifthen=False):
    initial_rep = create_datastructures_from_osil(filename)
    removedfloats = reformulate_floats_to_coef(initial_rep)
    removedproducts_rep = reformulate_products3(removedfloats)
    removeddivision_rep = reformulate_division(removedproducts_rep)
    reform_rep = reformulate_nonlinearities(removeddivision_rep, debug=False)
    onedim_rep = remove_products2(reform_rep)
    return onedim_rep
