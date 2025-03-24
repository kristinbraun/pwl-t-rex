import datastructure_nonlinearTree as nltree



def obtain_statistics(rep):
    stats = {
        "num_variables": len(rep.vars),
        "num_binary_variables": 0,
        "num_integer_variables": 0,
        "num_continuous_variables": 0,
        #'variable_bounds': [],
        "num_constraints": len(rep.cons),
        "num_nonlinear_1d": 0,
        "num_nonlinear_quad": len(rep.quadcons),
        "num_nonlinear_square": 0,
        "num_nonlinear_equations": len(rep.quadcons),
        "nonlinear_types": {},
    }

    for v in rep.vars:
        if v["type"] == "B":
            stats["num_binary_variables"] += 1
        if v["type"] == "I":
            stats["num_integer_variables"] += 1
        if v["type"] == "C":
            stats["num_continuous_variables"] += 1
        # stats['variable_bounds'] += [(v['type'], v['lb'], v['ub'])]

    # TODO hier den Typ der Nichtlinearität speichern (nl['expression'].operation.symbol)
    for nl in rep.nonlinearexprs:
        if nl["expression"].num_children == 0:
            continue
        if (nl["expression"].operation.symbol, nl["expression"].num_children) in stats[
            "nonlinear_types"
        ]:
            stats["nonlinear_types"][
                (nl["expression"].operation.symbol, nl["expression"].num_children)
            ] += 1
        else:
            stats["nonlinear_types"][
                (nl["expression"].operation.symbol, nl["expression"].num_children)
            ] = 1

        if nl["expression"].operation.symbol == "sum":
            onlylin = True
            for c in nl["expression"].children:
                if not isinstance(c, nltree.Variable) and not isinstance(
                    c, nltree.Number
                ):
                    onlylin = False
            if onlylin:
                continue
        if nl["expression"].num_children == 1:
            if isinstance(nl["expression"].children[0], nltree.Number):
                continue
            if isinstance(nl["expression"].children[0], nltree.Variable):
                if nl["expression"].operation.symbol == "square":
                    stats["num_nonlinear_square"] += 1
                    stats["num_nonlinear_quad"] += 1
                else:
                    stats["num_nonlinear_1d"] += 1
            stats["num_nonlinear_equations"] += 1
        else:
            if (
                nl["expression"].operation.symbol == "product"
                and nl["expression"].num_children == 2
            ):
                if isinstance(
                    nl["expression"].children[0], nltree.Variable
                ) and isinstance(nl["expression"].children[1], nltree.Variable):
                    stats["num_nonlinear_quad"] += 1
                    if (
                        nl["expression"].children[0].idx
                        == nl["expression"].children[1].idx
                    ):
                        stats["num_nonlinear_square"] += 1
            stats["num_nonlinear_equations"] += 1

    return stats
