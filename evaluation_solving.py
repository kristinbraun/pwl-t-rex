import numpy as np
import pyomo.environ as pyo

import pyomo.opt as popt
from gurobipy import GRB

GUR_OUTPUT = False
gur_tl = None


def solve_and_store_results(pyo_model, rep, gur=False):
    if not gur:
        opt = pyo.SolverFactory(
            "scip", executable="../../scip/scipoptsuite-8.0.1/scip/bin/scip"
        )
    elif gur:
        opt = pyo.SolverFactory("gurobi_persistent", solver_io="python")
        opt._set_instance(pyo_model)
        opt.options["NumericFocus"] = 2
        opt._lasttime = 0
    opt._sol_tuples = {}
    solving_results = {
        "status": None,
        "objective": None,
        "lower": None,
        "upper": None,
        "gap": None,
        "objectives": {},
        "time": None,
        "time_firstprimal": None,
        "info": None,
    }
    try:

        def my_callback(_, model, where):
            if where == GRB.Callback.MIPSOL:
                p_time = model.cbGet(GRB.Callback.RUNTIME)
                if not hasattr(model, "_firstprimal"):
                    model._firstprimal = p_time

            if where == GRB.Callback.MIPNODE:
                newprimal = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
                newdual = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                if (
                    not hasattr(model, "_currentprimal")
                    or newprimal != model._currentprimal
                    or not hasattr(model, "_currentdual")
                    or newdual != model._currentdual
                ):
                    p_time = model.cbGet(GRB.Callback.RUNTIME)
                    if p_time - model._lasttime >= 0.1:
                        model._lasttime = p_time
                        model._currentdual = newprimal
                        model._currentdual = newdual
                        model._sol_tuples[p_time] = (newprimal, newdual)

        if gur:
            opt.set_callback(my_callback)

        res = opt.solve(
            pyo_model, report_timing=False, tee=False, options={"TimeLimit": gur_tl}
        )

        if gur and hasattr(opt, "_firstprimal"):
            solving_results["time_firstprimal"] = opt._firstprimal

        if gur:
            solving_results["objectives"] = opt._sol_tuples

        # Case 1: optimal solution
        if (
            res.solver.termination_condition == popt.TerminationCondition.optimal
            and res.solver.status == popt.SolverStatus.ok
        ):
            solving_results["status"] = "OPTIMAL"
            solving_results["lower"] = res.problem.lower_bound
            solving_results["upper"] = res.problem.upper_bound
            if gur:
                solving_results["time"] = opt.get_model_attr("Runtime")
                solving_results["objectives"][solving_results["time"]] = (
                    solving_results["upper"],
                    solving_results["lower"],
                )
            else:
                solving_results["time"] = res.solver.time
            solving_results["objective"] = pyo.value(
                pyo_model.component("obj_" + rep.objs[0]["name"])
            )
            solving_results["gap"] = np.abs(
                res.problem.upper_bound - res.problem.lower_bound
            ) / (np.abs(solving_results["objective"]) + 1e-10)
        # Case 2: infeasible
        elif res.solver.termination_condition == popt.TerminationCondition.infeasible:
            solving_results["status"] = "INFEASIBLE"
            if gur:
                solving_results["time"] = opt.get_model_attr("Runtime")
                solving_results["objectives"][solving_results["time"]] = (
                    solving_results["upper"],
                    solving_results["lower"],
                )
            else:
                solving_results["time"] = res.solver.time
        # Case 3: unbounded
        elif res.solver.termination_condition == popt.TerminationCondition.unbounded:
            solving_results["status"] = "UNBOUNDED"
            if gur:
                solving_results["time"] = opt.get_model_attr("Runtime")
                solving_results["objectives"][solving_results["time"]] = (
                    solving_results["upper"],
                    solving_results["lower"],
                )
            else:
                solving_results["time"] = res.solver.time
        # Case 4: Timelimit exceeded (must be given in gur_tl)
        elif res.solver.termination_condition == popt.TerminationCondition.maxTimeLimit:
            solving_results["lower"] = res.problem.lower_bound
            solving_results["upper"] = res.problem.upper_bound
            solving_results["objective"] = pyo.value(
                pyo_model.component("obj_" + rep.objs[0]["name"])
            )
            solving_results["gap"] = np.abs(
                res.problem.upper_bound - res.problem.lower_bound
            ) / (np.abs(solving_results["objective"] + 1e-10))
            solving_results["status"] = "TIMELIMIT"
            if gur:
                solving_results["time"] = opt.get_model_attr("Runtime")
                solving_results["objectives"][solving_results["time"]] = (
                    solving_results["upper"],
                    solving_results["lower"],
                )
            else:
                solving_results["time"] = res.solver.time
        else:
            solving_results["lower"] = res.problem.lower_bound
            solving_results["upper"] = res.problem.upper_bound
            if gur:
                solving_results["time"] = opt.get_model_attr("Runtime")
                solving_results["objectives"][solving_results["time"]] = (
                    solving_results["upper"],
                    solving_results["lower"],
                )
            else:
                solving_results["time"] = res.solver.time
            solving_results["status"] = "ERROR"
    except Exception as exp:
        # Case: Other error
        solving_results["status"] = "ERROR"
        solving_results["info"] = exp

    return solving_results
