import pyomo.environ as pyo
import pyomo.opt as popt
import settings

import minlp_evaluate_solutions as minlp_eval



def solve_minlp_for_reference(init_model, init_rep):
    """Solve original MINLP with SCIP."""
    model = init_model.clone()
    try:
        opt = pyo.SolverFactory("scip", executable=settings.scip_executable)
        results = opt.solve(
            model,
            tee=settings.solver_output,
            options={"limits/time": settings.timelimit},
        )
        return minlp_eval._extract_solver_results(results, model, init_rep)
    except Exception as e:
        return {
            "status": "ERROR",
            "objective": None,
            "time": 0.0,
            "time_firstprimal": None,
            "info": str(e),
        }


def solve_minlp_warmstart(init_model, init_rep, mip_solution):
    """Solve original MINLP with SCIP using MIP solution as warmstart."""
    model = init_model.clone()

    for v in init_rep.vars:
        var_name = v["name"]
        if var_name not in mip_solution:
            continue
        var_component = model.component(var_name)
        mip_val = mip_solution[var_name]

        if v["type"] in ["B", "I"]:
            rounded_val = round(mip_val)
            if abs(mip_val - rounded_val) > settings.minlp_zero_tol:
                print(
                    f"Warning: Variable {var_name} rounded from {mip_val} to {rounded_val}"
                )
            mip_val = _clamp(rounded_val, v["lb"], v["ub"])

        var_component.set_value(mip_val)

    try:
        opt = pyo.SolverFactory("scip", executable=settings.scip_executable)
        results = opt.solve(
            model,
            tee=settings.solver_output,
            options={"limits/time": settings.timelimit},
        )
        return minlp_eval._extract_solver_results(results, model, init_rep)
    except Exception as e:
        return {
            "status": "ERROR",
            "objective": None,
            "time": 0.0,
            "time_firstprimal": None,
            "info": str(e),
        }



def solve_nlp_fixed(init_model, init_rep, mip_solution):
    """Solve original MINLP with Ipopt, fixing integer variables from MIP solution."""
    model = init_model.clone()

    for v in init_rep.vars:
        var_name = v["name"]
        if var_name not in mip_solution:
            continue
        var_component = model.component(var_name)
        mip_val = mip_solution[var_name]

        if v["type"] in ["B", "I"]:
            rounded_val = round(mip_val)
            fixed_val = _clamp(rounded_val, v["lb"], v["ub"])
            # Ipopt rejects integer domains even for fixed vars; treat as continuous
            var_component.domain = pyo.Reals
            var_component.fix(fixed_val)
        else:
            pass
            #var_component.set_value(mip_val) # this can lead to non-global optima when the initial solution is bad

    try:
        opt = pyo.SolverFactory("ipopt", executable=settings.ipopt_executable)
        results = opt.solve(
            model,
            tee=settings.solver_output,
            options={"max_cpu_time": settings.timelimit},
        )
        return minlp_eval._extract_solver_results(results, model, init_rep)
    except Exception as e:
        return {
            "status": "ERROR",
            "objective": None,
            "time": 0.0,
            "time_firstprimal": None,
            "info": str(e),
        }

def _clamp(val, lb, ub):
    if lb is not None and lb > float("-inf"):
        val = max(lb, val)
    if ub is not None and ub < float("inf"):
        val = min(ub, val)
    return val