import os
import json
import pyomo.environ as pyo
import pyomo.opt as popt
import settings
import numpy as np

def print_result(results, model_name):
    """Print solver results in a formatted way."""
    name_width = 25
    name_str = f"{model_name}:".ljust(name_width)

    if results is None:
        print(f"{name_str} No results found")
        return

    status = results.get("status")
    time = results.get("time")
    objective = results.get("objective")

    if status in ["ERROR"]:
        info = results.get("info")
        extra = f" ({info})" if info else ""
        print(f"{name_str} ERROR - no solution found.{extra}")
    elif status in ["INFEASIBLE", "UNBOUNDED"]:
        time_str = f"{time:<8.2f}" if time is not None else "n/a"
        print(f"{name_str} Runtime [s]: {time_str} Status: {status}")
    elif objective is not None:
        first_primal = results.get("time_firstprimal")
        first_primal_str = (
            f"{first_primal:<8.2f}" if first_primal is not None else "n/a"
        )
        time_str = f"{time:<8.2f}" if time is not None else "n/a"
        gap = results.get("gap")
        gap_str = f"{gap:<8.2f}" if gap is not None else "n/a"
        status_str = f"{status}".ljust(15)
        print(
            f"{name_str} Objective: {objective:<15.6f}  "
            f"Runtime [s]: {time_str}  "
            f"First primal: {first_primal_str}  "
            f"Gap: {gap_str}  "
            f"Status: {status_str}"
        )
    else:
        time_str = f"{time:<8.2f}" if time is not None else "n/a"
        print(f"{name_str} Runtime [s]: {time_str} Status: {status}")


def _json_default(obj):
    """Convert numpy scalars and similar types for json.dump."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _solution_filename():
    instance_name = os.path.basename(settings.testfile).replace(".osil", "")
    return os.path.join("solutions", f"{instance_name}_method{settings.method}.json")


def save_solution(mip_model, mip_rep, results_mip):
    """Save MIP solution to JSON file."""
    os.makedirs("solutions", exist_ok=True)
    filename = _solution_filename()

    solution_data = {
        "instance": settings.testfile,
        "method": settings.method,
        "status": results_mip["status"],
    }

    if results_mip.get("objective") is not None:
        solution_data["objective"] = results_mip["objective"]
    if results_mip.get("time") is not None:
        solution_data["time"] = results_mip["time"]
    if results_mip.get("gap") is not None:
        solution_data["gap"] = results_mip["gap"]

    # Extract all variable values if a primal solution is available
    if results_mip["status"] in ["OPTIMAL", "TIMELIMIT"] and results_mip.get(
        "objective"
    ) is not None:
        variables = {}
        if results_mip["time_firstprimal"] is not None:
            for v in mip_rep.vars:
                try:
                    val = pyo.value(mip_model.component(v["name"]))
                    if val is not None:
                        if abs(val) < settings.minlp_zero_tol:
                            val = 0.0
                        variables[v["name"]] = float(val)
                except Exception:
                    pass
        
        solution_data["variables"] = variables

    with open(filename, "w") as f:
        json.dump(solution_data, f, indent=2, default=_json_default)

    print(f"\nSolution saved to: {filename}")


def extract_original_variables(mip_model, init_rep):
    """Extract values of original MINLP variables from MIP solution."""
    solution = {}
    for v in init_rep.vars:
        var_name = v["name"]
        try:
            val = pyo.value(mip_model.component(var_name))
            if val is not None:
                solution[var_name] = float(val)
        except Exception:
            print(f"Warning: Original variable {var_name} not found in MIP solution")
    return solution



def _empty_solving_results():
    return {
        "status": None,
        "objective": None,
        "time": None,
        "time_firstprimal": None,
    }


def _extract_solver_results(results, model, init_rep):
    """Map a Pyomo solver result to a small status dict."""
    solving_results = _empty_solving_results()
    term = results.solver.termination_condition
    status = results.solver.status

    if (
        term == popt.TerminationCondition.optimal
        and status == popt.SolverStatus.ok
    ):
        solving_results["status"] = "OPTIMAL"
    elif term == popt.TerminationCondition.infeasible:
        solving_results["status"] = "INFEASIBLE"
    elif term == popt.TerminationCondition.unbounded:
        solving_results["status"] = "UNBOUNDED"
    elif term == popt.TerminationCondition.maxTimeLimit:
        solving_results["status"] = "TIMELIMIT"
    elif term == popt.TerminationCondition.maxIterations:
        solving_results["status"] = "ITERATION_LIMIT"
    else:
        solving_results["status"] = "ERROR"

    # Only read the objective if the solver reports a usable solution
    if solving_results["status"] in [
        "OPTIMAL",
        "TIMELIMIT",
        "ITERATION_LIMIT",
    ] and status in [popt.SolverStatus.ok, popt.SolverStatus.warning]:
        try:
            solving_results["objective"] = float(
                pyo.value(model.component("obj_" + init_rep.objs[0]["name"]))
            )
        except Exception:
            # Keep objective None if no loadable solution exists
            pass

    if hasattr(results.solver, "time") and results.solver.time is not None:
        solving_results["time"] = float(results.solver.time)
    else:
        solving_results["time"] = 0.0

    # Add the solver gap (relative optimality gap) to the results if available
    gap = np.abs(
                results.problem.upper_bound - results.problem.lower_bound
            ) / (np.abs(solving_results["objective"]) + 1e-10)
    solving_results["gap"] = gap

    return solving_results


def obtain_max_infeasibility(model, rep, mip_solution):
    """Obtain the maximum infeasibility of the initial solution."""
    model_used = model.clone()
    violations = {}

    # Set all variables in model_used to the values from mip_solution
    for v in rep.vars:
        var_name = v["name"]
        if var_name not in mip_solution:
            continue
        try:
            var_component = model_used.component(var_name)
        except Exception:
            continue
        mip_val = mip_solution[var_name]
        var_component.set_value(mip_val)


    # Iteriere über alle aktiven Nebenbedingungen im Modell
    for c in model_used.component_data_objects(pyo.Constraint, active=True):
        # Berechne den Wert der Gleichung/Ungleichung mit den aktuellen Startwerten
        try:
            body_val = pyo.value(c.body, exception=False)
        except ValueError:
            continue # Überspringen, falls Term nicht berechenbar (z.B. Division durch 0)

        if body_val is None:
            continue

        # Prüfe Verletzung der unteren Schranke (Lower Bound)
        if c.lower is not None:
            lb_viol = c.lower - body_val
            if lb_viol > 0:
                violations[c.name] = lb_viol
            elif c.name not in violations:
                violations[c.name] = 0

        # Prüfe Verletzung der oberen Schranke (Upper Bound)
        if c.upper is not None:
            ub_viol = body_val - c.upper
            if ub_viol > 0:
                violations[c.name] = ub_viol
            elif c.name not in violations:
                violations[c.name] = 0

    return violations



def append_postsolve_results(postsolve_results):
    """Append postsolve results to the existing JSON file."""
    filename = _solution_filename()

    with open(filename, "r") as f:
        data = json.load(f)

    for key, value in postsolve_results.items():
        data[key] = value

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)

    print(f"\nPostsolve results appended to: {filename}")
