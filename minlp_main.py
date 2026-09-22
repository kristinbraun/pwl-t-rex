import MIPRef_osilToOnedim as oto
import MIPRef_onedimToMIP as otm
import evaluation_solving as solving
import settings
import json
import os
import pyomo.environ as pyo
import pyomo.opt as popt


def run():
    print("Running MINLP solve mode:", settings.testfile)

    # Validate method is in valid range for solve mode
    if settings.method < 1 or settings.method > 8:
        print(f"ERROR: Method {settings.method} is not valid for solve mode.")
        print("Valid methods are 1-8 (see --help for details).")
        print("Use mode=compare for methods -5, -1, or 0.")
        return

    # Check if method 8 with relax 1 is disabled
    if settings.method == 8 and settings.relax == 1:
        print("ERROR: Method 8 (IntegerZigZag) is not compatible with relax=1.")
        print("Use relax=0 or relax=2 instead.")
        return

    # Build 1D representation (same logic as compare_main lines 12-21)
    if settings.find_1d:
        oned_rep = oto.obtain_1d_representation_chained_functions(settings.testfile)
        easy_rep = oned_rep
    else:
        oned_rep = oto.obtain_1d_representation(settings.testfile)
        easy_rep = otm.ease_model(oned_rep, use_univariate_functions=False)

    # Scale model if requested
    easy_rep = oto.scale_model(easy_rep)

    m_easy_rep = oto.create_pyomomodel_from_OSILdata(easy_rep)

    # Build PWL-MIP for the selected method
    print(f"Building PWL-MIP with method {settings.method}")
    mip_rep, breakpoints, breakpoint_info = otm.obtainMIPfrom1d(
        easy_rep, method=settings.method
    )

    # Create Pyomo model
    mip_model = oto.create_pyomomodel_from_OSILdata(mip_rep)

    # Solve with Gurobi
    print("Solving PWL-MIP with Gurobi... ", end="", flush=True)
    results_mip = solving.solve_and_store_results(mip_model, mip_rep, gur=True)
    print("Done\n")

    # Print results
    print_result(results_mip, "PWL-MIP")

    # Save solution to JSON
    save_solution(mip_model, mip_rep, results_mip)

    # Run post-solve steps if solution is primal feasible
    if results_mip["status"] in ["OPTIMAL", "TIMELIMIT"] and results_mip.get(
        "objective"
    ) is not None:
        # Build original MINLP model for warm-starting / fixing
        init_rep = oto.obtain_init_representation(settings.testfile)
        init_model = oto.create_pyomomodel_from_OSILdata(init_rep)

        # Extract MIP solution values for original variables
        mip_solution = extract_original_variables(mip_model, init_rep)

        postsolve_results = {}

        violations = obtain_max_infeasibility(init_model, init_rep, mip_solution)
        for constraint, violation in violations.items():
            print(f"Violation: {constraint} = {violation:.6f}")

        # 4a: SCIP with warmstart
        if settings.minlp_start:
            print("\n" + "=" * 70)
            print("Running SCIP with MIP warmstart...")
            print("=" * 70)
            result_scip = solve_with_scip_warmstart(init_model, init_rep, mip_solution)
            print_result(result_scip, "SCIP warmstart")
            postsolve_results["minlp_start"] = result_scip

        # 4b: Ipopt with fixed integers
        if settings.nlp_fixed:
            # Verify all integer variables have values from MIP solution
            missing_integers = [
                v["name"]
                for v in init_rep.vars
                if v["type"] in ["B", "I"] and v["name"] not in mip_solution
            ]
            if missing_integers:
                print(
                    "\nWarning: Cannot run Ipopt - missing integer variables "
                    f"in MIP solution: {missing_integers}"
                )
            else:
                print("\n" + "=" * 70)
                print("Running Ipopt with fixed integers from MIP...")
                print("=" * 70)
                result_ipopt = solve_with_ipopt_fixed(
                    init_model, init_rep, mip_solution
                )
                print_result(result_ipopt, "Ipopt fixed integers")
                postsolve_results["nlp_fixed"] = result_ipopt

        # Append postsolve results to JSON
        if postsolve_results:
            append_postsolve_results(postsolve_results)
    else:
        print("\nNo primal solution available, skipping post-solve steps.")


def print_result(results, model_name):
    """Print solver results in a formatted way."""
    name_width = 25
    name_str = f"{model_name}:".ljust(name_width)

    status = results.get("status")
    time = results.get("time")
    objective = results.get("objective")

    if status in ["ERROR"]:
        info = results.get("info")
        extra = f" ({info})" if info else ""
        print(f"{name_str} ERROR - no solution found.{extra}")
    elif status in ["INFEASIBLE", "UNBOUNDED"]:
        time_str = f"{time:<8.2f}" if time is not None else "n/a"
        print(f"{name_str} Status: {status:<12}  Runtime [s]: {time_str}")
    elif objective is not None:
        first_primal = results.get("time_firstprimal")
        first_primal_str = (
            f"{first_primal:<8.2f}" if first_primal is not None else "n/a"
        )
        time_str = f"{time:<8.2f}" if time is not None else "n/a"
        print(
            f"{name_str} Objective: {objective:<15.6f}  "
            f"Runtime [s]: {time_str}  "
            f"First primal: {first_primal_str}"
        )
    else:
        time_str = f"{time:<8.2f}" if time is not None else "n/a"
        print(f"{name_str} Status: {status}  Runtime [s]: {time_str}")


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
        for v in mip_rep.vars:
            try:
                val = pyo.value(mip_model.component(v["name"]))
                if val is not None:
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


def _clamp(val, lb, ub):
    if lb is not None and lb > float("-inf"):
        val = max(lb, val)
    if ub is not None and ub < float("inf"):
        val = min(ub, val)
    return val


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

    return solving_results


def solve_with_scip_warmstart(init_model, init_rep, mip_solution):
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
            if abs(mip_val - rounded_val) > 1e-5:
                print(
                    f"Warning: Variable {var_name} rounded from {mip_val} to {rounded_val}"
                )
            mip_val = _clamp(rounded_val, v["lb"], v["ub"])

        var_component.set_value(mip_val)

    try:
        #opt = pyo.SolverFactory("scip", executable=settings.scip_executable)
        opt = pyo.SolverFactory("scip", executable=settings.scip_executable)
        results = opt.solve(
            model,
            tee=settings.solver_output,
            options={"limits/time": settings.timelimit},
        )
        return _extract_solver_results(results, model, init_rep)
    except Exception as e:
        return {
            "status": "ERROR",
            "objective": None,
            "time": 0.0,
            "time_firstprimal": None,
            "info": str(e),
        }


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

    model_used.pprint()

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


def solve_with_ipopt_fixed(init_model, init_rep, mip_solution):
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
            print(f"Fixed integer variable '{var_name}' to value {fixed_val}")
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
        return _extract_solver_results(results, model, init_rep)
    except Exception as e:
        return {
            "status": "ERROR",
            "objective": None,
            "time": 0.0,
            "time_firstprimal": None,
            "info": str(e),
        }


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
