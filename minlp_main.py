import MIPRef_osilToOnedim as oto
import MIPRef_onedimToMIP as otm
import evaluation_solving as solving
import settings
import json
import os
import pyomo.environ as pyo
import pyomo.opt as popt

import minlp_evaluate_solutions as minlp_eval
import minlp_solve as minlp_solv


def solve_pwl_mip():
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
    return mip_model, mip_rep, results_mip   


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

    mip_model, mip_rep, results_mip = solve_pwl_mip()


    # Save solution to JSON
    minlp_eval.save_solution(mip_model, mip_rep, results_mip)

    # Run post-solve steps if solution is primal feasible
    if results_mip["status"] in ["OPTIMAL", "TIMELIMIT"] and results_mip.get(
        "objective"
    ) is not None and results_mip.get("time_firstprimal") is not None:
        # Build original MINLP model for warm-starting / fixing
        init_rep = oto.obtain_init_representation(settings.testfile)
        init_model = oto.create_pyomomodel_from_OSILdata(init_rep)

        # Extract MIP solution values for original variables
        mip_solution = minlp_eval.extract_original_variables(mip_model, init_rep)

        postsolve_results = {}

        violations = minlp_eval.obtain_max_infeasibility(init_model, init_rep, mip_solution)
        print("Max infeasibility: " + str(max(violations.values())))

        result_reference = None
        if settings.minlp_reference:
            print("\n" + "=" * 70)
            print("Running SCIP for reference MINLP...")
            print("=" * 70)
            result_reference = minlp_solv.solve_minlp_for_reference(init_model, init_rep)
            postsolve_results["minlp_reference"] = result_reference

        result_warmstart = None
        result_fixed = None
        # 4a: MINLP warmstart
        if settings.minlp_start:
            print("\n" + "=" * 70)
            print("Running SCIP with MIP warmstart...")
            print("=" * 70)
            result_warmstart = minlp_solv.solve_minlp_warmstart(init_model, init_rep, mip_solution)
            postsolve_results["minlp_start"] = result_warmstart

        # 4b: NLP with fixed integers
        if settings.nlp_fixed:
            # Verify all integer variables have values from MIP solution
            missing_integers = [
                v["name"]
                for v in init_rep.vars
                if v["type"] in ["B", "I"] and v["name"] not in mip_solution
            ]
            if missing_integers:
                print(
                    "\nWarning: Cannot run NLP - missing integer variables "
                    f"in MIP solution: {missing_integers}"
                )
            else:
                print("\n" + "=" * 70)
                print("Running NLP with fixed integers from MIP...")
                print("=" * 70)
                result_fixed = minlp_solv.solve_nlp_fixed(
                    init_model, init_rep, mip_solution
                )
                postsolve_results["nlp_fixed"] = result_fixed

        # Append postsolve results to JSON
        if postsolve_results:
            minlp_eval.append_postsolve_results(postsolve_results)

        # Print aggregated results
        print("\n" + "=" * 70)
        print("Aggregated results:")
        if settings.minlp_reference:
            minlp_eval.print_result(result_reference, "MINLP reference")
            print()
        minlp_eval.print_result(results_mip, "PWL-MIP")
        if settings.minlp_start:
            minlp_eval.print_result(result_warmstart, "MINLP warmstart")
        if settings.nlp_fixed:
            minlp_eval.print_result(result_fixed, "NLP fixed integers")
        print("=" * 70)
    else:
        print("\nNo primal solution available, skipping post-solve steps.")

