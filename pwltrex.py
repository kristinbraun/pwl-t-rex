import MIPRef_osilToOnedim as oto
import MIPRef_onedimToMIP as otm
import evaluation_solving as solving
import evaluation_statistics as stats
import numpy as np
import settings

settings.parse_cli()

print("Running", settings.testfile)

if settings.find_1d:
    oned_rep = oto.obtain_1d_representation_chained_functions(settings.testfile)
    easy_rep = oned_rep #otm.ease_model(oned_rep, use_univariate_functions=True)
else:
    oned_rep = oto.obtain_1d_representation(settings.testfile)
    easy_rep = otm.ease_model(oned_rep, use_univariate_functions=False)

easy_rep = oto.scale_model(easy_rep)

m = oto.create_pyomomodel_from_OSILdata(oned_rep)


init_rep = oto.obtain_init_representation(settings.testfile)
minit = oto.create_pyomomodel_from_OSILdata(init_rep)
if settings.init:
    if not settings.create:
        results_init = solving.solve_and_store_results(minit, init_rep, gur=False)
    else:
        results_init = {}
stats_init = stats.obtain_statistics(init_rep)


oned_prod = oto.obtain_1d_and_prod_representation(settings.testfile)
stats_1d_prod = stats.obtain_statistics(oned_prod)

if settings.oned:
    if not settings.create:
        results_1d = solving.solve_and_store_results(m, oned_rep, gur=True)
    else:
        results_1d = {}
stats_1d = stats.obtain_statistics(oned_rep)

if settings.disagg_convex:
    print("Disaggregated Convex Combination Model")
    mip_rep1, breakpoints1, breakpoint_info1 = otm.obtainMIPfrom1d(easy_rep, method=1)
    m1 = oto.create_pyomomodel_from_OSILdata(mip_rep1)

    stats_mip = stats.obtain_statistics(mip_rep1)

    if not settings.create:
        print("Solving... ", end="", flush=True)
        results_mip = solving.solve_and_store_results(m1, mip_rep1, gur=True)
        print("Done\n")
    else:
        results_mip = {}

if settings.log_disagg_convex:
    print("Logarithmic Disaggregated Convex Combination Model")
    mip_rep2, breakpoints2, breakpoint_info2 = otm.obtainMIPfrom1d(easy_rep, method=2)
    m2 = oto.create_pyomomodel_from_OSILdata(mip_rep2)

    stats_mip2 = stats.obtain_statistics(mip_rep2)

    if not settings.create:
        print("Solving... ", end="", flush=True)
        results_mip2 = solving.solve_and_store_results(m2, mip_rep2, gur=True)
        print("Done\n")
    else:
        results_mip2 = {}


if settings.agg_convex:
    print("Aggregated Convex Combination Model")

    mip_rep3, breakpoints3, breakpoint_info3 = otm.obtainMIPfrom1d(easy_rep, method=3)
    m3 = oto.create_pyomomodel_from_OSILdata(mip_rep3)

    stats_mip3 = stats.obtain_statistics(mip_rep3)

    if not settings.create:
        print("Solving... ", end="", flush=True)
        results_mip3 = solving.solve_and_store_results(m3, mip_rep3, gur=True)
        print("Done\n")
    else:
        results_mip3 = {}


if settings.log_agg_convex:
    print("Logarithmic Aggregated Convex Combination Model")
    mip_rep4, breakpoints4, breakpoint_info4 = otm.obtainMIPfrom1d(easy_rep, method=4)
    m4 = oto.create_pyomomodel_from_OSILdata(mip_rep4)

    stats_mip4 = stats.obtain_statistics(mip_rep4)

    if not settings.create:
        print("Solving... ", end="", flush=True)
        results_mip4 = solving.solve_and_store_results(m4, mip_rep4, gur=True)
        print("Done\n")
    else:
        results_mip4 = {}


if settings.delta:
    print("Incremental Method")
    mip_rep5, breakpoints5, breakpoint_info5 = otm.obtainMIPfrom1d(easy_rep, method=5)
    m5 = oto.create_pyomomodel_from_OSILdata(mip_rep5)
    m5.write("mip_rep5.lp")

    stats_mip5 = stats.obtain_statistics(mip_rep5)

    if not settings.create:
        print("Solving... ", end="", flush=True)
        results_mip5 = solving.solve_and_store_results(m5, mip_rep5, gur=True)
        print("Done\n")
    else:
        results_mip5 = {}


if settings.multiple_choice:
    print("Multiple Choice Model")
    mip_rep6, breakpoints6, breakpoint_info6 = otm.obtainMIPfrom1d(easy_rep, method=6)
    m6 = oto.create_pyomomodel_from_OSILdata(mip_rep6)

    stats_mip6 = stats.obtain_statistics(mip_rep6)

    if not settings.create:
        print("Solving... ", end="", flush=True)
        results_mip6 = solving.solve_and_store_results(m6, mip_rep6, gur=True)
        print("Done\n")
    else:
        results_mip6 = {}


if settings.binary:
    print("Binary Zig Zag Model")
    mip_rep7, breakpoints7, breakpoint_info7 = otm.obtainMIPfrom1d(easy_rep, method=7)
    m7 = oto.create_pyomomodel_from_OSILdata(mip_rep7)

    stats_mip7 = stats.obtain_statistics(mip_rep7)
    if not settings.create:
        print("Solving... ", end="", flush=True)
        results_mip7 = solving.solve_and_store_results(m7, mip_rep7, gur=True)
        print("Done\n")
    else:
        results_mip7 = {}


if settings.integer:
    print("Integer Zig Zag Model")
    mip_rep8, breakpoints8, breakpoint_info8 = otm.obtainMIPfrom1d(easy_rep, method=8)
    m8 = oto.create_pyomomodel_from_OSILdata(mip_rep8)

    stats_mip8 = stats.obtain_statistics(mip_rep8)

    if not settings.create:
        print("Solving... ", end="", flush=True)
        results_mip8 = solving.solve_and_store_results(m8, mip_rep8, gur=True)
        print("Done\n")
    else:
        results_mip8 = {}


def print_results(results, model_name, name_width=53):
    # Normalize the model name output for consistent alignment
    name_str = f"{model_name}:".ljust(name_width)
    if results["status"] in ["ERROR"]:
        print(f"{name_str} no solution found.")
    elif results["status"] in ["INFEASIBLE", "UNBOUNDED"]:
        print(
            f"{name_str} Runtime [s]: {results['time']:<8.2f}"
        )
    else:
        first_primal = results["time_firstprimal"]
        first_primal_str = (
            f"{first_primal:<8.2f}" if first_primal is not None else "n/a"
        )
        print(
            f"{name_str} Objective: {results['objective']:<15.6f}    Runtime [s]: {results['time']:<8.2f}, first primal: {first_primal_str}"
        )

if settings.init and not settings.create:
    print_results(results_init, "initial formulation")  
if settings.oned and not settings.create:
    print_results(results_1d, "1D formulation")
if settings.disagg_convex and not settings.create:
    print_results(results_mip, "disaggregated_convex_combination_model")
if settings.log_disagg_convex and not settings.create:
    print_results(results_mip2, "logarithmic_disaggregated_convex_combination_model")
if settings.agg_convex and not settings.create:
    print_results(results_mip3, "aggregated_convex_combination_model")
if settings.log_agg_convex and not settings.create:
    print_results(results_mip4, "logarithmic_aggregated_convex_combination_model")
if settings.delta and not settings.create:
    print_results(results_mip5, "incremental_method")
if settings.multiple_choice and not settings.create:
    print_results(results_mip6, "multiple_choice_model")
if settings.binary and not settings.create:
    print_results(results_mip7, "binary_zig_zag_model")
if settings.integer and not settings.create:
    print_results(results_mip8, "integer_zig_zag_model")

# Collect all times and objectives that were printed
results_dict = {}
if settings.disagg_convex and not settings.create:
    results_dict["disaggregated_convex_combination_model"] = (
        results_mip["time"] if results_mip["status"] in ["OPTIMAL", "TIMELIMIT"] else np.inf,
        results_mip["time_firstprimal"] if results_mip["time_firstprimal"] is not None else np.inf,
    )
if settings.log_disagg_convex and not settings.create:
    results_dict["logarithmic_disaggregated_convex_combination_model"] = (
        results_mip2["time"] if results_mip2["status"] in ["OPTIMAL", "TIMELIMIT"] else np.inf,
        results_mip2["time_firstprimal"] if results_mip2["time_firstprimal"] is not None else np.inf,
    )
if settings.agg_convex and not settings.create:
    results_dict["aggregated_convex_combination_model"] = (
        results_mip3["time"] if results_mip3["status"] in ["OPTIMAL", "TIMELIMIT"] else np.inf,
        results_mip3["time_firstprimal"] if results_mip3["time_firstprimal"] is not None else np.inf,
    )
if settings.log_agg_convex and not settings.create:
    results_dict["logarithmic_aggregated_convex_combination_model"] = (
        results_mip4["time"] if results_mip4["status"] in ["OPTIMAL", "TIMELIMIT"] else np.inf,
        results_mip4["time_firstprimal"] if results_mip4["time_firstprimal"] is not None else np.inf,
    )
if settings.delta and not settings.create:
    results_dict["incremental_method"] = (
        results_mip5["time"] if results_mip5["status"] in ["OPTIMAL", "TIMELIMIT"] else np.inf,
        results_mip5["time_firstprimal"] if results_mip5["time_firstprimal"] is not None else np.inf,
    )
if settings.multiple_choice and not settings.create:
    results_dict["multiple_choice_model"] = (
        results_mip6["time"] if results_mip6["status"] in ["OPTIMAL", "TIMELIMIT"] else np.inf,
        results_mip6["time_firstprimal"] if results_mip6["time_firstprimal"] is not None else np.inf,
    )
if settings.binary and not settings.create:
    results_dict["binary_zig_zag_model"] = (
        results_mip7["time"] if results_mip7["status"] in ["OPTIMAL", "TIMELIMIT"] else np.inf,
        results_mip7["time_firstprimal"] if results_mip7["time_firstprimal"] is not None else np.inf,
    )
if settings.integer and not settings.create:
    results_dict["integer_zig_zag_model"] = (
        results_mip8["time"] if results_mip8["status"] in ["OPTIMAL", "TIMELIMIT"] else np.inf,
        results_mip8["time_firstprimal"] if results_mip8["time_firstprimal"] is not None else np.inf,
    )

print(results_dict)

# Sort by time and print order
print("\nSorted by solution time:")

# Sort the ones with solutions by solution time
sorted_results = sorted(results_dict.items(), key=lambda x: x[1][0])
for i, (model, time) in enumerate(sorted_results, 1):
    model_name = model.replace("_", " ").title()
    print(f"{i:2d}. {model_name:<50} {time[0]:>8.2f}s")

# Sort by first primal time and print order
print("\nSorted by time until first primal solution was found:")
sorted_results = sorted(results_dict.items(), key=lambda x: x[1][1])
for i, (model, time) in enumerate(sorted_results, 1):
    model_name = model.replace("_", " ").title()
    print(f"{i:2d}. {model_name:<50} {time[1]:>8.2f}s")
