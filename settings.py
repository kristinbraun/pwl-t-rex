import argparse

# Default input parameters. parse_cli() overwrites these from the command line
# so that every module can read the effective values from this file.

# --- Instance ---
# Input instance (base name or path); resolved to testfile after parse_cli()
filename = None
# Directory that contains .osil instances
instances_dir = "instances/"
# Full path to the instance that is currently being solved
testfile = None

# --- Formulation ---
# Which MIP method should be used?
# -5: All, -1: Initial MINLP, 0: 1D-MINLP, 1: DisaggConvex, 2: LogDisaggConvex,
# 3: AggConvex, 4: LogAggConvex, 5: Incremental, 6: MultipleChoice,
# 7: BinaryZigZag, 8: IntegerZigZag
method = -5
# Should the model be scaled? 0: No, 1: Yes
scaling = 0



# --- Breakpoints / approximation ---
# How to create the breakpoints? 0: Fixed Epsilon, 1: Fixed Number of Breakpoints
breakpoint_creation = 0
# How much error is allowed? (used when breakpoint_creation == 0)
epsilon = 1
# How many breakpoints should be created per nonlinearity? (used when breakpoint_creation == 1)
breakpoint_number = 10
# Which relaxation method should be used?
# 0: Approximation, 1: Exact error for each segment, 2: Fixed error
relax = 2
# Should the breakpoints be perturbed? 0: No, 1: Yes
perturb_breakpoints = 0

# --- Solver ---
# Creating model without solving it
create = False
# Timelimit for MILPs in seconds
timelimit = 60
# Print solver output
solver_output = False
# SCIP executable used when Gurobi is not selected
scip_executable = "../../scip/scipoptsuite-8.0.1/scip/bin/scip"

# --- Derived method flags (set by resolve_derived_flags) ---
find_1d = False
oned = False
init = False
disagg_convex = False
log_disagg_convex = False
agg_convex = False
log_agg_convex = False
delta = False
multiple_choice = False
binary = False
integer = False


def resolve_derived_flags():
    global find_1d, oned, init, disagg_convex, log_disagg_convex
    global agg_convex, log_agg_convex, delta, multiple_choice, binary, integer

    find_1d = breakpoint_creation == 1

    oned = method == 0
    init = method == -1
    disagg_convex = method == 1
    log_disagg_convex = method == 2
    agg_convex = method == 3
    log_agg_convex = method == 4
    delta = method == 5
    multiple_choice = method == 6
    binary = method == 7
    integer = method == 8

    if method == -5:
        oned = False
        init = False
        disagg_convex = True
        log_disagg_convex = True
        agg_convex = True
        log_agg_convex = True
        delta = True
        multiple_choice = True
        binary = True
        integer = True

    integer = integer and (relax != 1)


resolve_derived_flags()


def parse_cli():
    global filename, testfile
    global method, scaling
    global breakpoint_creation, epsilon, breakpoint_number, relax
    global create, timelimit, solver_output

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "filename",
        action="store",
        type=str,
        help="Filename of the .osil instance (with or without extension)",
    )

    formulation = parser.add_argument_group("formulation")
    formulation.add_argument(
        "--method",
        action="store",
        type=int,
        default=method,
        help="Which MIP method should be used? (-5: All, -1: Initial MINLP, 0: 1D-MINLP, 1: DisaggConvex, 2: LogDisaggConvex, 3: AggConvex, 4: LogAggConvex, 5: Incremental, 6: MultipleChoice, 7: BinaryZigZag, 8: IntegerZigZag)",
    )
    formulation.add_argument(
        "--scaling",
        action="store",
        type=int,
        default=scaling,
        help="Should the model be scaled? 0: No, 1: Yes",
    )

    approximation = parser.add_argument_group("breakpoints / approximation")
    approximation.add_argument(
        "--breakpoint_creation",
        action="store",
        type=int,
        default=breakpoint_creation,
        help="How to create the breakpoints? 0: Fixed Epsilon, 1: Fixed Number of Breakpoints",
    )
    approximation.add_argument(
        "--epsilon",
        action="store",
        type=float,
        default=epsilon,
        help="How much error is allowed? (used when breakpoint_creation is 0)",
    )
    approximation.add_argument(
        "--breakpoint_number",
        action="store",
        type=int,
        default=breakpoint_number,
        help="How many breakpoints should be created per nonlinearity? (used when breakpoint_creation is 1)",
    )
    approximation.add_argument(
        "--relax",
        action="store",
        type=int,
        default=relax,
        help="Which relaxation method should be used? (0: Approximation, 1: Exact error for each segment, 2: Fixed error)",
    )
    approximation.add_argument(
        "--perturb_breakpoints",
        action="store",
        type=int,
        default=0,
        help="Should the breakpoints be perturbed? 0: No, 1: Yes",
    )

    solver = parser.add_argument_group("solver")
    solver.add_argument(
        "--create",
        action="store",
        type=int,
        default=0,
        help="Creating model without solving it. 0: No, 1: Yes",
    )
    solver.add_argument(
        "--timelimit",
        action="store",
        type=int,
        default=timelimit,
        help="Timelimit for MILPs in seconds",
    )
    solver.add_argument(
        "--solver_output",
        action="store",
        type=int,
        default=0,
        help="Print solver output. 0: No, 1: Yes",
    )

    args = parser.parse_args()

    filename = args.filename
    method = args.method
    scaling = args.scaling
    breakpoint_creation = args.breakpoint_creation
    epsilon = args.epsilon
    breakpoint_number = args.breakpoint_number
    relax = args.relax
    perturb_breakpoints = args.perturb_breakpoints
    create = False if args.create == 0 else True
    timelimit = args.timelimit
    solver_output = False if args.solver_output == 0 else True

    testfile = instances_dir + filename
    if ".osil" not in testfile:
        testfile = testfile + ".osil"

    resolve_derived_flags()
