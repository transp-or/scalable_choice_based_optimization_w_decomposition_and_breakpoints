import subprocess

# Define your parameters
N = 10           # sample size
R = 50          # number of draws
J_PSP = 1        # number of PSP alternatives
J_PUP = 1        # number of PUP alternatives
timelimit = 3600 # time limit in seconds
method = 8      # solver method 2: MILP, 3: QCQP, 4: QCLP, 5: Bnb, 6: BnBD, 7: BEA, 8: BHA, 82: ILS, 83: CMA-ES, 9: BEA with capacities, 10: BHA with caps, 11: ILS with caps 12: CMA-ES with caps
bp_bounds = 1    # bound improvements using breakpoints
valid_ineq = 1   # valid inequalities
minutes = 3      # time increase per duplicate alt
random_seed = 42 # random seed for sample and scenarios

print("")
print("Running BHA")
print("")

# Create the list of arguments
args = [
    "python",
    "solve_cpp.py",
    str(N),
    str(R),
    str(J_PSP),
    str(J_PUP),
    str(timelimit),
    str(8),
    "0",
    str(bp_bounds),
    str(valid_ineq),
    "0",
    str(minutes),
    "0",
    "0",
    "1",
    "1",
    str(random_seed)
]

# Run the script
subprocess.run(args)
