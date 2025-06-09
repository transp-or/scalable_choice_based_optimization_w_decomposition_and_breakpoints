import subprocess

# Define your parameters
N = 30           # sample size
R = 30            # number of draws
C = 28         # model specification 
D = 2           # dataset 

# From the BHAMSLE paper, Test 1: C=28 D=2, Test 2: C=1027 D=2, Test 3: C=2 D=4, Test 4: C=1032 D=4
# each using sample sizes 500 and 1000, and draw numbers from 1 to 3000

startSeed = 1    # starting panda seed
endSeed = 2     # ending panda seed

# Create the list of arguments
args = [
    "julia",
    "run_experimentLoc.jl",
    str(N),
    str(R),
    str(C),
    str(D),
    str(startSeed),
    str(endSeed)
]

# Run the script
subprocess.run(args, check=True, stderr=subprocess.DEVNULL)
