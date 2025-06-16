# Choice-Based Pricing Demo

This repository contains code for running the BHAMLSE algorithm presented in the following paper:

- Haering, T. and Bierlaire, M., 2025. *BHAMSLE: A Breakpoint Heuristic Algorithm for Maximum Simulated Likelihood Estimation of Advanced Discrete Choice Models*. Technical report TRANSP-OR 250404.

## 🔧 Setup

# create python environment (this may take a long time. Maybe faster to install packages on the go)
conda env create -f bhamsle.yml -n bhamsle

# activate python environment
conda activate bhamsle

# instantiate Julia environment
julia --project=. -e 'using Pkg; Pkg.instantiate()'


# inside the demo file you can then choose the model to estimate, the sample size and number of simulation draws
# by default it is set to use the first model presented in the paper, with a sample size of 50 and 50 simulation draws

# Important note: for the models involving mixed parameters, CMA-ES has very large runtimes (multiple hours)
# Thus the part relevant part in the code was commented out (lines 11208 to 11294 in run_experimentLoc.jl)
# To run *any of the tests* with the full comparison including CMA-ES, these lines should be uncommented

# run the demo file
python demo_bhamsle.py


