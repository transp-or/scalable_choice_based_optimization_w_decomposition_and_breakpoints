# Choice-Based Pricing Demo

This repository contains code for running choice-based pricing algorithms using Python and Julia.

The code is designed to help replicate the results presented in the following two papers:

- Haering, T., Legault, R., Torres, F., Ljubić, I. and Bierlaire, M., 2024. *Exact algorithms for continuous pricing with advanced discrete choice demand models*. OR Spectrum, pp.1–47.
- Haering, T., Legault, R., Torres, F. and Bierlaire, M., 2025. *Heuristics and Exact Algorithms for Choice-Based Capacitated and Uncapacitated Continuous Pricing*. Technical report TRANSP-OR 250508.

## 🔧 Setup

# create python environment 
# warning: this took several hours of package resolving on my machine. Might be easier to just run it on an empty environment and hope it works by just adding packages iteratively
conda env create -f pricing.yml -n pricing

# activate python environment
conda activate pricing

# instantiate Julia environment
julia --project=. -e 'using Pkg; Pkg.instantiate()'


# inside the demo file you can then adjust the instance size, algorithms etc 
# by default it is set to run the BHA algorithm, solving an instance with 10 individuals, 50 draws, and 2 controlled products

# run the demo file
python demo_pricing.py


