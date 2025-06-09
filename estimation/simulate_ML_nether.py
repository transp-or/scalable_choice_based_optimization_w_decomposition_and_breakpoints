import numpy as np
import pandas as pd
import random
import sys
import ast
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta, Variable, bioDraws, log, MonteCarlo, exp


def simulate_likelihood_mixed_nether(N, pandaSeed, beta, latent, R=1000):
    df = pd.read_csv('netherlands.dat', sep='\t')
    df = df[df['rp'] == 1]
    df['rail_time'] = df.rail_ivtt + df.rail_acc_time + df.rail_egr_time
    df['car_time'] = df.car_ivtt + df.car_walk_time

    if not N == 0:
        df_full = df.sample(N, random_state=pandaSeed)
    else:
        df_full = df

    database = db.Database('netherlands', df_full)
    globals().update(database.variables)

    ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)

    if 10 <= latent <= 19:
        seed_nr = 192
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        # 1 ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
        # 3 BETA_COST = Beta('BETA_COST', starting_point[1], None, None, 0)
        # 2 BETA_TIME = Beta('BETA_TIME', starting_point[2], None, None, 0)

        mix_inds = None
        if latent == 10:
            mix_inds = [[2, 4]]  # mix just time
            Z1_Beta_time_S = Beta('Z1_Beta_time_S', 1, None, None, 0)
            Beta_time_RND = BETA_TIME + Z1_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

            Car = Beta_time_RND * car_time + BETA_COST * car_cost
            Rail = ASC_RAIL + Beta_time_RND * rail_time + BETA_COST * rail_cost

        if latent == 11:
            mix_inds = [[2, 4], [3, 5]]  # mix time and costs
            Z1_Beta_time_S = Beta('Z1_Beta_time_S', 1, None, None, 0)
            Z2_Beta_cost_S = Beta('Z2_Beta_cost_S', 1, None, None, 0)
            Beta_time_RND = BETA_TIME + Z1_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')
            Beta_cost_RND = BETA_COST + Z2_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

            Car = Beta_time_RND * car_time + Beta_cost_RND * car_cost
            Rail = ASC_RAIL + Beta_time_RND * rail_time + Beta_cost_RND * rail_cost

        if latent == 12:
            mix_inds = [[2, 4], [3, 5]]  # mix time and costs and ASC rail
            Z1_Beta_time_S = Beta('Z1_Beta_time_S', 1, None, None, 0)
            Z2_Beta_cost_S = Beta('Z2_Beta_cost_S', 1, None, None, 0)
            Z3_ASC_Car_S = Beta('Z3_ASC_rail_S', 1, None, None, 0)
            Beta_time_RND = BETA_TIME + Z1_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')
            Beta_cost_RND = BETA_COST + Z2_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')
            ASC_rail_RND = ASC_RAIL + Z3_ASC_Car_S * bioDraws('ASC_rail_RND', 'NORMAL')

            Car = Beta_time_RND * car_time + Beta_cost_RND * car_cost
            Rail = ASC_rail_RND + Beta_time_RND * rail_time + Beta_cost_RND * rail_cost

        if latent == 13:
            mix_inds = [[3, 4]]  # mix only costs
            Z1_Beta_cost_S = Beta('Z1_Beta_cost_S', 1, None, None, 0)
            Beta_cost_RND = BETA_COST + Z1_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

            Car = BETA_TIME * car_time + Beta_cost_RND * car_cost
            Rail = ASC_RAIL + BETA_TIME * rail_time + Beta_cost_RND * rail_cost

        if latent == 14:
            mix_inds = [[1, 4]]  # mix ASC rail
            Z1_ASC_Car_S = Beta('Z1_ASC_rail_S', 1, None, None, 0)
            ASC_rail_RND = ASC_RAIL + Z1_ASC_Car_S * bioDraws('ASC_rail_RND', 'NORMAL')

            Car = BETA_TIME * car_time + BETA_COST * car_cost
            Rail = ASC_rail_RND + BETA_TIME * rail_time + BETA_COST * rail_cost

    V = {0: Car, 1: Rail}  # deterministic part: use utilities depending on mode indicator
    av = {0: 1, 1: 1}  # both are available to everyone

    prob = models.logit(V, av, choice)

    # numberOfDraws = 100000
    integral = MonteCarlo(prob)
    simulate = {
        'Integral': integral,
    }
    # Create the Biogeme object
    biosim = bio.BIOGEME(database, simulate, numberOfDraws=R)
    biosim.modelName = "nether_mixed_simul"

    betas = {
        'ASC_RAIL': beta[0],
        'BETA_COST': beta[1],
        'BETA_TIME': beta[2],
    }

    if latent == 10:
        betas['Z1_Beta_time_S'] = beta[3]
    elif latent == 11:
        betas['Z1_Beta_time_S'] = beta[3]
        betas['Z2_Beta_cost_S'] = beta[4]
    elif latent == 12:
        betas['Z1_Beta_time_S'] = beta[3]
        betas['Z2_Beta_cost_S'] = beta[4]
        betas['Z3_ASC_rail_S'] = beta[5]
    elif latent == 13:
        betas['Z1_Beta_cost_S'] = beta[3]
    elif latent == 14:
        betas['Z1_ASC_rail_S'] = beta[3]

    simresults = biosim.simulate(betas)
    logLikelihood = np.log(simresults["Integral"]).sum()

    return logLikelihood


if __name__ == "__main__":
    # Parse arguments from the command line
    N = int(sys.argv[1])
    pandaSeed = int(sys.argv[2])
    beta = eval(sys.argv[3])  # Convert the string representation of the list to an actual list
    latent = int(sys.argv[4])
    R = int(sys.argv[5])

    # Call the function and print the result
    logLikelihood = simulate_likelihood_mixed_nether(N, pandaSeed, beta, latent, R)
    print(logLikelihood)
