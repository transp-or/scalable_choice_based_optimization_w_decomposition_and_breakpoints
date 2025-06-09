import numpy as np
import pandas as pd
import random
import sys
import ast
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta, Variable, bioDraws, log, MonteCarlo, exp


def simulate_likelihood_mixed_telephone(N, pandaSeed, beta, latent, R=1000):
    df = pd.read_csv("telephone.dat", sep='\t')
    if not N == 0:
        df_full = df.sample(N, random_state=pandaSeed)
    else:
        df_full = df

    database = db.Database("telephone", df_full)

    # Define the variables
    choice = Variable('choice')
    cost1 = Variable('cost1')
    cost2 = Variable('cost2')
    cost3 = Variable('cost3')
    cost4 = Variable('cost4')
    cost5 = Variable('cost5')
    avail1 = Variable('avail1')
    avail2 = Variable('avail2')
    avail3 = Variable('avail3')
    avail4 = Variable('avail4')
    avail5 = Variable('avail5')

    log_cost1 = database.DefineVariable('log_cost1', log(cost1))
    log_cost2 = database.DefineVariable('log_cost2', log(cost2))
    log_cost3 = database.DefineVariable('log_cost3', log(cost3))
    log_cost4 = database.DefineVariable('log_cost4', log(cost4))
    log_cost5 = database.DefineVariable('log_cost5', log(cost5))

    ASC_BM = Beta('ASC_BM', 0, None, None, 0)
    ASC_EF = Beta('ASC_EF', 0, None, None, 0)
    ASC_LF = Beta('ASC_LF', 0, None, None, 0)
    ASC_MF = Beta('ASC_MF', 0, None, None, 0)
    B_COST = Beta('B_COST', 0, None, None, 0)

    seed_nr = 192
    random.seed(seed_nr)
    np.random.seed(seed_nr)

    # 1 ASC_BM = Beta('ASC_BM', starting_point[0], None, None, 0)
    # 2 ASC_EF = Beta('ASC_EF', starting_point[1], None, None, 0)
    # 3 ASC_LF = Beta('ASC_LF', starting_point[2], None, None, 0)
    # 4 ASC_MF = Beta('ASC_MF', starting_point[3], None, None, 0)
    # 5 B_COST = Beta('B_COST', starting_point[4], None, None, 0)

    if latent == 10:
        mix_inds = [[5, 6]]  # mix cost
        Z1_Beta_cost_S = Beta('Z1_Beta_cost_S', 1, None, None, 0)
        Beta_cost_RND = B_COST + Z1_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

        # Utilities
        V_BM = ASC_BM + Beta_cost_RND * log_cost1
        V_SM = Beta_cost_RND * log_cost2
        V_LF = ASC_LF + Beta_cost_RND * log_cost3
        V_EF = ASC_EF + Beta_cost_RND * log_cost4
        V_MF = ASC_MF + Beta_cost_RND * log_cost5
    if latent == 11:
        mix_inds = [[5, 6], [1, 7], [2, 8], [3, 9], [4, 10]]  # mix cost and ASCs
        Z1_Beta_cost_S = Beta('Z1_Beta_cost_S', 1, None, None, 0)
        Z2_ASC_BM_S = Beta('Z2_ASC_BM_S', 1, None, None, 0)
        Z3_ASC_EF_S = Beta('Z3_ASC_EF_S', 1, None, None, 0)
        Z4_ASC_LF_S = Beta('Z4_ASC_LF_S', 1, None, None, 0)
        Z5_ASC_MF_S = Beta('Z5_ASC_MF_S', 1, None, None, 0)

        Beta_cost_RND = B_COST + Z1_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')
        ASC_BM_RND = ASC_BM + Z2_ASC_BM_S * bioDraws('ASC_BM_RND', 'NORMAL')
        ASC_EF_RND = ASC_EF + Z3_ASC_EF_S * bioDraws('ASC_EF_RND', 'NORMAL')
        ASC_LF_RND = ASC_LF + Z4_ASC_LF_S * bioDraws('ASC_LF_RND', 'NORMAL')
        ASC_MF_RND = ASC_MF + Z5_ASC_MF_S * bioDraws('ASC_MF_RND', 'NORMAL')

        # Utilities
        V_BM = ASC_BM_RND + Beta_cost_RND * log_cost1
        V_SM = Beta_cost_RND * log_cost2
        V_LF = ASC_LF_RND + Beta_cost_RND * log_cost3
        V_EF = ASC_EF_RND + Beta_cost_RND * log_cost4
        V_MF = ASC_MF_RND + Beta_cost_RND * log_cost5
    if latent == 12:
        mix_inds = [[1, 6], [2, 7], [3, 8], [4, 9]]  # mix ASCs
        Z2_ASC_BM_S = Beta('Z2_ASC_BM_S', 1, None, None, 0)
        Z3_ASC_EF_S = Beta('Z3_ASC_EF_S', 1, None, None, 0)
        Z4_ASC_LF_S = Beta('Z4_ASC_LF_S', 1, None, None, 0)
        Z5_ASC_MF_S = Beta('Z5_ASC_MF_S', 1, None, None, 0)
        ASC_BM_RND = ASC_BM + Z2_ASC_BM_S * bioDraws('ASC_BM_RND', 'NORMAL')
        ASC_EF_RND = ASC_EF + Z3_ASC_EF_S * bioDraws('ASC_EF_RND', 'NORMAL')
        ASC_LF_RND = ASC_LF + Z4_ASC_LF_S * bioDraws('ASC_LF_RND', 'NORMAL')
        ASC_MF_RND = ASC_MF + Z5_ASC_MF_S * bioDraws('ASC_MF_RND', 'NORMAL')

        # Utilities
        V_BM = ASC_BM_RND + B_COST * log_cost1
        V_SM = B_COST * log_cost2
        V_LF = ASC_LF_RND + B_COST * log_cost3
        V_EF = ASC_EF_RND + B_COST * log_cost4
        V_MF = ASC_MF_RND + B_COST * log_cost5

    # Associate utility functions with the numbering of alternatives
    V = {1: V_BM, 2: V_SM, 3: V_LF, 4: V_EF, 5: V_MF}
    av = {1: avail1, 2: avail2, 3: avail3, 4: avail4, 5: avail5}
    prob = models.logit(V, av, choice)

    # numberOfDraws = 100000
    integral = MonteCarlo(prob)
    simulate = {
        'Integral': integral,
    }
    # Create the Biogeme object
    biosim = bio.BIOGEME(database, simulate, numberOfDraws=R)
    biosim.modelName = "telephone_mixed_simul"

    betas = {
        'ASC_BM': beta[0],
        'ASC_EF': beta[1],
        'ASC_LF': beta[2],
        'ASC_MF': beta[3],
        'B_COST': beta[4],
    }

    if latent == 10:
        betas['Z1_Beta_cost_S'] = beta[5]
    elif latent == 11:
        betas['Z1_Beta_cost_S'] = beta[5]
        betas['Z2_ASC_BM_S'] = beta[6]
        betas['Z3_ASC_EF_S'] = beta[7]
        betas['Z4_ASC_LF_S'] = beta[8]
        betas['Z5_ASC_MF_S'] = beta[9]
    elif latent == 12:
        betas['Z2_ASC_BM_S'] = beta[5]
        betas['Z3_ASC_EF_S'] = beta[6]
        betas['Z4_ASC_LF_S'] = beta[7]
        betas['Z5_ASC_MF_S'] = beta[8]

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
    logLikelihood = simulate_likelihood_mixed_telephone(N, pandaSeed, beta, latent, R)
    print(logLikelihood)
