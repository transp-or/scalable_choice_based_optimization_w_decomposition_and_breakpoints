import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import time
import random
import math
import contextlib
import os
import glob
import copy
import ast
import warnings

from scipy.special import logsumexp

import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.models import loglogit

import biogeme.messaging as msg
from biogeme.expressions import Beta, Variable, bioDraws, log, MonteCarlo, exp


from compute_biog_loglike_from_beta import compute_biog_loglike, compute_biog_loglike_latent, \
    simulate_likelihood_mixed_nether
from estimate_beta_biogeme import biogeme_estimate_beta_nether, biogeme_estimate_beta_london, \
    biogeme_estimate_beta_london_latent, biogeme_estimate_swissmetro, biogeme_estimate_swissmetro_mixed, \
    biogeme_estimate_beta_nether_mixed, biogeme_estimate_swissmetro_nested, biogeme_estimate_optima, \
    biogeme_estimate_telephone


N_range = [int(sys.argv[1])]
R_range = [int(sys.argv[2])]
R = R_range[-1]
latent = int(sys.argv[3])
pandaSeed = int(sys.argv[4])
MSeed = int(sys.argv[5])
TSeed = int(sys.argv[6])
try:
    MSeedLondon = int(sys.argv[7])
    TSeedLondon = int(sys.argv[8])
except IndexError:
    MSeedLondon = 0
    TSeedLondon = 0
try:
    MSeedSM = int(sys.argv[9])
except IndexError:
    MSeedSM = 0

try:
    OptimaSeed = int(sys.argv[10])
except IndexError:
    OptimaSeed = 0

try:
    starting_point = ast.literal_eval(sys.argv[11])
except (IndexError, ValueError, SyntaxError):
    starting_point = None

if MSeedSM > 0 and OptimaSeed > 0:
    MSeedSM = 0

TSeedSM = 0  # simplifying things

do_mixed = False

if 1020 <= latent <= 1029 or 1030 <= latent <= 1039:
    do_mixed = True
    mix_inds = None  # lets format the mix_inds in Julias indexing
    if MSeedSM == 1:  # SM
        if latent == 10:
            mix_inds = [[5, 6]]  # mix just time
        elif latent == 11:
            mix_inds = [[5, 6], [3, 7]]  # mix time and costs
        elif latent == 12:
            mix_inds = [[5, 6], [3, 7], [4, 8]]  # mix time and costs and headway
    elif MSeed == 1: # Nether
        # ASC RAIL, B COST, B TIME
        mix_inds = None
        if latent == 10:
            mix_inds = [[2, 4]]  # mix just time
        elif latent == 11:
            mix_inds = [[2, 4], [3, 5]]  # mix time and costs
        elif latent == 12:
            mix_inds = [[2, 4], [3, 5], [1, 6]]  # mix time and costs and ASC rail
        elif latent == 13:
            mix_inds = [[3, 4]]  # mix only costs
        elif latent == 14:
            mix_inds = [[1, 4]]  # mix ASC rail
    elif OptimaSeed == 1:  # Optima
        # ASC_PT = Beta('ASC_PT', 0.0, None, None, 1)
        #
        # 1 ASC_CAR = Beta('ASC_CAR', starting_point[0], None, None, 0)
        # 2 ASC_SM = Beta('ASC_SM', starting_point[1], None, None, 0)
        # 3 BETA_COST_HWH = Beta('BETA_COST_HWH', starting_point[2], None, None, 0)
        # 4 BETA_COST_OTHER = Beta('BETA_COST_OTHER', starting_point[3], None, None, 0)
        # 5 BETA_DIST = Beta('BETA_DIST', starting_point[4], None, None, 0)
        # 6 BETA_TIME_CAR = Beta('BETA_TIME_CAR', starting_point[5], None, None, 0)
        # 7 BETA_TIME_PT = Beta('BETA_TIME_PT', starting_point[6], None, None, 0)
        # 8 BETA_WAITING_TIME = Beta('BETA_WAITING_TIME', starting_point[7], None, None, 0)
        if latent == 10:
            mix_inds = [[6, 9], [7, 10]]  # mix just time
        elif latent == 11:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12]]  # mix time and costs
        elif latent == 12:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [5, 13]]  # mix time and costs and distance
        elif latent == 13:
            mix_inds = [[6, 9], [7, 10], [5, 11]]  # mix time and distance
        elif latent == 14:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [5, 13], [1, 14], [2, 15]]  # mix time and costs and
            # distance and ASCs
        elif latent == 15:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [1, 13], [2, 14]]  # mix time and costs and ASCs
        elif latent == 16:
            mix_inds = [[6, 9], [7, 10], [1, 11], [2, 12]]  # mix time and ASCs
        elif latent == 17:
            mix_inds = [[1, 9], [2, 10]]  # mix ASCs only
        elif latent == 18:
            mix_inds = [[1, 9]]  # mix car asc

    elif MSeedLondon == 1:  # London
        # ASC RAIL, B COST, B TIME
        if latent == 10:
            mix_inds = [[5, 6]]  # mix just time
        elif latent == 11:
            mix_inds = [[5, 6], [4, 7]] # mix time and costs
        elif latent == 12:
            mix_inds = [[4, 6]]  # mix costs
        elif latent == 18:
            mix_inds = [[2, 6]]  # mix car asc
    elif TSeedLondon == 1: # telephone
        if latent == 10:
            mix_inds = [[5, 6]]  # mix cost
        if latent == 11:
            mix_inds = [[5, 6], [1, 7], [2, 8], [3, 9], [4, 10]]  # mix cost and ASCs
        if latent == 12:
            mix_inds = [[1, 6], [2, 7], [3, 8], [4, 9]]  # mix ASCs

if 10 <= latent <= 19:
    do_mixed = True
    mix_inds = None  # lets format the mix_inds in Julias indexing
    if MSeedSM == 1:  # SM
        if latent == 10:
            mix_inds = [[5, 6]]  # mix just time
        elif latent == 11:
            mix_inds = [[5, 6], [3, 7]]  # mix time and costs
        elif latent == 12:
            mix_inds = [[5, 6], [3, 7], [4, 8]]  # mix time and costs and headway
    elif MSeed == 1: # Nether
        # ASC RAIL, B COST, B TIME
        mix_inds = None
        if latent == 10:
            mix_inds = [[2, 4]]  # mix just time
        elif latent == 11:
            mix_inds = [[2, 4], [3, 5]]  # mix time and costs
        elif latent == 12:
            mix_inds = [[2, 4], [3, 5], [1, 6]]  # mix time and costs and ASC rail
        elif latent == 13:
            mix_inds = [[3, 4]]  # mix only costs
        elif latent == 14:
            mix_inds = [[1, 4]]  # mix ASC rail
    elif OptimaSeed == 1:  # Optima
        # ASC_PT = Beta('ASC_PT', 0.0, None, None, 1)
        #
        # 1 ASC_CAR = Beta('ASC_CAR', starting_point[0], None, None, 0)
        # 2 ASC_SM = Beta('ASC_SM', starting_point[1], None, None, 0)
        # 3 BETA_COST_HWH = Beta('BETA_COST_HWH', starting_point[2], None, None, 0)
        # 4 BETA_COST_OTHER = Beta('BETA_COST_OTHER', starting_point[3], None, None, 0)
        # 5 BETA_DIST = Beta('BETA_DIST', starting_point[4], None, None, 0)
        # 6 BETA_TIME_CAR = Beta('BETA_TIME_CAR', starting_point[5], None, None, 0)
        # 7 BETA_TIME_PT = Beta('BETA_TIME_PT', starting_point[6], None, None, 0)
        # 8 BETA_WAITING_TIME = Beta('BETA_WAITING_TIME', starting_point[7], None, None, 0)
        if latent == 10:
            mix_inds = [[6, 9], [7, 10]]  # mix just time
        elif latent == 11:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12]]  # mix time and costs
        elif latent == 12:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [5, 13]]  # mix time and costs and distance
        elif latent == 13:
            mix_inds = [[6, 9], [7, 10], [5, 11]]  # mix time and distance
        elif latent == 14:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [5, 13], [1, 14], [2, 15]]  # mix time and costs and
            # distance and ASCs
        elif latent == 15:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [1, 13], [2, 14]]  # mix time and costs and ASCs
        elif latent == 16:
            mix_inds = [[6, 9], [7, 10], [1, 11], [2, 12]]  # mix time and ASCs
        elif latent == 17:
            mix_inds = [[1, 9], [2, 10]]  # mix ASCs only
        elif latent == 18:
            mix_inds = [[1, 9]]  # mix car asc

    elif MSeedLondon == 1:  # London
        # ASC RAIL, B COST, B TIME
        if latent == 10:
            mix_inds = [[5, 6]]  # mix just time
        elif latent == 11:
            mix_inds = [[5, 6], [4, 7]] # mix time and costs
        elif latent == 12:
            mix_inds = [[4, 6]]  # mix costs
        elif latent == 18:
            mix_inds = [[2, 6]]  # mix car asc
    elif TSeedLondon == 1: # telephone
        if latent == 10:
            mix_inds = [[5, 6]]  # mix cost
        if latent == 11:
            mix_inds = [[5, 6], [1, 7], [2, 8], [3, 9], [4, 10]]  # mix cost and ASCs
        if latent == 12:
            mix_inds = [[1, 6], [2, 7], [3, 8], [4, 9]]  # mix ASCs



    # in terms of x extension, mix_inds acts the same way as extra_inds
    # but maybe its too late and we already did all that in Julia by hand

run_latent = 1

# exactly not
# if do_mixed:
#     run_latent = 0

# nbOfDraws_normal_MonteCarlo = 10000
nbOfDraws_normal_MonteCarlo = R_range[-1]  # this is also a possibility eh.
nbOfDraws_sLL_Biog = 10000

michels_classes_SM = False
toms_extremists_SM = False

MichelSeed_SM = MSeedSM
TomSeed_SM = TSeedSM

if MichelSeed_SM > 0:
    michels_classes_SM = True
elif TomSeed_SM > 0:
    toms_extremists_SM = True

run_mixed = 0
run_nested = 0

errors = 0

opt_beta = None

logOfZer = -100
buffer = 0
ws = 0
warm_start = False
ht = 0
halton_sampling = False
foc = 3


def simulate_likelihood_mixed_latent_lpmc(N, pandaSeed, beta, latent, R=1000):
    df_full = pd.read_csv("lpmc.dat", sep='\t')
    df_full['rail_time'] = df_full.dur_pt_rail + df_full.dur_pt_bus + df_full.dur_pt_int + df_full.dur_pt_access
    df_full["car_cost"] = df_full.cost_driving_fuel + df_full.cost_driving_ccharge
    df_full["travel_mode_chosen"] = df_full.travel_mode - 1

    if not N == 0:
        df_full = df_full.sample(N, random_state=pandaSeed)
    else:
        df_full = df_full

    df_full = df_full.rename(columns={'start_time': 'start_time_new'})

    J = 4
    # 0: Walk
    # 1: Bike
    # 2: PT
    # 3: Car

    K = 5
    size = N

    x = np.zeros((J, size, K))
    for n in range(size):
        # intercept bike
        x[0, n, 0] = 0
        x[1, n, 0] = 1
        x[2, n, 0] = 0
        x[3, n, 0] = 0
        # intercept car
        x[0, n, 1] = 0
        x[1, n, 1] = 0
        x[2, n, 1] = 0
        x[3, n, 1] = 1
        # intercept pt
        x[0, n, 2] = 0
        x[1, n, 2] = 0
        x[2, n, 2] = 1
        x[3, n, 2] = 0
        # beta cost
        x[0, n, 3] = 0
        x[1, n, 3] = 0
        x[2, n, 3] = df_full['cost_transit'].values[n]
        x[3, n, 3] = df_full['car_cost'].values[n]
        # beta time
        x[0, n, 4] = df_full['dur_walking'].values[n]
        x[1, n, 4] = df_full['dur_cycling'].values[n]
        x[2, n, 4] = df_full['rail_time'].values[n]
        x[3, n, 4] = df_full['dur_driving'].values[n]

    #                    Value  Rob. Std err  Rob. t-test  Rob. p-value
    # ASC_Bike       -3.433393      0.244194   -14.060101  0.000000e+00
    # ASC_Car        -1.036667      0.208348    -4.975655  6.502737e-07
    # ASC_PB         -0.413903      0.126875    -3.262290  1.105160e-03
    # Beta_cost      -0.217605      0.030447    -7.146905  8.875123e-13
    # Beta_time      -5.053744      0.659223    -7.666217  1.776357e-14
    # Z1_Beta_time_S -0.783698      1.047497    -0.748163  4.543621e-01

    # or
    #                    Value  Rob. Std err  Rob. t-test  Rob. p-value
    # ASC_Bike       -3.403574      0.217726   -15.632339  0.000000e+00
    # ASC_Car        -0.999725      0.168735    -5.924826  3.126280e-09
    # ASC_PB         -0.402967      0.117863    -3.418958  6.286141e-04
    # Beta_cost      -0.216995      0.030199    -7.185558  6.692424e-13
    # Beta_time      -4.882945      0.413365   -11.812668  0.000000e+00
    # Z1_Beta_cost_S  0.001563      0.006335     0.246671  8.051631e-01

    # we'll tackle the following specifications:
    # most importantly:
    # C = 1025: class 1 mixed costs, class 2 new beta time
    # C = 1032: class 1 mixed costs, class 2 new beta time, class 3 lazy
    # C = 1033: class 1 mixed costs, class 2 new beta time, class 3 no car

    # C = 1020: class 1 mixed time, class 2 new ASCs and no car
    # C = 1021: class 1 mixed time, class 2 new ASCs and lazy
    # C = 1022: class 1 mixed time, class 2 new ASCs
    # C = 1023: class 1 mixed time, class 2 no car
    # C = 1024: class 1 mixed time, class 2 lazy
    # C = 1030: class 1 mixed time, class 2 new ASCs, class 3 no car
    # C = 1031: class 1 mixed time, class 2 new ASCs, class 3 lazy

    # all checked and verified

    seed_nr = pandaSeed + 1
    random.seed(seed_nr)
    np.random.seed(seed_nr)

    # ben remember to treat 25 differently.
    # and 30s we'll do after

    if 1020 <= latent <= 1025:
        seed = 1
        runNuM = 1
        seedEnd = seed + runNuM - 1

        if latent <= 1024:
            base_beta = [-3.433393, -1.036667, -0.413903, -0.217605, -5.053744, -0.783698]
        else:
            base_beta = [-3.4035735503211497, -0.9997250334091937, -0.40296715846675113, -0.21699454211574326,
                         -4.882944640779013, 0.0015626834662186557]

        #                    Value  Rob. Std err  Rob. t-test  Rob. p-value
        # ASC_Bike       -3.403574      0.217726   -15.632339  0.000000e+00
        # ASC_Car        -0.999725      0.168735    -5.924826  3.126280e-09
        # ASC_PB         -0.402967      0.117863    -3.418958  6.286141e-04
        # Beta_cost      -0.216995      0.030199    -7.185558  6.692424e-13
        # Beta_time      -4.882945      0.413365   -11.812668  0.000000e+00
        # Z1_Beta_cost_S  0.001563      0.006335     0.246671  8.051631e-01

        while seed <= seedEnd:
            seed += 1

            # define the new ASCs (consider new ASC balances)
            new_beta = copy.copy(base_beta)
            # 0 bike
            # 1 car
            # 2 PB
            # we are just copying. Keep them all even if we
            # dont use them, adjust it with indices later.

            if latent == 1020:  # new bike, PB. Class 2 has no car
                # so make class 1 have more car people?
                # so make class 2 like cars less already? Maybe
                new_beta[0] = base_beta[0] * 2  # bike more attractive
                new_beta[2] = base_beta[2] * 2  # PB more attractiveness
            if latent == 1021:  # new PB only. Class 2 hates walk and bike
                new_beta[2] = base_beta[2] * 3  # increase PB attractiveness
            if latent == 1022:  # new bike, car, PB. Make class 2 eco-friendly
                new_beta[0] = base_beta[0] * 3  # incr. bike attractiveness
                new_beta[1] = base_beta[1] * 0.2  # reduce car attractiveness
                new_beta[2] = base_beta[2] * 3  # incr. PB attractiveness
            if latent == 1025:  # new beta time
                new_beta[4] = base_beta[4] * 0.2  # as done before, they care LESS about time

            # worth mentioning: 1023 and 1024 have no new params in latent

            population = [1, 2]
            weights = [0.7, 0.3]
            random.seed(seed)
            class1ers = 0
            class2ers = 0

            for n in range(N):
                # Draw a random number from 1 to 2
                n_class = random.choices(population, weights=weights, k=1)[0]

                if n_class == 1:
                    # k = 0: ASC_Bike
                    # k = 1: ASC_Car
                    # k = 2: ASC_PT
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_C2 or time_STD or cost_STD
                    class1ers += 1
                    # all have the same class 1, except 1025
                    if latent <= 1024:
                        u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[0, n, 4] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[1, n, 4] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[2, n, 4] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_3 = sum(x[3, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[3, n, 4] * base_beta[
                            5] * np.random.normal(0, 1)
                    else:  # 1025 mixes costs
                        u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[0, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[1, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[2, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_3 = sum(x[3, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[3, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2, u_3]

                    # Find the index of the maximum utility
                    best_alternative = max(range(4), key=lambda i: utilities[i])

                    df_full.loc[df_full.index[n], 'travel_mode'] = best_alternative + 1

                elif n_class == 2:
                    class2ers += 1
                    u_0 = sum(x[0, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_3 = sum(x[3, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])

                    # C = 1033: class 1 mixed costs, class 2 new beta time, class 3 no car
                    # C = 1030: class 1 mixed time, class 2 new ASCs, class 3 no car
                    # C = 1031: class 1 mixed time, class 2 new ASCs, class 3 lazy

                    # Put the utilities into a list
                    if latent == 1020 or latent == 1023:  # no car
                        utilities = [u_0, u_1, u_2, -9000000]
                    elif latent == 1021 or latent == 1024:  # lazy
                        utilities = [-9000000, -9000000, u_2, u_3]
                    else:  # 1022, 1025
                        utilities = [u_0, u_1, u_2, u_3]

                    # Find the index of the maximum utility
                    best_alternative = max(range(4), key=lambda i: utilities[i])

                    df_full.loc[df_full.index[n], 'travel_mode'] = best_alternative + 1
            print("# class1 = ", class1ers)
            print("# class2 = ", class2ers)
    if 1030 <= latent <= 1033:
        seed = 1
        runNuM = 1
        seedEnd = seed + runNuM - 1

        if latent <= 1031:
            base_beta = [-3.433393, -1.036667, -0.413903, -0.217605, -5.053744, -0.783698]
        else:
            base_beta = [-3.4035735503211497, -0.9997250334091937, -0.40296715846675113, -0.21699454211574326,
                         -4.882944640779013, 0.0015626834662186557]

        #                    Value  Rob. Std err  Rob. t-test  Rob. p-value
        # ASC_Bike       -3.403574      0.217726   -15.632339  0.000000e+00
        # ASC_Car        -0.999725      0.168735    -5.924826  3.126280e-09
        # ASC_PB         -0.402967      0.117863    -3.418958  6.286141e-04
        # Beta_cost      -0.216995      0.030199    -7.185558  6.692424e-13
        # Beta_time      -4.882945      0.413365   -11.812668  0.000000e+00
        # Z1_Beta_cost_S  0.001563      0.006335     0.246671  8.051631e-01

        while seed <= seedEnd:
            seed += 1

            # define the new ASCs (consider new ASC balances)
            new_beta = copy.copy(base_beta)
            # 0 bike
            # 1 car
            # 2 PB
            # we are just copying. Keep them all even if we
            # dont use them, adjust it with indices later.

            # C = 1033: class 1 mixed costs, class 2 new beta time, class 3 no car
            # C = 1030: class 1 mixed time, class 2 new ASCs, class 3 no car
            # C = 1031: class 1 mixed time, class 2 new ASCs, class 3 lazy
            # C = 1032: class 1 mixed costs, class 2 new beta time, class 3 lazy

            if latent == 1030:  # new bike, car, PB. class 3 no car
                # thus make class 2 carlovers
                new_beta[0] = base_beta[0] * 0.2  # decr bike attractiveness
                new_beta[1] = base_beta[1] * 3  # incr car attractiveness
                new_beta[2] = base_beta[2] * 0.2  # incr. PB attractiveness
            if latent == 1031:  # new bike, car, PB. class 3 lazy
                # thus make class 3 eco
                new_beta[0] = base_beta[0] * 3  # incr. bike attractiveness
                new_beta[1] = base_beta[1] * 0.2  # reduce car attractiveness
                new_beta[2] = base_beta[2] * 3  # incr. PB attractiveness
            if latent == 1032 or latent == 1033:  # new beta time
                new_beta[4] = base_beta[4] * 0.2  # as done before, they care LESS about time

            # worth mentioning: 1023 and 1024 have no new params in latent

            population = [1, 2, 3]
            weights = [0.5, 0.3, 0.2]
            random.seed(seed)
            class1ers = 0
            class2ers = 0
            class3ers = 0

            for n in range(N):
                # Draw a random number from 1 to 2
                n_class = random.choices(population, weights=weights, k=1)[0]

                if n_class == 1:
                    # k = 0: ASC_Bike
                    # k = 1: ASC_Car
                    # k = 2: ASC_PT
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_C2 or time_STD or cost_STD
                    class1ers += 1
                    # all have the same class 1, except 1025
                    if latent <= 1031:
                        u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[0, n, 4] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[1, n, 4] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[2, n, 4] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_3 = sum(x[3, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[3, n, 4] * base_beta[
                            5] * np.random.normal(0, 1)
                    else:  # 1025 mixes costs
                        u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[0, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[1, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[2, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_3 = sum(x[3, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[3, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2, u_3]

                    # Find the index of the maximum utility
                    best_alternative = max(range(4), key=lambda i: utilities[i])

                    df_full.loc[df_full.index[n], 'travel_mode'] = best_alternative + 1

                elif n_class == 2:
                    class2ers += 1
                    u_0 = sum(x[0, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_3 = sum(x[3, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2, u_3]

                    # Find the index of the maximum utility
                    best_alternative = max(range(4), key=lambda i: utilities[i])

                    df_full.loc[df_full.index[n], 'travel_mode'] = best_alternative + 1
                elif n_class == 3:
                    class2ers += 3
                    u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])
                    u_3 = sum(x[3, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])

                    # C = 1033: class 1 mixed costs, class 2 new beta time, class 3 no car
                    # C = 1030: class 1 mixed time, class 2 new ASCs, class 3 no car
                    # C = 1031: class 1 mixed time, class 2 new ASCs, class 3 lazy

                    # Put the utilities into a list
                    if latent == 1030 or latent == 1033:  # no car
                        utilities = [u_0, u_1, u_2, -9000000]
                    elif latent == 1031 or latent == 1032:  # lazy
                        utilities = [-9000000, -9000000, u_2, u_3]

                    # Find the index of the maximum utility
                    best_alternative = max(range(4), key=lambda i: utilities[i])

                    df_full.loc[df_full.index[n], 'travel_mode'] = best_alternative + 1
            print("# class1 = ", class1ers)
            print("# class2 = ", class2ers)
            print("# class3 = ", class3ers)

    database = db.Database('lpmc', df_full)
    globals().update(database.variables)

    # Choice
    travel_mode_chosen = travel_mode

    # Parameters to be estimated
    ASC_Walk = Beta('ASC_Walk', 0, None, None, 1)
    ASC_Bike = Beta('ASC_Bike', 0, None, None, 0)
    ASC_PB = Beta('ASC_PB', 0, None, None, 0)
    ASC_Car = Beta('ASC_Car', 0, None, None, 0)
    Beta_cost = Beta('Beta_cost', 0, None, None, 0)
    Beta_time = Beta('Beta_time', 0, None, None, 0)

    # Define here arithmetic expressions for variables that are not directly available from the data
    Car_TT = database.DefineVariable('Car_TT', dur_driving)
    Walk_TT = database.DefineVariable('Walk_TT', dur_walking)
    PB_TT = database.DefineVariable('PB_TT', dur_pt_rail + dur_pt_bus + dur_pt_int + dur_pt_access)
    Bike_TT = database.DefineVariable('Bike_TT', dur_cycling)

    Car_cost = database.DefineVariable('Car_cost', cost_driving_fuel + cost_driving_ccharge)
    PB_cost = database.DefineVariable('PB_cost', cost_transit)

    seed_nr = pandaSeed + 1
    random.seed(seed_nr)
    np.random.seed(seed_nr)

    # ASC_Walk = Beta('ASC_Walk', 0, None, None, 1)
    #
    # 1 ASC_Bike = Beta('ASC_Bike', starting_point[0], None, None, 0)
    # 2 ASC_Car = Beta('ASC_Car', starting_point[1], None, None, 0)
    # 3 ASC_PB = Beta('ASC_PB', starting_point[2], None, None, 0)
    # 4 Beta_cost = Beta('Beta_cost', starting_point[3], None, None, 0)
    # 5 Beta_time = Beta('Beta_time', starting_point[4], None, None, 0)

    if latent == 1020:
        Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
        Beta_time_RND = Beta_time + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

        Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
        Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', 0, None, None, 0)
        Z4_ASC_PB = Beta('Z4_ASC_PB', 0, None, None, 0)

        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
        PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
        Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

        Walk_class_2 = Z1_ASC_WALK + Walk_TT * Beta_time
        Bike_class_2 = Z2_ASC_BIKE + Bike_TT * Beta_time
        Car_class_2 = 0
        PB_class_2 = Z4_ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}
        av2 = {1: 1, 2: 1, 3: 1, 4: 0}

        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av2, travel_mode_chosen)
        )
    if latent == 1021:
        Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
        Beta_time_RND = Beta_time + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

        Z4_ASC_PB = Beta('Z4_ASC_PB', 0, None, None, 0)

        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
        PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
        Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

        Walk_class_2 = Walk_TT * Beta_time
        Bike_class_2 = Bike_TT * Beta_time
        Car_class_2 = Car_TT * Beta_time + Car_cost * Beta_cost
        PB_class_2 = Z4_ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}
        av2 = {1: 0, 2: 0, 3: 1, 4: 1}

        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av2, travel_mode_chosen)
        )
    if latent == 1022 or latent == 1027:
        Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
        Beta_time_RND = Beta_time + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

        Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
        Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', 0, None, None, 0)
        Z3_ASC_CAR = Beta('Z3_ASC_CAR', 0, None, None, 0)
        Z4_ASC_PB = Beta('Z4_ASC_PB', 0, None, None, 0)

        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
        PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
        Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

        Walk_class_2 = Z1_ASC_WALK + Walk_TT * Beta_time
        Bike_class_2 = Z2_ASC_BIKE + Bike_TT * Beta_time
        Car_class_2 = Z3_ASC_CAR + Car_TT * Beta_time + Car_cost * Beta_cost
        PB_class_2 = Z4_ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen)
        )
    if latent == 1023:
        Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
        Beta_time_RND = Beta_time + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
        PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
        Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

        # walk is normalized. not having car leaves walk, bike, and goddamn I thought I was doing swissmetro damit
        Walk_class_2 = ASC_Walk + Walk_TT * Beta_time
        Bike_class_2 = ASC_Bike + Bike_TT * Beta_time
        Car_class_2 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}
        av2 = {1: 1, 2: 1, 3: 1, 4: 0}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av2, travel_mode_chosen)
        )
    if latent == 1024:
        Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
        Beta_time_RND = Beta_time + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
        PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
        Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

        # walk is normalized. not having car leaves walk, bike, and goddamn I thought I was doing swissmetro damit
        Walk_class_2 = ASC_Walk + Walk_TT * Beta_time
        Bike_class_2 = ASC_Bike + Bike_TT * Beta_time
        Car_class_2 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}
        av2 = {1: 0, 2: 0, 3: 1, 4: 1}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av2, travel_mode_chosen)
        )
    if latent == 1025:
        Z01_Beta_cost_S = Beta('Z01_Beta_cost_S', 1, None, None, 0)
        Beta_cost_RND = Beta_cost + Z01_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost_RND
        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost_RND

        Z1_Beta_time_C2 = Beta('Z1_Beta_time_C2', 0, None, None, 0)

        Walk_class_2 = ASC_Walk + Walk_TT * Z1_Beta_time_C2
        Bike_class_2 = ASC_Bike + Bike_TT * Z1_Beta_time_C2
        Car_class_2 = ASC_Car + Car_TT * Z1_Beta_time_C2 + Car_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Z1_Beta_time_C2 + PB_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen)
        )
    if latent == 1032:
        Z01_Beta_cost_S = Beta('Z01_Beta_cost_S', 1, None, None, 0)
        Beta_cost_RND = Beta_cost + Z01_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost_RND
        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost_RND

        Z1_Beta_time_C2 = Beta('Z1_Beta_time_C2', 0, None, None, 0)

        Walk_class_2 = ASC_Walk + Walk_TT * Z1_Beta_time_C2
        Bike_class_2 = ASC_Bike + Bike_TT * Z1_Beta_time_C2
        Car_class_2 = ASC_Car + Car_TT * Z1_Beta_time_C2 + Car_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Z1_Beta_time_C2 + PB_cost * Beta_cost

        Walk_class_3 = ASC_Walk + Walk_TT * Beta_time
        Bike_class_3 = ASC_Bike + Bike_TT * Beta_time
        Car_class_3 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        PB_class_3 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}
        av3 = {1: 0, 2: 0, 3: 1, 4: 1}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen) +
                P3 * models.logit(V_class_3, av3, travel_mode_chosen)
        )
    if latent == 1033:
        Z01_Beta_cost_S = Beta('Z01_Beta_cost_S', 1, None, None, 0)
        Beta_cost_RND = Beta_cost + Z01_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost_RND
        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost_RND

        Z1_Beta_time_C2 = Beta('Z1_Beta_time_C2', 0, None, None, 0)

        Walk_class_2 = ASC_Walk + Walk_TT * Z1_Beta_time_C2
        Bike_class_2 = ASC_Bike + Bike_TT * Z1_Beta_time_C2
        Car_class_2 = ASC_Car + Car_TT * Z1_Beta_time_C2 + Car_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Z1_Beta_time_C2 + PB_cost * Beta_cost

        Walk_class_3 = ASC_Walk + Walk_TT * Beta_time
        Bike_class_3 = ASC_Bike + Bike_TT * Beta_time
        Car_class_3 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        PB_class_3 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}
        av3 = {1: 1, 2: 1, 3: 1, 4: 0}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen) +
                P3 * models.logit(V_class_3, av3, travel_mode_chosen)
        )

    if latent == 1030:
        Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
        Beta_time_RND = Beta_time + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

        Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
        Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', 0, None, None, 0)
        Z3_ASC_CAR = Beta('Z3_ASC_CAR', 0, None, None, 0)
        Z4_ASC_PB = Beta('Z4_ASC_PB', 0, None, None, 0)

        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
        PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
        Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

        Walk_class_2 = Z1_ASC_WALK + Walk_TT * Beta_time
        Bike_class_2 = Z2_ASC_BIKE + Bike_TT * Beta_time
        Car_class_2 = Z3_ASC_CAR + Car_TT * Beta_time + Car_cost * Beta_cost
        PB_class_2 = Z4_ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

        Walk_class_3 = ASC_Walk + Walk_TT * Beta_time
        Bike_class_3 = ASC_Bike + Bike_TT * Beta_time
        PB_class_3 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        Car_class_3 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}
        av3 = {1: 1, 2: 1, 3: 1, 4: 0}

        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen) +
                P3 * models.logit(V_class_3, av3, travel_mode_chosen)
        )
    if latent == 1031:
        Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
        Beta_time_RND = Beta_time + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

        Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
        Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', 0, None, None, 0)
        Z3_ASC_CAR = Beta('Z3_ASC_CAR', 0, None, None, 0)
        Z4_ASC_PB = Beta('Z4_ASC_PB', 0, None, None, 0)

        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
        PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
        Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

        Walk_class_2 = Z1_ASC_WALK + Walk_TT * Beta_time
        Bike_class_2 = Z2_ASC_BIKE + Bike_TT * Beta_time
        Car_class_2 = Z3_ASC_CAR + Car_TT * Beta_time + Car_cost * Beta_cost
        PB_class_2 = Z4_ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

        Walk_class_3 = ASC_Walk + Walk_TT * Beta_time
        Bike_class_3 = ASC_Bike + Bike_TT * Beta_time
        PB_class_3 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        Car_class_3 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}
        av3 = {1: 0, 2: 0, 3: 1, 4: 1}

        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen) +
                P3 * models.logit(V_class_3, av3, travel_mode_chosen)
        )
    if latent == 1034:
        Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
        Beta_time_RND = Beta_cost + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
        PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
        Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

        Z1_Beta_cost_C2 = Beta('Z1_Beta_cost_C2', 0, None, None, 0)

        Walk_class_2 = ASC_Walk + Walk_TT * Beta_time
        Bike_class_2 = ASC_Bike + Bike_TT * Beta_time
        Car_class_2 = ASC_Car + Car_TT * Beta_time + Car_cost * Z1_Beta_cost_C2
        PB_class_2 = ASC_PB + PB_TT * Beta_time + PB_cost * Z1_Beta_cost_C2

        Walk_class_3 = ASC_Walk + Walk_TT * Beta_time
        Bike_class_3 = ASC_Bike + Bike_TT * Beta_time
        Car_class_3 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        PB_class_3 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}
        av3 = {1: 0, 2: 0, 3: 1, 4: 1}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen) +
                P3 * models.logit(V_class_3, av3, travel_mode_chosen)
        )
    if latent == 1026:
        Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
        Beta_time_RND = Beta_cost + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
        PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
        Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

        Z1_Beta_cost_C2 = Beta('Z1_Beta_cost_C2', 0, None, None, 0)

        Walk_class_2 = ASC_Walk + Walk_TT * Beta_time
        Bike_class_2 = ASC_Bike + Bike_TT * Beta_time
        Car_class_2 = ASC_Car + Car_TT * Beta_time + Car_cost * Z1_Beta_cost_C2
        PB_class_2 = ASC_PB + PB_TT * Beta_time + PB_cost * Z1_Beta_cost_C2

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen)
        )

    # numberOfDraws = 100000
    integral = MonteCarlo(prob)
    simulate = {
        'Integral': integral,
    }
    # Create the Biogeme object
    biosim = bio.BIOGEME(database, simulate, numberOfDraws=R, seed=seed_nr)
    biosim.modelName = "lpmc_mixed_latent_simul"

    betas = {
        'ASC_Bike': beta[0],
        'ASC_Car': beta[1],
        'ASC_PB': beta[2],
        'Beta_cost': beta[3],
        'Beta_time': beta[4]
    }

    if latent == 1020:
        betas['Z01_Beta_time_S'] = beta[5]
        betas['Z2_ASC_BIKE'] = beta[6]
        betas['Z4_ASC_PB'] = beta[7]
        betas['prob_class_1'] = beta[8]
    elif latent == 1021:
        betas['Z01_Beta_time_S'] = beta[5]
        betas['Z4_ASC_PB'] = beta[6]
        betas['prob_class_1'] = beta[7]
    elif latent == 1021:
        betas['Z01_Beta_time_S'] = beta[5]
        betas['Z4_ASC_PB'] = beta[6]
        betas['prob_class_1'] = beta[7]
    elif latent == 1022 or latent == 1027:
        betas['Z01_Beta_time_S'] = beta[5]
        betas['Z2_ASC_BIKE'] = beta[6]
        betas['Z3_ASC_CAR'] = beta[7]
        betas['Z4_ASC_PB'] = beta[8]
        betas['prob_class_1'] = beta[9]
    elif latent == 1023 or latent == 1024:
        betas['Z01_Beta_time_S'] = beta[5]
        betas['prob_class_1'] = beta[6]
    elif latent == 1025:
        betas['Z01_Beta_cost_S'] = beta[5]
        betas['Z1_Beta_time_C2'] = beta[6]
        betas['prob_class_1'] = beta[7]
    elif latent == 1032:
        betas['Z01_Beta_cost_S'] = beta[5]
        betas['Z1_Beta_time_C2'] = beta[6]
        betas['prob_class_1'] = beta[7]
        betas['prob_class_2'] = beta[8]
    elif latent == 1033:
        betas['Z01_Beta_cost_S'] = beta[5]
        betas['Z1_Beta_time_C2'] = beta[6]
        betas['prob_class_1'] = beta[7]
        betas['prob_class_2'] = beta[8]
    elif latent == 1034:
        betas['Z01_Beta_time_S'] = beta[5]
        betas['Z1_Beta_cost_C2'] = beta[6]
        betas['prob_class_1'] = beta[7]
        betas['prob_class_2'] = beta[8]
    elif latent == 1026:
        betas['Z01_Beta_time_S'] = beta[5]
        betas['Z1_Beta_cost_C2'] = beta[6]
        betas['prob_class_1'] = beta[7]
    elif latent == 1030:
        betas['Z01_Beta_time_S'] = beta[5]
        betas['Z2_ASC_BIKE'] = beta[6]
        betas['Z3_ASC_CAR'] = beta[7]
        betas['Z4_ASC_PB'] = beta[8]
        betas['prob_class_1'] = beta[9]
        betas['prob_class_2'] = beta[10]
    elif latent == 1031:
        betas['Z01_Beta_time_S'] = beta[5]
        betas['Z2_ASC_BIKE'] = beta[6]
        betas['Z3_ASC_CAR'] = beta[7]
        betas['Z4_ASC_PB'] = beta[8]
        betas['prob_class_1'] = beta[9]
        betas['prob_class_2'] = beta[10]

    simresults = biosim.simulate(betas)
    logLikelihood = np.log(simresults["Integral"]).sum()

    return logLikelihood


def simulate_likelihood_mixed_lpmc(N, pandaSeed, beta, latent, R=1000):
    df_full = pd.read_csv("lpmc.dat", sep='\t')
    df_full['rail_time'] = df_full.dur_pt_rail + df_full.dur_pt_bus + df_full.dur_pt_int + df_full.dur_pt_access
    df_full["car_cost"] = df_full.cost_driving_fuel + df_full.cost_driving_ccharge
    df_full["travel_mode_chosen"] = df_full.travel_mode - 1

    if not N == 0:
        df_full = df_full.sample(N, random_state=pandaSeed)
    else:
        df_full = df_full

    df_full = df_full.rename(columns={'start_time': 'start_time_new'})
    database = db.Database('lpmc', df_full)
    globals().update(database.variables)

    # Choice
    travel_mode_chosen = travel_mode

    # Parameters to be estimated
    ASC_Walk = Beta('ASC_Walk', 0, None, None, 1)
    ASC_Bike = Beta('ASC_Bike', 0, None, None, 0)
    ASC_PB = Beta('ASC_PB', 0, None, None, 0)
    ASC_Car = Beta('ASC_Car', 0, None, None, 0)
    Beta_cost = Beta('Beta_cost', 0, None, None, 0)
    Beta_time = Beta('Beta_time', 0, None, None, 0)

    # Define here arithmetic expressions for variables that are not directly available from the data
    Car_TT = database.DefineVariable('Car_TT', dur_driving)
    Walk_TT = database.DefineVariable('Walk_TT', dur_walking)
    PB_TT = database.DefineVariable('PB_TT', dur_pt_rail + dur_pt_bus + dur_pt_int + dur_pt_access)
    Bike_TT = database.DefineVariable('Bike_TT', dur_cycling)

    Car_cost = database.DefineVariable('Car_cost', cost_driving_fuel + cost_driving_ccharge)
    PB_cost = database.DefineVariable('PB_cost', cost_transit)

    if 10 <= latent <= 19:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        # ASC_Walk = Beta('ASC_Walk', 0, None, None, 1)
        #
        # 1 ASC_Bike = Beta('ASC_Bike', starting_point[0], None, None, 0)
        # 2 ASC_Car = Beta('ASC_Car', starting_point[1], None, None, 0)
        # 3 ASC_PB = Beta('ASC_PB', starting_point[2], None, None, 0)
        # 4 Beta_cost = Beta('Beta_cost', starting_point[3], None, None, 0)
        # 5 Beta_time = Beta('Beta_time', starting_point[4], None, None, 0)

        mix_inds = None
        if latent == 10:
            mix_inds = [[5, 6]]  # mix just time
            if starting_point is not None:
                Z1_Beta_time_S = Beta('Z1_Beta_time_S', starting_point[5], None, None, 0)
            else:
                Z1_Beta_time_S = Beta('Z1_Beta_time_S', 1, None, None, 0)
            Beta_time_RND = Beta_time + Z1_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

            Walk = ASC_Walk + Walk_TT * Beta_time_RND
            Bike = ASC_Bike + Bike_TT * Beta_time_RND
            PB = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
            Car = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

        if latent == 11:
            mix_inds = [[5, 6], [4, 7]]  # mix time and costs
            if starting_point is not None:
                Z1_Beta_time_S = Beta('Z1_Beta_time_S', starting_point[5], None, None, 0)
                Z1_Beta_cost_S = Beta('Z1_Beta_cost_S', starting_point[6], None, None, 0)
            else:
                Z1_Beta_time_S = Beta('Z1_Beta_time_S', 1, None, None, 0)
                Z1_Beta_cost_S = Beta('Z1_Beta_cost_S', 1, None, None, 0)
            Beta_time_RND = Beta_time + Z1_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')
            Beta_cost_RND = Beta_cost + Z1_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

            Walk = ASC_Walk + Walk_TT * Beta_time_RND
            Bike = ASC_Bike + Bike_TT * Beta_time_RND
            PB = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost_RND
            Car = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost_RND

        if latent == 12:
            mix_inds = [[4, 6]]  # mix only costs
            if starting_point is not None:
                Z1_Beta_cost_S = Beta('Z1_Beta_cost_S', starting_point[5], None, None, 0)
            else:
                Z1_Beta_cost_S = Beta('Z1_Beta_cost_S', 1, None, None, 0)
            Beta_cost_RND = Beta_cost + Z1_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

            Walk = ASC_Walk + Walk_TT * Beta_time
            Bike = ASC_Bike + Bike_TT * Beta_time
            PB = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost_RND
            Car = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost_RND

        if latent == 18:
            mix_inds = [[2, 6]]  # mix ASC car
            if starting_point is not None:
                Z1_ASC_Car_S = Beta('Z1_ASC_Car_S', starting_point[5], None, None, 0)
            else:
                Z1_ASC_Car_S = Beta('Z1_ASC_Car_S', 1, None, None, 0)
            ASC_Car_RND = ASC_Car + Z1_ASC_Car_S * bioDraws('ASC_Car_RND', 'NORMAL')

            Walk = ASC_Walk + Walk_TT * Beta_time
            Bike = ASC_Bike + Bike_TT * Beta_time
            PB = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
            Car = ASC_Car_RND + Car_TT * Beta_time + Car_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V = {1: Walk,
             2: Bike,
             3: PB,
             4: Car}

        # Associate the availability conditions with the alternatives
        av = {1: 1,
              2: 1,
              3: 1,
              4: 1}

        # Conditional to B_TIME_RND, we have a logit model (called the kernel)
        prob = models.logit(V, av, travel_mode_chosen)

    # numberOfDraws = 100000
    integral = MonteCarlo(prob)
    simulate = {
        'Integral': integral,
    }
    # Create the Biogeme object
    biosim = bio.BIOGEME(database, simulate, numberOfDraws=R)
    biosim.modelName = "lpmc_mixed_simul"

    betas = {
        'ASC_Bike': beta[0],
        'ASC_Car': beta[1],
        'ASC_PB': beta[2],
        'Beta_cost': beta[3],
        'Beta_time': beta[4]
    }

    if latent == 10:
        betas['Z1_Beta_time_S'] = beta[5]
    elif latent == 11:
        betas['Z1_Beta_time_S'] = beta[5]
        betas['Z1_Beta_cost_S'] = beta[6]
    elif latent == 12:
        betas['Z1_Beta_cost_S'] = beta[5]
    elif latent == 18:
        betas['Z1_ASC_Car_S'] = beta[5]

    simresults = biosim.simulate(betas)
    logLikelihood = np.log(simresults["Integral"]).sum()

    return logLikelihood


def simulate_likelihood_mixed_latent_swissmetro(N, pandaSeed, beta, latent, R=1000):
    df_full = pd.read_csv('swissmetro.dat', sep='\t')
    df_full = df_full.loc[
        ~((df_full["PURPOSE"] != 1) & (df_full["PURPOSE"] != 3) | (df_full["CHOICE"] == 0) > 0)]

    if not N == 0:
        df_full = df_full.sample(N, random_state=pandaSeed)
    else:
        df_full = df_full

    # synthetic choices
    J = 3
    # 0: Train
    # 1: SM
    # 2: Car

    K = 5

    size = N
    x = np.zeros((J, size, K))

    # Add custom columns directly
    df_full['SM_COST'] = df_full['SM_CO'] * (df_full['GA'] == 0)
    df_full['TRAIN_COST'] = df_full['TRAIN_CO'] * (df_full['GA'] == 0)
    df_full['TRAIN_TT_SCALED'] = df_full['TRAIN_TT'] / 100.0
    df_full['TRAIN_COST_SCALED'] = df_full['TRAIN_COST'] / 100
    df_full['SM_TT_SCALED'] = df_full['SM_TT'] / 100.0
    df_full['SM_COST_SCALED'] = df_full['SM_COST'] / 100
    df_full['CAR_TT_SCALED'] = df_full['CAR_TT'] / 100
    df_full['CAR_CO_SCALED'] = df_full['CAR_CO'] / 100
    df_full['TRAIN_HE_SCALED'] = df_full['TRAIN_HE'] / 1000
    df_full['SM_HE_SCALED'] = df_full['SM_HE'] / 1000

    for n in range(size):
        # k = 1: ASC_Car
        # k = 2: ASC_Train
        # k = 3: Beta_time
        # k = 4: Beta_cost
        # k = 5: Beta_headway
        # k = 6: Prob1

        # train, SM, car

        # intercept car
        x[0, n, 0] = 0
        x[1, n, 0] = 0
        x[2, n, 0] = 1
        # intercept train
        x[0, n, 1] = 1
        x[1, n, 1] = 0
        x[2, n, 1] = 0
        # beta cost
        x[0, n, 2] = df_full['TRAIN_COST_SCALED'].values[n]
        x[1, n, 2] = df_full['SM_COST_SCALED'].values[n]
        x[2, n, 2] = df_full['CAR_CO_SCALED'].values[n]
        # beta headway
        x[0, n, 3] = df_full['TRAIN_HE_SCALED'].values[n]
        x[1, n, 3] = df_full['SM_HE_SCALED'].values[n]
        x[2, n, 3] = 0
        # beta time
        x[0, n, 4] = df_full['TRAIN_TT_SCALED'].values[n]
        x[1, n, 4] = df_full['SM_TT_SCALED'].values[n]
        x[2, n, 4] = df_full['CAR_TT_SCALED'].values[n]

    base_beta = [-0.12339153095665459, -0.012223798401333768, -1.5146992381080648, -5.181104734705727,
                 -2.2057779141945284, 1.186726809265002]
    seed_nr = pandaSeed + 1
    random.seed(seed_nr)
    np.random.seed(seed_nr)

    if 1020 <= latent <= 1023:
        seed = 1
        runNuM = 1
        seedEnd = seed + runNuM - 1

        while seed <= seedEnd:
            seed += 1

            # define the new ASCs (ASC_car at ind 0, ASC_train at ind 1)
            new_beta = copy.copy(base_beta)
            if latent == 1020:  # no car
                new_beta[1] = base_beta[1] * 0.33333  # reduce train attractiveness
            if latent == 1021:  # no SM
                new_beta[0] = base_beta[0] * 0.33333  # reduce car attractiveness
            if latent == 1022:  # has both
                new_beta[0] = base_beta[0] * 0.33333  # reduce car attractiveness
                new_beta[1] = base_beta[1] * 0.33333  # reduce train attractiveness

            N = len(df_full)  # maybe works

            population = [1, 2]
            weights = [0.25, 0.75]
            random.seed(seed)
            class1ers = 0
            class2ers = 0

            for n in range(N):
                # Draw a random number from 1 to 2
                n_class = random.choices(population, weights=weights, k=1)[0]

                if n_class == 1:
                    # k = 0: ASC_Car
                    # k = 1: ASC_Train
                    # k = 2: Beta_headway
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_std
                    class1ers += 1
                    u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[0, n, 4] * base_beta[
                        5] * np.random.normal(0, 1)
                    u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[1, n, 4] * base_beta[
                        5] * np.random.normal(0, 1)
                    u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[2, n, 4] * base_beta[
                        5] * np.random.normal(0, 1)

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2]

                    # Find the index of the maximum utility
                    best_alternative = max(range(3), key=lambda i: utilities[i])

                    df_full.loc[df_full.index[n], 'CHOICE'] = best_alternative + 1

                elif n_class == 2:
                    # k = 0: ASC_Car
                    # k = 1: ASC_Train
                    # k = 2: Beta_headway
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_std
                    class2ers += 1

                    u_0 = sum(x[0, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])

                    # Put the utilities into a list
                    if latent == 1020 or latent == 1023:  # no car
                        utilities = [u_0, u_1, -900000]
                    elif latent == 1021 or latent == 1024:  # no SM
                        utilities = [u_0, -900000, u_2]
                    else:  # 1022 has all av in class 2
                        utilities = [u_0, u_1, u_2]

                    # Find the index of the maximum utility
                    best_alternative = max(range(3), key=lambda i: utilities[i])

                    df_full.loc[df_full.index[n], 'CHOICE'] = best_alternative + 1
            print("# class1 = ", class1ers)
            print("# class2 = ", class2ers)

    if 1030 <= latent <= 1031:
        seed = 1
        runNuM = 1
        seedEnd = seed + runNuM - 1

        while seed <= seedEnd:
            seed += 1

            # define the new ASCs (ASC_car at ind 0, ASC_train at ind 1)
            new_beta = copy.copy(base_beta)
            if latent == 1030:  # has both, but no car in the last group
                new_beta[0] = base_beta[0] * 1.33333  # increase car attractiveness
                new_beta[1] = base_beta[1] * 0.33333  # reduce train attractiveness
            if latent == 1031:  # has both, but no SM in the last group
                new_beta[0] = base_beta[0] * 0.33333  # reduce car attractiveness
                new_beta[1] = base_beta[1] * 0.33333  # reduce train attractiveness

            N = len(df_full)  # maybe works

            population = [1, 2, 3]
            weights = [0.25, 0.5, 0.25]
            random.seed(seed)
            class1ers = 0
            class2ers = 0
            class3ers = 0

            for n in range(N):
                # Draw a random number from 1 to 3
                n_class = random.choices(population, weights=weights, k=1)[0]

                if n_class == 1:
                    # k = 0: ASC_Car
                    # k = 1: ASC_Train
                    # k = 2: Beta_headway
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_std
                    class1ers += 1
                    u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[0, n, 4] * base_beta[
                        5] * np.random.normal(0, 1)
                    u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[1, n, 4] * base_beta[
                        5] * np.random.normal(0, 1)
                    u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[2, n, 4] * base_beta[
                        5] * np.random.normal(0, 1)

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2]

                    # Find the index of the maximum utility
                    best_alternative = max(range(3), key=lambda i: utilities[i])

                    df_full.loc[df_full.index[n], 'CHOICE'] = best_alternative + 1
                elif n_class == 2:
                    # k = 0: ASC_Car
                    # k = 1: ASC_Train
                    # k = 2: Beta_headway
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_std
                    class2ers += 1

                    u_0 = sum(x[0, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2]

                    # Find the index of the maximum utility
                    best_alternative = max(range(3), key=lambda i: utilities[i])

                    df_full.loc[df_full.index[n], 'CHOICE'] = best_alternative + 1
                elif n_class == 3:
                    # k = 0: ASC_Car
                    # k = 1: ASC_Train
                    # k = 2: Beta_headway
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_std
                    class3ers += 1

                    u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])

                    # Put the utilities into a list
                    if latent == 1030:  # no car
                        utilities = [u_0, u_1, -900000]
                    elif latent == 1031:  # no SM
                        utilities = [u_0, -900000, u_2]

                    # Find the index of the maximum utility
                    best_alternative = max(range(3), key=lambda i: utilities[i])

                    df_full.loc[df_full.index[n], 'CHOICE'] = best_alternative + 1
            print("# class1 = ", class1ers)
            print("# class2 = ", class2ers)
            print("# class3 = ", class3ers)

    database = db.Database('swissmetro', df_full)

    seed_nr = pandaSeed + 1
    random.seed(seed_nr)
    np.random.seed(seed_nr)

    PURPOSE = Variable('PURPOSE')
    CHOICE = Variable('CHOICE')
    SM_CO = Variable('SM_CO')
    TRAIN_CO = Variable('TRAIN_CO')
    TRAIN_TT = Variable('TRAIN_TT')
    GA = Variable('GA')
    CAR_AV = Variable('CAR_AV')
    TRAIN_AV = Variable('TRAIN_AV')
    SP = Variable('SP')
    SM_TT = Variable('SM_TT')
    CAR_TT = Variable('CAR_TT')
    CAR_CO = Variable('CAR_CO')
    TRAIN_HE = Variable('TRAIN_HE')
    SM_HE = Variable('SM_HE')
    SM_AV = Variable('SM_AV')

    # exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
    # database.remove(exclude)

    CAR_AV_SP = database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
    TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))

    TRAIN_TT_SCALED = Variable('TRAIN_TT_SCALED')
    TRAIN_COST_SCALED = Variable('TRAIN_COST_SCALED')
    SM_TT_SCALED = Variable('SM_TT_SCALED')
    SM_COST_SCALED = Variable('SM_COST_SCALED')
    CAR_TT_SCALED = Variable('CAR_TT_SCALED')
    CAR_CO_SCALED = Variable('CAR_CO_SCALED')
    TRAIN_HE_SCALED = Variable('TRAIN_HE_SCALED')
    SM_HE_SCALED = Variable('SM_HE_SCALED')

    # Parameters to be estimated
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
    ASC_SM = Beta('ASC_SM', 0, None, None, 1)
    B_COST = Beta('B_COST', 0, None, None, 0)
    B_HE = Beta('B_HE', 0, None, None, 0)
    B_TIME = Beta('B_TIME', 0, None, None, 0)

    seed_nr = pandaSeed + 1
    random.seed(seed_nr)
    np.random.seed(seed_nr)

    if latent == 1020:
        Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
        B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

        Z2_ASC_TRAIN = Beta('Z2_ASC_TRAIN', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = Z2_ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        # av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        # av2 = {1: TRAIN_AV_SP, 2: SM_AV, 3: 0}

        av = {1: 1, 2: 1, 3: 1}
        av2 = {1: 1, 2: 1, 3: 0}

        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)
        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av2, CHOICE)
        )
    if latent == 1021:
        Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
        B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

        Z1_ASC_CAR = Beta('Z1_ASC_CAR', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = Z1_ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        # av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        # av2 = {1: TRAIN_AV_SP, 2: 0, 3: CAR_AV_SP}

        av = {1: 1, 2: 1, 3: 1}
        av2 = {1: 1, 2: 0, 3: 1}

        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av2, CHOICE)
        )

    if latent == 1022 or latent == 1027:
        Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
        B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

        Z02_ASC_SM = Beta('Z02_ASC_SM', 0, None, None, 1)
        Z1_ASC_CAR = Beta('Z1_ASC_CAR', 0, None, None, 0)
        Z2_ASC_TRAIN = Beta('Z2_ASC_TRAIN', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = Z2_ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = Z02_ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = Z1_ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        # av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        av = {1: 1, 2: 1, 3: 1}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )
    if latent == 1023:
        Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
        B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

        # ASC SM is normalized. Removing Car leaves ASC Train to be estimated, and SM fixed. Works.

        TRAIN_class_1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        # av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        # av2 = {1: TRAIN_AV_SP, 2: SM_AV, 3: 0}

        av = {1: 1, 2: 1, 3: 1}
        av2 = {1: 1, 2: 1, 3: 0}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av2, CHOICE)
        )
    if latent == 1024:
        Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
        B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

        # ASC SM is normalized. Removing Car leaves ASC Train to be estimated, and SM fixed. Works.

        TRAIN_class_1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        # av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        # av2 = {1: TRAIN_AV_SP, 2: 0, 3: CAR_AV_SP}

        av = {1: 1, 2: 1, 3: 1}
        av2 = {1: 1, 2: 0, 3: 1}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av2, CHOICE)
        )
    if latent == 1025:
        Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
        B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

        Z1_B_TIME_C2 = Beta('Z1_B_TIME_C2', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + Z1_B_TIME_C2 * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + Z1_B_TIME_C2 * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + Z1_B_TIME_C2 * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

    if latent == 1026:
        Z01_B_COST_S = Beta('Z01_B_COST_S', 1, None, None, 0)
        B_COST_RND = B_COST + Z01_B_COST_S * bioDraws('B_COST_RND', 'NORMAL')

        Z1_B_TIME_C2 = Beta('Z1_B_TIME_C2', 0, None, None, 0)

        # ASC SM is normalized. Removing Car leaves ASC Train to be estimated, and SM fixed. Works.

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST_RND * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + Z1_B_TIME_C2 * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST_RND * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + Z1_B_TIME_C2 * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST_RND * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + Z1_B_TIME_C2 * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )
    if latent == 1028:
        Z01_B_HE_S = Beta('Z01_B_HE_S', 1, None, None, 0)
        B_HE_RND = B_HE + Z01_B_HE_S * bioDraws('B_HE_RND', 'NORMAL')

        Z1_B_TIME_C2 = Beta('Z1_B_TIME_C2', 0, None, None, 0)

        # ASC SM is normalized. Removing Car leaves ASC Train to be estimated, and SM fixed. Works.

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE_RND * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + Z1_B_TIME_C2 * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE_RND * SM_HE_SCALED
        SM_class_2 = ASC_SM + Z1_B_TIME_C2 * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + Z1_B_TIME_C2 * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

    if latent == 1030:
        Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
        B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

        Z1_ASC_CAR = Beta('Z1_ASC_CAR', 0, None, None, 0)
        Z2_ASC_TRAIN = Beta('Z2_ASC_TRAIN', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = Z2_ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_3 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_3 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = Z1_ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}

        # Associate the availability conditions with the alternatives
        # av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        # av3 = {1: TRAIN_AV_SP, 2: SM_AV, 3: 0}

        av = {1: 1, 2: 1, 3: 1}
        av3 = {1: 1, 2: 1, 3: 0}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE) +
                P3 * models.logit(V_class_3, av3, CHOICE)
        )
    if latent == 1031:
        Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
        B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

        Z1_ASC_CAR = Beta('Z1_ASC_CAR', 0, None, None, 0)
        Z2_ASC_TRAIN = Beta('Z2_ASC_TRAIN', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = Z2_ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_3 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_3 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = Z1_ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}

        # Associate the availability conditions with the alternatives
        # av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        # av3 = {1: TRAIN_AV_SP, 2: 0, 3: CAR_AV_SP}

        av = {1: 1, 2: 1, 3: 1}
        av3 = {1: 1, 2: 0, 3: 1}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE) +
                P3 * models.logit(V_class_3, av3, CHOICE)
        )

    # numberOfDraws = 100000
    integral = MonteCarlo(prob)
    simulate = {
        'Integral': integral,
    }
    # Create the Biogeme object
    biosim = bio.BIOGEME(database, simulate, numberOfDraws=R, seed=seed_nr)
    biosim.modelName = "swissmetro_mixed_latent_simul"

    betas = {
        'ASC_CAR': beta[0],
        'ASC_TRAIN': beta[1],
        'B_COST': beta[2],
        'B_HE': beta[3],
        'B_TIME': beta[4]
    }

    if latent == 1020:
        betas['Z01_B_TIME_S'] = beta[5]
        betas['Z2_ASC_TRAIN'] = beta[6]
        betas['prob_class_1'] = beta[7]
    elif latent == 1021:
        betas['Z01_B_TIME_S'] = beta[5]
        betas['Z1_ASC_CAR'] = beta[6]
        betas['prob_class_1'] = beta[7]
    elif latent == 1022 or latent == 1027:
        betas['Z01_B_TIME_S'] = beta[5]
        betas['Z1_ASC_CAR'] = beta[6]
        betas['Z2_ASC_TRAIN'] = beta[7]
        betas['prob_class_1'] = beta[8]
    elif latent == 1023:
        betas['Z01_B_TIME_S'] = beta[5]
        betas['prob_class_1'] = beta[6]
    elif latent == 1024:
        betas['Z01_B_TIME_S'] = beta[5]
        betas['prob_class_1'] = beta[6]
    elif latent == 1025:
        betas['Z01_B_TIME_S'] = beta[5]
        betas['Z1_B_TIME_C2'] = beta[6]
        betas['prob_class_1'] = beta[7]
    elif latent == 1026:
        betas['Z01_B_COST_S'] = beta[5]
        betas['Z1_B_TIME_C2'] = beta[6]
        betas['prob_class_1'] = beta[7]
    elif latent == 1028:
        betas['Z01_B_HE_S'] = beta[5]
        betas['Z1_B_TIME_C2'] = beta[6]
        betas['prob_class_1'] = beta[7]
    elif latent == 1030:
        betas['Z01_B_TIME_S'] = beta[5]
        betas['Z1_ASC_CAR'] = beta[6]
        betas['Z2_ASC_TRAIN'] = beta[7]
        betas['prob_class_1'] = beta[8]
        betas['prob_class_2'] = beta[9]
    elif latent == 1031:
        betas['Z01_B_TIME_S'] = beta[5]
        betas['Z1_ASC_CAR'] = beta[6]
        betas['Z2_ASC_TRAIN'] = beta[7]
        betas['prob_class_1'] = beta[8]
        betas['prob_class_2'] = beta[9]

    simresults = biosim.simulate(betas)
    logLikelihood = np.log(simresults["Integral"]).sum()

    return logLikelihood


def simulate_likelihood_mixed_swissmetro(N, pandaSeed, beta, mix_inds, R=1000):
    df_full = pd.read_csv('swissmetro.dat', sep='\t')
    df_full = df_full.loc[
        ~((df_full["PURPOSE"] != 1) & (df_full["PURPOSE"] != 3) | (df_full["CHOICE"] == 0) > 0)]

    if not N == 0:
        df_full = df_full.sample(N, random_state=pandaSeed)
    else:
        df_full = df_full

    database = db.Database('swissmetro', df_full)

    seed_nr = pandaSeed + 1
    random.seed(seed_nr)
    np.random.seed(seed_nr)

    PURPOSE = Variable('PURPOSE')
    CHOICE = Variable('CHOICE')
    SM_CO = Variable('SM_CO')
    TRAIN_CO = Variable('TRAIN_CO')
    TRAIN_TT = Variable('TRAIN_TT')
    GA = Variable('GA')
    CAR_AV = Variable('CAR_AV')
    TRAIN_AV = Variable('TRAIN_AV')
    SP = Variable('SP')
    SM_TT = Variable('SM_TT')
    CAR_TT = Variable('CAR_TT')
    CAR_CO = Variable('CAR_CO')
    TRAIN_HE = Variable('TRAIN_HE')
    SM_HE = Variable('SM_HE')
    SM_AV = Variable('SM_AV')

    exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
    database.remove(exclude)

    SM_COST = SM_CO * (GA == 0)
    TRAIN_COST = TRAIN_CO * (GA == 0)

    CAR_AV_SP = database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
    TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))

    TRAIN_TT_SCALED = database.DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0)
    TRAIN_COST_SCALED = database.DefineVariable(
        'TRAIN_COST_SCALED', TRAIN_COST / 100
    )
    SM_TT_SCALED = database.DefineVariable('SM_TT_SCALED', SM_TT / 100.0)
    SM_COST_SCALED = database.DefineVariable('SM_COST_SCALED', SM_COST / 100)
    CAR_TT_SCALED = database.DefineVariable('CAR_TT_SCALED', CAR_TT / 100)
    CAR_CO_SCALED = database.DefineVariable('CAR_CO_SCALED', CAR_CO / 100)
    TRAIN_HE_SCALED = database.DefineVariable('TRAIN_HE_SCALED', TRAIN_HE / 1000)
    SM_HE_SCALED = database.DefineVariable('SM_HE_SCALED', SM_HE / 1000)

    # Parameters to be estimated
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
    ASC_SM = Beta('ASC_SM', 0, None, None, 1)
    B_COST = Beta('B_COST', 0, None, None, 0)
    B_HE = Beta('B_HE', 0, None, None, 0)
    B_TIME = Beta('B_TIME', 0, None, None, 0)

    firsts = [tupel[0] for tupel in mix_inds]

    if 5 in firsts:
        Z1_B_TIME_S = Beta('Z1_B_TIME_S', 1, None, None, 0)
        B_TIME_RND = B_TIME + Z1_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')
    if 3 in firsts:
        Z2_B_COST_S = Beta('Z2_B_COST_S', 1, None, None, 0)
        B_COST_RND = B_COST + Z2_B_COST_S * bioDraws('B_COST_RND', 'NORMAL')
    if 4 in firsts:
        Z3_B_HE_S = Beta('Z3_B_HE_S', 1, None, None, 0)
        B_HE_RND = B_HE + Z3_B_HE_S * bioDraws('B_HE_RND', 'NORMAL')

    # Definition of the utility functions
    if mix_inds == [[5, 6]]:
        V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
    elif mix_inds == [[5, 6], [3, 7]]:
        V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST_RND * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST_RND * SM_COST_SCALED + B_HE * SM_HE_SCALED
        V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST_RND * CAR_CO_SCALED
    elif mix_inds == [[5, 6], [3, 7], [4, 8]]:
        V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST_RND * TRAIN_COST_SCALED + B_HE_RND * TRAIN_HE_SCALED
        V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST_RND * SM_COST_SCALED + B_HE_RND * SM_HE_SCALED
        V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST_RND * CAR_CO_SCALED

    # Associate utility functions with the numbering of alternatives
    V = {1: V1, 2: V2, 3: V3}

    # Associate the availability conditions with the alternatives
    av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

    # Conditional to B_TIME_RND, we have a logit model (called the kernel)
    prob = models.logit(V, av, CHOICE)

    # numberOfDraws = 100000
    integral = MonteCarlo(prob)
    simulate = {
        'Integral': integral,
    }
    # Create the Biogeme object
    biosim = bio.BIOGEME(database, simulate, numberOfDraws=R)
    biosim.modelName = "swissmetro_mixed_simul"

    betas = {
        'ASC_CAR': beta[0],
        'ASC_TRAIN': beta[1],
        'B_COST': beta[2],
        'B_HE': beta[3],
        'B_TIME': beta[4]
    }

    beta_idx = 4
    if 5 in firsts:
        beta_idx += 1
        betas['Z1_B_TIME_S'] = beta[beta_idx]
    if 3 in firsts:
        beta_idx += 1
        betas['Z2_B_COST_S'] = beta[beta_idx]
    if 4 in firsts:
        beta_idx += 1
        betas['Z3_B_HE_S'] = beta[beta_idx]

    simresults = biosim.simulate(betas)
    logLikelihood = np.log(simresults["Integral"]).sum()

    return logLikelihood


def compute_sLL_mixed(x, y, mix_inds, av, epsilon, beta, R=100):
    mix_inds = [[index - 1 for index in pair] for pair in mix_inds]
    L = len(mix_inds)
    K = len(beta) - L
    J, N, _ = epsilon.shape
    logOfZero = -100

    epsilon = np.random.gumbel(loc=0, scale=1, size=(J, N * R))
    normal_epsilon = np.random.normal(loc=0, scale=1, size=(L, N * R))

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    Uinr = dict()
    omega = dict()

    for n in range(N):
        for r in range(R):
            for i in av_alts[n]:
                Uinr[i, n, r] = sum(beta[k] * x[i, n, k] for k in range(K)) \
                                + sum(beta[l[1]] * x[i, n, l[0]] * normal_epsilon[ll, n * N + r] for ll, l in enumerate(mix_inds)) \
                                + epsilon[i, n * N + r]

            max_util = np.max([Uinr[i, n, r] for i in av_alts[n]])
            for i in av_alts[n]:
                if Uinr[i, n, r] == max_util:
                    omega[i, n, r] = 1
                else:
                    omega[i, n, r] = 0

    # compute objective value
    s_dict = dict()
    z_dict = dict()
    obj_dict = dict()
    for n in range(N):
        for i in av_alts[n]:
            s_dict[i, n] = sum(omega[i, n, r] for r in range(R))
            if s_dict[i, n] > 0:
                z_dict[i, n] = np.log(s_dict[i, n])
            else:
                z_dict[i, n] = logOfZero
        obj_dict[n] = sum(y[i, n] * z_dict[i, n] for i in av_alts[n])
    total_obj = sum(obj_dict[n] for n in range(N)) - N * np.log(R)

    return total_obj


def compute_biog_loglike_mixed(x, y, av, beta, mix_inds, R=10000):
    J, N, K = x.shape
    Pin = np.zeros((J, N))  # Store choice probabilities

    mix_inds = [[index - 1 for index in pair] for pair in mix_inds]

    # Extract mean and std for mixed beta coefficients
    distr_params = [k[0] for k in mix_inds]
    means = np.array([beta[k] for k in distr_params])
    stds = np.array([beta[k[1]] for k in mix_inds])

    for n in range(N):
        av_n = np.where(av[:, n] == 1)[0]  # Indices of available alternatives
        R_draws = np.random.randn(len(means), R)  # Precompute R samples for efficiency
        beta_samples = means[:, None] + stds[:, None] * R_draws  # Generate R samples for random coefficients

        # Compute utilities for all available alternatives and all simulation draws
        Vin_samples = []
        for i in av_n:
            deterministic_utility = sum(beta[k] * x[i, n, k] for k in range(K))
            random_utility = sum(beta_samples[d, :] * x[i, n, distr_params[d]] for d in range(len(distr_params)))
            Vin_samples.append(deterministic_utility + random_utility)
        Vin_samples = np.array(Vin_samples)  # Shape: (len(av_n), R)

        # Compute log-sum-exp for numerical stability across simulation draws
        log_denom = logsumexp(Vin_samples, axis=0)  # Compute for each simulation draw

        # Calculate the probabilities for each alternative
        integral_sum = np.exp(Vin_samples - log_denom)  # Numerator minus log-denominator
        mean_integral_sum = np.mean(integral_sum, axis=1)  # Average across R draws
        Pin[av_n, n] = mean_integral_sum  # Store probabilities for available alternatives

    # Compute the log-likelihood
    biog_obj = 0
    for n in range(N):
        av_n = np.where(av[:, n] == 1)[0]
        for i in av_n:
            if Pin[i, n] > 0:  # Avoid log(0) errors
                biog_obj += y[i, n] * np.log(Pin[i, n])

    return biog_obj


def compute_biog_LL(x, y, av, beta_orig):
    J, N, K = x.shape
    Vin = np.zeros((J, N))
    Pin = np.zeros((J, N))

    beta = beta_orig.copy()  # Ensure beta is not modified outside the function

    for n in range(N):
        for i in [i for i in range(J) if av[i][n] == 1]:
            Vin[i, n] = sum(beta[k] * x[i, n, k] for k in range(K))

    for n in range(N):
        # Find the maximum Vin value for the current n and c (log-sum-exp trick) for valid indices
        max_vin = max(Vin[i, n] for i in [i for i in range(J) if av[i][n] == 1])

        # Compute the denominator using the log-sum-exp trick for valid indices
        denom = sum(np.exp(Vin[i, n] - max_vin) for i in [i for i in range(J) if av[i][n] == 1])

        # Compute Pin, applying the same shift
        for i in [i for i in range(J) if av[i][n] == 1]:
            Pin[i, n] = np.exp(Vin[i, n] - max_vin) / denom

    # Compute the overall log-likelihood
    biog_obj = 0
    for n in range(N):
        for i in [i for i in range(J) if av[i][n] == 1]:
            log_prob = -100 if Pin[i, n] == 0 else np.log(Pin[i, n])
            biog_obj += y[i, n] * log_prob

    return biog_obj


def compute_biog_LL_mixed(x, y, av, beta_orig, mix_inds, R=100):
    """
    Computes the log-likelihood for a mixed logit model using Monte Carlo simulation.

    Parameters:
    - x: Input data, shape (J, N, K).
    - y: Observed choices, binary matrix of shape (J, N).
    - av: Availability matrix, binary matrix of shape (J, N).
    - beta_orig: Vector of estimated parameters (mean and standard deviations).
    - mix_inds: List of lists, where each sublist contains the index of a parameter and its standard deviation.
    - R: Number of Monte Carlo draws.

    Returns:
    - Log-likelihood value for the mixed logit model.
    """
    seed_nr = pandaSeed + 1
    random.seed(seed_nr)
    np.random.seed(seed_nr)

    mix_inds = [[index - 1 for index in pair] for pair in mix_inds]
    J, N, K = x.shape
    num_mixed = len(mix_inds)  # Number of mixed parameters
    num_means = len(beta_orig) - num_mixed  # Length of the mean vector

    beta_mean = beta_orig[:num_means]  # Extract means (first part of beta_orig)
    beta_std = np.zeros(num_means)  # Initialize standard deviations (aligned with beta_mean)

    for mean_idx, std_idx in mix_inds:
        beta_std[mean_idx] = beta_orig[std_idx]  # Assign stdv to the correct index

    # Monte Carlo draws for random effects
    random_effects = np.random.normal(0, 1, size=(R, num_means))  # R samples for each mean parameter

    # Compute utility for each draw
    Vin = np.zeros((J, N, R))
    for r in range(R):
        # Adjust beta with random effects for mixed parameters
        beta_r = beta_mean + random_effects[r] * beta_std

        for n in range(N):
            for i in [i for i in range(J) if av[i][n] == 1]:
                Vin[i, n, r] = sum(beta_r[k] * x[i, n, k] for k in range(num_means))

    # Compute probabilities for each draw
    Pin = np.zeros((J, N, R))
    for r in range(R):
        for n in range(N):
            max_vin = max(Vin[i, n, r] for i in [i for i in range(J) if av[i][n] == 1])
            denom = sum(np.exp(Vin[i, n, r] - max_vin) for i in [i for i in range(J) if av[i][n] == 1])
            for i in [i for i in range(J) if av[i][n] == 1]:
                Pin[i, n, r] = np.exp(Vin[i, n, r] - max_vin) / denom

    # Average probabilities across all draws
    P_avg = np.mean(Pin, axis=2)

    # Compute the overall log-likelihood
    biog_obj = 0
    for n in range(N):
        for i in [i for i in range(J) if av[i][n] == 1]:
            log_prob = -100 if P_avg[i, n] == 0 else np.log(P_avg[i, n])
            biog_obj += y[i, n] * log_prob

    return biog_obj


def compute_biog_loglike_latent2classes(x, y, av, beta_orig, class_1_ks, class_2_ks, extra_inds):
    ext_x = x.copy()

    # Extend x for latent classes
    for pair in extra_inds:
        old_idx = pair[0]  # Adjusted for 0-based indexing in Python
        slice_to_add = ext_x[:, :, old_idx]  # Extract the slice
        slice_to_add = np.reshape(slice_to_add, (ext_x.shape[0], ext_x.shape[1], 1))  # Reshape to (I, J, 1)
        ext_x = np.concatenate((ext_x, slice_to_add), axis=2)  # Append along the third dimension

    J, N, _ = ext_x.shape
    C = 2
    Vin = np.zeros((J, N, C))
    Pin = np.zeros((J, N, C))

    beta = beta_orig.copy()  # Ensure beta is not modified outside the function
    latent_beta = beta[-1]
    beta = beta[:-1]

    denom = np.exp(latent_beta) + 1
    P1 = np.exp(latent_beta) / denom
    P2 = 1 / denom

    prob = [P1, P2]

    for n in range(N):
        for i in [i for i in range(J) if av[i][n] == 1]:
            for c in range(C):
                if c == 0:
                    Vin[i, n, 0] = sum(beta[k] * ext_x[i, n, k] for k in class_1_ks)
                else:
                    Vin[i, n, 1] = sum(beta[k] * ext_x[i, n, k] for k in class_2_ks)

    for n in range(N):
        for c in range(C):
            # Find the maximum Vin value for the current n and c (log-sum-exp trick) for valid indices
            max_vin = max(Vin[i, n, c] for i in [i for i in range(J) if av[i][n] == 1])

            # Compute the denominator using the log-sum-exp trick for valid indices
            denom = sum(np.exp(Vin[i, n, c] - max_vin) for i in [i for i in range(J) if av[i][n] == 1])

            # Compute Pin, applying the same shift
            for i in [i for i in range(J) if av[i][n] == 1]:
                Pin[i, n, c] = np.exp(Vin[i, n, c] - max_vin) / denom

    # Compute the overall log-likelihood
    biog_obj = 0
    for n in range(N):
        for i in [i for i in range(J) if av[i][n] == 1]:
            sum_prob = sum(Pin[i, n, c] * prob[c] for c in range(C))
            log_sum = -100 if sum_prob == 0 else np.log(sum_prob)
            biog_obj += y[i, n] * log_sum

    return biog_obj


def compute_biog_loglike_latent3classes(x, y, av, beta_orig, class_1_ks, class_2_ks, class_3_ks, biogeme=False):
    J, N, K = x.shape
    C = 3
    Vin = np.zeros((J, N, C))
    Pin = np.zeros((J, N, C))

    beta = beta_orig.copy()  # Ensure beta is not modified outside the function

    latent_beta = [beta[-2], beta[-1]]
    beta = beta[:(len(beta_orig) - 2)]

    if biogeme:
        denom = np.exp(latent_beta[0]) + np.exp(latent_beta[1]) + 1
        P1 = np.exp(latent_beta[0]) / denom
        P2 = np.exp(latent_beta[1]) / denom
        P3 = 1 / denom
    else:
        P1 = latent_beta[0]
        P2 = latent_beta[1] - latent_beta[0]
        P3 = 1 - latent_beta[1]

    prob = [P1, P2, P3]

    for n in range(N):
        for i in [i for i in range(J) if av[i][n] == 1]:
            for c in range(C):
                if c == 0:
                    Vin[i, n, 0] = sum(beta[k] * x[i, n, k] for k in class_1_ks)
                elif c == 1:
                    Vin[i, n, 1] = sum(beta[k] * x[i, n, k] for k in class_2_ks)
                elif c == 2:
                    Vin[i, n, 2] = sum(beta[k] * x[i, n, k] for k in class_3_ks)

    for n in range(N):
        for c in range(C):
            # Find the maximum Vin value for the current n and c (log-sum-exp trick) for valid indices
            max_vin = max(Vin[i, n, c] for i in [i for i in range(J) if av[i][n] == 1])

            # Compute the denominator using the log-sum-exp trick for valid indices
            denom = sum(np.exp(Vin[i, n, c] - max_vin) for i in [i for i in range(J) if av[i][n] == 1])

            # Compute Pin, applying the same shift
            for i in [i for i in range(J) if av[i][n] == 1]:
                Pin[i, n, c] = np.exp(Vin[i, n, c] - max_vin) / denom

    # Compute the overall log-likelihood
    biog_obj = 0
    for n in range(N):
        for i in [i for i in range(J) if av[i][n] == 1]:
            sum_prob = sum(Pin[i, n, c] * prob[c] for c in range(C))
            log_sum = -100 if sum_prob == 0 else np.log(sum_prob)
            biog_obj += y[i, n] * log_sum

    return biog_obj


def choice_data_netherlands(size, df_cut, latent, epsilon, intercept=True, cost=True, probit=False,
                            michels_classes=False, toms_extremists=False, michelSeed=263, tomSeed=4,
                            starting_point=None, R=None):
    J = 2
    K = 3
    N = size
    x = np.zeros((J, size, K))
    for n in range(size):
        # intercept for rail
        x[0, n, 0] = 0
        x[1, n, 0] = 1
        # beta cost
        x[0, n, 1] = df_cut['car_cost'].values[n]
        x[1, n, 1] = df_cut['rail_cost'].values[n]
        # beta time
        x[0, n, 2] = df_cut['car_time'].values[n]
        x[1, n, 2] = df_cut['rail_time'].values[n]

    av = np.ones((J, N), dtype=int)
    y = np.array([df_cut.choice == i for i in range(J)]).astype(int)

    # if toms_extremists:
    #
    #     # # intercept for rail
    #     # x[0, n, 0] = 0
    #     # x[1, n, 0] = 1
    #     # # beta cost
    #     # x[0, n, 1] = df_cut['car_cost'].values[n]
    #     # x[1, n, 1] = df_cut['rail_cost'].values[n]
    #     # # beta time
    #     # x[0, n, 2] = df_cut['car_time'].values[n]
    #     # x[1, n, 2] = df_cut['rail_time'].values[n]
    #
    #     # y[0, n] = car yes / no
    #     # y[1, n] = rail yes / no
    #
    #     N = len(df_cut)  # maybe works
    #
    #     # dont forget that our version needs a bunch! of x adjustement too.
    #     # basically an additional
    #     # x[1, n, 3], x[2, n, 3] for class2 beta_time and
    #     # x[1, n, 4], x[2, n, 4] for class3 beta_cost
    #
    #     # now lets do our version
    #
    #     class_2_goal = math.ceil(N * 0.3)  # 30% of people
    #     class_3_goal = math.ceil(N * 0.2)  # 20% of people
    #
    #     class_2_counter = 0
    #     class_3_counter = 0
    #
    #     n_ind = 0
    #
    #     random.seed(tomSeed)
    #     # 2 gives at least cool probs
    #     # 4 is a more typical example
    #     # ah but I think what we see is that it struggles more! haha
    #
    #     # Generate a list of integers from 0 to 227
    #     numbers = list(range(228))
    #     # Shuffle the list in place
    #     random.shuffle(numbers)
    #
    #     while (class_2_counter < class_2_goal or class_3_counter < class_3_goal) and (n_ind <= N):
    #         n = numbers[n_ind]
    #         reassigned = False
    #         n_car_time = x[0, n, 2]
    #         n_rail_time = x[1, n, 2]
    #         n_car_cost = x[0, n, 1]
    #         n_rail_cost = x[1, n, 1]
    #
    #         if class_2_counter < class_2_goal:
    #             if (n_car_time < n_rail_time and y[0, n] == 1) or (n_rail_time < n_car_time and y[1, n] == 1):
    #                 # if he chose less time alternative and there is not enough class 2's: make him class 2
    #                 if n_car_time < n_rail_time:
    #                     y[0, n] = 0
    #                     y[1, n] = 1
    #                 else:  # make him choose more time intensive option
    #                     y[0, n] = 1
    #                     y[1, n] = 0
    #                 class_2_counter += 1
    #                 reassigned = True
    #         if not reassigned:
    #             if class_3_counter < class_3_goal:
    #                 if (n_car_cost < n_rail_cost and y[0, n] == 1) or (n_rail_cost < n_car_cost and y[1, n] == 1):
    #                     # if he chose less cost alternative and there is not enough class 3's: make him class 3
    #                     if n_car_cost < n_rail_cost:
    #                         y[0, n] = 0
    #                         y[1, n] = 1
    #                     else:  # make him choose more time intensive option
    #                         y[0, n] = 1
    #                         y[1, n] = 0
    #                     class_3_counter += 1
    #         n_ind += 1
    #     print("# class1 = ", N - class_2_counter - class_3_counter)
    #     print("# class2 = ", class_2_counter)
    #     print("# class3 = ", class_3_counter)
    # elif michels_classes:
    #     # 12360 too very long
    #     #
    #     failure = True
    #     seed = michelSeed
    #     # seed 263 is bad one hihi
    #     # 3 is pretty good? (really good LL) and so is 5 (ok LL)?
    #     tot_betas = np.zeros(5)
    #     tot_LL = 0
    #     runNuM = 1
    #     seedEnd = seed + runNuM - 1
    #
    #     while seed <= seedEnd:
    #         seed += 1
    #         print("seed = ", seed)
    #         # first lets get the parameters without latent classes to get logit estimates
    #         # use these to compute highest utility
    #         Logit_beta, Logit_loglike, _, _ = biogeme_estimate_beta_nether(df_cut,
    #                                                                        latent=0,
    #                                                                        intercept=intercept,
    #                                                                        cost=cost)
    #         # print("Logit Beta = ", Logit_beta)
    #         N = len(df_cut)  # maybe works
    #
    #         population = [1, 2, 3]
    #         weights = [0.5, 0.3, 0.2]  # Probabilities for each number
    #         # crazy results: 3, 4
    #         # better: 14
    #         random.seed(seed)
    #         class1ers = 0
    #         class2ers = 0
    #         class3ers = 0
    #         for n in range(N):
    #             # Draw a random number from 1 to 3
    #             n_class = random.choices(population, weights=weights, k=1)[0]
    #
    #             # now assign y based on the highest utility given the class
    #             # aah look, for this we need to have estimated a beta already.
    #             # this is why it was done the way it was before.. smert
    #
    #             # class 2 has only time, class 3 has only cost lets say
    #             # (to keep it a bit in line with our classes)
    #
    #             if n_class == 1:
    #                 class1ers += 1
    #                 # intercept for rail: 0, beta cost: 1, beta time: 2
    #                 u_0 = x[0, n, 0] * Logit_beta[0] + x[0, n, 1] * Logit_beta[1] + x[0, n, 2] * Logit_beta[2]
    #                 u_1 = x[1, n, 0] * Logit_beta[0] + x[1, n, 1] * Logit_beta[1] + x[1, n, 2] * Logit_beta[2]
    #                 if u_0 > u_1:
    #                     y[0, n] = 1
    #                     y[1, n] = 0
    #                 else:
    #                     y[0, n] = 0
    #                     y[1, n] = 1
    #             elif n_class == 2:
    #                 class2ers += 1
    #                 # intercept for rail: 0, beta time: 2
    #                 u_0 = x[0, n, 0] * Logit_beta[0] + x[0, n, 2] * Logit_beta[2]
    #                 u_1 = x[1, n, 0] * Logit_beta[0] + x[1, n, 2] * Logit_beta[2]
    #                 if u_0 > u_1:
    #                     y[0, n] = 1
    #                     y[1, n] = 0
    #                 else:
    #                     y[0, n] = 0
    #                     y[1, n] = 1
    #             else:
    #                 class3ers += 1
    #                 # intercept for rail: 0, beta cost: 1
    #                 u_0 = x[0, n, 0] * Logit_beta[0] + x[0, n, 1] * Logit_beta[1]
    #                 u_1 = x[1, n, 0] * Logit_beta[0] + x[1, n, 1] * Logit_beta[1]
    #                 if u_0 > u_1:
    #                     y[0, n] = 1
    #                     y[1, n] = 0
    #                 else:
    #                     y[0, n] = 0
    #                     y[1, n] = 1
    #
    #         print("# class1 = ", class1ers)
    #         print("# class2 = ", class2ers)
    #         print("# class3 = ", class3ers)

        #     # update the choice column in df
        #     df_cut['choice'] = np.where(y[0, :] == 1, 0, 1)
        #
        #     biog_beta, biog_loglike, beta_confs, loglike_conf = biogeme_estimate_beta_nether(df_cut,
        #                                                                                      latent=latent,
        #                                                                                      intercept=intercept,
        #                                                                                      cost=cost,
        #                                                                                      toms_extremists=toms_extremists,
        #                                                                                      michels_classes=michels_classes)
        #     tot_betas += biog_beta
        #     tot_LL += biog_loglike
        #     failure = False
        #     for i in range(len(biog_beta)):
        #         if abs(biog_beta[i]) > 15:
        #             failure = True
        #     if not failure:
        #         print("NOT A FAILURE AT seed = ", seed)
        # av_betas = tot_betas / runNuM
        # denom = np.exp(av_betas[3]) + np.exp(av_betas[4]) + 1
        # P11 = np.exp(av_betas[3]) / denom
        # P22 = np.exp(av_betas[4]) / denom
        # P33 = 1 / denom
        # av_probs = [P11, P22, P33]
        # av_LL = tot_LL / runNuM
        # print("avg. betas = ", av_betas)
        # print("avg. probs = ", av_probs)
        # print("avg. LL = ", av_LL)

    # update the choice column in df
    # df_cut['choice'] = np.where(y[0, :] == 1, 0, 1)

    # and only NOW estimate
    start_time_biog = time.time()
    try:
        biog_beta, biog_loglike, beta_confs, loglike_conf = biogeme_estimate_beta_nether(df_cut,
                                                                                         latent=latent,
                                                                                         intercept=intercept,
                                                                                         cost=cost,
                                                                                         toms_extremists=toms_extremists,
                                                                                         michels_classes=michels_classes,
                                                                                         starting_point=starting_point,
                                                                                         R=R,
                                                                                         pandaSeed=pandaSeed)
    except RuntimeError:
        print("BioGeme crashed")
        biog_beta = [5, 5, 5] + [5 for l in range(len(latent - 1))]
        biog_loglike = 5
        loglike_conf = [-5, 5]
        beta_confs = dict()
        for k in range(len(biog_beta)):
            beta_confs[k] = [-5, 5]
    time_biog = time.time() - start_time_biog

    print(f"Biogeme latent estimated biog_beta = {biog_beta} with biog_loglike = {biog_loglike} in {time_biog} seconds")

    # starttime = time.time()
    # Logit_beta, Logit_loglike, _, _ = biogeme_estimate_beta_nether(df_cut,
    #                                                                latent=0,
    #                                                                intercept=intercept,
    #                                                                cost=cost)
    # enddtime = time.time() - starttime
    #
    # print(
    #     f"Biogeme NONlatent estimated biog_beta = {Logit_beta} with biog_loglike = {Logit_loglike} in {enddtime} seconds")

    # File extensions to clean up
    extensions = ["*.iter", "*.html", "*.pickle"]

    # Iterate over each extension and remove matching files
    for ext in extensions:
        files = glob.glob(ext)
        for file in files:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass

    return x, y, J, K, biog_beta, biog_loglike, beta_confs, loglike_conf, time_biog, av


def choice_data_netherlands_mixed(size, df_cut, R):
    global biogeme_sim_loglike
    print("Estimate Biogeme model")
    start_time_biog = time.time()
    # R_biog = 1000
    biog_beta, biog_loglike, beta_confs, loglike_conf, df = biogeme_estimate_beta_nether_mixed(df_cut, R, pandaSeed=pandaSeed)
    print(f"Time to estimate Biogeme model = {time.time() - start_time_biog}s")
    biog_time = time.time() - start_time_biog
    # print(f"BioG: Simulating likelihood with {nbOfDraws_sLL_Biog} draws...")
    start_time = time.time()
    biogeme_sim_loglike = simulate_likelihood_mixed_nether(df_cut, biog_beta, numberOfDraws=nbOfDraws_sLL_Biog)
    print(f"Done ({round((time.time() - start_time), 2)}s)")
    print(f"Biogeme Biog sim. loglike = {biogeme_sim_loglike}")

    time_biog = time.time() - start_time_biog
    # biog_beta = [20]
    # biog_loglike = [20]
    # beta_confs = [[20, 20], [20, 20], [20, 20], [20, 20], [20, 20]]
    # loglike_conf = [[20, 20], [20, 20], [20, 20], [20, 20], [20, 20]]
    print(f"Biogeme estimated biog_beta = {biog_beta} with biog_loglike = {biog_loglike} in {time_biog} seconds")

    J = 2
    K = 4
    x = np.zeros((J, size, K))
    for n in range(size):
        # intercept for rail
        x[0, n, 0] = 0
        x[1, n, 0] = 1
        # beta cost
        x[0, n, 1] = df_cut['car_cost'].values[n]
        x[1, n, 1] = df_cut['rail_cost'].values[n]
        # beta time
        x[0, n, 2] = df_cut['car_time'].values[n]
        x[1, n, 2] = df_cut['rail_time'].values[n]
        # beta time_s
        x[0, n, 3] = df_cut['car_time'].values[n]
        x[1, n, 3] = df_cut['rail_time'].values[n]

    y = np.array([df_cut.choice == i for i in range(J)]).astype(int)

    return x, y, J, K, biog_beta, biog_loglike, beta_confs, loglike_conf, time_biog


def choice_data_london(size, df_cut, latent, michels_classes=False, toms_extremists=False, michelSeed=0,
                       tomSeed=0, one_param=False, probit=False, starting_point=None, R=None):
    J = 4
    # 0: Walk
    # 1: Bike
    # 2: PT
    # 3: Car

    N = len(df_cut)  # maybe works
    
    K = 5
    
    x = np.zeros((J, size, K))
    for n in range(size):
        # intercept bike
        x[0, n, 0] = 0
        x[1, n, 0] = 1
        x[2, n, 0] = 0
        x[3, n, 0] = 0
        # intercept car
        x[0, n, 1] = 0
        x[1, n, 1] = 0
        x[2, n, 1] = 0
        x[3, n, 1] = 1
        # intercept pt
        x[0, n, 2] = 0
        x[1, n, 2] = 0
        x[2, n, 2] = 1
        x[3, n, 2] = 0
        # beta cost
        x[0, n, 3] = 0
        x[1, n, 3] = 0
        x[2, n, 3] = df_cut['cost_transit'].values[n]
        x[3, n, 3] = df_cut['car_cost'].values[n]
        # beta time
        x[0, n, 4] = df_cut['dur_walking'].values[n]
        x[1, n, 4] = df_cut['dur_cycling'].values[n]
        x[2, n, 4] = df_cut['rail_time'].values[n]
        x[3, n, 4] = df_cut['dur_driving'].values[n]

    #                    Value  Rob. Std err  Rob. t-test  Rob. p-value
    # ASC_Bike       -3.433393      0.244194   -14.060101  0.000000e+00
    # ASC_Car        -1.036667      0.208348    -4.975655  6.502737e-07
    # ASC_PB         -0.413903      0.126875    -3.262290  1.105160e-03
    # Beta_cost      -0.217605      0.030447    -7.146905  8.875123e-13
    # Beta_time      -5.053744      0.659223    -7.666217  1.776357e-14
    # Z1_Beta_time_S -0.783698      1.047497    -0.748163  4.543621e-01

    # or
    #                    Value  Rob. Std err  Rob. t-test  Rob. p-value
    # ASC_Bike       -3.403574      0.217726   -15.632339  0.000000e+00
    # ASC_Car        -0.999725      0.168735    -5.924826  3.126280e-09
    # ASC_PB         -0.402967      0.117863    -3.418958  6.286141e-04
    # Beta_cost      -0.216995      0.030199    -7.185558  6.692424e-13
    # Beta_time      -4.882945      0.413365   -11.812668  0.000000e+00
    # Z1_Beta_cost_S  0.001563      0.006335     0.246671  8.051631e-01

    # we'll tackle the following specifications:
    # most importantly:
    # C = 1025: class 1 mixed costs, class 2 new beta time
    # C = 1032: class 1 mixed costs, class 2 new beta time, class 3 lazy
    # C = 1033: class 1 mixed costs, class 2 new beta time, class 3 no car

    # C = 1020: class 1 mixed time, class 2 new ASCs and no car
    # C = 1021: class 1 mixed time, class 2 new ASCs and lazy
    # C = 1022: class 1 mixed time, class 2 new ASCs
    # C = 1023: class 1 mixed time, class 2 no car
    # C = 1024: class 1 mixed time, class 2 lazy
    # C = 1030: class 1 mixed time, class 2 new ASCs, class 3 no car
    # C = 1031: class 1 mixed time, class 2 new ASCs, class 3 lazy

    # Implement these
    # C = 1034: class 1 mixed time, class 2 new beta cost, class 3 lazy
    # C = 1026: class 1 mixed time, class 2 new beta cost

    # all checked and verified

    seed_nr = pandaSeed + 1
    random.seed(seed_nr)
    np.random.seed(seed_nr)

    # ben remember to treat 25 differently.
    # and 30s we'll do after

    if latent == 2:
        seed = 1
        runNuM = 1
        seedEnd = seed + runNuM - 1

        while seed <= seedEnd:
            seed += 1
            # print("seed = ", seed)
            # first lets get the parameters without latent classes to get logit estimates
            # use these to compute highest utility
            df_cut_copy = copy.copy(df_cut)
            Logit_beta, Logit_loglike, _, _, timestamp = biogeme_estimate_beta_london(df_cut_copy)

            # Remove output files
            time.sleep(2)
            # File extensions to clean up
            extensions = ["*.iter", "*.html", "*.pickle"]

            # Iterate over each extension and remove matching files
            for ext in extensions:
                files = glob.glob(ext)
                for file in files:
                    try:
                        os.remove(file)
                    except FileNotFoundError:
                        pass

            # print("Logit Beta = ", Logit_beta)
            N = len(df_cut)  # maybe works

            population = [1, 2]
            weights = [0.7, 0.3]  # Probabilities for each number
            # crazy results: 3, 4
            # better: 14
            random.seed(seed)
            class1ers = 0
            class2ers = 0

            Logit_beta_C2 = copy.copy(Logit_beta)
            Logit_beta_C2.append(Logit_beta[4] / 5)

            print("Logit_beta", Logit_beta)
            print("Logit_beta_C2", Logit_beta_C2)

            for n in range(N):
                # Draw a random number from 1 to 3
                n_class = random.choices(population, weights=weights, k=1)[0]

                # now assign y based on the highest utility given the class
                # aah look, for this we need to have estimated a beta already.
                # this is why it was done the way it was before.. smert

                # class 2 has only time, class 3 has only cost lets say
                # (to keep it a bit in line with our classes

                if n_class == 1:
                    # k = 0: ASC_Bike
                    # k = 1: ASC_Car
                    # k = 2: ASC_PT
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time C2
                    class1ers += 1
                    u_0 = sum(x[0, n, i] * Logit_beta_C2[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * Logit_beta_C2[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * Logit_beta_C2[i] for i in [0, 1, 2, 3, 4])
                    u_3 = sum(x[3, n, i] * Logit_beta_C2[i] for i in [0, 1, 2, 3, 4])

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2, u_3]

                    # Find the index of the maximum utility
                    best_alternative = max(range(4), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'travel_mode'] = best_alternative + 1

                elif n_class == 2:
                    # k = 0: ASC_Bike
                    # k = 1: ASC_Car
                    # k = 2: ASC_PT
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    class2ers += 1
                    u_0 = sum(x[0, n, i] * Logit_beta_C2[i] for i in [0, 1, 2, 3]) + x[0, n, 4] * Logit_beta_C2[5]
                    u_1 = sum(x[1, n, i] * Logit_beta_C2[i] for i in [0, 1, 2, 3]) + x[1, n, 4] * Logit_beta_C2[5]
                    u_2 = sum(x[2, n, i] * Logit_beta_C2[i] for i in [0, 1, 2, 3]) + x[2, n, 4] * Logit_beta_C2[5]
                    u_3 = sum(x[3, n, i] * Logit_beta_C2[i] for i in [0, 1, 2, 3]) + x[3, n, 4] * Logit_beta_C2[5]

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2, u_3]

                    # Find the index of the maximum utility
                    best_alternative = max(range(4), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'travel_mode'] = best_alternative + 1
            print("# class1 = ", class1ers)
            print("# class2 = ", class2ers)

    if 1020 <= latent <= 1025:
        seed = 1
        runNuM = 1
        seedEnd = seed + runNuM - 1

        if latent <= 1024:
            base_beta = [-3.433393, -1.036667, -0.413903, -0.217605, -5.053744, -0.783698]
        else:
            base_beta = [-3.4035735503211497, -0.9997250334091937, -0.40296715846675113, -0.21699454211574326, -4.882944640779013, 0.0015626834662186557]

        #                    Value  Rob. Std err  Rob. t-test  Rob. p-value
        # ASC_Bike       -3.403574      0.217726   -15.632339  0.000000e+00
        # ASC_Car        -0.999725      0.168735    -5.924826  3.126280e-09
        # ASC_PB         -0.402967      0.117863    -3.418958  6.286141e-04
        # Beta_cost      -0.216995      0.030199    -7.185558  6.692424e-13
        # Beta_time      -4.882945      0.413365   -11.812668  0.000000e+00
        # Z1_Beta_cost_S  0.001563      0.006335     0.246671  8.051631e-01

        while seed <= seedEnd:
            seed += 1

            # define the new ASCs (consider new ASC balances)
            new_beta = copy.copy(base_beta)
            # 0 bike
            # 1 car
            # 2 PB
            # we are just copying. Keep them all even if we
            # dont use them, adjust it with indices later.

            if latent == 1020:  # new bike, PB. Class 2 has no car
                # so make class 1 have more car people?
                # so make class 2 like cars less already? Maybe
                new_beta[0] = base_beta[0] * 2  # bike more attractive
                new_beta[2] = base_beta[2] * 2  # PB more attractiveness
            if latent == 1021:  # new PB only. Class 2 hates walk and bike
                new_beta[2] = base_beta[2] * 3  # increase PB attractiveness
            if latent == 1022:  # new bike, car, PB. Make class 2 eco-friendly
                new_beta[0] = base_beta[0] * 3  # incr. bike attractiveness
                new_beta[1] = base_beta[1] * 0.2  # reduce car attractiveness
                new_beta[2] = base_beta[2] * 3  # incr. PB attractiveness
            if latent == 1025:  # new beta time
                new_beta[4] = base_beta[4] * 0.2  # as done before, they care LESS about time

            # worth mentioning: 1023 and 1024 have no new params in latent

            population = [1, 2]
            weights = [0.7, 0.3]
            random.seed(seed)
            class1ers = 0
            class2ers = 0

            for n in range(N):
                # Draw a random number from 1 to 2
                n_class = random.choices(population, weights=weights, k=1)[0]

                if n_class == 1:
                    # k = 0: ASC_Bike
                    # k = 1: ASC_Car
                    # k = 2: ASC_PT
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_C2 or time_STD or cost_STD
                    class1ers += 1
                    # all have the same class 1, except 1025
                    if latent <= 1024:
                        u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[0, n, 4] * base_beta[5] * np.random.normal(0, 1)
                        u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[1, n, 4] * base_beta[5] * np.random.normal(0, 1)
                        u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[2, n, 4] * base_beta[5] * np.random.normal(0, 1)
                        u_3 = sum(x[3, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[3, n, 4] * base_beta[5] * np.random.normal(0, 1)
                    else:  # 1025 mixes costs
                        u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[0, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[1, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[2, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_3 = sum(x[3, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[3, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2, u_3]

                    # Find the index of the maximum utility
                    best_alternative = max(range(4), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'travel_mode'] = best_alternative + 1

                elif n_class == 2:
                    class2ers += 1
                    u_0 = sum(x[0, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_3 = sum(x[3, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])

                    # C = 1033: class 1 mixed costs, class 2 new beta time, class 3 no car
                    # C = 1030: class 1 mixed time, class 2 new ASCs, class 3 no car
                    # C = 1031: class 1 mixed time, class 2 new ASCs, class 3 lazy

                    # Put the utilities into a list
                    if latent == 1020 or latent == 1023:  # no car
                        utilities = [u_0, u_1, u_2, -9000000]
                    elif latent == 1021 or latent == 1024:  # lazy
                        utilities = [-9000000, -9000000, u_2, u_3]
                    else:  # 1022, 1025
                        utilities = [u_0, u_1, u_2, u_3]

                    # Find the index of the maximum utility
                    best_alternative = max(range(4), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'travel_mode'] = best_alternative + 1
            print("# class1 = ", class1ers)
            print("# class2 = ", class2ers)
    if 1030 <= latent <= 1033:
        seed = 1
        runNuM = 1
        seedEnd = seed + runNuM - 1

        if latent <= 1031:
            base_beta = [-3.433393, -1.036667, -0.413903, -0.217605, -5.053744, -0.783698]
        else:
            base_beta = [-3.4035735503211497, -0.9997250334091937, -0.40296715846675113, -0.21699454211574326,
                         -4.882944640779013, 0.0015626834662186557]

        #                    Value  Rob. Std err  Rob. t-test  Rob. p-value
        # ASC_Bike       -3.403574      0.217726   -15.632339  0.000000e+00
        # ASC_Car        -0.999725      0.168735    -5.924826  3.126280e-09
        # ASC_PB         -0.402967      0.117863    -3.418958  6.286141e-04
        # Beta_cost      -0.216995      0.030199    -7.185558  6.692424e-13
        # Beta_time      -4.882945      0.413365   -11.812668  0.000000e+00
        # Z1_Beta_cost_S  0.001563      0.006335     0.246671  8.051631e-01

        while seed <= seedEnd:
            seed += 1

            # define the new ASCs (consider new ASC balances)
            new_beta = copy.copy(base_beta)
            # 0 bike
            # 1 car
            # 2 PB
            # we are just copying. Keep them all even if we
            # dont use them, adjust it with indices later.

            # C = 1033: class 1 mixed costs, class 2 new beta time, class 3 no car
            # C = 1030: class 1 mixed time, class 2 new ASCs, class 3 no car
            # C = 1031: class 1 mixed time, class 2 new ASCs, class 3 lazy
            # C = 1032: class 1 mixed costs, class 2 new beta time, class 3 lazy

            if latent == 1030:  # new bike, car, PB. class 3 no car
                # thus make class 2 carlovers
                new_beta[0] = base_beta[0] * 0.2  # decr bike attractiveness
                new_beta[1] = base_beta[1] * 3  # incr car attractiveness
                new_beta[2] = base_beta[2] * 0.2  # incr. PB attractiveness
            if latent == 1031:  # new bike, car, PB. class 3 lazy
                # thus make class 3 eco
                new_beta[0] = base_beta[0] * 3  # incr. bike attractiveness
                new_beta[1] = base_beta[1] * 0.2  # reduce car attractiveness
                new_beta[2] = base_beta[2] * 3  # incr. PB attractiveness
            if latent == 1032 or latent == 1033:  # new beta time
                new_beta[4] = base_beta[4] * 0.2  # as done before, they care LESS about time

            # worth mentioning: 1023 and 1024 have no new params in latent

            population = [1, 2, 3]
            weights = [0.5, 0.3, 0.2]
            random.seed(seed)
            class1ers = 0
            class2ers = 0
            class3ers = 0

            for n in range(N):
                # Draw a random number from 1 to 2
                n_class = random.choices(population, weights=weights, k=1)[0]

                if n_class == 1:
                    # k = 0: ASC_Bike
                    # k = 1: ASC_Car
                    # k = 2: ASC_PT
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_C2 or time_STD or cost_STD
                    class1ers += 1
                    # all have the same class 1, except 1025
                    if latent <= 1031:
                        u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[0, n, 4] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[1, n, 4] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[2, n, 4] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_3 = sum(x[3, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[3, n, 4] * base_beta[
                            5] * np.random.normal(0, 1)
                    else:  # 1025 mixes costs
                        u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[0, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[1, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[2, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)
                        u_3 = sum(x[3, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[3, n, 3] * base_beta[
                            5] * np.random.normal(0, 1)

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2, u_3]

                    # Find the index of the maximum utility
                    best_alternative = max(range(4), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'travel_mode'] = best_alternative + 1

                elif n_class == 2:
                    class2ers += 1
                    u_0 = sum(x[0, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_3 = sum(x[3, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2, u_3]

                    # Find the index of the maximum utility
                    best_alternative = max(range(4), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'travel_mode'] = best_alternative + 1
                elif n_class == 3:
                    class2ers += 3
                    u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])
                    u_3 = sum(x[3, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])

                    # C = 1033: class 1 mixed costs, class 2 new beta time, class 3 no car
                    # C = 1030: class 1 mixed time, class 2 new ASCs, class 3 no car
                    # C = 1031: class 1 mixed time, class 2 new ASCs, class 3 lazy

                    # Put the utilities into a list
                    if latent == 1030 or latent == 1033:  # no car
                        utilities = [u_0, u_1, u_2, -9000000]
                    elif latent == 1031 or latent == 1032:  # lazy
                        utilities = [-9000000, -9000000, u_2, u_3]

                    # Find the index of the maximum utility
                    best_alternative = max(range(4), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'travel_mode'] = best_alternative + 1
            print("# class1 = ", class1ers)
            print("# class2 = ", class2ers)
            print("# class3 = ", class3ers)

    # and ONLY NOW compute y
    y = np.array([df_cut.travel_mode == i for i in range(1, J + 1)]).astype(int)
    av = np.ones((J, size), dtype=int)

    # and only NOW estimate
    start_time_biog = time.time()
    biog_beta, biog_loglike, beta_confs, loglike_conf, signi, timestamp = biogeme_estimate_beta_london_latent(df_cut,
                                                                                                              latent,
                                                                                                              michels_classes,
                                                                                                              toms_extremists,
                                                                                                              starting_point,
                                                                                                              R=R,
                                                                                                              pandaSeed=pandaSeed)

    time_biog = time.time() - start_time_biog
    print(f"Biogeme estimated biog_beta = {biog_beta} with biog_loglike = {biog_loglike} in {time_biog} seconds")

    # Remove output files
    time.sleep(2)
    # try:
    #     os.remove(f"lpmc_latent_{timestamp}.html")
    #     os.remove(f"lpmc_latent_{timestamp}.pickle")
    #     os.remove(f"__lpmc_latent_{timestamp}.iter")
    # except FileNotFoundError:
    #     os.remove(f"lpmc_latent_{timestamp}~00.html")
    #     os.remove(f"lpmc_latent_{timestamp}~00.html")
    #     os.remove(f"lpmc_latent_{timestamp}~00.pickle")
    #     os.remove(f"__lpmc_latent_{timestamp}~00.iter")
    
    # File extensions to clean up
    extensions = ["*.iter", "*.html", "*.pickle"]

    # Iterate over each extension and remove matching files
    for ext in extensions:
        files = glob.glob(ext)
        for file in files:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass

    # print(f"Biogeme estimated biog_beta =  {biog_beta} with biog_loglike = {biog_loglike}")

    return x, y, J, K, biog_beta, biog_loglike, beta_confs, loglike_conf, time_biog, av


def choice_data_swissmetro(size, df_cut, latent=1, michels_classes=False, toms_extremists=False, michelSeed=0,
                           tomSeed=0, starting_point=None, R=None):
    N = len(df_cut)

    # loading custom attributes
    _, _, _, _, _, _ = biogeme_estimate_swissmetro(df_cut, 0, False, False, True, pandaSeed=pandaSeed)

    J = 3
    # 0: Train
    # 1: SM
    # 2: Car

    K = 5
    # k = 0: ASC_Car
    # k = 1: ASC_Train
    # k = 2: Beta_time
    # k = 3: Beta_cost
    # k = 4: Beta_headway

    x = np.zeros((J, size, K))
    for n in range(size):
        # k = 1: ASC_Car
        # k = 2: ASC_Train
        # k = 3: Beta_time
        # k = 4: Beta_cost
        # k = 5: Beta_headway
        # k = 6: Prob1

        # train, SM, car

        # intercept car
        x[0, n, 0] = 0
        x[1, n, 0] = 0
        x[2, n, 0] = 1
        # intercept train
        x[0, n, 1] = 1
        x[1, n, 1] = 0
        x[2, n, 1] = 0
        # beta cost
        x[0, n, 2] = df_cut['TRAIN_COST_SCALED'].values[n]
        x[1, n, 2] = df_cut['SM_COST_SCALED'].values[n]
        x[2, n, 2] = df_cut['CAR_CO_SCALED'].values[n]
        # beta headway
        x[0, n, 3] = df_cut['TRAIN_HE_SCALED'].values[n]
        x[1, n, 3] = df_cut['SM_HE_SCALED'].values[n]
        x[2, n, 3] = 0
        # beta time
        x[0, n, 4] = df_cut['TRAIN_TT_SCALED'].values[n]
        x[1, n, 4] = df_cut['SM_TT_SCALED'].values[n]
        x[2, n, 4] = df_cut['CAR_TT_SCALED'].values[n]

    #                 Value  Rob. Std err  Rob. t-test  Rob. p-value
    # ASC_CAR     -0.123392      0.138956    -0.887989  3.745464e-01
    # ASC_TRAIN   -0.012224      0.205194    -0.059572  9.524966e-01
    # B_COST      -1.514699      0.201045    -7.534138  4.907186e-14
    # B_HE        -5.181105      2.586589    -2.003064  4.517039e-02
    # B_TIME      -2.205778      0.273753    -8.057560  8.881784e-16
    # Z1_B_TIME_S  1.186727      0.298090     3.981104  6.859593e-05
    #
    # Loglike = -747.8505023
    # biog_beta = [-0.12339153095665459, -0.012223798401333768, -1.5146992381080648, -5.181104734705727, -2.2057779141945284, 1.186726809265002]

    # lets gather our thoughts. We will implement a synthetic choice generation here, using the 1000, 1000 mixed logit
    # with differnet seeds as a base model. Probably not worse or better than anything else.

    # we'll tackle the following specifications:
    # C = 1020: class 1 mixed time, class 2 new ASCs and no car
    # C = 1021: class 1 mixed time, class 2 new ASCs and no SM
    # C = 1022: class 1 mixed time, class 2 new ASCs
    # C = 1023: class 1 mixed time, class 2 no car
    # C = 1024: class 1 mixed time, class 2 no SM
    # ( C = 1025: class 1 mixed time, class 2 no car but with ORGANIC choices, excluded from this conditional )
    # ( C = 1026: class 1 mixed time, class 2 no SM  but with ORGANIC choices, excluded from this conditional )
    # ( C = 1027: class 1 mixed time, class 2 new ASCs but with ORGANIC choices, excluded from this conditional )

    # implement this
    # FOR ALL THREE NEW MODELS
    # 1. except it below from av filtrage (DONE)
    # 2. do estimate (DONE)
    # ---> take the stupid stuff into account...
    # ah or something we could do is to just rename other models
    # to this. 1025, 1026, 1028
    # 3. do simulate (DONE)
    # 4. copy simulate (DONE)
    # 5. do julia (DONE)
    # 6. do NPZstart
    # 7. test in login node
    # ( C = 1025 / 10291: class 1 mixed time, class 2 new time but with ORGANIC choices, excluded from this conditional )
    # ( C = 1026 / 10292: class 1 mixed costs, class 2 new time but with ORGANIC choices, excluded from this conditional )
    # ( C = 1028 / 10293: class 1 mixed headway, class 2 new time but with ORGANIC choices, excluded from this conditional )

    # Implemented, including simulation. 23/25 and 24/26 refer to the same model.

    # C = 1030: class 1 mixed time, class 2 new ASCs, class 3 no car
    # C = 1031: class 1 mixed time, class 2 new ASCs, class 3 no SM

    # df_cut_copy = copy.copy(df_cut)
    # Logit_beta, Logit_loglike, _, _, _, timestamp = biogeme_estimate_swissmetro(df_cut_copy, 0, False, False,
    #                                                                             False)
    # # Remove output files
    # time.sleep(2)
    # # File extensions to clean up
    # extensions = ["*.iter", "*.html", "*.pickle"]
    # # Iterate over each extension and remove matching files
    # for ext in extensions:
    #     files = glob.glob(ext)
    #     for file in files:
    #         try:
    #             os.remove(file)
    #         except FileNotFoundError:
    #             pass

    base_beta = [-0.12339153095665459, -0.012223798401333768, -1.5146992381080648, -5.181104734705727,
                 -2.2057779141945284, 1.186726809265002]
    # ASC_CAR     -0.123392      0.138956    -0.887989  3.745464e-01
    # ASC_TRAIN   -0.012224      0.205194    -0.059572  9.524966e-01
    # B_COST      -1.514699      0.201045    -7.534138  4.907186e-14
    # B_HE        -5.181105      2.586589    -2.003064  4.517039e-02
    # B_TIME      -2.205778      0.273753    -8.057560  8.881784e-16
    # Z1_B_TIME_S  1.186727      0.298090     3.981104  6.859593e-05
    seed_nr = pandaSeed + 1
    random.seed(seed_nr)
    np.random.seed(seed_nr)

    if 1020 <= latent <= 1024:
        seed = michelSeed
        runNuM = 1
        seedEnd = seed + runNuM - 1

        while seed <= seedEnd:
            seed += 1

            # define the new ASCs (ASC_car at ind 0, ASC_train at ind 1)
            new_beta = copy.copy(base_beta)
            if latent == 1020:  # no car
                new_beta[1] = base_beta[1] * 0.33333  # reduce train attractiveness
            if latent == 1021:  # no SM
                new_beta[0] = base_beta[0] * 0.33333  # reduce car attractiveness
            if latent == 1022:  # has both
                new_beta[0] = base_beta[0] * 0.33333  # reduce car attractiveness
                new_beta[1] = base_beta[1] * 0.33333  # reduce train attractiveness

            # worth mentioning: 1023 and 1024 have no new params in latent

            N = len(df_cut)  # maybe works

            population = [1, 2]
            weights = [0.25, 0.75]
            random.seed(seed)
            class1ers = 0
            class2ers = 0

            for n in range(N):
                # Draw a random number from 1 to 2
                n_class = random.choices(population, weights=weights, k=1)[0]

                if n_class == 1:
                    # k = 0: ASC_Car
                    # k = 1: ASC_Train
                    # k = 2: Beta_headway
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_std
                    class1ers += 1
                    u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[0, n, 4] * base_beta[5] * np.random.normal(0, 1)
                    u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[1, n, 4] * base_beta[5] * np.random.normal(0, 1)
                    u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[2, n, 4] * base_beta[5] * np.random.normal(0, 1)

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2]

                    # Find the index of the maximum utility
                    best_alternative = max(range(3), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'CHOICE'] = best_alternative + 1

                elif n_class == 2:
                    # k = 0: ASC_Car
                    # k = 1: ASC_Train
                    # k = 2: Beta_headway
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_std
                    class2ers += 1

                    u_0 = sum(x[0, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])

                    # Put the utilities into a list
                    if latent == 1020 or latent == 1023:  # no car
                        utilities = [u_0, u_1, -900000]
                    elif latent == 1021 or latent == 1024:  # no SM
                        utilities = [u_0, -900000, u_2]
                    else:  # 1022 has all av in class 2
                        utilities = [u_0, u_1, u_2]

                    # Find the index of the maximum utility
                    best_alternative = max(range(3), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'CHOICE'] = best_alternative + 1
            print("# class1 = ", class1ers)
            print("# class2 = ", class2ers)

    if 1030 <= latent <= 1031:
        seed = michelSeed
        runNuM = 1
        seedEnd = seed + runNuM - 1

        while seed <= seedEnd:
            seed += 1

            # define the new ASCs (ASC_car at ind 0, ASC_train at ind 1)
            new_beta = copy.copy(base_beta)
            if latent == 1030:  # has both, but no car in the last group
                new_beta[0] = base_beta[0] * 1.33333  # increase car attractiveness
                new_beta[1] = base_beta[1] * 0.33333  # reduce train attractiveness
            if latent == 1031:  # has both, but no SM in the last group
                new_beta[0] = base_beta[0] * 0.33333  # reduce car attractiveness
                new_beta[1] = base_beta[1] * 0.33333  # reduce train attractiveness

            N = len(df_cut)  # maybe works

            population = [1, 2, 3]
            weights = [0.25, 0.5, 0.25]
            random.seed(seed)
            class1ers = 0
            class2ers = 0
            class3ers = 0

            for n in range(N):
                # Draw a random number from 1 to 3
                n_class = random.choices(population, weights=weights, k=1)[0]

                if n_class == 1:
                    # k = 0: ASC_Car
                    # k = 1: ASC_Train
                    # k = 2: Beta_headway
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_std
                    class1ers += 1
                    u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[0, n, 4] * base_beta[
                        5] * np.random.normal(0, 1)
                    u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[1, n, 4] * base_beta[
                        5] * np.random.normal(0, 1)
                    u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4]) + x[2, n, 4] * base_beta[
                        5] * np.random.normal(0, 1)

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2]

                    # Find the index of the maximum utility
                    best_alternative = max(range(3), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'CHOICE'] = best_alternative + 1

                elif n_class == 2:
                    # k = 0: ASC_Car
                    # k = 1: ASC_Train
                    # k = 2: Beta_headway
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_std
                    class2ers += 1

                    u_0 = sum(x[0, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * new_beta[i] for i in [0, 1, 2, 3, 4])

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2]

                    # Find the index of the maximum utility
                    best_alternative = max(range(3), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'CHOICE'] = best_alternative + 1
                elif n_class == 3:
                    # k = 0: ASC_Car
                    # k = 1: ASC_Train
                    # k = 2: Beta_headway
                    # k = 3: Beta_cost
                    # k = 4: Beta_time
                    # k = 5: Beta_time_std
                    class3ers += 1

                    u_0 = sum(x[0, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])
                    u_1 = sum(x[1, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])
                    u_2 = sum(x[2, n, i] * base_beta[i] for i in [0, 1, 2, 3, 4])

                    # Put the utilities into a list
                    if latent == 1030:  # no car
                        utilities = [u_0, u_1, -90000]
                    elif latent == 1031:  # no SM
                        utilities = [u_0, -90000, u_2]

                    # Find the index of the maximum utility
                    best_alternative = max(range(3), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'CHOICE'] = best_alternative + 1
            print("# class1 = ", class1ers)
            print("# class2 = ", class2ers)
            print("# class3 = ", class3ers)

    # and ONLY NOW compute y
    y = np.array([df_cut.CHOICE == i + 1 for i in range(J)]).astype(int)
    
    # av = np.ones((J, size), dtype=int) # lets not deal with availabilites here, since its synthetic
    av = np.array([df_cut.TRAIN_AV_SP == 1, df_cut.SM_AV == 1, df_cut.CAR_AV_SP == 1]).astype(int)

    if latent not in (1025, 1026, 1028, 28, 229, 29, 10291, 10292, 10293):
        av = np.ones((J, N), dtype=int)

    start_time_biog = time.time()
    biog_beta, biog_loglike, _, _, _, timestamp = biogeme_estimate_swissmetro(df_cut, latent, michels_classes,
                                                                              toms_extremists,
                                                                              False, starting_point, R=R, pandaSeed=pandaSeed)
    # Remove output files
    time.sleep(3)
    # File extensions to clean up
    extensions = ["*.iter", "*.html", "*.pickle"]

    # Iterate over each extension and remove matching files
    for ext in extensions:
        files = glob.glob(ext)
        for file in files:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass

    time_biog = time.time() - start_time_biog
    print(f"Biogeme estimated biog_beta = {biog_beta} with biog_loglike = {biog_loglike} in {time_biog} seconds")

    # print(f"Biogeme estimated biog_beta =  {biog_beta} with biog_loglike = {biog_loglike}")

    return x, y, av, J, K, biog_beta, biog_loglike, None, None, time_biog


def choice_data_telephone(size, df_cut, latent=1, TSeedLondon=0, starting_point=None, R=None):
    N = len(df_cut)

    # loading custom attributes
    _, _, _, _, _, _ = biogeme_estimate_telephone(df_cut, 0, True, None, pandaSeed=pandaSeed)

    J = 5
    # V = {1: V_BM, 2: V_SM, 3: V_LF, 4: V_EF, 5: V_MF}

    K = 5

    # 0 ASC_BM = Beta('ASC_BM', 0, None, None, 0)
    # 1 ASC_EF = Beta('ASC_EF', 0, None, None, 0)
    # 2 ASC_LF = Beta('ASC_LF', 0, None, None, 0)
    # 3 ASC_MF = Beta('ASC_MF', 0, None, None, 0)
    # 4 B_COST = Beta('B_COST', 0, None, None, 0)

    x = np.zeros((J, size, K))
    for n in range(size):
        # 0 ASC_BM = Beta('ASC_BM', 0, None, None, 0)
        # 1 ASC_EF = Beta('ASC_EF', 0, None, None, 0)
        # 2 ASC_LF = Beta('ASC_LF', 0, None, None, 0)
        # 3 ASC_MF = Beta('ASC_MF', 0, None, None, 0)
        # 4 B_COST = Beta('B_COST', 0, None, None, 0)

        # this results in the following classes
        # 2:
        # class_1_ks = [1, 2, 3, 4, 5]
        # class_2_ks = [1, 2, 3, 4, 6]
        # prob_inds = [7]

        # V = {1: V_BM, 2: V_SM, 3: V_LF, 4: V_EF, 5: V_MF}

        # ASC_BM
        x[0, n, 0] = 1
        x[1, n, 0] = 0
        x[2, n, 0] = 0
        x[3, n, 0] = 0
        x[4, n, 0] = 0
        # ASC_EF
        x[0, n, 1] = 0
        x[1, n, 1] = 0
        x[2, n, 1] = 0
        x[3, n, 1] = 1
        x[4, n, 1] = 0
        # ASC_LF
        x[0, n, 2] = 0
        x[1, n, 2] = 0
        x[2, n, 2] = 1
        x[3, n, 2] = 0
        x[4, n, 2] = 0
        # ASC_MF
        x[0, n, 3] = 0
        x[1, n, 3] = 0
        x[2, n, 3] = 0
        x[3, n, 3] = 0
        x[4, n, 3] = 1
        # B_COST
        x[0, n, 4] = df_cut['log_cost1'].values[n]
        x[1, n, 4] = df_cut['log_cost2'].values[n]
        x[2, n, 4] = df_cut['log_cost3'].values[n]
        x[3, n, 4] = df_cut['log_cost4'].values[n]
        x[4, n, 4] = df_cut['log_cost5'].values[n]

    y = np.array([df_cut.choice == i + 1 for i in range(J)]).astype(int)
    av = np.array([df_cut.avail1 == 1, df_cut.avail2 == 1, df_cut.avail3 == 1, df_cut.avail4 == 1, df_cut.avail5 == 1]).astype(int)

    start_time_biog = time.time()
    biog_beta, biog_loglike, _, _, _, timestamp = biogeme_estimate_telephone(df_cut, latent, False, starting_point, R=R, pandaSeed=pandaSeed)
    # Remove output files
    time.sleep(3)
    # try:
    #     os.remove(f"telephone_latent_{latent}_{timestamp}.html")
    #     os.remove(f"telephone_latent_{latent}_{timestamp}.pickle")
    #     os.remove(f"__telephone_latent_{latent}_{timestamp}.iter")
    # except FileNotFoundError:
    #     pass
        
    # File extensions to clean up
    extensions = ["*.iter", "*.html", "*.pickle"]

    # Iterate over each extension and remove matching files
    for ext in extensions:
        files = glob.glob(ext)
        for file in files:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
                
    time_biog = time.time() - start_time_biog
    print(f"Biogeme estimated biog_beta = {biog_beta} with biog_loglike = {biog_loglike} in {time_biog} seconds")

    # print(f"Biogeme estimated biog_beta =  {biog_beta} with biog_loglike = {biog_loglike}")

    return x, y, av, J, K, biog_beta, biog_loglike, None, None, time_biog


def choice_data_optima(size, df_cut, latent=1, OptimaSeed=0, starting_point=None, R=None):
    N = len(df_cut)

    # loading custom attributes
    _, _, _, _, _, _ = biogeme_estimate_optima(df_cut, 0, True, None, pandaSeed=pandaSeed)

    J = 3
    # 0: PT
    # 1: CAR
    # 2: SM (soft modes, walk bike etc)

    K = 8

    # k = 0: ASC_CAR
    # k = 1: ASC_SM
    # k = 2: BETA_COST_HWH
    # k = 3: BETA_COST_OTHER
    # k = 4: BETA_DIST
    # k = 5: BETA_TIME_CAR
    # k = 6: BETA_TIME_PT
    # k = 7: BETA_WAITING_TIME

    x = np.zeros((J, size, K))
    for n in range(size):
        # k = 1: ASC_CAR
        # k = 2: ASC_SM
        # k = 3: BETA_COST_HWH
        # k = 4: BETA_COST_OTHER
        # k = 5: BETA_DIST
        # k = 6: BETA_TIME_CAR
        # k = 7: BETA_TIME_PT
        # k = 8: BETA_WAITING_TIME

        # this results in the following classes

        # 2:
        # class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
        # class_2_ks = [1, 2, 3, 4, 5] # ignores all travel and waiting times
        # prob_inds = [9]

        # 22:
        # class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
        # class_2_ks = [1, 2, 3, 4, 5, 7, 8] # ignores only CAR travel time
        # prob_inds = [9]

        # 3:
        # class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
        # class_2_ks = [1, 2, 3, 4, 5]
        # class_3_ks = [1, 2, 5, 6, 7, 8] # ignores all costs
        # prob_inds = [9, 10]

        # V0 = ASC_PT + \
        #      BETA_TIME_PT * TimePT_scaled + \
        #      BETA_WAITING_TIME * WaitingTimePT + \
        #      BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
        #      BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        #
        # V1 = ASC_CAR + \
        #      BETA_TIME_CAR * TimeCar_scaled + \
        #      BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
        #      BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        #
        # V2 = ASC_SM + BETA_DIST * distance_km_scaled

        # PT, CAR, SM

        # ASC_CAR
        x[0, n, 0] = 0
        x[1, n, 0] = 1
        x[2, n, 0] = 0
        # ASC_SM
        x[0, n, 1] = 0
        x[1, n, 1] = 0
        x[2, n, 1] = 1
        # BETA_COST_HWH
        x[0, n, 2] = df_cut['MarginalCostPT_scaled'].values[n] * df_cut['PurpHWH'].values[n]
        x[1, n, 2] = df_cut['CostCarCHF_scaled'].values[n] * df_cut['PurpHWH'].values[n]
        x[2, n, 2] = 0
        # BETA_COST_OTHER
        x[0, n, 3] = df_cut['MarginalCostPT_scaled'].values[n] * df_cut['PurpOther'].values[n]
        x[1, n, 3] = df_cut['CostCarCHF_scaled'].values[n] * df_cut['PurpOther'].values[n]
        x[2, n, 3] = 0
        # BETA_DIST
        x[0, n, 4] = 0
        x[1, n, 4] = 0
        x[2, n, 4] = df_cut['distance_km_scaled'].values[n]
        # BETA_TIME_CAR
        x[0, n, 5] =0
        x[1, n, 5] = df_cut['TimeCar_scaled'].values[n]
        x[2, n, 5] = 0
        # BETA_TIME_PT
        x[0, n, 6] = df_cut['TimePT_scaled'].values[n]
        x[1, n, 6] = 0
        x[2, n, 6] = 0
        # BETA_WAITING_TIME
        x[0, n, 7] = df_cut['WaitingTimePT'].values[n]
        x[1, n, 7] = 0
        x[2, n, 7] = 0

    if latent == 29:
        seed = OptimaSeed
        runNuM = 1
        seedEnd = seed + runNuM - 1

        while seed <= seedEnd:
            seed += 1
            df_cut_copy = copy.copy(df_cut)
            Logit_beta, Logit_loglike, _, _, _, timestamp = biogeme_estimate_optima(df_cut, 1, False, None, pandaSeed=pandaSeed)

            # Remove output files
            time.sleep(2)
            # File extensions to clean up
            extensions = ["*.iter", "*.html", "*.pickle"]

            # Iterate over each extension and remove matching files
            for ext in extensions:
                files = glob.glob(ext)
                for file in files:
                    try:
                        os.remove(file)
                    except FileNotFoundError:
                        pass

            N = len(df_cut)  # maybe works

            population = [1, 2]
            weights = [0.7, 0.3]
            random.seed(seed)
            class1ers = 0
            class2ers = 0

            # k = 0: ASC_CAR
            # k = 1: ASC_SM
            # k = 2: BETA_COST_HWH
            # k = 3: BETA_COST_OTHER
            # k = 4: BETA_DIST
            # k = 5: BETA_TIME_CAR
            # k = 6: BETA_TIME_PT
            # k = 7: BETA_WAITING_TIME

            # k = 8: BETA_TIME_PT_C2
            # k = 9: BETA_TIME_CAR_C2

            Logit_beta_C2 = copy.copy(Logit_beta)
            Logit_beta_C2.append(Logit_beta[6] / 5)
            Logit_beta_C2.append(Logit_beta[5] / 5)

            print("Logit_beta", Logit_beta)
            print("Logit_beta_C2", Logit_beta_C2)

            for n in range(N):
                # Draw a random number from 1 to 3
                n_class = random.choices(population, weights=weights, k=1)[0]

                if n_class == 1:
                    # k = 0: ASC_CAR
                    # k = 1: ASC_SM
                    # k = 2: BETA_COST_HWH
                    # k = 3: BETA_COST_OTHER
                    # k = 4: BETA_DIST
                    # k = 5: BETA_TIME_CAR
                    # k = 6: BETA_TIME_PT
                    # k = 7: BETA_WAITING_TIME

                    # k = 8: BETA_TIME_PT_C2
                    # k = 9: BETA_TIME_CAR_C2
                    class1ers += 1
                    u_0 = sum(x[0, n, i] * Logit_beta[i] for i in [0, 1, 2, 3, 4, 5, 6, 7])
                    u_1 = sum(x[1, n, i] * Logit_beta[i] for i in [0, 1, 2, 3, 4, 5, 6, 7])
                    u_2 = sum(x[2, n, i] * Logit_beta[i] for i in [0, 1, 2, 3, 4, 5, 6, 7])

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2]

                    # Find the index of the maximum utility
                    best_alternative = max(range(3), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'Choice'] = best_alternative

                elif n_class == 2:
                    # k = 0: ASC_CAR
                    # k = 1: ASC_SM
                    # k = 2: BETA_COST_HWH
                    # k = 3: BETA_COST_OTHER
                    # k = 4: BETA_DIST
                    # k = 5: BETA_TIME_CAR
                    # k = 6: BETA_TIME_PT
                    # k = 7: BETA_WAITING_TIME

                    # k = 8: BETA_TIME_PT_C2
                    # k = 9: BETA_TIME_CAR_C2
                    class2ers += 1
                    u_0 = sum(x[0, n, i] * Logit_beta_C2[i] for i in [0, 1, 2, 3, 4, 7]) + x[0, n, 5] * Logit_beta_C2[9] + x[0, n, 6] * Logit_beta_C2[8]
                    u_1 = sum(x[1, n, i] * Logit_beta_C2[i] for i in [0, 1, 2, 3, 4, 7]) + x[1, n, 5] * Logit_beta_C2[9] + x[1, n, 6] * Logit_beta_C2[8]
                    u_2 = sum(x[2, n, i] * Logit_beta_C2[i] for i in [0, 1, 2, 3, 4, 7]) + x[2, n, 5] * Logit_beta_C2[9] + x[2, n, 6] * Logit_beta_C2[8]

                    # Put the utilities into a list
                    utilities = [u_0, u_1, u_2]

                    # Find the index of the maximum utility
                    best_alternative = max(range(3), key=lambda i: utilities[i])

                    df_cut.loc[df_cut.index[n], 'Choice'] = best_alternative
            print("# class1 = ", class1ers)
            print("# class2 = ", class2ers)

    # and ONLY NOW compute y
    y = np.array([df_cut.Choice == i for i in range(J)]).astype(int)
    av = np.ones((J, N), dtype=int)

    start_time_biog = time.time()
    biog_beta, biog_loglike, _, _, _, timestamp = biogeme_estimate_optima(df_cut, latent, False, starting_point, R=R, pandaSeed=pandaSeed)
    # Remove output files
    time.sleep(3)
    # File extensions to clean up
    extensions = ["*.iter", "*.html", "*.pickle"]

    # Iterate over each extension and remove matching files
    for ext in extensions:
        files = glob.glob(ext)
        for file in files:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass

    time_biog = time.time() - start_time_biog
    print(f"Biogeme estimated biog_beta = {biog_beta} with biog_loglike = {biog_loglike} in {time_biog} seconds")

    # print(f"Biogeme estimated biog_beta =  {biog_beta} with biog_loglike = {biog_loglike}")

    return x, y, av, J, K, biog_beta, biog_loglike, None, None, time_biog


def choice_data_swissmetro_mixed(size, df_cut, mix_inds, R):
    N = len(df_cut)

    # loading custom attributes
    _ = biogeme_estimate_swissmetro_mixed(df_cut, mix_inds, R, loadAttr=True, pandaSeed=pandaSeed)

    print("Estimate Biogeme model")
    start_time_biog = time.time()
    biog_beta, biog_loglike, beta_confs, loglike_conf, timeBiog = biogeme_estimate_swissmetro_mixed(df_cut, mix_inds, R,
                                                                                              loadAttr=False, pandaSeed=pandaSeed)
    print(f"Time to estimate Biogeme model = {time.time() - start_time_biog}s")
    biog_time = time.time() - start_time_biog
    # print(f"BioG: Simulating likelihood with {nbOfDraws_sLL_Biog} draws...")

    J = 3
    # 0: Train
    # 1: SM
    # 2: Car

    K = 5
    # k = 0: ASC_Car
    # k = 1: ASC_Train
    # k = 2: Beta_time
    # k = 3: Beta_cost
    # k = 4: Beta_headway

    x = np.zeros((J, size, K))
    for n in range(size):
        # train, SM, car

        # intercept car
        x[0, n, 0] = 0
        x[1, n, 0] = 0
        x[2, n, 0] = 1
        # intercept train
        x[0, n, 1] = 1
        x[1, n, 1] = 0
        x[2, n, 1] = 0
        # beta cost
        x[0, n, 2] = df_cut['TRAIN_COST_SCALED'].values[n]
        x[1, n, 2] = df_cut['SM_COST_SCALED'].values[n]
        x[2, n, 2] = df_cut['CAR_CO_SCALED'].values[n]
        # beta headway
        x[0, n, 3] = df_cut['TRAIN_HE_SCALED'].values[n]
        x[1, n, 3] = df_cut['SM_HE_SCALED'].values[n]
        x[2, n, 3] = 0
        # beta time
        x[0, n, 4] = df_cut['TRAIN_TT_SCALED'].values[n]
        x[1, n, 4] = df_cut['SM_TT_SCALED'].values[n]
        x[2, n, 4] = df_cut['CAR_TT_SCALED'].values[n]

    y = np.array([df_cut.CHOICE == i + 1 for i in range(J)]).astype(int)
    av = np.array([df_cut.TRAIN_AV_SP == 1, df_cut.SM_AV == 1, df_cut.CAR_AV_SP == 1]).astype(int)

    return x, y, av, J, K, biog_beta, biog_loglike, beta_confs, loglike_conf, biog_time


def choice_data_swissmetro_nested(size, df_cut, R):
    # global biogeme_sim_loglike
    print("Estimate Biogeme model")
    start_time_biog = time.time()
    # R_biog = 1000
    biog_beta, biog_loglike, beta_confs, loglike_conf, df = biogeme_estimate_swissmetro_nested(df_cut, R)

    print(f"Time to estimate Biogeme model = {time.time() - start_time_biog}s")
    biog_time = time.time() - start_time_biog
    # print(f"BioG: Simulating likelihood with {nbOfDraws_sLL_Biog} draws...")
    # start_time = time.time()
    # biogeme_sim_loglike = simulate_likelihood_mixed_swissmetro(df_cut, biog_beta, numberOfDraws=nbOfDraws_sLL_Biog)
    # print(f"Done ({round((time.time()-start_time), 2)}s)")
    # print(f"Biogeme Biog sim. loglike = {biogeme_sim_loglike}")

    # print(f"Biogeme estimated biog_beta =  {biog_beta} with biog_loglike = {biog_loglike}")

    print("Compute choice data")
    st_time = time.time()

    J = 3
    # 0: Train
    # 1: SM
    # 2: Car

    K = 5
    # k = 0: ASC_Car
    # k = 1: ASC_Train
    # k = 2: Beta_cost
    # k = 3: Beta_time
    # k = 4: Beta_time_s

    x = np.empty(shape=(J, size, K))
    for n in range(size):
        # intercept car
        x[0, n, 0] = 0
        x[1, n, 0] = 0
        x[2, n, 0] = 1
        # intercept train
        x[0, n, 1] = 1
        x[1, n, 1] = 0
        x[2, n, 1] = 0
        # beta cost
        x[0, n, 2] = df['TRAIN_COST_SCALED'].values[n]
        x[1, n, 2] = df['SM_COST_SCALED'].values[n]
        x[2, n, 2] = df['CAR_CO_SCALED'].values[n]
        # beta time mean
        x[0, n, 3] = df_cut['TRAIN_TT_SCALED'].values[n]
        x[1, n, 3] = df_cut['SM_TT_SCALED'].values[n]
        x[2, n, 3] = df_cut['CAR_TT_SCALED'].values[n]
        # beta time standard deviation
        x[0, n, 4] = df_cut['TRAIN_TT_SCALED'].values[n]
        x[1, n, 4] = df_cut['SM_TT_SCALED'].values[n]
        x[2, n, 4] = df_cut['CAR_TT_SCALED'].values[n]

    y = np.array([df_cut.CHOICE == i + 1 for i in range(J)]).astype(int)
    av = dict()
    av[0] = np.array(df_cut.TRAIN_AV_SP == 1).astype(int)
    av[1] = np.array(df_cut.SM_AV == 1).astype(int)
    av[2] = np.array(df_cut.CAR_AV_SP == 1).astype(int)

    print(f"Time to compute choice data: {time.time() - st_time}s")

    return x, y, av, J, K, biog_beta, biog_loglike, beta_confs, loglike_conf, biog_time


def print_results(N, R, x, y, total_time, iteration, bestbeta, best_loglike, mixed=False):
    print("N = ", N, "R = ", R, "Total time = ", total_time, ", Iterations = ",
          iteration, ", Best Beta = ", bestbeta, ", loglike = ", best_loglike, "biog_loglike = ",
          compute_biog_LL(x, y, bestbeta, mixed))


def mle_benders(data, cplex, N_range, R_range, errors, foc=2, logOfZer=-100, latent=1, opt_beta=None, Halton=False,
                probit=False, time_limit=10800, csv=False, plotting=False):
    J = None
    if data == "nether":
        df = pd.read_csv('netherlands.dat', sep='\t')
        df = df[df['rp'] == 1]
        df['rail_time'] = df.rail_ivtt + df.rail_acc_time + df.rail_egr_time
        df['car_time'] = df.car_ivtt + df.car_walk_time

        if not N_range[-1] == 0:
            df_full = df.sample(N_range[-1], random_state=pandaSeed)
        else:
            df_full = df

        J = 2
    elif data == "lpmc":
        df = pd.read_csv("lpmc.dat", sep='\t')
        df['rail_time'] = df.dur_pt_rail + df.dur_pt_bus + df.dur_pt_int + df.dur_pt_access
        df["car_cost"] = df.cost_driving_fuel + df.cost_driving_ccharge
        df["travel_mode_chosen"] = df.travel_mode - 1

        if not N_range[-1] == 0:
            df_full = df.sample(N_range[-1], random_state=pandaSeed)
        else:
            df_full = df

        J = 4

    elif data == "swissmetro":
        df_full = pd.read_csv('swissmetro.dat', sep='\t')
        df_full = df_full.loc[
            ~((df_full["PURPOSE"] != 1) & (df_full["PURPOSE"] != 3) | (df_full["CHOICE"] == 0) > 0)]

        if not N_range[-1] == 0:
            df_full = df_full.sample(N_range[-1], random_state=pandaSeed)
        else:
            df_full = df_full

        J = 3

    elif data == "optima":
        print("read data")
        st_time = time.time()
        df_full = pd.read_csv("optima.dat", sep='\t')

        # Exclude observations such that the chosen alternative is -1
        df_full = df_full[df_full["Choice"] != -1]

        # MAX N = 1906

        if not N_range[-1] == 0:
            df_full = df_full.sample(N_range[-1], random_state=pandaSeed)
        else:
            df_full = df_full

        J = 3
        print(f"Time to read data: {time.time() - st_time}s")

    elif data == "telephone":
        print("read data")
        st_time = time.time()
        df_full = pd.read_csv("telephone.dat", sep='\t')

        # MAX N = 434

        if not N_range[-1] == 0:
            df_full = df_full.sample(N_range[-1], random_state=pandaSeed)
        else:
            df_full = df_full

        J = 5
        print(f"Time to read data: {time.time() - st_time}s")

    # compute epsilon
    if probit:
        multi = np.random.multivariate_normal(np.zeros(N_range[-1]), np.eye(N_range[-1]), (J, R_range[-1]))
        grande_epsilon = np.empty(shape=(J, N_range[-1], R_range[-1]))

        for i in range(J):
            for n in range(N_range[-1]):
                for r in range(R_range[-1]):
                    grande_epsilon[i, n, r] = multi[i, r, n]
        np.save("epsilon_random_probit.npy", grande_epsilon)

    elif halton_sampling:
        grande_epsilon = np.load(f"epsilon_halton_2_200_1000.npy")
    else:
        if not ((1020 <= latent <= 1029) or (1030 <= latent <= 1039)):
            grande_epsilon = np.random.gumbel(loc=0, scale=1, size=(J, N_range[-1], R_range[-1]))
            epsilon = grande_epsilon
        else:
            grande_epsilon = None
            epsilon = grande_epsilon

    # plt.plot(grande_epsilon[0, :, :], marker='.', markersize=10, linestyle='None')
    # plt.show()

    for N in N_range:
        latent_indices = None  # INITIALIZED as none. Probably in run_latent we will change that
        if data == "nether":
            if run_latent:

                if latent == 2:
                    latent_indices = [2]
                elif latent >= 3:
                    latent_indices = [2, 1]

                # synthetic_choice_latent = False  # interesting

                signi = True  # up for debate

                # if synthetic_choice_latent:
                #     epsilon = grande_epsilon[:, 0:N, 0:R_range[-1]]
                #     y = generate_synthetic_latent_observed_choices(df_full, N, R_range[-1], epsilon, latent,
                #                                                    latent_indices)
                #     choices_y = [y[1, n] for n in range(N)]
                #     df_full['choice'] = choices_y
                #     print(choices_y)

                michels_classes = False
                toms_extremists = False

                MichelSeed = MSeed
                TomSeed = TSeed

                if MichelSeed > 0:
                    michels_classes = True
                elif TomSeed > 0:
                    toms_extremists = True

                x, y, J, K, biog_beta, biog_loglike, beta_confs, loglike_conf, timeBiog, av = choice_data_netherlands(N,
                                                                                                                  df_full,
                                                                                                                  latent=latent,
                                                                                                                  epsilon=grande_epsilon,
                                                                                                                  intercept=True,
                                                                                                                  cost=True,
                                                                                                                  michels_classes=michels_classes,
                                                                                                                  toms_extremists=toms_extremists,
                                                                                                                  michelSeed=MichelSeed,
                                                                                                                  tomSeed=TomSeed,
                                                                                                                  starting_point=starting_point,
                                                                                                                  R=nbOfDraws_normal_MonteCarlo)

            # elif run_mixed:
            #     x, y, J, K, biog_beta, biog_loglike, beta_confs, _, biog_time = choice_data_netherlands_mixed(N,
            #                                                                                                   df_full,
            #                                                                                                   nbOfDraws_normal_MonteCarlo)
            #     mixed_index = 2
            # else:
            #     x, y, J, K, biog_beta, biog_loglike, beta_confs, loglike_conf = choice_data_netherlands(N,
            #                                                                                             df_full,
            #                                                                                             latent=latent,
            #                                                                                             intercept=False,
            #                                                                                             cost=False,
            #                                                                                             probit=False)

            # av = dict()
            # for i in range(J):
            #     inds = dict()
            #     for n in range(N):
            #         inds[n] = 1
            #     av[i] = inds
        elif data == "lpmc":
            michels_classes_London = False
            toms_extremists_London = False

            MichelSeed_London = MSeedLondon
            TomSeed_London = TSeedLondon

            if MichelSeed_London > 0:
                michels_classes_London = True
            elif TomSeed_London > 0:
                toms_extremists_London = True

            x, y, J, K, biog_beta, biog_loglike, beta_confs, loglike_conf, timeBiog, av = choice_data_london(N,
                                                                                                         df_full,
                                                                                                         latent=latent,
                                                                                                         # epsilon=grande_epsilon,
                                                                                                         michels_classes=michels_classes_London,
                                                                                                         toms_extremists=toms_extremists_London,
                                                                                                         michelSeed=MichelSeed_London,
                                                                                                         tomSeed=TomSeed_London,
                                                                                                         starting_point=starting_point,
                                                                                                         R=nbOfDraws_normal_MonteCarlo
                                                                                                         )

            # av = dict()
            # for i in range(J):
            #     inds = dict()
            #     for n in range(N):
            #         inds[n] = 1
            #     av[i] = inds

            if latent == 2:
                latent_indices = [4]
            elif latent >= 3:
                latent_indices = [4, 3]

        elif data == "swissmetro":
            # if do_mixed:
            #     x, y, av, J, K, biog_beta, biog_loglike, beta_confs, _, timeBiog = choice_data_swissmetro_mixed(N,
            #                                                                                                      df_full,
            #                                                                                                      mix_inds,
            #                                                                                                      nbOfDraws_normal_MonteCarlo)

            if run_latent:
                x, y, av, J, K, biog_beta, biog_loglike, _, _, timeBiog = choice_data_swissmetro(N,
                                                                                                 df_full,
                                                                                                 latent=latent,
                                                                                                 michels_classes=michels_classes_SM,
                                                                                                 toms_extremists=toms_extremists_SM,
                                                                                                 michelSeed=MichelSeed_SM,
                                                                                                 tomSeed=TomSeed_SM,
                                                                                                 starting_point=starting_point,
                                                                                                 R=nbOfDraws_normal_MonteCarlo)
                if latent == 2:
                    latent_indices = [2]
                elif latent >= 3:
                    latent_indices = [2, 3]
        elif data == "optima":
            if run_latent:
                x, y, av, J, K, biog_beta, biog_loglike, _, _, timeBiog = choice_data_optima(N,
                                                                                             df_full,
                                                                                             latent=latent,
                                                                                             OptimaSeed=OptimaSeed,
                                                                                             starting_point=starting_point,
                                                                                             R=nbOfDraws_normal_MonteCarlo)
        elif data == "telephone":
            if run_latent:
                x, y, av, J, K, biog_beta, biog_loglike, _, _, timeBiog = choice_data_telephone(N,
                                                                                             df_full,
                                                                                             latent=latent,
                                                                                             TSeedLondon=TSeedLondon,
                                                                                             starting_point=starting_point,
                                                                                             R=nbOfDraws_normal_MonteCarlo)
        if run_latent and not ((1020 <= latent <= 1029) or (1030 <= latent <= 1039)):
            class_draws = np.random.uniform(low=0, high=1, size=(N, R))
        else:
            class_draws = None
            # np.save(f"class_draws_nether_{datetime.now()}.npy", class_draws)
        # if do_mixed:
        #     H = len(mix_inds)
        #     sigma = np.random.normal(loc=0, scale=1, size=(H, N, R))

        exporto = True
        if exporto:
            if data == "nether":
                if do_mixed:
                    av = np.ones((J, N), dtype=int)
                    np.savez(f"input_data_{N}_mixed_latent_{pandaSeed}_{latent}_nether.npz",
                             x=x, y=y, av=av)
                    print(f"input_data_{N}_mixed_latent_{pandaSeed}_{latent}_nether.npz saved")
                elif michels_classes:
                    np.savez(f"input_data_{N}_{R}_latent_{pandaSeed}_nether_michels_classes_{MichelSeed}.npz", x=x, y=y,
                             epsilon=epsilon, class_draws=class_draws)
                    print(f"input_data_{N}_{R}_latent_{pandaSeed}_nether_michels_classes_{MichelSeed}.npz saved")
                elif toms_extremists:
                    np.savez(f"input_data_{N}_{R}_latent_{pandaSeed}_nether_toms_extremists_{TomSeed}.npz", x=x, y=y,
                             epsilon=epsilon, class_draws=class_draws)
                    print(f"input_data_{N}_{R}_latent_{pandaSeed}_nether_toms_extremists_{TomSeed}.npz saved")
                else:
                    np.savez(f"input_data_{N}_{R}_latent_{pandaSeed}_nether.npz", x=x, y=y,
                             epsilon=epsilon, class_draws=class_draws)
                    print(f"input_data_{N}_{R}_latent_{pandaSeed}_nether.npz saved")

            elif data == "lpmc":
                if do_mixed:
                    av = np.ones((J, N), dtype=int)
                    np.savez(f"input_data_{N}_mixed_latent_{pandaSeed}_{latent}_lpmc.npz",
                             x=x, y=y, av=av)
                    print(f"input_data_{N}_mixed_latent_{pandaSeed}_{latent}_lpmc.npz saved")
                elif michels_classes_London:
                    np.savez(f"input_data_{N}_{R}_latent_{pandaSeed}_lpmc_Michel_{MichelSeed_London}.npz", x=x,
                             y=y,
                             av=av,
                             epsilon=epsilon,
                             class_draws=class_draws)
                    print(f"input_data_{N}_{R}_latent_{pandaSeed}_lpmc_Michel_{MichelSeed_London}.npz saved")
                elif toms_extremists_London:
                    np.savez(f"input_data_{N}_{R}_latent_{pandaSeed}_lpmc_Tom_{TomSeed_London}.npz", x=x, y=y,
                             epsilon=epsilon,
                             class_draws=class_draws)
                    print(f"input_data_{N}_{R}_latent_{pandaSeed}_lpmc_Tom_{TomSeed_London}.npz saved")
                else:
                    np.savez(f"input_data_{N}_{R}_latent_{pandaSeed}_lpmc.npz", x=x, y=y, epsilon=epsilon,
                             class_draws=class_draws)
                    print(f"input_data_{N}_{R}_latent_{pandaSeed}_lpmc.npz saved")
            elif data == "swissmetro":
                if do_mixed:
                    np.savez(f"input_data_{N}_mixed_latent_{pandaSeed}_{latent}_SM.npz",
                             x=x, y=y, av=av)
                    print(f"input_data_{N}_mixed_latent_{pandaSeed}_{latent}_SM.npz saved")
                elif michels_classes_SM:
                    np.savez(f"input_data_{N}_{R}_latent_{pandaSeed}_SM_Michel_{MichelSeed_SM}.npz", x=x, y=y,
                             av=av,
                             epsilon=epsilon,
                             class_draws=class_draws)
                    print(f"input_data_{N}_{R}_latent_{pandaSeed}_SM_Michel_{MichelSeed_SM}.npz saved")
                elif toms_extremists_SM:
                    np.savez(f"input_data_{N}_{R}_latent_{pandaSeed}_SM_Tom_{TomSeed_SM}.npz", x=x, y=y,
                             av=av,
                             epsilon=epsilon,
                             class_draws=class_draws)
                    print(f"input_data_{N}_{R}_latent_{pandaSeed}_SM_Tom_{TomSeed_SM}.npz saved")
                else:
                    np.savez(f"input_data_{N}_{R}_latent_{pandaSeed}_SM.npz", x=x, y=y, av=av, epsilon=epsilon,
                             class_draws=class_draws)
                    print(f"input_data_{N}_{R}_latent_{pandaSeed}_SM.npz saved")
            elif data == "optima":
                if do_mixed:
                    np.savez(f"input_data_{N}_mixed_latent_{pandaSeed}_{latent}_optima.npz",
                             x=x, y=y, av=av)
                    print(f"input_data_{N}_mixed_latent_{pandaSeed}_{latent}_optima.npz saved")
                else:
                    np.savez(f"input_data_{N}_{R}_latent_{pandaSeed}_optima_{OptimaSeed}.npz", x=x, y=y,
                             av=av,
                             epsilon=epsilon,
                             class_draws=class_draws)
                    print(f"input_data_{N}_{R}_latent_{pandaSeed}_optima_{OptimaSeed}.npz saved")
            elif data == "telephone":
                if do_mixed:
                    np.savez(f"input_data_{N}_mixed_latent_{pandaSeed}_{latent}_telephone.npz",
                             x=x, y=y, av=av)
                    print(f"input_data_{N}_mixed_latent_{pandaSeed}_{latent}_telephone.npz saved")
                else:
                    np.savez(f"input_data_{N}_{R}_latent_{pandaSeed}_telephone_{TSeedLondon}.npz", x=x, y=y,
                             av=av,
                             epsilon=epsilon,
                             class_draws=class_draws)
                print(f"input_data_{N}_{R}_latent_{pandaSeed}_telephone_{TSeedLondon}.npz saved")
            time.sleep(3)

            # print(f"bio LL, R={R}: : ", biog_loglike)

            # LL_R = R

            # startt = time.time()
            #
            # class_1_ks = [0, 1, 2, 3, 4]
            # class_2_ks = [0, 1, 2, 3, 5]
            #
            # class_1_ks = [0, 1, 2, 3, 4]
            # class_2_ks = [0, 1, 3, 5, 6]
            # prob_inds = [7]
            # extra_inds = [[2, 5], [4, 6]]
            #
            # log_likelihood_bio = compute_biog_loglike_latent2classes(x, y, av, biog_beta, class_1_ks, class_2_ks, extra_inds)

            # if data == "swissmetro":
            #     if 10 <= latent <= 19:
            #         log_likelihood_bio = simulate_likelihood_mixed_swissmetro(N, pandaSeed, biog_beta, mix_inds, R=LL_R)  # BIOGEME
            #     else:
            #         log_likelihood_bio = simulate_likelihood_mixed_latent_swissmetro(N, pandaSeed, biog_beta, latent, R=LL_R)  # BIOGEME
            # else:
            #     if 10 <= latent <= 19:
            #         log_likelihood_bio = simulate_likelihood_mixed_lpmc(N, pandaSeed, biog_beta, mix_inds, R=LL_R)  # BIOGEME
            #     else:
            #         log_likelihood_bio = simulate_likelihood_mixed_latent_lpmc(N, pandaSeed, biog_beta, latent, R=LL_R)  # BIOGEME

            # endt = time.time() - startt
            # print(f"computed LL with BioSim, R={LL_R}: ", log_likelihood_bio, "in ", round(endt, 2), "s")

            # class_1_ks = [0, 1, 2, 3, 4]
            # class_2_ks = [0, 1, 2, 3]
            # class_3_ks = [0, 1, 2]
            #
            # computed_LL = compute_biog_loglike_latent3classes(x, y, av, biog_beta, class_1_ks, class_2_ks, class_3_ks,
            #                                                   biogeme=True)

            # class_1_ks = [0, 1, 2, 3, 4]
            # class_2_ks = [0, 1, 2, 3, 5]
            #
            # computed_LL = compute_biog_loglike_latent2classes(x, y, av, biog_beta, class_1_ks, class_2_ks, biogeme=True)

            #
            # print("biog_loglike = ", biog_loglike)
            # print("computed_LL = ", computed_LL)

            return biog_beta, timeBiog, biog_loglike
        else:
            return biog_beta, None, None


# A context manager to suppress print statements
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if MSeed > 0:
    dataname = "nether"
elif MSeedLondon > 0:
    dataname = "lpmc"
elif TSeedLondon > 0:
    dataname = "telephone"
elif MSeedSM > 0:
    dataname = "swissmetro"
elif OptimaSeed > 0:
    dataname = "optima"

suppresso = True

# # Suppress RuntimeWarnings
# warnings.filterwarnings('ignore', category=RuntimeWarning)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='biogeme')

if suppresso:
    with suppress_output():
        biog_beta_time = mle_benders(data=dataname,
                                     cplex=False,
                                     N_range=N_range,
                                     R_range=R_range,
                                     errors=errors,
                                     foc=foc,
                                     logOfZer=logOfZer,
                                     latent=latent,
                                     opt_beta=opt_beta,
                                     Halton=False,
                                     probit=False,
                                     time_limit=10800,
                                     csv=False,
                                     plotting=False)
else:
    biog_beta_time = mle_benders(data=dataname,
                                 cplex=False,
                                 N_range=N_range,
                                 R_range=R_range,
                                 errors=errors,
                                 foc=foc,
                                 logOfZer=logOfZer,
                                 latent=latent,
                                 opt_beta=opt_beta,
                                 Halton=False,
                                 probit=False,
                                 time_limit=10800,
                                 csv=False,
                                 plotting=False)
print(biog_beta_time)
