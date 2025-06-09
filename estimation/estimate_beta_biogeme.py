import os
import glob
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import bioMax, bioNormalCdf, Elem
import biogeme.models as models
from biogeme.models import loglogit
import multiprocessing as mp

import biogeme.messaging as msg
from biogeme.expressions import bioDraws, MonteCarlo

import numpy as np
import random
import time
from biogeme.expressions import (
    Beta,
    Variable,
    exp,
    log,
)


def biogeme_estimate_beta_nether(df, latent, intercept=False, cost=False, probit=False, altSpec=False,
                                 toms_extremists=False, michels_classes=False, starting_point=None, R=None, pandaSeed=1):
    database = db.Database('netherlands', df)
    globals().update(database.variables)

    if starting_point is not None:
        ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
        BETA_COST = Beta('BETA_COST', starting_point[1], None, None, 0)
        BETA_TIME = Beta('BETA_TIME', starting_point[2], None, None, 0)
    else:
        ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
        BETA_COST = Beta('BETA_COST', 0, None, None, 0)
        BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)

    if 10 <= latent <= 19:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        # 1 ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
        # 3 BETA_COST = Beta('BETA_COST', starting_point[1], None, None, 0)
        # 2 BETA_TIME = Beta('BETA_TIME', starting_point[2], None, None, 0)

        mix_inds = None
        if latent == 10:
            mix_inds = [[2, 4]]  # mix just time
            if starting_point is not None:
                Z1_Beta_time_S = Beta('Z1_Beta_time_S', starting_point[3], None, None, 0)
            else:
                Z1_Beta_time_S = Beta('Z1_Beta_time_S', 1, None, None, 0)
            Beta_time_RND = BETA_TIME  + Z1_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

            Car = Beta_time_RND * car_time + BETA_COST * car_cost
            Rail = ASC_RAIL + Beta_time_RND * rail_time + BETA_COST * rail_cost

        if latent == 11:
            mix_inds = [[2, 4], [3, 5]]  # mix time and costs
            if starting_point is not None:
                Z1_Beta_time_S = Beta('Z1_Beta_time_S', starting_point[3], None, None, 0)
                Z2_Beta_cost_S = Beta('Z2_Beta_cost_S', starting_point[4], None, None, 0)
            else:
                Z1_Beta_time_S = Beta('Z1_Beta_time_S', 1, None, None, 0)
                Z2_Beta_cost_S = Beta('Z2_Beta_cost_S', 1, None, None, 0)
            Beta_time_RND = BETA_TIME + Z1_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')
            Beta_cost_RND = BETA_COST + Z2_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

            Car = Beta_time_RND * car_time + Beta_cost_RND * car_cost
            Rail = ASC_RAIL + Beta_time_RND * rail_time + Beta_cost_RND * rail_cost

        if latent == 12:
            mix_inds = [[2, 4], [3, 5]]  # mix time and costs and ASC rail
            if starting_point is not None:
                Z1_Beta_time_S = Beta('Z1_Beta_time_S', starting_point[3], None, None, 0)
                Z2_Beta_cost_S = Beta('Z2_Beta_cost_S', starting_point[4], None, None, 0)
                Z3_ASC_Car_S = Beta('Z3_ASC_rail_S', starting_point[5], None, None, 0)
            else:
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
            if starting_point is not None:
                Z1_Beta_cost_S = Beta('Z1_Beta_cost_S', starting_point[3], None, None, 0)
            else:
                Z1_Beta_cost_S = Beta('Z1_Beta_cost_S', 1, None, None, 0)
            Beta_cost_RND = BETA_COST + Z1_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

            Car = BETA_TIME * car_time + Beta_cost_RND * car_cost
            Rail = ASC_RAIL + BETA_TIME * rail_time + Beta_cost_RND * rail_cost

        if latent == 14:
            mix_inds = [[1, 4]]  # mix ASC rail
            if starting_point is not None:
                Z1_ASC_Car_S = Beta('Z1_ASC_rail_S', starting_point[3], None, None, 0)
            else:
                Z1_ASC_Car_S = Beta('Z1_ASC_rail_S', 1, None, None, 0)
            ASC_rail_RND = ASC_RAIL + Z1_ASC_Car_S * bioDraws('ASC_rail_RND', 'NORMAL')

            Car = BETA_TIME * car_time + BETA_COST * car_cost
            Rail = ASC_rail_RND + BETA_TIME * rail_time + BETA_COST * rail_cost

    V = {0: Car, 1: Rail}  # deterministic part: use utilities depending on mode indicator
    av = {0: 1, 1: 1}  # both are available to everyone

    prob = models.logit(V, av, choice)
    logprob = log(MonteCarlo(prob))

    # Create the Biogeme object
    print(f"Nb. of MonteCarlo draws = {R}")
    biogeme = bio.BIOGEME(
        database, logprob, numberOfDraws=R, seed=pandaSeed, numberOfThreads=1
    )
    biogeme.modelName = 'nether_mixed'
    biogeme.saveIterations = False

    # Estimate the parameters
    st_time = time.time()
    results = biogeme.estimate()
    elapsed = time.time() - st_time
    print(f"Estimating Biogeme model takes {elapsed}s")
    pandasResults = results.getEstimatedParameters()
    biog_loglike = round(results.data.logLike, 8)
    print(pandasResults)

    print("")
    print(f"Loglike = {biog_loglike}")

    # Get the results
    biog_beta = list(pandasResults["Value"])

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

    # if michels_classes:
    #     if latent == 2:
    #         # class 1 both
    #         # class 2 ignores time
    #
    #         if starting_point is not None:
    #             ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', starting_point[1], None, None, 0)
    #             BETA_COST = Beta('BETA_COST', starting_point[2], None, None, 0)
    #         else:
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #
    #         # Utilities
    #         Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
    #         Car_class_2 = BETA_COST * car_cost
    #
    #         Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #         Rail_class_2 = ASC_RAIL + BETA_COST * rail_cost
    #
    #         V_class_1 = {0: Car_class_1,
    #                      1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #         V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #
    #         av = {0: 1, 1: 1}  # both are available to everyone
    #
    #         # specify likelihood function
    #         # Class membership
    #         if starting_point is not None:
    #             prob_class_1 = Beta('prob_class_1', starting_point[3], None, None, 0)
    #         else:
    #             prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
    #
    #         denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)
    #
    #         P1 = exp(prob_class_1) / denom
    #         P2 = 1 / denom
    #
    #         prob = (
    #                 P1 * models.logit(V_class_1, av, choice) +
    #                 P2 * models.logit(V_class_2, av, choice)
    #         )
    #
    #     if latent == 29:
    #         if starting_point is not None:
    #             ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
    #             BETA_COST = Beta('BETA_COST', starting_point[1], None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', starting_point[2], None, None, 0)
    #             Z_BETA_TIME_2 = Beta('Z_BETA_TIME_2', starting_point[3], None, None, 0)
    #         else:
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #             Z_BETA_TIME_2 = Beta('Z_BETA_TIME_2', 0, None, None, 0)
    #
    #         # Utilities
    #         Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
    #         Car_class_2 = Z_BETA_TIME_2 * car_time + BETA_COST * car_cost
    #
    #         Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #         Rail_class_2 = ASC_RAIL + Z_BETA_TIME_2 * rail_time + BETA_COST * rail_cost
    #
    #         V_class_1 = {0: Car_class_1,
    #                      1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #         V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #
    #         av = {0: 1, 1: 1}  # both are available to everyone
    #
    #         # specify likelihood function
    #         # Class membership
    #         if starting_point is not None:
    #             prob_class_1 = Beta('prob_class_1', starting_point[4], None, None, 0)
    #         else:
    #             prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
    #
    #         denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)
    #
    #         P1 = exp(prob_class_1) / denom
    #         P2 = 1 / denom
    #
    #         prob = (
    #                 P1 * models.logit(V_class_1, av, choice) +
    #                 P2 * models.logit(V_class_2, av, choice)
    #         )
    #     if latent == 229:
    #         if starting_point is not None:
    #             ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
    #             BETA_COST = Beta('BETA_COST', starting_point[1], None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', starting_point[2], None, None, 0)
    #             Z_BETA_COST_2 = Beta('Z_BETA_COST_2', starting_point[3], None, None, 0)
    #             Z_BETA_TIME_2 = Beta('Z_BETA_TIME_2', starting_point[4], None, None, 0)
    #
    #         else:
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #             Z_BETA_COST_2 = Beta('Z_BETA_COST_2', 0, None, None, 0)
    #             Z_BETA_TIME_2 = Beta('Z_BETA_TIME_2', 0, None, None, 0)
    #
    #         # Utilities
    #         Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
    #         Car_class_2 = Z_BETA_TIME_2 * car_time + Z_BETA_COST_2 * car_cost
    #
    #         Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #         Rail_class_2 = ASC_RAIL + Z_BETA_TIME_2 * rail_time + Z_BETA_COST_2 * rail_cost
    #
    #         V_class_1 = {0: Car_class_1,
    #                      1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #         V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #
    #         av = {0: 1, 1: 1}  # both are available to everyone
    #
    #         # specify likelihood function
    #         # Class membership
    #         if starting_point is not None:
    #             prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
    #         else:
    #             prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
    #
    #         denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)
    #
    #         P1 = exp(prob_class_1) / denom
    #         P2 = 1 / denom
    #
    #         prob = (
    #                 P1 * models.logit(V_class_1, av, choice) +
    #                 P2 * models.logit(V_class_2, av, choice)
    #         )
    #     if latent == 28 or latent == 282:
    #         if starting_point is not None:
    #             ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
    #             BETA_COST = Beta('BETA_COST', starting_point[1], None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', starting_point[2], None, None, 0)
    #             Z_ASC_RAIL_2 = Beta("Z_ASC_RAIL_2", starting_point[3], None, None, 0)
    #         else:
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #             Z_ASC_RAIL_2 = Beta('Z_ASC_RAIL_2', 0, None, None, 0)
    #
    #         # Utilities
    #         Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
    #         Car_class_2 = BETA_TIME * car_time + BETA_COST * car_cost
    #
    #         Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #         Rail_class_2 = Z_ASC_RAIL_2 + BETA_TIME * rail_time + BETA_COST * rail_cost
    #
    #         V_class_1 = {0: Car_class_1,
    #                      1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #         V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #
    #         av = {0: 1, 1: 1}  # both are available to everyone
    #
    #         # specify likelihood function
    #         # Class membership
    #         if starting_point is not None:
    #             prob_class_1 = Beta('prob_class_1', starting_point[4], None, None, 0)
    #         else:
    #             prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
    #
    #         denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)
    #
    #         P1 = exp(prob_class_1) / denom
    #         P2 = 1 / denom
    #
    #         prob = (
    #                 P1 * models.logit(V_class_1, av, choice) +
    #                 P2 * models.logit(V_class_2, av, choice)
    #         )
    #     if latent == 39:
    #         if starting_point is not None:
    #             ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', starting_point[1], None, None, 0)
    #             BETA_COST = Beta('BETA_COST', starting_point[2], None, None, 0)
    #             Z_BETA_TIME_2 = Beta('Z_BETA_TIME_2', starting_point[3], None, None, 0)
    #             Z_BETA_COST_2 = Beta('Z_BETA_COST_2', starting_point[4], None, None, 0)
    #         else:
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #             Z_BETA_TIME_2 = Beta('Z_BETA_TIME_2', 0, None, None, 0)
    #             Z_BETA_COST_2 = Beta('Z_BETA_COST_2', 0, None, None, 0)
    #
    #         # Utilities
    #         Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
    #         Car_class_2 = Z_BETA_TIME_2 * car_time + BETA_COST * car_cost
    #         Car_class_3 = BETA_TIME * car_time + Z_BETA_COST_2 * car_cost
    #
    #         Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #         Rail_class_2 = ASC_RAIL + Z_BETA_TIME_2 * rail_time + BETA_COST * rail_cost
    #         Rail_class_3 = ASC_RAIL + BETA_TIME * rail_time + Z_BETA_COST_2 * rail_cost
    #
    #         V_class_1 = {0: Car_class_1,
    #                      1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #         V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #         V_class_3 = {0: Car_class_3, 1: Rail_class_3}
    #
    #         av = {0: 1, 1: 1}  # both are available to everyone
    #
    #         # specify likelihood function
    #         # Class membership
    #         if starting_point is not None:
    #             prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
    #             prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
    #         else:
    #             prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
    #             prob_class_2 = Beta('prob_class_2', 0, None, None, 0)
    #
    #         denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)
    #
    #         P1 = exp(prob_class_1) / denom
    #         P2 = exp(prob_class_2) / denom
    #         P3 = 1 / denom
    #
    #         prob = (
    #                 P1 * models.logit(V_class_1, av, choice) +
    #                 P2 * models.logit(V_class_2, av, choice) +
    #                 P3 * models.logit(V_class_3, av, choice)
    #         )
    #
    #     if latent == 22:
    #         # class 1 both
    #         # class 2 ignores cost
    #
    #         if starting_point is not None:
    #             ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', starting_point[1], None, None, 0)
    #             BETA_COST = Beta('BETA_COST', starting_point[2], None, None, 0)
    #         else:
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #
    #         # Utilities
    #         Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
    #         Car_class_2 = BETA_TIME * car_time
    #
    #         Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #         Rail_class_2 = ASC_RAIL + BETA_TIME * rail_time
    #
    #         V_class_1 = {0: Car_class_1,
    #                      1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #         V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #
    #         av = {0: 1, 1: 1}  # both are available to everyone
    #
    #         # specify likelihood function
    #         # Class membership
    #         if starting_point is not None:
    #             prob_class_1 = Beta('prob_class_1', starting_point[3], None, None, 0)
    #         else:
    #             prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
    #
    #         denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)
    #
    #         P1 = exp(prob_class_1) / denom
    #         P2 = 1 / denom
    #
    #         prob = (
    #                 P1 * models.logit(V_class_1, av, choice) +
    #                 P2 * models.logit(V_class_2, av, choice)
    #         )
    #     if latent == 23:
    #         # class 1 both
    #         # class 2 only ASCs
    #
    #         if starting_point is not None:
    #             ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', starting_point[1], None, None, 0)
    #             BETA_COST = Beta('BETA_COST', starting_point[2], None, None, 0)
    #         else:
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #
    #         # Utilities
    #         Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
    #         Car_class_2 = 0
    #
    #         Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #         Rail_class_2 = ASC_RAIL
    #
    #         V_class_1 = {0: Car_class_1,
    #                      1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #         V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #
    #         av = {0: 1, 1: 1}  # both are available to everyone
    #
    #         # specify likelihood function
    #         # Class membership
    #         if starting_point is not None:
    #             prob_class_1 = Beta('prob_class_1', starting_point[3], None, None, 0)
    #         else:
    #             prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
    #
    #         denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)
    #
    #         P1 = exp(prob_class_1) / denom
    #         P2 = 1 / denom
    #
    #         prob = (
    #                 P1 * models.logit(V_class_1, av, choice) +
    #                 P2 * models.logit(V_class_2, av, choice)
    #         )
    #     elif latent == 3:
    #         # class 1 both
    #         # class 2 no time
    #         # class 3 no cost
    #
    #         if starting_point is not None:
    #             ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', starting_point[1], None, None, 0)
    #             BETA_COST = Beta('BETA_COST', starting_point[2], None, None, 0)
    #         else:
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #
    #         # Utilities
    #         Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
    #         Car_class_2 = BETA_COST * car_cost
    #         Car_class_3 = BETA_TIME * car_time
    #
    #         Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #         Rail_class_2 = ASC_RAIL + BETA_COST * rail_cost
    #         Rail_class_3 = ASC_RAIL + BETA_TIME * rail_time
    #
    #         V_class_1 = {0: Car_class_1,
    #                      1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #         V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #         V_class_3 = {0: Car_class_3, 1: Rail_class_3}
    #
    #         av = {0: 1, 1: 1}  # both are available to everyone
    #
    #         # specify likelihood function
    #         # Class membership
    #         if starting_point is not None:
    #             prob_class_1 = Beta('prob_class_1', starting_point[3], None, None, 0)
    #             prob_class_2 = Beta('prob_class_2', starting_point[4], None, None, 0)
    #         else:
    #             prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
    #             prob_class_2 = Beta('prob_class_2', 0, None, None, 0)
    #
    #         denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)
    #
    #         P1 = exp(prob_class_1) / denom
    #         P2 = exp(prob_class_2) / denom
    #         P3 = 1 / denom
    #
    #         prob = (
    #                 P1 * models.logit(V_class_1, av, choice) +
    #                 P2 * models.logit(V_class_2, av, choice) +
    #                 P3 * models.logit(V_class_3, av, choice)
    #         )
    #     elif latent == 31:
    #         # class 1 both
    #         # class 2 no time
    #         # class 3 no cost
    #
    #         if starting_point is not None:
    #             ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', starting_point[1], None, None, 0)
    #             BETA_COST = Beta('BETA_COST', starting_point[2], None, None, 0)
    #         else:
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #
    #         # Utilities
    #         Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
    #         Car_class_2 = BETA_COST * car_cost
    #         Car_class_3 = 0
    #
    #         Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #         Rail_class_2 = ASC_RAIL + BETA_COST * rail_cost
    #         Rail_class_3 = ASC_RAIL
    #
    #         V_class_1 = {0: Car_class_1,
    #                      1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #         V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #         V_class_3 = {0: Car_class_3, 1: Rail_class_3}
    #
    #         av = {0: 1, 1: 1}  # both are available to everyone
    #
    #         # specify likelihood function
    #         # Class membership
    #         if starting_point is not None:
    #             prob_class_1 = Beta('prob_class_1', starting_point[3], None, None, 0)
    #             prob_class_2 = Beta('prob_class_2', starting_point[4], None, None, 0)
    #         else:
    #             prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
    #             prob_class_2 = Beta('prob_class_2', 0, None, None, 0)
    #
    #         denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)
    #
    #         P1 = exp(prob_class_1) / denom
    #         P2 = exp(prob_class_2) / denom
    #         P3 = 1 / denom
    #
    #         prob = (
    #                 P1 * models.logit(V_class_1, av, choice) +
    #                 P2 * models.logit(V_class_2, av, choice) +
    #                 P3 * models.logit(V_class_3, av, choice)
    #         )
    #     elif latent == 32:
    #         # class 1 both
    #         # class 2 no time
    #         # class 3 no cost
    #
    #         if starting_point is not None:
    #             ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', starting_point[1], None, None, 0)
    #             BETA_COST = Beta('BETA_COST', starting_point[2], None, None, 0)
    #         else:
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #
    #         # Utilities
    #         Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
    #         Car_class_2 = BETA_TIME * car_time
    #         Car_class_3 = 0
    #
    #         Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #         Rail_class_2 = ASC_RAIL + BETA_TIME * rail_time
    #         Rail_class_3 = ASC_RAIL
    #
    #         V_class_1 = {0: Car_class_1,
    #                      1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #         V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #         V_class_3 = {0: Car_class_3, 1: Rail_class_3}
    #
    #         av = {0: 1, 1: 1}  # both are available to everyone
    #
    #         # specify likelihood function
    #         # Class membership
    #         if starting_point is not None:
    #             prob_class_1 = Beta('prob_class_1', starting_point[3], None, None, 0)
    #             prob_class_2 = Beta('prob_class_2', starting_point[4], None, None, 0)
    #         else:
    #             prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
    #             prob_class_2 = Beta('prob_class_2', 0, None, None, 0)
    #
    #         denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)
    #
    #         P1 = exp(prob_class_1) / denom
    #         P2 = exp(prob_class_2) / denom
    #         P3 = 1 / denom
    #
    #         prob = (
    #                 P1 * models.logit(V_class_1, av, choice) +
    #                 P2 * models.logit(V_class_2, av, choice) +
    #                 P3 * models.logit(V_class_3, av, choice)
    #         )
    #
    #     logprob = log(prob)
    #     biogeme = bio.BIOGEME(database, logprob)
    #     biogeme.modelName = "binary_netherlands"
    #     start_time = time.time()
    #     results = biogeme.estimate()
    #
    #     biog_loglike = round(results.data.logLike, 8)
    #     pandasResults = results.getEstimatedParameters()
    #     biog_beta = list(pandasResults["Value"])
    #     vpi = biog_beta[-2:]
    #     betas = list(pandasResults.index)
    #     print(pandasResults)
    #
    #     print("")
    #     print(f"Loglike = {biog_loglike}")
    #     print("Estimated class parameters:")
    #
    #     signi = False
    #
    #     if latent == 2 or latent == 22 or latent == 23:
    #         denom = np.exp(biog_beta[3]) + 1
    #
    #         P11 = np.exp(biog_beta[3]) / denom
    #         P22 = 1 / denom
    #         print("prob 1 = ", P11)
    #         print("prob 2 = ", P22)
    #
    #         print(f"Estimation time = {time.time() - start_time}s")
    #     elif latent == 3 or latent == 31:
    #         denom = np.exp(biog_beta[3]) + np.exp(biog_beta[4]) + 1
    #
    #         P11 = np.exp(biog_beta[3]) / denom
    #         P22 = np.exp(biog_beta[4]) / denom
    #         P33 = 1 / denom
    #         print("prob 1 = ", P11)
    #         print("prob 2 = ", P22)
    #         print("prob 3 = ", P33)
    #
    #         pvals = list(pandasResults["Rob. p-value"])
    #         firstp = pvals[3]
    #         secp = pvals[4]
    #
    #         a_level = 0.05
    #
    #         if firstp < a_level and secp < a_level and P11 > 0.1 and P22 > 0.1 and P33 > 0.1:
    #             signi = True
    #
    #         print("")
    #         print(f"Estimation time = {time.time() - start_time}s")
    #
    #         # biog_beta = biog_beta[0:len(biog_beta) - 2]
    #         # biog_beta = biog_beta + [p1, p2]
    #
    #         # loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
    #         loglike_conf = [-5, 5]
    #         beta_confs = dict()
    #         for k in range(len(betas)):
    #             beta_confs[k] = [-5, 5]  # we honestly don't care about this now
    #             # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]
    # elif toms_extremists:
    #     # class 1 both
    #     # class 2 have their own time beta
    #     # class 3 have their own cost beta
    #
    #     if starting_point is not None:
    #         ASC_RAIL = Beta("ASC_RAIL", starting_point[0], None, None, 0)
    #         BETA_TIME = Beta('BETA_TIME', starting_point[1], None, None, 0)
    #         BETA_COST = Beta('BETA_COST', starting_point[2], None, None, 0)
    #         BETA_TIME_C2 = Beta('BETA_TIME_C2', starting_point[3], None, None, 0)
    #         BETA_COST_C3 = Beta('BETA_COST_C3', starting_point[4], None, None, 0)
    #     else:
    #         ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #         BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #         BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #         BETA_TIME_C2 = Beta('BETA_TIME_C2', 0, None, None, 0)
    #         BETA_COST_C3 = Beta('BETA_COST_C3', 0, None, None, 0)
    #
    #     # Utilities
    #     Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
    #     Car_class_2 = BETA_TIME_C2 * car_time + BETA_COST * car_cost
    #     Car_class_3 = BETA_TIME * car_time + BETA_COST_C3 * car_cost
    #
    #     Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #     Rail_class_2 = ASC_RAIL + BETA_TIME_C2 * rail_time + BETA_COST * rail_cost
    #     Rail_class_3 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST_C3 * rail_cost
    #
    #     V_class_1 = {0: Car_class_1,
    #                  1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #     V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #     V_class_3 = {0: Car_class_3, 1: Rail_class_3}
    #
    #     av = {0: 1, 1: 1}  # both are available to everyone
    #
    #     # specify likelihood function
    #     # Class membership
    #     if starting_point is not None:
    #         prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
    #         prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
    #     else:
    #         prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
    #         prob_class_2 = Beta('prob_class_2', 0, None, None, 0)
    #
    #     denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)
    #
    #     P1 = exp(prob_class_1) / denom
    #     P2 = exp(prob_class_2) / denom
    #     P3 = 1 / denom
    #
    #     prob = (
    #             P1 * models.logit(V_class_1, av, choice) +
    #             P2 * models.logit(V_class_2, av, choice) +
    #             P3 * models.logit(V_class_3, av, choice)
    #     )
    #
    #     logprob = log(prob)
    #     biogeme = bio.BIOGEME(database, logprob)
    #     biogeme.modelName = "binary_netherlands"
    #     start_time = time.time()
    #     results = biogeme.estimate()
    #
    #     biog_loglike = round(results.data.logLike, 8)
    #     pandasResults = results.getEstimatedParameters()
    #     biog_beta = list(pandasResults["Value"])
    #     vpi = biog_beta[-2:]
    #     betas = list(pandasResults.index)
    #     print(pandasResults)
    #
    #     print("")
    #     print(f"Loglike = {biog_loglike}")
    #     print("Estimated class parameters:")
    #
    #     signi = False
    #
    #     denom = np.exp(biog_beta[5]) + np.exp(biog_beta[6]) + 1
    #
    #     P11 = np.exp(biog_beta[5]) / denom
    #     P22 = np.exp(biog_beta[6]) / denom
    #     P33 = 1 / denom
    #     print("prob 1 = ", P11)
    #     print("prob 1 = ", P22)
    #     print("prob 1 = ", P33)
    #
    #     pvals = list(pandasResults["Rob. p-value"])
    #     firstp = pvals[5]
    #     secp = pvals[6]
    #
    #     a_level = 0.05
    #
    #     if firstp < a_level and secp < a_level and P11 > 0.1 and P22 > 0.1 and P33 > 0.1:
    #         signi = True
    #
    #     print("")
    #
    #     print(f"Estimation time = {time.time() - start_time}s")
    #
    #     # biog_beta = biog_beta[0:len(biog_beta) - 2]
    #     # biog_beta = biog_beta + [p1, p2]
    #
    #     # loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
    #     loglike_conf = [-5, 5]
    #     beta_confs = dict()
    #     for k in range(len(betas)):
    #         beta_confs[k] = [-5, 5]  # we honestly don't care about this now
    #         # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]
    # else:
    #     if altSpec:
    #         K = 0
    #         K += 1
    #         BETA_TIME_CAR = Beta('BETA_TIME_CAR', 0, None, None, 0)
    #         K += 1
    #         BETA_TIME_RAIL = Beta('BETA_TIME_RAIL', 0, None, None, 0)
    #         K += 1
    #         ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #         # K += 1
    #         # BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #
    #         # Utilities
    #         Car = BETA_TIME_CAR * car_time
    #         Rail = ASC_RAIL + BETA_TIME_RAIL * rail_time
    #
    #         V = {0: Car, 1: Rail}  # deterministic part: use utilities depending on mode indicator
    #         av = {0: 1, 1: 1}  # both are available to everyone
    #
    #         logprob = loglogit(V, av, choice)
    #
    #         # specify formulas for simulation
    #         formulas = {
    #             "BETA_TIME_CAR": BETA_TIME_CAR,
    #             "BETA_TIME_RAIL": BETA_TIME_RAIL,
    #             "ASC_RAIL": ASC_RAIL,
    #             "loglike": logprob
    #         }
    #
    #         biogeme = bio.BIOGEME(database, formulas)
    #         biogeme.modelName = "binary_netherlands"
    #
    #         start_time = time.time()
    #         if K > 1:
    #             results = biogeme.estimate(bootstrap=100)
    #         else:
    #             results = biogeme.estimate()
    #
    #         biog_loglike = round(results.data.logLike, 8)
    #         pandasResults = results.getEstimatedParameters()
    #         betas = list(pandasResults.index)
    #         biog_beta = list(pandasResults["Value"])
    #
    #         # Get confidence intervals on beta and loglike
    #         if K > 1:
    #             b = results.getBetasForSensitivityAnalysis(betas, useBootstrap=False, size=100)
    #         else:
    #             b = results.getBetasForSensitivityAnalysis(betas, useBootstrap=False, size=100)
    #         # simulatedValues = biogeme.simulate(results.getBetaValues())
    #
    #         # results.getEstimatedParameters()
    #
    #         left, right = biogeme.confidenceIntervals(b, 0.9)
    #
    #         loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
    #         beta_confs = dict()
    #         for i in range(K):
    #             beta_confs[i] = [left[betas[i]].iloc[0], right[betas[i]].iloc[0]]
    #
    #         print(pandasResults)
    #
    #         print("")
    #         print(f"Loglike = {biog_loglike}")
    #         print("")
    #         print(f"Estimation time = {time.time() - start_time}s")
    #     else:
    #         if not latent >= 1:
    #             K = 0
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             K += 1
    #             if intercept:
    #                 K += 1
    #                 ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             if cost:
    #                 K += 1
    #                 BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #
    #             # Utilities
    #             if cost:
    #                 Car = BETA_TIME * car_time + BETA_COST * car_cost
    #             else:
    #                 Car = BETA_TIME * car_time
    #             if intercept:
    #                 if cost:
    #                     Rail = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #                 else:
    #                     Rail = ASC_RAIL + BETA_TIME * rail_time
    #             else:
    #                 if cost:
    #                     Rail = BETA_TIME * rail_time + BETA_COST * rail_cost
    #                 else:
    #                     Rail = BETA_TIME * rail_time
    #
    #             V = {0: Car, 1: Rail}  # deterministic part: use utilities depending on mode indicator
    #             av = {0: 1, 1: 1}  # both are available to everyone
    #
    #             # specify likelihood function
    #             if probit:
    #                 # Associate choice probability with the numbering of alternatives
    #                 P = {0: bioNormalCdf(V[0] - V[1]), 1: bioNormalCdf(V[1] - V[0])}
    #
    #                 # Definition of the model. This is the contribution of each
    #                 # observation to the log likelihood function.
    #                 logprob = log(Elem(P, choice))
    #             else:
    #                 # The choice model is a logit, with availability conditions
    #                 logprob = loglogit(V, av, choice)
    #
    #             # specify formulas for simulation
    #             if intercept:
    #                 if cost:
    #                     formulas = {
    #                         "BETA_TIME": BETA_TIME,
    #                         "BETA_COST": BETA_COST,
    #                         "ASC_RAIL": ASC_RAIL,
    #                         "loglike": logprob
    #                     }
    #                 else:
    #                     formulas = {
    #                         "BETA_TIME": BETA_TIME,
    #                         "ASC_RAIL": ASC_RAIL,
    #                         "loglike": logprob
    #                     }
    #             else:
    #                 if cost:
    #                     formulas = {
    #                         "BETA_TIME": BETA_TIME,
    #                         "BETA_COST": BETA_COST,
    #                         "loglike": logprob
    #                     }
    #                 else:
    #                     formulas = {
    #                         "BETA_TIME": BETA_TIME,
    #                         "loglike": logprob
    #                     }
    #
    #             biogeme = bio.BIOGEME(database, formulas)
    #             biogeme.modelName = "binary_netherlands"
    #
    #             start_time = time.time()
    #             if K > 1:
    #                 results = biogeme.estimate(bootstrap=100)
    #             else:
    #                 results = biogeme.estimate()
    #
    #             biog_loglike = round(results.data.logLike, 8)
    #             pandasResults = results.getEstimatedParameters()
    #             betas = list(pandasResults.index)
    #             biog_beta = list(pandasResults["Value"])
    #
    #             # Get confidence intervals on beta and loglike
    #             if K > 1:
    #                 b = results.getBetasForSensitivityAnalysis(betas, useBootstrap=False, size=100)
    #             else:
    #                 b = results.getBetasForSensitivityAnalysis(betas, useBootstrap=False, size=100)
    #             # simulatedValues = biogeme.simulate(results.getBetaValues())
    #
    #             # results.getEstimatedParameters()
    #
    #             left, right = biogeme.confidenceIntervals(b, 0.9)
    #
    #             loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
    #             beta_confs = dict()
    #             for i in range(K):
    #                 beta_confs[i] = [left[betas[i]].iloc[0], right[betas[i]].iloc[0]]
    #
    #             print(pandasResults)
    #
    #             print("")
    #             print(f"Loglike = {biog_loglike}")
    #             print("")
    #             print(f"Estimation time = {time.time() - start_time}s")
    #
    #         elif latent == 2:
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #
    #             # Utilities
    #             Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
    #             Car_class_2 = BETA_COST * car_cost
    #
    #             Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #             Rail_class_2 = ASC_RAIL + BETA_COST * rail_cost
    #
    #             V_class_1 = {0: Car_class_1,
    #                          1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #             V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #
    #             av = {0: 1, 1: 1}  # both are available to everyone
    #
    #             # specify likelihood function
    #             prob_class_1 = Beta('prob_class_1', 0.5, 0, 1, 0)
    #             prob_class_2 = 1 - prob_class_1
    #
    #             prob = (
    #                     prob_class_1 * models.logit(V_class_1, av, choice) +
    #                     prob_class_2 * models.logit(V_class_2, av, choice)
    #             )
    #
    #             logprob = log(prob)
    #             biogeme = bio.BIOGEME(database, logprob)
    #             biogeme.modelName = "binary_netherlands"
    #             start_time = time.time()
    #             results = biogeme.estimate()
    #
    #             biog_loglike = round(results.data.logLike, 8)
    #             pandasResults = results.getEstimatedParameters()
    #             biog_beta = list(pandasResults["Value"])
    #             betas = list(pandasResults.index)
    #             print(pandasResults)
    #
    #             print("")
    #             print(f"Loglike = {biog_loglike}")
    #             print("")
    #             print(f"Estimation time = {time.time() - start_time}s")
    #
    #             # loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
    #             loglike_conf = [-5, 5]
    #             beta_confs = dict()
    #             for k in range(len(betas)):
    #                 beta_confs[k] = [-5, 5]  # we honestly don't care about this now
    #                 # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]
    #
    #         elif latent == 3:
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #
    #             # Utilities
    #             Car_class_1 = BETA_TIME * car_time
    #             Car_class_2 = BETA_COST * car_cost
    #             Car_class_3 = BETA_TIME * car_time + BETA_COST * car_cost
    #
    #             Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time
    #             Rail_class_2 = ASC_RAIL + BETA_COST * rail_cost
    #             Rail_class_3 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #
    #             V_class_1 = {0: Car_class_1,
    #                          1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #             V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #             V_class_3 = {0: Car_class_3, 1: Rail_class_3}
    #
    #             av = {0: 1, 1: 1}  # both are available to everyone
    #
    #             # specify likelihood function
    #             # Michels suggestion
    #             vp1 = Beta("vp1", 0, None, None, 0)
    #             vp2 = Beta("vp2", 0, None, None, 0)
    #             vp3 = 0
    #
    #             vp = {1: vp1, 2: vp2, 3: vp3}
    #
    #             prob_class_1 = models.logit(vp, None, 1)
    #             prob_class_2 = models.logit(vp, None, 2)
    #             prob_class_3 = models.logit(vp, None, 3)
    #
    #             prob = (
    #                     prob_class_1 * models.logit(V_class_1, av, choice) +
    #                     prob_class_2 * models.logit(V_class_2, av, choice) +
    #                     prob_class_3 * models.logit(V_class_3, av, choice)
    #             )
    #
    #             logprob = log(prob)
    #             biogeme = bio.BIOGEME(database, logprob)
    #             biogeme.modelName = "binary_netherlands"
    #             start_time = time.time()
    #             results = biogeme.estimate()
    #
    #             biog_loglike = round(results.data.logLike, 8)
    #             pandasResults = results.getEstimatedParameters()
    #             biog_beta = list(pandasResults["Value"])
    #             vpi = biog_beta[-2:]
    #             betas = list(pandasResults.index)
    #             print(pandasResults)
    #
    #             print("")
    #             print(f"Loglike = {biog_loglike}")
    #             print("Estimated class parameters:")
    #             p1 = np.exp(vpi[0]) / (np.exp(vpi[0]) + np.exp(vpi[1]) + np.exp(0))
    #             p2 = np.exp(vpi[1]) / (np.exp(vpi[0]) + np.exp(vpi[1]) + np.exp(0))
    #             p3 = 1 - p1 - p2
    #             p1 = round(p1, 8)
    #             p2 = round(p2, 8)
    #             p3 = round(p3, 8)
    #             print([p1, p2, p3], "sum = ", p1 + p2 + p3)
    #             print("")
    #             print(f"Estimation time = {time.time() - start_time}s")
    #
    #             # biog_beta = biog_beta[0:len(biog_beta) - 2]
    #             # biog_beta = biog_beta + [p1, p2]
    #
    #             # loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
    #             loglike_conf = [-5, 5]
    #             beta_confs = dict()
    #             for k in range(len(betas)):
    #                 beta_confs[k] = [-5, 5]  # we honestly don't care about this now
    #                 # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]
    #
    #         elif latent == 4:
    #             BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
    #             ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
    #             BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    #
    #             # Utilities
    #             Car_class_1 = BETA_TIME * car_time
    #             Car_class_2 = BETA_COST * car_cost
    #             Car_class_3 = BETA_TIME * car_time + BETA_COST * car_cost
    #             Car_class_4 = 0
    #
    #             Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time
    #             Rail_class_2 = ASC_RAIL + BETA_COST * rail_cost
    #             Rail_class_3 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
    #             Rail_class_4 = ASC_RAIL
    #
    #             V_class_1 = {0: Car_class_1,
    #                          1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
    #             V_class_2 = {0: Car_class_2, 1: Rail_class_2}
    #             V_class_3 = {0: Car_class_3, 1: Rail_class_3}
    #             V_class_4 = {0: Car_class_4, 1: Rail_class_4}
    #
    #             av = {0: 1, 1: 1}  # both are available to everyone
    #
    #             # specify likelihood function
    #             # prob_class_1 = Beta('prob_class_1', 0.25, 0, 1, 0)
    #             # prob_class_2 = Beta('prob_class_2', 0.25, 0, 1, 0)
    #             # prob_class_3 = Beta('prob_class_3', 0.25, 0, 1, 0)
    #             # prob_class_4 = 1 - prob_class_1 - prob_class_2 - prob_class_3
    #
    #             # Michels suggestion
    #             vp1 = Beta("vp1", 0, None, None, 0)
    #             vp2 = Beta("vp2", 0, None, None, 0)
    #             vp3 = Beta("vp3", 0, None, None, 0)
    #             vp4 = 0
    #
    #             vp = {1: vp1, 2: vp2, 3: vp3, 4: vp4}
    #
    #             prob_class_1 = models.logit(vp, None, 1)
    #             prob_class_2 = models.logit(vp, None, 2)
    #             prob_class_3 = models.logit(vp, None, 3)
    #             prob_class_4 = models.logit(vp, None, 4)
    #
    #             prob = (
    #                     prob_class_1 * models.logit(V_class_1, av, choice) +
    #                     prob_class_2 * models.logit(V_class_2, av, choice) +
    #                     prob_class_3 * models.logit(V_class_3, av, choice) +
    #                     prob_class_4 * models.logit(V_class_4, av, choice)
    #             )
    #
    #             logprob = log(prob)
    #             biogeme = bio.BIOGEME(database, logprob)
    #             biogeme.modelName = "binary_netherlands"
    #             start_time = time.time()
    #             results = biogeme.estimate()
    #
    #             biog_loglike = round(results.data.logLike, 8)
    #             pandasResults = results.getEstimatedParameters()
    #             biog_beta = list(pandasResults["Value"])
    #             vpi = biog_beta[-3:]
    #             betas = list(pandasResults.index)
    #             print(pandasResults)
    #
    #             print("")
    #             print(f"Loglike = {biog_loglike}")
    #             print("Estimated class parameters:")
    #             p1 = np.exp(vpi[0]) / (np.exp(vpi[0]) + np.exp(vpi[1]) + np.exp(vpi[2]) + np.exp(0))
    #             p2 = np.exp(vpi[1]) / (np.exp(vpi[0]) + np.exp(vpi[1]) + np.exp(vpi[2]) + np.exp(0))
    #             p3 = np.exp(vpi[2]) / (np.exp(vpi[0]) + np.exp(vpi[1]) + np.exp(vpi[2]) + np.exp(0))
    #             p4 = 1 - p1 - p2 - p3
    #             p1 = round(p1, 8)
    #             p2 = round(p2, 8)
    #             p3 = round(p3, 8)
    #             p4 = round(p4, 8)
    #             print([p1, p2, p3, p4], "sum = ", p1 + p2 + p3 + p4)
    #             print("")
    #             print(f"Estimation time = {time.time() - start_time}s")
    #
    #             biog_beta = biog_beta[0:len(biog_beta) - 3]
    #             biog_beta = biog_beta + [p1, p2, p3]
    #
    #             # loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
    #             loglike_conf = [-5, 5]
    #             beta_confs = dict()
    #             for k in range(len(betas)):
    #                 beta_confs[k] = [-5, 5]  # we honestly don't care about this now
    #                 # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]

    # Remove output files
    # os.remove("binary_netherlands.html")
    # os.remove("binary_netherlands.pickle")
    # os.remove("__binary_netherlands.iter")

    # # File extensions to clean up
    # extensions = ["*.iter", "*.html", "*.pickle"]
    #
    # # Iterate over each extension and remove matching files
    # for ext in extensions:
    #     files = glob.glob(ext)
    #     for file in files:
    #         try:
    #             os.remove(file)
    #         except FileNotFoundError:
    #             pass

    return biog_beta, biog_loglike, None, None


def biogeme_estimate_beta_nether_mixed(df, R, pandaSeed=1):
    print(f"Setting up BioGeme ML model with {R} draws")
    st_time = time.time()

    database = db.Database('netherlands', df)
    # pd.set_option('display.max_columns', 50)
    # print(df)

    CHOICE = Variable("choice")
    CAR_TIME = Variable("car_time")
    RAIL_TIME = Variable("rail_time")
    CAR_COST = Variable("car_cost")
    RAIL_COST = Variable("rail_cost")

    CAR_TIME_SCALED = database.DefineVariable('CAR_TIME_SCALED', CAR_TIME / 1)
    RAIL_TIME_SCALED = database.DefineVariable('RAIL_TIME_SCALED', RAIL_TIME / 1)
    CAR_COST_SCALED = database.DefineVariable('CAR_COST_SCALED', CAR_COST / 1)
    RAIL_COST_SCALED = database.DefineVariable('RAIL_COST_SCALED', RAIL_COST / 1)

    # Define a random parameter, normally distributed, designed to be used
    # for Monte-Carlo simulation
    BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)

    # It is advised not to use 0 as starting value for the following parameter.
    BETA_TIME_S = Beta('BETA_TIME_S', 1, None, None, 0)
    BETA_TIME_RND = BETA_TIME + BETA_TIME_S * bioDraws('BETA_TIME_RND', 'NORMAL')

    ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)

    BETA_COST = Beta('BETA_COST', 0, None, None, 0)

    # Utilities
    Car = BETA_TIME_RND * CAR_TIME_SCALED + BETA_COST * CAR_COST_SCALED
    Rail = ASC_RAIL + BETA_TIME_RND * RAIL_TIME_SCALED + BETA_COST * RAIL_COST_SCALED

    V = {0: Car, 1: Rail}  # deterministic part: use utilities depending on mode indicator
    av = {0: 1, 1: 1}  # both are available to everyone

    # specify likelihood function
    prob = models.logit(V, av, CHOICE)

    # We integrate over BETA_TIME_RND using Monte-Carlo
    logprob = log(MonteCarlo(prob))

    # Define level of verbosity
    logger = msg.bioMessage()
    # logger.setSilent()
    # logger.setWarning()
    logger.setGeneral()
    # logger.setDetailed()

    # Create the Biogeme object
    print(f"Nb. of MonteCarlo draws = {R}")
    biogeme = bio.BIOGEME(
        database, logprob, numberOfDraws=R, seed=pandaSeed, numberOfThreads=1)
    biogeme.modelName = 'mixed_binary_netherlands'

    elapsed = time.time() - st_time
    print(f"Setting up Biogeme model takes {elapsed}s")

    # Estimate the parameters
    print(f"Solving Biogeme model")
    st_time = time.time()
    results = biogeme.estimate()

    elapsed = time.time() - st_time
    print(f"Solving Biogeme model takes {elapsed}s")
    print("Extract Biogeme results")
    st_time = time.time()
    pandasResults = results.getEstimatedParameters()
    biog_loglike = round(results.data.logLike, 8)
    print(pandasResults)

    print("")
    print(f"Loglike = {biog_loglike}")

    betas = list(pandasResults.index)

    loglike_conf = [-5, 5]
    beta_confs = dict()
    for k in range(len(betas)):
        beta_confs[k] = [-5, 5]  # we honestly don't care about this now
        # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]

    # Get the results
    biog_beta = list(pandasResults["Value"])

    # Remove output files
    # os.remove("mixed_binary_netherlands.html")
    # os.remove("mixed_binary_netherlands.pickle")
    # os.remove("__mixed_binary_netherlands.iter")

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

    elapsed = time.time() - st_time
    print(f"Extracting Biogeme results takes {elapsed}s")

    return biog_beta, biog_loglike, beta_confs, loglike_conf, df


def biogeme_estimate_beta_london(df, one_param=False, probit=False):
    database = db.Database('lpmc', df)
    globals().update(database.variables)

    # Choice
    travel_mode_chosen = travel_mode

    # Parameters to be estimated
    beta_names = []
    if one_param:
        Beta_time = Beta('Beta_time', 0, None, None, 0)
        beta_names.append("Beta_time")
    else:
        ASC_Walk = Beta('ASC_Walk', 0, None, None, 1)
        ASC_Bike = Beta('ASC_Bike', 0, None, None, 0)
        ASC_PB = Beta('ASC_PB', 0, None, None, 0)
        ASC_Car = Beta('ASC_Car', 0, None, None, 0)
        Beta_cost = Beta('Beta_cost', 0, None, None, 0)
        Beta_time = Beta('Beta_time', 0, None, None, 0)

        beta_names.append("ASC_Bike")
        beta_names.append("ASC_PB")
        beta_names.append("ASC_Car")
        beta_names.append("Beta_cost")
        beta_names.append("Beta_time")

    # Define here arithmetic expressions for variables that are not directly available from the data
    Car_TT = database.DefineVariable('Car_TT', dur_driving)
    Walk_TT = database.DefineVariable('Walk_TT', dur_walking)
    PB_TT = database.DefineVariable('PB_TT', dur_pt_rail + dur_pt_bus + dur_pt_int + dur_pt_access)
    Bike_TT = database.DefineVariable('Bike_TT', dur_cycling)

    Car_cost = database.DefineVariable('Car_cost', cost_driving_fuel + cost_driving_ccharge)
    PB_cost = database.DefineVariable('PB_cost', cost_transit)

    # Utilities
    if one_param:
        Walk = Walk_TT * Beta_time
        Bike = Bike_TT * Beta_time
        PB = PB_TT * Beta_time
        Car = Car_TT * Beta_time
    else:
        Walk = ASC_Walk + Walk_TT * Beta_time
        Bike = ASC_Bike + Bike_TT * Beta_time
        PB = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        Car = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost

    V = {1: Walk, 2: Bike, 3: PB, 4: Car}
    av = {1: 1, 2: 1, 3: 1, 4: 1}

    if probit:
        # to be implemented

        # Associate choice probability with the numbering of alternatives
        max_1_2 = bioMax(V[1], V[2])
        max_1_2_3 = bioMax(max_1_2, V[3])
        max_V = bioMax(max_1_2_3, V[4])
        P = {1: bioNormalCdf(V[1] - max_V), 2: bioNormalCdf(V[2] - max_V),
             3: bioNormalCdf(V[3] - max_V), 4: bioNormalCdf(V[4] - max_V)}

        # Definition of the model. This is the contribution of each
        # observation to the log likelihood function.
        logprob = log(Elem(P, travel_mode_chosen))
    else:
        # The choice model is a logit, with availability conditions
        logprob = loglogit(V, av, travel_mode_chosen)

    if one_param:
        formulas = {
            "Beta_time": Beta_time,
            "loglike": logprob
        }
    else:
        formulas = {
            "ASC_Bike": ASC_Bike,
            "ASC_PB": ASC_PB,
            "ASC_Car": ASC_Car,
            "Beta_cost": Beta_cost,
            "Beta_time": Beta_time,
            "loglike": logprob
        }

    biogeme = bio.BIOGEME(database, formulas, numberOfThreads=1)
    timestamp = time.time()
    biogeme.modelName = f"lpmc_MNL_{timestamp}"

    startttime = time.time()
    results = biogeme.estimate()
    estimtime = time.time() - startttime
    # Get the results
    biog_loglike = round(results.data.logLike, 8)
    pandasResults = results.getEstimatedParameters()
    biog_beta = list(pandasResults["Value"])
    # beta_stderr = round(pandasResults["Std err"][0], 8)

    print(pandasResults)
    print(f"Likelihood: {biog_loglike}")
    print(f"Estimation time: {round(estimtime, 2)}s")

    return biog_beta, biog_loglike, None, None, None


def biogeme_estimate_beta_london_latent(df, latent, michels_classes, toms_extremists, starting_point=None, R=None, pandaSeed=1):
    df = df.rename(columns={'start_time': 'start_time_new'})
    database = db.Database('lpmc', df)
    globals().update(database.variables)

    # Choice
    travel_mode_chosen = travel_mode

    # Parameters to be estimated
    if starting_point is not None:

        # order :
        # ASC_Bike     -507.267922     38.420324   -13.203114  0.000000e+00
        # ASC_Car       -10.120993      1.836423    -5.511255  3.562844e-08
        # ASC_PB       -301.604857     22.654939   -13.312985  0.000000e+00
        # Beta_cost    -377.416129     28.753510   -13.125915  0.000000e+00
        # Beta_time    -309.155372     23.979588   -12.892439  0.000000e+00
        # prob_class_1   16.265855      1.000395    16.259432  0.000000e+00
        # prob_class_2   

        ASC_Walk = Beta('ASC_Walk', 0, None, None, 1)
        ASC_Bike = Beta('ASC_Bike', starting_point[0], None, None, 0)
        ASC_Car = Beta('ASC_Car', starting_point[1], None, None, 0)
        ASC_PB = Beta('ASC_PB', starting_point[2], None, None, 0)
        Beta_cost = Beta('Beta_cost', starting_point[3], None, None, 0)
        Beta_time = Beta('Beta_time', starting_point[4], None, None, 0)
        if latent == 2:
            Beta_time_C2 = Beta('Beta_time_C2', starting_point[5], None, None, 0)
        # Beta_WKD = Beta('Beta_WKD', 0, None, None, 0)
        # Beta_EB = Beta('Beta_EB', 0, None, None, 0)
    else:
        ASC_Walk = Beta('ASC_Walk', 0, None, None, 1)
        ASC_Bike = Beta('ASC_Bike', 0, None, None, 0)
        ASC_PB = Beta('ASC_PB', 0, None, None, 0)
        ASC_Car = Beta('ASC_Car', 0, None, None, 0)
        Beta_cost = Beta('Beta_cost', 0, None, None, 0)
        Beta_time = Beta('Beta_time', 0, None, None, 0)
        if latent == 2:
            Beta_time_C2 = Beta('Beta_time_C2', 0, None, None, 0)
        # Beta_WKD = Beta('Beta_WKD', 0, None, None, 0)
        # Beta_EB = Beta('Beta_EB', 0, None, None, 0)
    # if (not michels_classes) and (not toms_extremists):
    #     Beta_RH = Beta('Beta_RH', 0, None, None, 0)
    #     RushHour = database.DefineVariable('RushHour', ((start_time_new > 7) * (start_time_new < 9)) + (
    #             (start_time_new > 17) * (start_time_new < 19)))

    # Define here arithmetic expressions for variables that are not directly available from the data
    Car_TT = database.DefineVariable('Car_TT', dur_driving)
    Walk_TT = database.DefineVariable('Walk_TT', dur_walking)
    PB_TT = database.DefineVariable('PB_TT', dur_pt_rail + dur_pt_bus + dur_pt_int + dur_pt_access)
    Bike_TT = database.DefineVariable('Bike_TT', dur_cycling)

    Car_cost = database.DefineVariable('Car_cost', cost_driving_fuel + cost_driving_ccharge)
    PB_cost = database.DefineVariable('PB_cost', cost_transit)

    if 1020 <= latent <= 1029 or 1030 <= latent <= 1039:
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
        if latent == 1030:
            # lets say
            # 1.) mixed time
            # 2.) ASCs separate
            # 3.) Dont cosider car

            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [4, 5, 7, 8, 9]
            class_3_ks = [1, 2, 3, 4, 5]
            prob_inds = [10]
            extra_inds = [[1, 7], [2, 8], [3, 9]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
            class_3_av = [1, 2, 3]

            if starting_point is not None:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', starting_point[5], None, None, 0)
            else:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
            Beta_time_RND = Beta_time + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

            Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
            Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
            PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
            Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

            if starting_point is not None:
                Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
                Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', starting_point[6], None, None, 0)
                Z3_ASC_CAR = Beta('Z3_ASC_CAR', starting_point[7], None, None, 0)
                Z4_ASC_PB = Beta('Z4_ASC_PB', starting_point[8], None, None, 0)
            else:
                Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
                Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', 0, None, None, 0)
                Z3_ASC_CAR = Beta('Z3_ASC_CAR', 0, None, None, 0)
                Z4_ASC_PB = Beta('Z4_ASC_PB', 0, None, None, 0)

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

            # Class membership
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[9], None, None, 0)
                prob_class_2 = Beta('prob_class_2', starting_point[10], None, None, 0)
            else:
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

            logprob = log(MonteCarlo(prob))

        if latent == 1031:
            # lets say
            # 1.) mixed time
            # 2.) ASCs separate
            # 3.) Dont cosider walk and bike

            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [4, 5, 7, 8, 9]
            class_3_ks = [1, 2, 3, 4, 5]
            prob_inds = [10]
            extra_inds = [[1, 7], [2, 8], [3, 9]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
            class_3_av = [3, 4]

            if starting_point is not None:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', starting_point[5], None, None, 0)
            else:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
            Beta_time_RND = Beta_time + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

            Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
            Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
            PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
            Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

            if starting_point is not None:
                Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
                Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', starting_point[6], None, None, 0)
                Z3_ASC_CAR = Beta('Z3_ASC_CAR', starting_point[7], None, None, 0)
                Z4_ASC_PB = Beta('Z4_ASC_PB', starting_point[8], None, None, 0)
            else:
                Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
                Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', 0, None, None, 0)
                Z3_ASC_CAR = Beta('Z3_ASC_CAR', 0, None, None, 0)
                Z4_ASC_PB = Beta('Z4_ASC_PB', 0, None, None, 0)

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

            # Class membership
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[9], None, None, 0)
                prob_class_2 = Beta('prob_class_2', starting_point[10], None, None, 0)
            else:
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

            logprob = log(MonteCarlo(prob))

        if latent == 1020:
            # lets say
            # 1.) mixed time
            # 2.) ASCs separate, no car

            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [4, 5, 7, 8]
            prob_inds = [9]
            extra_inds = [[1, 7], [3, 8]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3]

            if starting_point is not None:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', starting_point[5], None, None, 0)
            else:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
            Beta_time_RND = Beta_time + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

            Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
            Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
            PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
            Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

            if starting_point is not None:
                Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
                Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', starting_point[6], None, None, 0)
                Z4_ASC_PB = Beta('Z4_ASC_PB', starting_point[7], None, None, 0)
            else:
                Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
                Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', 0, None, None, 0)
                Z4_ASC_PB = Beta('Z4_ASC_PB', 0, None, None, 0)

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

            # Class membership
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                    P2 * models.logit(V_class_2, av2, travel_mode_chosen)
            )

            logprob = log(MonteCarlo(prob))

        if latent == 1021:
            # lets say
            # 1.) mixed time
            # 2.) ASCs separate, lazy

            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [4, 5, 7]
            prob_inds = [8]
            extra_inds = [[3, 7]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [3, 4]

            if starting_point is not None:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', starting_point[5], None, None, 0)
            else:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
            Beta_time_RND = Beta_time + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

            Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
            Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
            PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
            Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

            if starting_point is not None:
                Z4_ASC_PB = Beta('Z4_ASC_PB', starting_point[6], None, None, 0)
            else:
                Z4_ASC_PB = Beta('Z4_ASC_PB', 0, None, None, 0)

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

            # Class membership
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                    P2 * models.logit(V_class_2, av2, travel_mode_chosen)
            )

            logprob = log(MonteCarlo(prob))

        if latent == 1022 or latent == 1027:
            # lets say
            # 1.) mixed time
            # 2.) ASCs separate

            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [4, 5, 7, 8, 9]
            prob_inds = [10]
            extra_inds = [[1, 7], [2, 8], [3, 9]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]

            if starting_point is not None:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', starting_point[5], None, None, 0)
            else:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
            Beta_time_RND = Beta_time + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

            Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
            Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
            PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
            Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

            if starting_point is not None:
                Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
                Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', starting_point[6], None, None, 0)
                Z3_ASC_CAR = Beta('Z3_ASC_CAR', starting_point[7], None, None, 0)
                Z4_ASC_PB = Beta('Z4_ASC_PB', starting_point[8], None, None, 0)
            else:
                Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
                Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', 0, None, None, 0)
                Z3_ASC_CAR = Beta('Z3_ASC_CAR', 0, None, None, 0)
                Z4_ASC_PB = Beta('Z4_ASC_PB', 0, None, None, 0)

            Walk_class_2 = Z1_ASC_WALK + Walk_TT * Beta_time
            Bike_class_2 = Z2_ASC_BIKE + Bike_TT * Beta_time
            Car_class_2 = Z3_ASC_CAR + Car_TT * Beta_time + Car_cost * Beta_cost
            PB_class_2 = Z4_ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

            # Associate utility functions with the numbering of alternatives
            V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
            V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

            # Associate the availability conditions with the alternatives (everything is av to all)
            av = {1: 1, 2: 1, 3: 1, 4: 1}

            # Class membership
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[9], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                    P2 * models.logit(V_class_2, av, travel_mode_chosen)
            )

            logprob = log(MonteCarlo(prob))

        if latent == 1023:
            # lets say
            # 1.) mixed time
            # 2.) no car

            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 4, 5]
            prob_inds = [7]
            extra_inds = []
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3]

            if starting_point is not None:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', starting_point[5], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[6], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                    P2 * models.logit(V_class_2, av2, travel_mode_chosen)
            )

            logprob = log(MonteCarlo(prob))

        if latent == 1024:
            # lets say
            # 1.) mixed time
            # 2.) lazy

            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 4, 5]
            prob_inds = [7]
            extra_inds = []
            class_1_av = [1, 2, 3, 4]
            class_2_av = [3, 4]

            if starting_point is not None:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', starting_point[5], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[6], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                    P2 * models.logit(V_class_2, av2, travel_mode_chosen)
            )

            logprob = log(MonteCarlo(prob))
        if latent == 1025:
            # lets say
            # 1.) mixed costs
            # 2.) new beta time

            # ASC_Walk = Beta('ASC_Walk', 0, None, None, 1)
            #
            # 1 ASC_Bike = Beta('ASC_Bike', starting_point[0], None, None, 0)
            # 2 ASC_Car = Beta('ASC_Car', starting_point[1], None, None, 0)
            # 3 ASC_PB = Beta('ASC_PB', starting_point[2], None, None, 0)
            # 4 Beta_cost = Beta('Beta_cost', starting_point[3], None, None, 0)
            # 5 Beta_time = Beta('Beta_time', starting_point[4], None, None, 0)

            mix_inds = [[4, 6]]  # mix costs in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 4, 7]  # only mean costs, new beta time
            prob_inds = [8]
            extra_inds = [[5, 7]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]

            if starting_point is not None:
                Z01_Beta_cost_S = Beta('Z01_Beta_cost_S', starting_point[5], None, None, 0)
            else:
                Z01_Beta_cost_S = Beta('Z01_Beta_cost_S', 1, None, None, 0)
            Beta_cost_RND = Beta_cost + Z01_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

            Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
            Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
            PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost_RND
            Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost_RND

            if starting_point is not None:
                Z1_Beta_time_C2 = Beta('Z1_Beta_time_C2', starting_point[6], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                    P2 * models.logit(V_class_2, av, travel_mode_chosen)
            )

            logprob = log(MonteCarlo(prob))
        if latent == 1032:
            # lets say
            # 1.) mixed costs
            # 2.) new beta time
            # 3.) lazy

            # ASC_Walk = Beta('ASC_Walk', 0, None, None, 1)
            #
            # 1 ASC_Bike = Beta('ASC_Bike', starting_point[0], None, None, 0)
            # 2 ASC_Car = Beta('ASC_Car', starting_point[1], None, None, 0)
            # 3 ASC_PB = Beta('ASC_PB', starting_point[2], None, None, 0)
            # 4 Beta_cost = Beta('Beta_cost', starting_point[3], None, None, 0)
            # 5 Beta_time = Beta('Beta_time', starting_point[4], None, None, 0)

            mix_inds = [[4, 6]]  # mix costs in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 4, 7]  # only mean costs, new beta time
            class_3_ks = [1, 2, 3, 4, 5]  # only mean costs, new beta time
            prob_inds = [8, 9]
            extra_inds = [[5, 7]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
            class_3_av = [3, 4]

            if starting_point is not None:
                Z01_Beta_cost_S = Beta('Z01_Beta_cost_S', starting_point[5], None, None, 0)
            else:
                Z01_Beta_cost_S = Beta('Z01_Beta_cost_S', 1, None, None, 0)
            Beta_cost_RND = Beta_cost + Z01_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

            Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
            Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
            PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost_RND
            Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost_RND

            if starting_point is not None:
                Z1_Beta_time_C2 = Beta('Z1_Beta_time_C2', starting_point[6], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
                prob_class_2 = Beta('prob_class_2', starting_point[8], None, None, 0)
            else:
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

            logprob = log(MonteCarlo(prob))
        if latent == 1033:
            # lets say
            # 1.) mixed costs
            # 2.) new beta time
            # 3.) no car

            # ASC_Walk = Beta('ASC_Walk', 0, None, None, 1)
            #
            # 1 ASC_Bike = Beta('ASC_Bike', starting_point[0], None, None, 0)
            # 2 ASC_Car = Beta('ASC_Car', starting_point[1], None, None, 0)
            # 3 ASC_PB = Beta('ASC_PB', starting_point[2], None, None, 0)
            # 4 Beta_cost = Beta('Beta_cost', starting_point[3], None, None, 0)
            # 5 Beta_time = Beta('Beta_time', starting_point[4], None, None, 0)

            mix_inds = [[4, 6]]  # mix costs in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 4, 7]  # only mean costs, new beta time
            class_3_ks = [1, 2, 3, 4, 5]  # only mean costs, new beta time
            prob_inds = [8, 9]
            extra_inds = [[5, 7]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
            class_3_av = [1, 2, 3]

            if starting_point is not None:
                Z01_Beta_cost_S = Beta('Z01_Beta_cost_S', starting_point[5], None, None, 0)
            else:
                Z01_Beta_cost_S = Beta('Z01_Beta_cost_S', 1, None, None, 0)
            Beta_cost_RND = Beta_cost + Z01_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

            Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
            Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
            PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost_RND
            Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost_RND

            if starting_point is not None:
                Z1_Beta_time_C2 = Beta('Z1_Beta_time_C2', starting_point[6], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
                prob_class_2 = Beta('prob_class_2', starting_point[8], None, None, 0)
            else:
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

            logprob = log(MonteCarlo(prob))
        if latent == 1034:
            # lets say
            # 1.) mixed time
            # 2.) new beta cost
            # 3.) lazy

            # ASC_Walk = Beta('ASC_Walk', 0, None, None, 1)
            #
            # 1 ASC_Bike = Beta('ASC_Bike', starting_point[0], None, None, 0)
            # 2 ASC_Car = Beta('ASC_Car', starting_point[1], None, None, 0)
            # 3 ASC_PB = Beta('ASC_PB', starting_point[2], None, None, 0)
            # 4 Beta_cost = Beta('Beta_cost', starting_point[3], None, None, 0)
            # 5 Beta_time = Beta('Beta_time', starting_point[4], None, None, 0)

            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 5, 7]  # new beta cost
            class_3_ks = [1, 2, 3, 4, 5]  # lazy
            prob_inds = [8, 9]
            extra_inds = [[4, 7]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
            class_3_av = [3, 4]

            if starting_point is not None:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', starting_point[5], None, None, 0)
            else:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
            Beta_time_RND = Beta_cost + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

            Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
            Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
            PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
            Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

            if starting_point is not None:
                Z1_Beta_cost_C2 = Beta('Z1_Beta_cost_C2', starting_point[6], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
                prob_class_2 = Beta('prob_class_2', starting_point[8], None, None, 0)
            else:
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

            logprob = log(MonteCarlo(prob))
        if latent == 1026:
            # lets say
            # 1.) mixed time
            # 2.) new beta cost

            # ASC_Walk = Beta('ASC_Walk', 0, None, None, 1)
            #
            # 1 ASC_Bike = Beta('ASC_Bike', starting_point[0], None, None, 0)
            # 2 ASC_Car = Beta('ASC_Car', starting_point[1], None, None, 0)
            # 3 ASC_PB = Beta('ASC_PB', starting_point[2], None, None, 0)
            # 4 Beta_cost = Beta('Beta_cost', starting_point[3], None, None, 0)
            # 5 Beta_time = Beta('Beta_time', starting_point[4], None, None, 0)

            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 5, 7]  # new beta cost
            prob_inds = [8]
            extra_inds = [[4, 7]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]

            if starting_point is not None:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', starting_point[5], None, None, 0)
            else:
                Z01_Beta_time_S = Beta('Z01_Beta_time_S', 1, None, None, 0)
            Beta_time_RND = Beta_cost + Z01_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

            Walk_class_1 = ASC_Walk + Walk_TT * Beta_time_RND
            Bike_class_1 = ASC_Bike + Bike_TT * Beta_time_RND
            PB_class_1 = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
            Car_class_1 = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

            if starting_point is not None:
                Z1_Beta_cost_C2 = Beta('Z1_Beta_cost_C2', starting_point[6], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                    P2 * models.logit(V_class_2, av, travel_mode_chosen)
            )

            logprob = log(MonteCarlo(prob))

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
                Z2_Beta_cost_S = Beta('Z2_Beta_cost_S', starting_point[6], None, None, 0)
            else:
                Z1_Beta_time_S = Beta('Z1_Beta_time_S', 1, None, None, 0)
                Z2_Beta_cost_S = Beta('Z2_Beta_cost_S', 1, None, None, 0)
            Beta_time_RND = Beta_time + Z1_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')
            Beta_cost_RND = Beta_cost + Z2_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

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

        # We integrate over B_TIME_RND using Monte-Carlo
        logprob = log(MonteCarlo(prob))

    if latent == 2:
        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk + Walk_TT * Beta_time_C2

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike + Bike_TT * Beta_time_C2

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Beta_time_C2 + PB_cost * Beta_cost

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = ASC_Car + Car_TT * Beta_time_C2 + Car_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen)
        )

    if latent == 29:

        if starting_point is not None:
            Z_Beta_time_2 = Beta('Z_Beta_time_2', starting_point[5], None, None, 0)
        else:
            Z_Beta_time_2 = Beta('Z_Beta_time_2', 0, None, None, 0)

        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk + Walk_TT * Z_Beta_time_2

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike + Bike_TT * Z_Beta_time_2

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Z_Beta_time_2 + PB_cost * Beta_cost

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = ASC_Car + Car_TT * Z_Beta_time_2 + Car_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen)
        )

    if latent == 229:

        if starting_point is not None:
            Z_Beta_cost_2 = Beta('Z_Beta_cost_2', starting_point[5], None, None, 0)
            Z_Beta_time_2 = Beta('Z_Beta_time_2', starting_point[6], None, None, 0)
        else:
            Z_Beta_cost_2 = Beta('Z_Beta_cost_2', 0, None, None, 0)
            Z_Beta_time_2 = Beta('Z_Beta_time_2', 0, None, None, 0)

        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk + Walk_TT * Z_Beta_time_2

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike + Bike_TT * Z_Beta_time_2

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Z_Beta_time_2 + PB_cost * Z_Beta_cost_2

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = ASC_Car + Car_TT * Z_Beta_time_2 + Car_cost * Z_Beta_cost_2

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen)
        )

    if latent == 28:

        if starting_point is not None:
            Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
            Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', starting_point[5], None, None, 0)
            Z3_ASC_PB = Beta('Z3_ASC_PB', starting_point[6], None, None, 0)
            Z4_ASC_CAR = Beta('Z4_ASC_CAR', starting_point[7], None, None, 0)
        else:
            Z1_ASC_WALK = Beta('Z1_ASC_WALK', 0, None, None, 1)
            Z2_ASC_BIKE = Beta('Z2_ASC_BIKE', 0, None, None, 0)
            Z3_ASC_PB = Beta('Z3_ASC_PB', 0, None, None, 0)
            Z4_ASC_CAR = Beta('Z4_ASC_CAR', 0, None, None, 0)

        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = Z1_ASC_WALK + Walk_TT * Beta_time

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = Z2_ASC_BIKE + Bike_TT * Beta_time

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = Z3_ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = Z4_ASC_CAR + Car_TT * Beta_time + Car_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen)
        )
    if latent == 282:

        if starting_point is not None:
            Z4_ASC_CAR = Beta('Z4_ASC_CAR', starting_point[5], None, None, 0)
        else:
            Z4_ASC_CAR = Beta('Z4_ASC_CAR', 0, None, None, 0)

        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk + Walk_TT * Beta_time

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike + Bike_TT * Beta_time

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = Z4_ASC_CAR + Car_TT * Beta_time + Car_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen)
        )

    if latent == 21:
        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_cost * Beta_cost

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = ASC_Car + Car_cost * Beta_cost

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen)
        )
    if latent == 22:
        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk + Walk_TT * Beta_time

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike + Bike_TT * Beta_time

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Beta_time

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = ASC_Car + Car_TT * Beta_time

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen)
        )
    if latent == 23:
        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = ASC_PB

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = ASC_Car

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen)
        )
    if latent == 3:
        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk
        Walk_class_3 = ASC_Walk + Walk_TT * Beta_time

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike
        Bike_class_3 = ASC_Bike + Bike_TT * Beta_time

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_cost * Beta_cost
        PB_class_3 = ASC_PB + PB_TT * Beta_time

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = ASC_Car + Car_cost * Beta_cost
        Car_class_3 = ASC_Car + Car_TT * Beta_time

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen) +
                P3 * models.logit(V_class_3, av, travel_mode_chosen)
        )

    if latent == 39:
        if starting_point is not None:
            Z1_Beta_time_2 = Beta('Z1_Beta_time_2', starting_point[5], None, None, 0)
            Z2_Beta_cost_2 = Beta('Z2_Beta_cost_2', starting_point[6], None, None, 0)
        else:
            Z1_Beta_time_2 = Beta('Z1_Beta_time_2', 0, None, None, 0)
            Z2_Beta_cost_2 = Beta('Z2_Beta_cost_2', 0, None, None, 0)

        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk + Walk_TT * Z1_Beta_time_2
        Walk_class_3 = ASC_Walk + Walk_TT * Beta_time

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike + Bike_TT * Z1_Beta_time_2
        Bike_class_3 = ASC_Bike + Bike_TT * Beta_time

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Z1_Beta_time_2 + PB_cost * Beta_cost
        PB_class_3 = ASC_PB + PB_TT * Beta_time + PB_cost * Z2_Beta_cost_2

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = ASC_Car + Car_TT * Z1_Beta_time_2 + Car_cost * Beta_cost
        Car_class_3 = ASC_Car + Car_TT * Beta_time + Car_cost * Z2_Beta_cost_2
        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen) +
                P3 * models.logit(V_class_3, av, travel_mode_chosen)
        )

    if latent == 31:
        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk
        Walk_class_3 = ASC_Walk

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike
        Bike_class_3 = ASC_Bike

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_cost * Beta_cost
        PB_class_3 = ASC_PB

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = ASC_Car + Car_cost * Beta_cost
        Car_class_3 = ASC_Car

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen) +
                P3 * models.logit(V_class_3, av, travel_mode_chosen)
        )
    if latent == 32:
        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk
        Walk_class_3 = ASC_Walk

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike
        Bike_class_3 = ASC_Bike

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Beta_time
        PB_class_3 = ASC_PB

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = ASC_Car + Car_TT * Beta_time
        Car_class_3 = ASC_Car

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen) +
                P3 * models.logit(V_class_3, av, travel_mode_chosen)
        )
    if latent == 4:
        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_3 = ASC_Walk
        Walk_class_4 = ASC_Walk

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_3 = ASC_Bike
        Bike_class_4 = ASC_Bike

        PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_2 = ASC_PB + PB_TT * Beta_time
        PB_class_3 = ASC_PB + PB_cost * Beta_cost
        PB_class_4 = ASC_PB

        Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_2 = ASC_Car + Car_TT * Beta_time
        Car_class_3 = ASC_Car + Car_cost * Beta_cost
        Car_class_4 = ASC_Car

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}
        V_class_4 = {1: Walk_class_4, 2: Bike_class_4, 3: PB_class_4, 4: Car_class_4}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
            prob_class_3 = Beta('prob_class_3', starting_point[7], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)
            prob_class_3 = Beta('prob_class_3', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + exp(
            prob_class_3) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = exp(prob_class_3) / denom
        P4 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, travel_mode_chosen) +
                P2 * models.logit(V_class_2, av, travel_mode_chosen) +
                P3 * models.logit(V_class_3, av, travel_mode_chosen) +
                P4 * models.logit(V_class_4, av, travel_mode_chosen)
        )
        # elif toms_extremists:
        #     # Add parameters to be estimated
        #     if starting_point is not None:
        #         Beta_time_C2 = Beta('Beta_time_C2', starting_point[5], None, None, 0)
        #         Beta_cost_C3 = Beta('Beta_cost_C3', starting_point[6], None, None, 0)
        #     else:
        #         Beta_time_C2 = Beta('Beta_time_C2', 0, None, None, 0)
        #         Beta_cost_C3 = Beta('Beta_cost_C3', 0, None, None, 0)
        #
        #     # Utilities
        #     Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        #     Walk_class_2 = ASC_Walk + Walk_TT * Beta_time_C2
        #     Walk_class_3 = ASC_Walk + Walk_TT * Beta_time
        #
        #     Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        #     Bike_class_2 = ASC_Bike + Bike_TT * Beta_time_C2
        #     Bike_class_3 = ASC_Bike + Bike_TT * Beta_time
        #
        #     PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        #     PB_class_2 = ASC_PB + PB_TT * Beta_time_C2 + PB_cost * Beta_cost
        #     PB_class_3 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost_C3
        #
        #     Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        #     Car_class_2 = ASC_Car + Car_TT * Beta_time_C2 + Car_cost * Beta_cost
        #     Car_class_3 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost_C3
        #
        #     # Associate utility functions with the numbering of alternatives
        #     V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        #     V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        #     V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}
        #
        #     # Associate the availability conditions with the alternatives (everything is av to all)
        #     av = {1: 1, 2: 1, 3: 1, 4: 1}
        #
        #     # Class membership
        #     if starting_point is not None:
        #         prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
        #         prob_class_2 = Beta('prob_class_2', starting_point[8], None, None, 0)
        #     else:
        #         prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        #         prob_class_2 = Beta('prob_class_2', 0, None, None, 0)
        #
        #     denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)
        #
        #     P1 = exp(prob_class_1) / denom
        #     P2 = exp(prob_class_2) / denom
        #     P3 = 1 / denom
        #
        #     prob = (
        #             P1 * models.logit(V_class_1, av, travel_mode_chosen) +
        #             P2 * models.logit(V_class_2, av, travel_mode_chosen) +
        #             P3 * models.logit(V_class_3, av, travel_mode_chosen)
        #     )
        # else:
        #     seed_nr = pandaSeed + 1
        #     random.seed(seed_nr)
        #     np.random.seed(seed_nr)
        #     # factor = 1 means its inactive
        #     stressfactor = 1.6666  # class 2 value time more
        #     poor = 1  # class 3 cares a lot about costs
        #
        #     # what if class 3 cares a lot about whether or not its the weekend?
        #     # on the weekend they will be more likely to walk and use pt maybe?
        #     # because they have time and maybe less people?
        #
        #     # what if some people really care if its early in the morning? Maybe thats when they want to use PT or not?
        #     # Or maybe a group of people that cares about rush hour so they dont use PT then? ot car then?
        #
        #     # Utilities
        #     Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        #     Walk_class_2 = ASC_Walk + Walk_TT * stressfactor * Beta_time
        #     Walk_class_3 = ASC_Walk + Walk_TT * Beta_time
        #
        #     Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        #     Bike_class_2 = ASC_Bike + Bike_TT * stressfactor * Beta_time
        #     Bike_class_3 = ASC_Bike + Bike_TT * Beta_time
        #
        #     PB_class_1 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        #     PB_class_2 = ASC_PB + PB_TT * Beta_time * stressfactor + PB_cost * Beta_cost
        #     PB_class_3 = ASC_PB + PB_TT * Beta_time + PB_cost * poor * Beta_cost
        #
        #     Car_class_1 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        #     Car_class_2 = ASC_Car + Car_TT * Beta_time * stressfactor + Car_cost * Beta_cost
        #     Car_class_3 = ASC_Car + Car_TT * Beta_time + Car_cost * poor * Beta_cost + RushHour * Beta_RH
        #
        #     # Associate utility functions with the numbering of alternatives
        #     V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        #     V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        #     V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}
        #
        #     # Associate the availability conditions with the alternatives (everything is av to all)
        #     av = {1: 1, 2: 1, 3: 1, 4: 1}
        #
        #     # Class membership
        #     prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        #     prob_class_2 = Beta('prob_class_2', 0, None, None, 0)
        #
        #     denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)
        #
        #     P1 = exp(prob_class_1) / denom
        #     P2 = exp(prob_class_2) / denom
        #     P3 = 1 / denom
        #
        #     prob = (
        #             P1 * models.logit(V_class_1, av, travel_mode_chosen) +
        #             P2 * models.logit(V_class_2, av, travel_mode_chosen) +
        #             P3 * models.logit(V_class_3, av, travel_mode_chosen)
        #     )

    if latent == 89:
        # Utilities
        Walk_class_1 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_2 = ASC_Walk
        Walk_class_3 = ASC_Walk + Walk_TT * Beta_time
        Walk_class_4 = ASC_Walk

        Bike_class_1 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_2 = ASC_Bike
        Bike_class_3 = ASC_Bike + Bike_TT * Beta_time
        Bike_class_4 = ASC_Bike

        PB_class_1 = ASC_PB + PB_TT * Beta_time
        PB_class_2 = ASC_PB + PB_cost * Beta_cost
        PB_class_3 = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost
        PB_class_4 = ASC_PB

        Car_class_1 = ASC_Car + Car_TT * Beta_time
        Car_class_2 = ASC_Car + Car_cost * Beta_cost
        Car_class_3 = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost
        Car_class_4 = ASC_Car

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: Walk_class_1, 2: Bike_class_1, 3: PB_class_1, 4: Car_class_1}
        V_class_2 = {1: Walk_class_2, 2: Bike_class_2, 3: PB_class_2, 4: Car_class_2}
        V_class_3 = {1: Walk_class_3, 2: Bike_class_3, 3: PB_class_3, 4: Car_class_3}
        V_class_4 = {1: Walk_class_4, 2: Bike_class_4, 3: PB_class_4, 4: Car_class_4}

        # Associate the availability conditions with the alternatives (everything is av to all)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Class membership
        prob_class_1 = Beta('prob_class_1', 0.25, 0, 1, 0)
        prob_class_2 = Beta('prob_class_2', 0.25, 0, 1, 0)
        prob_class_3 = Beta('prob_class_3', 0.25, 0, 1, 0)
        prob_class_4 = 1 - (prob_class_1 + prob_class_2 + prob_class_3)

        prob = (
                prob_class_1 * models.logit(V_class_1, av, travel_mode_chosen) +
                prob_class_2 * models.logit(V_class_2, av, travel_mode_chosen) +
                prob_class_3 * models.logit(V_class_3, av, travel_mode_chosen) +
                prob_class_4 * models.logit(V_class_4, av, travel_mode_chosen)
        )

    if 10 <= latent <= 19 or 1020 <= latent <= 1039:
        # Create the Biogeme object
        print(f"Nb. of MonteCarlo draws = {R}")
        biogeme = bio.BIOGEME(
            database, logprob, numberOfDraws=R, seed=seed_nr, numberOfThreads=1
        )
        if 1020 <= latent <= 1039:
            biogeme.modelName = 'lpmc_mixed_latent'
        else:
            biogeme.modelName = 'lpmc_mixed'
        biogeme.saveIterations = False

        # Estimate the parameters
        st_time = time.time()
        results = biogeme.estimate()
        elapsed = time.time() - st_time
        print(f"Estimating Biogeme model takes {elapsed}s")
        pandasResults = results.getEstimatedParameters()
        biog_loglike = round(results.data.logLike, 8)
        print(pandasResults)

        print("")
        print(f"Loglike = {biog_loglike}")

        # Get the results
        biog_beta = list(pandasResults["Value"])

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
        return biog_beta, biog_loglike, None, None, df, None

    logprob = log(prob)
    biogeme = bio.BIOGEME(database, logprob, numberOfThreads=1)
    timestamp = time.time()
    biogeme.modelName = f'lpmc_latent_{timestamp}'
    start_time = time.time()
    results = biogeme.estimate()

    # Get the results in a pandas table
    pandasResults = results.getEstimatedParameters()
    biog_loglike = round(results.data.logLike, 8)
    betas = list(pandasResults.index)
    print(pandasResults)

    print("")
    print(f"Loglike = {biog_loglike}")
    print("")
    print(f"Estimation time = {time.time() - start_time}s")

    # Get the results
    biog_beta = list(pandasResults["Value"])

    signi = False

    if latent == 3 or latent == 4 or latent == 2:
        if michels_classes:
            if latent == 2:
                denom = np.exp(biog_beta[6]) + 1

                P11 = np.exp(biog_beta[6]) / denom
                P22 = 1 / denom
                print("prob 1 = ", P11)
                print("prob 2 = ", P22)
            if latent == 3:
                denom = np.exp(biog_beta[5]) + np.exp(biog_beta[6]) + 1

                P11 = np.exp(biog_beta[5]) / denom
                P22 = np.exp(biog_beta[6]) / denom
                P33 = 1 / denom
                print("prob 1 = ", P11)
                print("prob 2 = ", P22)
                print("prob 3 = ", P33)
            if latent == 4:
                denom = np.exp(biog_beta[5]) + np.exp(biog_beta[6]) + np.exp(biog_beta[7]) + 1

                P11 = np.exp(biog_beta[5]) / denom
                P22 = np.exp(biog_beta[6]) / denom
                P33 = np.exp(biog_beta[7]) / denom
                P44 = 1 / denom
                print("prob 1 = ", P11)
                print("prob 2 = ", P22)
                print("prob 3 = ", P33)
                print("prob 4 = ", P44)
        elif toms_extremists:
            denom = np.exp(biog_beta[7]) + np.exp(biog_beta[8]) + 1

            P11 = np.exp(biog_beta[7]) / denom
            P22 = np.exp(biog_beta[8]) / denom
            P33 = 1 / denom
            print("prob 1 = ", P11)
            print("prob 2 = ", P22)
            print("prob 3 = ", P33)
        else:
            denom = np.exp(biog_beta[6]) + np.exp(biog_beta[7]) + 1

            P11 = np.exp(biog_beta[6]) / denom
            P22 = np.exp(biog_beta[7]) / denom
            P33 = 1 / denom
            print("prob 1 = ", P11)
            print("prob 2 = ", P22)
            print("prob 3 = ", P33)

            pvals = list(pandasResults["Rob. p-value"])
            firstp = pvals[6]
            secp = pvals[7]

            a_level = 0.05

            if firstp < a_level and secp < a_level and P11 > 0.1 and P22 > 0.1 and P33 > 0.1:
                signi = True

    # loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
    loglike_conf = [-5, 5]
    beta_confs = dict()
    for k in range(len(betas)):
        beta_confs[k] = [-5, 5]  # we honestly don't care about this now
        # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]

    print("biog_beta = ", biog_beta)

    return biog_beta, biog_loglike, beta_confs, loglike_conf, signi, timestamp


def biogeme_estimate_telephone(df, latent, loadAttr=False, starting_point=None, R=None, pandaSeed=1):
    database = db.Database("telephone", df)
    globals().update(database.variables)

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

    if loadAttr:
        log_cost1 = database.DefineVariable('log_cost1', log(cost1))
        log_cost2 = database.DefineVariable('log_cost2', log(cost2))
        log_cost3 = database.DefineVariable('log_cost3', log(cost3))
        log_cost4 = database.DefineVariable('log_cost4', log(cost4))
        log_cost5 = database.DefineVariable('log_cost5', log(cost5))
        return None, None, None, None, df, None
    else:
        log_cost1 = Variable('log_cost1')
        log_cost2 = Variable('log_cost2')
        log_cost3 = Variable('log_cost3')
        log_cost4 = Variable('log_cost4')
        log_cost5 = Variable('log_cost5')

    if starting_point is not None:
        ASC_BM = Beta('ASC_BM', starting_point[0], None, None, 0)
        ASC_EF = Beta('ASC_EF', starting_point[1], None, None, 0)
        ASC_LF = Beta('ASC_LF', starting_point[2], None, None, 0)
        ASC_MF = Beta('ASC_MF', starting_point[3], None, None, 0)
        B_COST = Beta('B_COST', starting_point[4], None, None, 0)
    else:
        ASC_BM = Beta('ASC_BM', 0, None, None, 0)
        ASC_EF = Beta('ASC_EF', 0, None, None, 0)
        ASC_LF = Beta('ASC_LF', 0, None, None, 0)
        ASC_MF = Beta('ASC_MF', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)

    if 10 <= latent <= 19:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        # 1 ASC_BM = Beta('ASC_BM', starting_point[0], None, None, 0)
        # 2 ASC_EF = Beta('ASC_EF', starting_point[1], None, None, 0)
        # 3 ASC_LF = Beta('ASC_LF', starting_point[2], None, None, 0)
        # 4 ASC_MF = Beta('ASC_MF', starting_point[3], None, None, 0)
        # 5 B_COST = Beta('B_COST', starting_point[4], None, None, 0)

        mix_inds = None
        if latent == 10:
            mix_inds = [[5, 6]]  # mix cost
            if starting_point is not None:
                Z1_Beta_cost_S = Beta('Z1_Beta_cost_S', starting_point[5], None, None, 0)
            else:
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
            if starting_point is not None:
                Z1_Beta_cost_S = Beta('Z1_Beta_cost_S', starting_point[5], None, None, 0)
                Z2_ASC_BM_S = Beta('Z2_ASC_BM_S', starting_point[6], None, None, 0)
                Z3_ASC_EF_S = Beta('Z3_ASC_EF_S', starting_point[7], None, None, 0)
                Z4_ASC_LF_S = Beta('Z4_ASC_LF_S', starting_point[8], None, None, 0)
                Z5_ASC_MF_S = Beta('Z5_ASC_MF_S', starting_point[9], None, None, 0)
            else:
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
            if starting_point is not None:
                Z2_ASC_BM_S = Beta('Z2_ASC_BM_S', starting_point[5], None, None, 0)
                Z3_ASC_EF_S = Beta('Z3_ASC_EF_S', starting_point[6], None, None, 0)
                Z4_ASC_LF_S = Beta('Z4_ASC_LF_S', starting_point[7], None, None, 0)
                Z5_ASC_MF_S = Beta('Z5_ASC_MF_S', starting_point[8], None, None, 0)
            else:
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
        logprob = log(MonteCarlo(prob))

    if latent == 229:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z1_B_COST_2 = Beta('Z1_B_COST_2', starting_point[5], None, None, 0)
        else:
            Z1_B_COST_2 = Beta('Z1_B_COST_2', 0, None, None, 0)

        # Utilities
        V_BM_class_1 = ASC_BM + B_COST * log_cost1
        V_BM_class_2 = ASC_BM + Z1_B_COST_2 * log_cost1

        V_SM_class_1 = B_COST * log_cost2
        V_SM_class_2 = Z1_B_COST_2 * log_cost2

        V_LF_class_1 = ASC_LF + B_COST * log_cost3
        V_LF_class_2 = ASC_LF + Z1_B_COST_2 * log_cost3

        V_EF_class_1 = ASC_EF + B_COST * log_cost4
        V_EF_class_2 = ASC_EF + Z1_B_COST_2 * log_cost4

        V_MF_class_1 = ASC_MF + B_COST * log_cost5
        V_MF_class_2 = ASC_MF + Z1_B_COST_2 * log_cost5

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: V_BM_class_1, 2: V_SM_class_1, 3: V_LF_class_1, 4: V_EF_class_1, 5: V_MF_class_1}
        V_class_2 = {1: V_BM_class_2, 2: V_SM_class_2, 3: V_LF_class_2, 4: V_EF_class_2, 5: V_MF_class_2}
        av = {1: avail1, 2: avail2, 3: avail3, 4: avail4, 5: avail5}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, choice) +
                P2 * models.logit(V_class_2, av, choice)
        )

        logprob = log(prob)

    if latent == 28:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z1_ASC_BM = Beta('Z1_ASC_BM', starting_point[5], None, None, 0)
            Z2_ASC_EF = Beta('Z2_ASC_EF', starting_point[6], None, None, 0)
            Z3_ASC_LF = Beta('Z3_ASC_LF', starting_point[7], None, None, 0)
            Z4_ASC_MF = Beta('Z4_ASC_MF', starting_point[8], None, None, 0)
        else:
            Z1_ASC_BM = Beta('Z1_ASC_BM', 0, None, None, 0)
            Z2_ASC_EF = Beta('Z2_ASC_EF', 0, None, None, 0)
            Z3_ASC_LF = Beta('Z3_ASC_LF', 0, None, None, 0)
            Z4_ASC_MF = Beta('Z4_ASC_MF', 0, None, None, 0)

        # Utilities
        V_BM_class_1 = ASC_BM + B_COST * log_cost1
        V_BM_class_2 = Z1_ASC_BM + B_COST * log_cost1

        V_SM_class_1 = B_COST * log_cost2
        V_SM_class_2 = B_COST * log_cost2

        V_LF_class_1 = ASC_LF + B_COST * log_cost3
        V_LF_class_2 = Z3_ASC_LF + B_COST * log_cost3

        V_EF_class_1 = ASC_EF + B_COST * log_cost4
        V_EF_class_2 = Z2_ASC_EF + B_COST * log_cost4

        V_MF_class_1 = ASC_MF + B_COST * log_cost5
        V_MF_class_2 = Z4_ASC_MF + B_COST * log_cost5

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: V_BM_class_1, 2: V_SM_class_1, 3: V_LF_class_1, 4: V_EF_class_1, 5: V_MF_class_1}
        V_class_2 = {1: V_BM_class_2, 2: V_SM_class_2, 3: V_LF_class_2, 4: V_EF_class_2, 5: V_MF_class_2}
        av = {1: avail1, 2: avail2, 3: avail3, 4: avail4, 5: avail5}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[9], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, choice) +
                P2 * models.logit(V_class_2, av, choice)
        )

        logprob = log(prob)

    if latent == 282:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        # Utilities
        V_BM_class_1 = ASC_BM + B_COST * log_cost1
        V_BM_class_2 = ASC_BM

        V_SM_class_1 = B_COST * log_cost2
        V_SM_class_2 = 0

        V_LF_class_1 = ASC_LF + B_COST * log_cost3
        V_LF_class_2 = ASC_LF

        V_EF_class_1 = ASC_EF + B_COST * log_cost4
        V_EF_class_2 = ASC_EF

        V_MF_class_1 = ASC_MF + B_COST * log_cost5
        V_MF_class_2 = ASC_MF

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: V_BM_class_1, 2: V_SM_class_1, 3: V_LF_class_1, 4: V_EF_class_1, 5: V_MF_class_1}
        V_class_2 = {1: V_BM_class_2, 2: V_SM_class_2, 3: V_LF_class_2, 4: V_EF_class_2, 5: V_MF_class_2}
        av = {1: avail1, 2: avail2, 3: avail3, 4: avail4, 5: avail5}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, choice) +
                P2 * models.logit(V_class_2, av, choice)
        )

        logprob = log(prob)

    if latent == 0:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        # Utilities
        V_BM_class_1 = ASC_BM + B_COST * log_cost1
        V_SM_class_1 = B_COST * log_cost2
        V_LF_class_1 = ASC_LF + B_COST * log_cost3
        V_EF_class_1 = ASC_EF + B_COST * log_cost4
        V_MF_class_1 = ASC_MF + B_COST * log_cost5

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: V_BM_class_1, 2: V_SM_class_1, 3: V_LF_class_1, 4: V_EF_class_1, 5: V_MF_class_1}
        av = {1: avail1, 2: avail2, 3: avail3, 4: avail4, 5: avail5}

        logprob = loglogit(V_class_1, av, choice)

    if 10 <= latent <= 19:
        # Create the Biogeme object
        print(f"Nb. of MonteCarlo draws = {R}")
        biogeme = bio.BIOGEME(
            database, logprob, numberOfDraws=R, seed=pandaSeed, numberOfThreads=1
        )
        biogeme.modelName = 'telephone_mixed'
        biogeme.saveIterations = False

        # Estimate the parameters
        st_time = time.time()
        results = biogeme.estimate()
        elapsed = time.time() - st_time
        print(f"Estimating Biogeme model takes {elapsed}s")
        pandasResults = results.getEstimatedParameters()
        biog_loglike = round(results.data.logLike, 8)
        print(pandasResults)

        print("")
        print(f"Loglike = {biog_loglike}")

        # Get the results
        biog_beta = list(pandasResults["Value"])

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
    else:
        biogeme = bio.BIOGEME(database, logprob, numberOfThreads=1)
        timestamp = time.time()
        biogeme.modelName = f'telephone_latent_{latent}_{timestamp}'
        start_time = time.time()
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        results = biogeme.estimate()

        # Get the results in a pandas table
        pandasResults = results.getEstimatedParameters()
        biog_loglike = round(results.data.logLike, 8)
        print(pandasResults)

        print("")
        print(f"Loglike = {biog_loglike}")
        print("")
        print(f"Estimation time = {time.time() - start_time}s")

        # Get confidence intervals on beta and loglike
        betas = list(pandasResults.index)  # get BetaNames

        loglike_conf = [-5, 5]
        beta_confs = dict()
        for k in range(len(betas)):
            beta_confs[k] = [-5, 5]  # we honestly don't care about this now

        # Get the results
        biog_beta = list(pandasResults["Value"])

    return biog_beta, biog_loglike, None, None, df, None


def biogeme_estimate_optima(df, latent, loadAttr=False, starting_point=None, R=None, pandaSeed=1):
    database = db.Database("optima", df)

    TimePT = Variable('TimePT')
    Choice = Variable('Choice')
    TimeCar = Variable('TimeCar')
    MarginalCostPT = Variable('MarginalCostPT')
    CostCarCHF = Variable('CostCarCHF')
    distance_km = Variable('distance_km')
    TripPurpose = Variable('TripPurpose')
    WaitingTimePT = Variable('WaitingTimePT')

    if starting_point is not None:
        ASC_PT = Beta('ASC_PT', 0.0, None, None, 1)

        ASC_CAR = Beta('ASC_CAR', starting_point[0], None, None, 0)
        ASC_SM = Beta('ASC_SM', starting_point[1], None, None, 0)
        BETA_COST_HWH = Beta('BETA_COST_HWH', starting_point[2], None, None, 0)
        BETA_COST_OTHER = Beta('BETA_COST_OTHER', starting_point[3], None, None, 0)
        BETA_DIST = Beta('BETA_DIST', starting_point[4], None, None, 0)
        BETA_TIME_CAR = Beta('BETA_TIME_CAR', starting_point[5], None, None, 0)
        BETA_TIME_PT = Beta('BETA_TIME_PT', starting_point[6], None, None, 0)
        BETA_WAITING_TIME = Beta('BETA_WAITING_TIME', starting_point[7], None, None, 0)
    else:
        ASC_PT = Beta('ASC_PT', 0.0, None, None, 1)

        ASC_CAR = Beta('ASC_CAR', 0.0, None, None, 0)
        ASC_SM = Beta('ASC_SM', 0.0, None, None, 0)
        BETA_COST_HWH = Beta('BETA_COST_HWH', 0.0, None, None, 0)
        BETA_COST_OTHER = Beta('BETA_COST_OTHER', 0.0, None, None, 0)
        BETA_DIST = Beta('BETA_DIST', 0.0, None, None, 0)
        BETA_TIME_CAR = Beta('BETA_TIME_CAR', 0.0, None, None, 0)
        BETA_TIME_PT = Beta('BETA_TIME_PT', 0.0, None, None, 0)
        BETA_WAITING_TIME = Beta('BETA_WAITING_TIME', 0.0, None, None, 0)

    if loadAttr:
        TimePT_scaled = database.DefineVariable('TimePT_scaled', TimePT / 200)
        TimeCar_scaled = database.DefineVariable('TimeCar_scaled', TimeCar / 200)
        MarginalCostPT_scaled = \
            database.DefineVariable('MarginalCostPT_scaled', MarginalCostPT / 10)
        CostCarCHF_scaled = \
            database.DefineVariable('CostCarCHF_scaled', CostCarCHF / 10)
        distance_km_scaled = \
            database.DefineVariable('distance_km_scaled', distance_km / 5)
        PurpHWH = database.DefineVariable('PurpHWH', TripPurpose == 1)
        PurpOther = database.DefineVariable('PurpOther', TripPurpose != 1)
        return None, None, None, None, df, None
    else:
        TimePT_scaled = Variable('TimePT_scaled')
        TimeCar_scaled = Variable('TimeCar_scaled')
        MarginalCostPT_scaled = Variable('MarginalCostPT_scaled')
        CostCarCHF_scaled = Variable('CostCarCHF_scaled')
        distance_km_scaled = Variable('distance_km_scaled')
        PurpHWH = Variable('PurpHWH')
        PurpOther = Variable('PurpOther')

    if 10 <= latent <= 19:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

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

        mix_inds = None
        if latent == 10:
            mix_inds = [[6, 9], [7, 10]]  # mix just time
            if starting_point is not None:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', starting_point[8], None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', starting_point[9], None, None, 0)
            else:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', 1, None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', 1, None, None, 0)
            BETA_TIME_PT_RND = BETA_TIME_PT + Z1_BETA_TIME_PT_S * bioDraws('BETA_TIME_PT_RND', 'NORMAL')
            BETA_TIME_CAR_RND = BETA_TIME_CAR + Z2_BETA_TIME_CAR_S * bioDraws('BETA_TIME_CAR_RND', 'NORMAL')

            V0 = ASC_PT + \
                 BETA_TIME_PT_RND * TimePT_scaled + \
                 BETA_WAITING_TIME * WaitingTimePT + \
                 BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                 BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
            V1 = ASC_CAR + \
                 BETA_TIME_CAR_RND * TimeCar_scaled + \
                 BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                 BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
            V2 = ASC_SM + BETA_DIST * distance_km_scaled

        if latent == 18:
            mix_inds = [[1, 9]]  # mix just ASC Car
            if starting_point is not None:
                Z1_ASC_CAR_S = Beta('Z1_ASC_CAR_S', starting_point[8], None, None, 0)
            else:
                Z1_ASC_CAR_S = Beta('Z1_ASC_CAR_S', 1, None, None, 0)
            ASC_CAR_RND = ASC_CAR + Z1_ASC_CAR_S * bioDraws('ASC_CAR_RND', 'NORMAL')

            V0 = ASC_PT + \
                 BETA_TIME_PT * TimePT_scaled + \
                 BETA_WAITING_TIME * WaitingTimePT + \
                 BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                 BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
            V1 = ASC_CAR_RND + \
                 BETA_TIME_CAR * TimeCar_scaled + \
                 BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                 BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
            V2 = ASC_SM + BETA_DIST * distance_km_scaled

        elif latent == 11:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12]]  # mix time and costs
            if starting_point is not None:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', starting_point[8], None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', starting_point[9], None, None, 0)
                Z3_BETA_COST_HWH_S = Beta('Z3_BETA_COST_HWH_S', starting_point[10], None, None, 0)
                Z4_BETA_COST_OTHER_S = Beta('Z4_BETA_TIME_CAR_S', starting_point[11], None, None, 0)
            else:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', 1, None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', 1, None, None, 0)
                Z3_BETA_COST_HWH_S = Beta('Z3_BETA_COST_HWH_S', 1, None, None, 0)
                Z4_BETA_COST_OTHER_S = Beta('Z4_BETA_TIME_CAR_S', 1, None, None, 0)
            BETA_TIME_PT_RND = BETA_TIME_PT + Z1_BETA_TIME_PT_S * bioDraws('BETA_TIME_PT_RND', 'NORMAL')
            BETA_TIME_CAR_RND = BETA_TIME_CAR + Z2_BETA_TIME_CAR_S * bioDraws('BETA_TIME_CAR_RND', 'NORMAL')
            BETA_COST_HWH_RND = BETA_COST_HWH + Z3_BETA_COST_HWH_S * bioDraws('BETA_COST_HWH_RND', 'NORMAL')
            BETA_COST_OTHER_RND = BETA_COST_OTHER + Z4_BETA_COST_OTHER_S * bioDraws('BETA_COST_OTHER_RND', 'NORMAL')

            V0 = ASC_PT + \
                 BETA_TIME_PT_RND * TimePT_scaled + \
                 BETA_WAITING_TIME * WaitingTimePT + \
                 BETA_COST_HWH_RND * MarginalCostPT_scaled * PurpHWH + \
                 BETA_COST_OTHER_RND * MarginalCostPT_scaled * PurpOther
            V1 = ASC_CAR + \
                 BETA_TIME_CAR_RND * TimeCar_scaled + \
                 BETA_COST_HWH_RND * CostCarCHF_scaled * PurpHWH + \
                 BETA_COST_OTHER_RND * CostCarCHF_scaled * PurpOther
            V2 = ASC_SM + BETA_DIST * distance_km_scaled

        elif latent == 12:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [5, 13]]  # mix time and costs and distance
            if starting_point is not None:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', starting_point[8], None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', starting_point[9], None, None, 0)
                Z3_BETA_COST_HWH_S = Beta('Z3_BETA_COST_HWH_S', starting_point[10], None, None, 0)
                Z4_BETA_COST_OTHER_S = Beta('Z4_BETA_TIME_CAR_S', starting_point[11], None, None, 0)
                Z5_BETA_DIST_S = Beta('Z5_BETA_DIST_S', starting_point[12], None, None, 0)
            else:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', 1, None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', 1, None, None, 0)
                Z3_BETA_COST_HWH_S = Beta('Z3_BETA_COST_HWH_S', 1, None, None, 0)
                Z4_BETA_COST_OTHER_S = Beta('Z4_BETA_TIME_CAR_S', 1, None, None, 0)
                Z5_BETA_DIST_S = Beta('Z5_BETA_DIST_S', 1, None, None, 0)
            BETA_TIME_PT_RND = BETA_TIME_PT + Z1_BETA_TIME_PT_S * bioDraws('BETA_TIME_PT_RND', 'NORMAL')
            BETA_TIME_CAR_RND = BETA_TIME_CAR + Z2_BETA_TIME_CAR_S * bioDraws('BETA_TIME_CAR_RND', 'NORMAL')
            BETA_COST_HWH_RND = BETA_COST_HWH + Z3_BETA_COST_HWH_S * bioDraws('BETA_COST_HWH_RND', 'NORMAL')
            BETA_COST_OTHER_RND = BETA_COST_OTHER + Z4_BETA_COST_OTHER_S * bioDraws('BETA_COST_OTHER_RND', 'NORMAL')
            BETA_DIST_RND = BETA_DIST + Z5_BETA_DIST_S * bioDraws('BETA_DIST_RND', 'NORMAL')

            V0 = ASC_PT + \
                 BETA_TIME_PT_RND * TimePT_scaled + \
                 BETA_WAITING_TIME * WaitingTimePT + \
                 BETA_COST_HWH_RND * MarginalCostPT_scaled * PurpHWH + \
                 BETA_COST_OTHER_RND * MarginalCostPT_scaled * PurpOther
            V1 = ASC_CAR + \
                 BETA_TIME_CAR_RND * TimeCar_scaled + \
                 BETA_COST_HWH_RND * CostCarCHF_scaled * PurpHWH + \
                 BETA_COST_OTHER_RND * CostCarCHF_scaled * PurpOther
            V2 = ASC_SM + BETA_DIST_RND * distance_km_scaled

        elif latent == 13:
            mix_inds = [[6, 9], [7, 10], [5, 11]]  # mix time and distance
            if starting_point is not None:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', starting_point[8], None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', starting_point[9], None, None, 0)
                Z5_BETA_DIST_S = Beta('Z5_BETA_DIST_S', starting_point[10], None, None, 0)
            else:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', 1, None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', 1, None, None, 0)
                Z5_BETA_DIST_S = Beta('Z5_BETA_DIST_S', 1, None, None, 0)
            BETA_TIME_PT_RND = BETA_TIME_PT + Z1_BETA_TIME_PT_S * bioDraws('BETA_TIME_PT_RND', 'NORMAL')
            BETA_TIME_CAR_RND = BETA_TIME_CAR + Z2_BETA_TIME_CAR_S * bioDraws('BETA_TIME_CAR_RND', 'NORMAL')
            BETA_DIST_RND = BETA_DIST + Z5_BETA_DIST_S * bioDraws('BETA_DIST_RND', 'NORMAL')

            V0 = ASC_PT + \
                 BETA_TIME_PT_RND * TimePT_scaled + \
                 BETA_WAITING_TIME * WaitingTimePT + \
                 BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                 BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
            V1 = ASC_CAR + \
                 BETA_TIME_CAR_RND * TimeCar_scaled + \
                 BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                 BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
            V2 = ASC_SM + BETA_DIST_RND * distance_km_scaled

        elif latent == 14:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [5, 13], [1, 14], [2, 15]]  # mix time and costs and
            # distance and ASCs
            if starting_point is not None:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', starting_point[8], None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', starting_point[9], None, None, 0)
                Z3_BETA_COST_HWH_S = Beta('Z3_BETA_COST_HWH_S', starting_point[10], None, None, 0)
                Z4_BETA_COST_OTHER_S = Beta('Z4_BETA_TIME_CAR_S', starting_point[11], None, None, 0)
                Z5_BETA_DIST_S = Beta('Z5_BETA_DIST_S', starting_point[12], None, None, 0)
                Z6_ASC_CAR_S = Beta('Z6_ASC_CAR_S', starting_point[13], None, None, 0)
                Z7_ASC_SM_S = Beta('Z7_ASC_SM_S', starting_point[14], None, None, 0)
            else:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', 1, None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', 1, None, None, 0)
                Z3_BETA_COST_HWH_S = Beta('Z3_BETA_COST_HWH_S', 1, None, None, 0)
                Z4_BETA_COST_OTHER_S = Beta('Z4_BETA_TIME_CAR_S', 1, None, None, 0)
                Z5_BETA_DIST_S = Beta('Z5_BETA_DIST_S', 1, None, None, 0)
                Z6_ASC_CAR_S = Beta('Z6_ASC_CAR_S', 1, None, None, 0)
                Z7_ASC_SM_S = Beta('Z7_ASC_SM_S', 1, None, None, 0)
            BETA_TIME_PT_RND = BETA_TIME_PT + Z1_BETA_TIME_PT_S * bioDraws('BETA_TIME_PT_RND', 'NORMAL')
            BETA_TIME_CAR_RND = BETA_TIME_CAR + Z2_BETA_TIME_CAR_S * bioDraws('BETA_TIME_CAR_RND', 'NORMAL')
            BETA_COST_HWH_RND = BETA_COST_HWH + Z3_BETA_COST_HWH_S * bioDraws('BETA_COST_HWH_RND', 'NORMAL')
            BETA_COST_OTHER_RND = BETA_COST_OTHER + Z4_BETA_COST_OTHER_S * bioDraws('BETA_COST_OTHER_RND', 'NORMAL')
            BETA_DIST_RND = BETA_DIST + Z5_BETA_DIST_S * bioDraws('BETA_DIST_RND', 'NORMAL')
            ASC_CAR_RND = ASC_CAR + Z6_ASC_CAR_S * bioDraws('ASC_CAR_RND', 'NORMAL')
            ASC_SM_RND = ASC_SM + Z7_ASC_SM_S * bioDraws('ASC_SM_RND', 'NORMAL')

            V0 = ASC_PT + \
                 BETA_TIME_PT_RND * TimePT_scaled + \
                 BETA_WAITING_TIME * WaitingTimePT + \
                 BETA_COST_HWH_RND * MarginalCostPT_scaled * PurpHWH + \
                 BETA_COST_OTHER_RND * MarginalCostPT_scaled * PurpOther
            V1 = ASC_CAR_RND + \
                 BETA_TIME_CAR_RND * TimeCar_scaled + \
                 BETA_COST_HWH_RND * CostCarCHF_scaled * PurpHWH + \
                 BETA_COST_OTHER_RND * CostCarCHF_scaled * PurpOther
            V2 = ASC_SM_RND + BETA_DIST_RND * distance_km_scaled

        elif latent == 15:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [1, 13], [2, 14]]  # mix time and costs and ASCs
            if starting_point is not None:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', starting_point[8], None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', starting_point[9], None, None, 0)
                Z3_BETA_COST_HWH_S = Beta('Z3_BETA_COST_HWH_S', starting_point[10], None, None, 0)
                Z4_BETA_COST_OTHER_S = Beta('Z4_BETA_TIME_CAR_S', starting_point[11], None, None, 0)
                Z6_ASC_CAR_S = Beta('Z6_ASC_CAR_S', starting_point[12], None, None, 0)
                Z7_ASC_SM_S = Beta('Z7_ASC_SM_S', starting_point[13], None, None, 0)
            else:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', 1, None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', 1, None, None, 0)
                Z3_BETA_COST_HWH_S = Beta('Z3_BETA_COST_HWH_S', 1, None, None, 0)
                Z4_BETA_COST_OTHER_S = Beta('Z4_BETA_TIME_CAR_S', 1, None, None, 0)
                Z6_ASC_CAR_S = Beta('Z6_ASC_CAR_S', 1, None, None, 0)
                Z7_ASC_SM_S = Beta('Z7_ASC_SM_S', 1, None, None, 0)
            BETA_TIME_PT_RND = BETA_TIME_PT + Z1_BETA_TIME_PT_S * bioDraws('BETA_TIME_PT_RND', 'NORMAL')
            BETA_TIME_CAR_RND = BETA_TIME_CAR + Z2_BETA_TIME_CAR_S * bioDraws('BETA_TIME_CAR_RND', 'NORMAL')
            BETA_COST_HWH_RND = BETA_COST_HWH + Z3_BETA_COST_HWH_S * bioDraws('BETA_COST_HWH_RND', 'NORMAL')
            BETA_COST_OTHER_RND = BETA_COST_OTHER + Z4_BETA_COST_OTHER_S * bioDraws('BETA_COST_OTHER_RND', 'NORMAL')
            ASC_CAR_RND = ASC_CAR + Z6_ASC_CAR_S * bioDraws('ASC_CAR_RND', 'NORMAL')
            ASC_SM_RND = ASC_SM + Z7_ASC_SM_S * bioDraws('ASC_SM_RND', 'NORMAL')

            V0 = ASC_PT + \
                 BETA_TIME_PT_RND * TimePT_scaled + \
                 BETA_WAITING_TIME * WaitingTimePT + \
                 BETA_COST_HWH_RND * MarginalCostPT_scaled * PurpHWH + \
                 BETA_COST_OTHER_RND * MarginalCostPT_scaled * PurpOther
            V1 = ASC_CAR_RND + \
                 BETA_TIME_CAR_RND * TimeCar_scaled + \
                 BETA_COST_HWH_RND * CostCarCHF_scaled * PurpHWH + \
                 BETA_COST_OTHER_RND * CostCarCHF_scaled * PurpOther
            V2 = ASC_SM_RND + BETA_DIST * distance_km_scaled

        elif latent == 16:
            mix_inds = [[6, 9], [7, 10], [1, 11], [2, 12]]  # mix time and ASCs
            if starting_point is not None:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', starting_point[8], None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', starting_point[9], None, None, 0)
                Z6_ASC_CAR_S = Beta('Z6_ASC_CAR_S', starting_point[10], None, None, 0)
                Z7_ASC_SM_S = Beta('Z7_ASC_SM_S', starting_point[11], None, None, 0)
            else:
                Z1_BETA_TIME_PT_S = Beta('Z1_BETA_TIME_PT_S', 1, None, None, 0)
                Z2_BETA_TIME_CAR_S = Beta('Z2_BETA_TIME_CAR_S', 1, None, None, 0)
                Z6_ASC_CAR_S = Beta('Z6_ASC_CAR_S', 1, None, None, 0)
                Z7_ASC_SM_S = Beta('Z7_ASC_SM_S', 1, None, None, 0)
            BETA_TIME_PT_RND = BETA_TIME_PT + Z1_BETA_TIME_PT_S * bioDraws('BETA_TIME_PT_RND', 'NORMAL')
            BETA_TIME_CAR_RND = BETA_TIME_CAR + Z2_BETA_TIME_CAR_S * bioDraws('BETA_TIME_CAR_RND', 'NORMAL')
            ASC_CAR_RND = ASC_CAR + Z6_ASC_CAR_S * bioDraws('ASC_CAR_RND', 'NORMAL')
            ASC_SM_RND = ASC_SM + Z7_ASC_SM_S * bioDraws('ASC_SM_RND', 'NORMAL')

            V0 = ASC_PT + \
                 BETA_TIME_PT_RND * TimePT_scaled + \
                 BETA_WAITING_TIME * WaitingTimePT + \
                 BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                 BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
            V1 = ASC_CAR_RND + \
                 BETA_TIME_CAR_RND * TimeCar_scaled + \
                 BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                 BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
            V2 = ASC_SM_RND + BETA_DIST * distance_km_scaled

        elif latent == 17:
            mix_inds = [[1, 9], [2, 10]]  # mix ASCs only
            if starting_point is not None:
                Z6_ASC_CAR_S = Beta('Z6_ASC_CAR_S', starting_point[8], None, None, 0)
                Z7_ASC_SM_S = Beta('Z7_ASC_SM_S', starting_point[9], None, None, 0)
            else:
                Z6_ASC_CAR_S = Beta('Z6_ASC_CAR_S', 1, None, None, 0)
                Z7_ASC_SM_S = Beta('Z7_ASC_SM_S', 1, None, None, 0)
            ASC_CAR_RND = ASC_CAR + Z6_ASC_CAR_S * bioDraws('ASC_CAR_RND', 'NORMAL')
            ASC_SM_RND = ASC_SM + Z7_ASC_SM_S * bioDraws('ASC_SM_RND', 'NORMAL')

            V0 = ASC_PT + \
                 BETA_TIME_PT * TimePT_scaled + \
                 BETA_WAITING_TIME * WaitingTimePT + \
                 BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                 BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
            V1 = ASC_CAR_RND + \
                 BETA_TIME_CAR * TimeCar_scaled + \
                 BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                 BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
            V2 = ASC_SM_RND + BETA_DIST * distance_km_scaled
        else:
            V0 = ASC_PT + \
                 BETA_TIME_PT * TimePT_scaled + \
                 BETA_WAITING_TIME * WaitingTimePT + \
                 BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                 BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther

            V1 = ASC_CAR + \
                 BETA_TIME_CAR * TimeCar_scaled + \
                 BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                 BETA_COST_OTHER * CostCarCHF_scaled * PurpOther

            V2 = ASC_SM + BETA_DIST * distance_km_scaled

        # Associate utility functions with the numbering of alternatives
        V = {0: V0,
             1: V1,
             2: V2}

        # Associate the availability conditions with the alternatives
        av = {0: 1,
              1: 1,
              2: 1}

        # Conditional to B_TIME_RND, we have a logit model (called the kernel)
        prob = models.logit(V, av, Choice)

        # We integrate over B_TIME_RND using Monte-Carlo
        logprob = log(MonteCarlo(prob))

    if latent == 2:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        V0_class_1 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_2 = ASC_PT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther

        V1_class_1 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_2 = ASC_CAR + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther

        V2_class_1 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_2 = ASC_SM + BETA_DIST * distance_km_scaled

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {0: V0_class_1, 1: V1_class_1, 2: V2_class_1}
        V_class_2 = {0: V0_class_2, 1: V1_class_2, 2: V2_class_2}

        av = {0: 1, 1: 1, 2: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, Choice) +
                P2 * models.logit(V_class_2, av, Choice)
        )

        logprob = log(prob)

    if latent == 29:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z1_BETA_TIME_PT_2 = Beta('Z1_BETA_TIME_PT_2', starting_point[8], None, None, 0)
            Z2_BETA_TIME_CAR_2 = Beta('Z2_BETA_TIME_CAR_2', starting_point[9], None, None, 0)
        else:
            Z1_BETA_TIME_PT_2 = Beta('Z1_BETA_TIME_PT_2', 0, None, None, 0)
            Z2_BETA_TIME_CAR_2 = Beta('Z2_BETA_TIME_CAR_2', 0, None, None, 0)

        V0_class_1 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_2 = ASC_PT + \
                     Z1_BETA_TIME_PT_2 * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther

        V1_class_1 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_2 = ASC_CAR + \
                     Z2_BETA_TIME_CAR_2 * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther

        V2_class_1 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_2 = ASC_SM + BETA_DIST * distance_km_scaled

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {0: V0_class_1, 1: V1_class_1, 2: V2_class_1}
        V_class_2 = {0: V0_class_2, 1: V1_class_2, 2: V2_class_2}

        av = {0: 1, 1: 1, 2: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[10], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, Choice) +
                P2 * models.logit(V_class_2, av, Choice)
        )

        logprob = log(prob)

    if latent == 229:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z1_BETA_COST_HWH_2 = Beta('Z1_BETA_COST_HWH_2', starting_point[8], None, None, 0)
            Z2_BETA_COST_OTHER_2 = Beta('Z2_BETA_COST_OTHER_2', starting_point[9], None, None, 0)
            Z3_BETA_TIME_PT_2 = Beta('Z3_BETA_TIME_PT_2', starting_point[10], None, None, 0)
            Z4_BETA_TIME_CAR_2 = Beta('Z4_BETA_TIME_CAR_2', starting_point[11], None, None, 0)
        else:
            Z1_BETA_COST_HWH_2 = Beta('Z1_BETA_COST_HWH_2', 0, None, None, 0)
            Z2_BETA_COST_OTHER_2 = Beta('Z2_BETA_COST_OTHER_2', 0, None, None, 0)
            Z3_BETA_TIME_PT_2 = Beta('Z3_BETA_TIME_PT_2', 0, None, None, 0)
            Z4_BETA_TIME_CAR_2 = Beta('Z4_BETA_TIME_CAR_2', 0, None, None, 0)

        V0_class_1 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_2 = ASC_PT + \
                     Z3_BETA_TIME_PT_2 * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     Z1_BETA_COST_HWH_2 * MarginalCostPT_scaled * PurpHWH + \
                     Z2_BETA_COST_OTHER_2 * MarginalCostPT_scaled * PurpOther

        V1_class_1 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_2 = ASC_CAR + \
                     Z4_BETA_TIME_CAR_2 * TimeCar_scaled + \
                     Z1_BETA_COST_HWH_2 * CostCarCHF_scaled * PurpHWH + \
                     Z2_BETA_COST_OTHER_2 * CostCarCHF_scaled * PurpOther

        V2_class_1 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_2 = ASC_SM + BETA_DIST * distance_km_scaled

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {0: V0_class_1, 1: V1_class_1, 2: V2_class_1}
        V_class_2 = {0: V0_class_2, 1: V1_class_2, 2: V2_class_2}

        av = {0: 1, 1: 1, 2: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[12], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, Choice) +
                P2 * models.logit(V_class_2, av, Choice)
        )

        logprob = log(prob)

    if latent == 28:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z1_ASC_PT_2 = Beta('Z1_ASC_PT_2', 0, None, None, 1)
            Z2_ASC_CAR_2 = Beta('Z2_ASC_CAR_2', starting_point[8], None, None, 0)
            Z3_ASC_SM_2 = Beta('Z3_ASC_SM_2', starting_point[9], None, None, 0)
        else:
            Z1_ASC_PT_2 = Beta('Z1_ASC_PT_2', 0, None, None, 1)
            Z2_ASC_CAR_2 = Beta('Z2_ASC_CAR_2', 0, None, None, 0)
            Z3_ASC_SM_2 = Beta('Z3_ASC_SM_2', 0, None, None, 0)

        V0_class_1 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_2 = Z1_ASC_PT_2 + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther

        V1_class_1 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_2 = Z2_ASC_CAR_2 + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther

        V2_class_1 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_2 = Z3_ASC_SM_2 + BETA_DIST * distance_km_scaled

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {0: V0_class_1, 1: V1_class_1, 2: V2_class_1}
        V_class_2 = {0: V0_class_2, 1: V1_class_2, 2: V2_class_2}

        av = {0: 1, 1: 1, 2: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[10], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, Choice) +
                P2 * models.logit(V_class_2, av, Choice)
        )

        logprob = log(prob)

    if latent == 282:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z2_ASC_CAR_2 = Beta('Z2_ASC_CAR_2', starting_point[8], None, None, 0)
        else:
            Z2_ASC_CAR_2 = Beta('Z2_ASC_CAR_2', 0, None, None, 0)

        V0_class_1 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_2 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther

        V1_class_1 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_2 = Z2_ASC_CAR_2 + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther

        V2_class_1 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_2 = ASC_SM + BETA_DIST * distance_km_scaled

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {0: V0_class_1, 1: V1_class_1, 2: V2_class_1}
        V_class_2 = {0: V0_class_2, 1: V1_class_2, 2: V2_class_2}

        av = {0: 1, 1: 1, 2: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[9], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, Choice) +
                P2 * models.logit(V_class_2, av, Choice)
        )

        logprob = log(prob)

    if latent == 259:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z1_BETA_TIME_PT_2 = Beta('Z1_BETA_TIME_PT_2', starting_point[8], None, None, 0)
            Z2_BETA_TIME_CAR_2 = Beta('Z2_BETA_TIME_CAR_2', starting_point[9], None, None, 0)
            Z3_BETA_COST_HWH_2 = Beta('Z3_BETA_COST_HWH_2', starting_point[10], None, None, 0)
        else:
            Z1_BETA_TIME_PT_2 = Beta('Z1_BETA_TIME_PT_2', 0, None, None, 0)
            Z2_BETA_TIME_CAR_2 = Beta('Z2_BETA_TIME_CAR_2', 0, None, None, 0)
            Z3_BETA_COST_HWH_2 = Beta('Z3_BETA_COST_HWH_2', starting_point[10], None, None, 0)

        V0_class_1 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_2 = ASC_PT + \
                     Z1_BETA_TIME_PT_2 * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     Z3_BETA_COST_HWH_2 * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther

        V1_class_1 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_2 = ASC_CAR + \
                     Z2_BETA_TIME_CAR_2 * TimeCar_scaled + \
                     Z3_BETA_COST_HWH_2 * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther

        V2_class_1 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_2 = ASC_SM + BETA_DIST * distance_km_scaled

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {0: V0_class_1, 1: V1_class_1, 2: V2_class_1}
        V_class_2 = {0: V0_class_2, 1: V1_class_2, 2: V2_class_2}

        av = {0: 1, 1: 1, 2: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[11], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, Choice) +
                P2 * models.logit(V_class_2, av, Choice)
        )

        logprob = log(prob)

    if latent == 22:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        V0_class_1 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_2 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT

        V1_class_1 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_2 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled

        V2_class_1 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_2 = ASC_SM + BETA_DIST * distance_km_scaled

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {0: V0_class_1, 1: V1_class_1, 2: V2_class_1}
        V_class_2 = {0: V0_class_2, 1: V1_class_2, 2: V2_class_2}

        av = {0: 1, 1: 1, 2: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, Choice) +
                P2 * models.logit(V_class_2, av, Choice)
        )

        logprob = log(prob)
    if latent == 23:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        V0_class_1 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_2 = ASC_PT

        V1_class_1 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_2 = ASC_CAR

        V2_class_1 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_2 = ASC_SM

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {0: V0_class_1, 1: V1_class_1, 2: V2_class_1}
        V_class_2 = {0: V0_class_2, 1: V1_class_2, 2: V2_class_2}

        av = {0: 1, 1: 1, 2: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, Choice) +
                P2 * models.logit(V_class_2, av, Choice)
        )

        logprob = log(prob)

    if latent == 3:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        V0_class_1 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_2 = ASC_PT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_3 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT

        V1_class_1 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_2 = ASC_CAR + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_3 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled

        V2_class_1 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_2 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_3 = ASC_SM + BETA_DIST * distance_km_scaled

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {0: V0_class_1, 1: V1_class_1, 2: V2_class_1}
        V_class_2 = {0: V0_class_2, 1: V1_class_2, 2: V2_class_2}
        V_class_3 = {0: V0_class_3, 1: V1_class_3, 2: V2_class_3}

        av = {0: 1, 1: 1, 2: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[9], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, Choice) +
                P2 * models.logit(V_class_2, av, Choice) +
                P3 * models.logit(V_class_3, av, Choice)
        )
        logprob = log(prob)

    if latent == 39:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z1_BETA_TIME_PT_2 = Beta('Z1_BETA_TIME_PT_2', starting_point[8], None, None, 0)
            Z2_BETA_TIME_CAR_2 = Beta('Z2_BETA_TIME_CAR_2', starting_point[9], None, None, 0)
            Z3_BETA_COST_HWH_2 = Beta('Z3_BETA_COST_HWH_2', starting_point[10], None, None, 0)
            Z4_BETA_COST_OTHER_2 = Beta('Z4_BETA_COST_OTHER_2', starting_point[11], None, None, 0)
        else:
            Z1_BETA_TIME_PT_2 = Beta('Z1_BETA_TIME_PT_2', 0, None, None, 0)
            Z2_BETA_TIME_CAR_2 = Beta('Z2_BETA_TIME_CAR_2', 0, None, None, 0)
            Z3_BETA_COST_HWH_2 = Beta('Z3_BETA_COST_HWH_2', 0, None, None, 0)
            Z4_BETA_COST_OTHER_2 = Beta('Z4_BETA_COST_OTHER_2', 0, None, None, 0)

        V0_class_1 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_2 = ASC_PT + \
                     Z1_BETA_TIME_PT_2 * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_3 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     Z3_BETA_COST_HWH_2 * MarginalCostPT_scaled * PurpHWH + \
                     Z4_BETA_COST_OTHER_2 * MarginalCostPT_scaled * PurpOther

        V1_class_1 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_2 = ASC_CAR + \
                     Z2_BETA_TIME_CAR_2 * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_3 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     Z3_BETA_COST_HWH_2 * CostCarCHF_scaled * PurpHWH + \
                     Z4_BETA_COST_OTHER_2 * CostCarCHF_scaled * PurpOther

        V2_class_1 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_2 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_3 = ASC_SM + BETA_DIST * distance_km_scaled

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {0: V0_class_1, 1: V1_class_1, 2: V2_class_1}
        V_class_2 = {0: V0_class_2, 1: V1_class_2, 2: V2_class_2}
        V_class_3 = {0: V0_class_3, 1: V1_class_3, 2: V2_class_3}

        av = {0: 1, 1: 1, 2: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[12], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[13], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, Choice) +
                P2 * models.logit(V_class_2, av, Choice) +
                P3 * models.logit(V_class_3, av, Choice)
        )
        logprob = log(prob)

    if latent == 31:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        V0_class_1 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_2 = ASC_PT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_3 = ASC_PT

        V1_class_1 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_2 = ASC_CAR + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_3 = ASC_CAR

        V2_class_1 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_2 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_3 = ASC_SM

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {0: V0_class_1, 1: V1_class_1, 2: V2_class_1}
        V_class_2 = {0: V0_class_2, 1: V1_class_2, 2: V2_class_2}
        V_class_3 = {0: V0_class_3, 1: V1_class_3, 2: V2_class_3}

        av = {0: 1, 1: 1, 2: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[9], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, Choice) +
                P2 * models.logit(V_class_2, av, Choice) +
                P3 * models.logit(V_class_3, av, Choice)
        )
        logprob = log(prob)
    if latent == 32:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        V0_class_1 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT + \
                     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
                     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
        V0_class_2 = ASC_PT + \
                     BETA_TIME_PT * TimePT_scaled + \
                     BETA_WAITING_TIME * WaitingTimePT
        V0_class_3 = ASC_PT

        V1_class_1 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled + \
                     BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
                     BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
        V1_class_2 = ASC_CAR + \
                     BETA_TIME_CAR * TimeCar_scaled
        V1_class_3 = ASC_CAR

        V2_class_1 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_2 = ASC_SM + BETA_DIST * distance_km_scaled
        V2_class_3 = ASC_SM

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {0: V0_class_1, 1: V1_class_1, 2: V2_class_1}
        V_class_2 = {0: V0_class_2, 1: V1_class_2, 2: V2_class_2}
        V_class_3 = {0: V0_class_3, 1: V1_class_3, 2: V2_class_3}

        av = {0: 1, 1: 1, 2: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[9], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, Choice) +
                P2 * models.logit(V_class_2, av, Choice) +
                P3 * models.logit(V_class_3, av, Choice)
        )
        logprob = log(prob)
    if latent == 4:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_3 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_4 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_3 = ASC_SM + B_TIME * SM_TT_SCALED + B_HE * SM_HE_SCALED
        SM_class_4 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_COST * CAR_CO_SCALED
        CAR_class_3 = ASC_CAR + B_TIME * CAR_TT_SCALED
        CAR_class_4 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}
        V_class_4 = {1: TRAIN_class_4, 2: SM_class_4, 3: CAR_class_4}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
            prob_class_3 = Beta('prob_class_3', starting_point[7], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, 0, 1, 0)
            prob_class_2 = Beta('prob_class_2', 0, 0, 1, 0)
            prob_class_3 = Beta('prob_class_3', 0, 0, 1, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + exp(prob_class_3) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = exp(prob_class_3) / denom
        P4 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE) +
                P3 * models.logit(V_class_3, av, CHOICE) +
                P4 * models.logit(V_class_4, av, CHOICE)
        )

        logprob = log(prob)
    if latent == 5:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_3 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_4 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
        TRAIN_class_5 = ASC_TRAIN

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_3 = ASC_SM + B_TIME * SM_TT_SCALED + B_HE * SM_HE_SCALED
        SM_class_4 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
        SM_class_5 = ASC_SM

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_COST * CAR_CO_SCALED
        CAR_class_3 = ASC_CAR + B_TIME * CAR_TT_SCALED
        CAR_class_4 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_5 = ASC_CAR

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}
        V_class_4 = {1: TRAIN_class_4, 2: SM_class_4, 3: CAR_class_4}
        V_class_5 = {1: TRAIN_class_5, 2: SM_class_5, 3: CAR_class_5}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
            prob_class_3 = Beta('prob_class_3', starting_point[7], None, None, 0)
            prob_class_4 = Beta('prob_class_4', starting_point[8], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, 0, 1, 0)
            prob_class_2 = Beta('prob_class_2', 0, 0, 1, 0)
            prob_class_3 = Beta('prob_class_3', 0, 0, 1, 0)
            prob_class_4 = Beta('prob_class_4', 0, 0, 1, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + exp(prob_class_3) + exp(
            prob_class_4) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = exp(prob_class_3) / denom
        P4 = exp(prob_class_4) / denom
        P5 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE) +
                P3 * models.logit(V_class_3, av, CHOICE) +
                P4 * models.logit(V_class_4, av, CHOICE) +
                P5 * models.logit(V_class_5, av, CHOICE)
        )

        logprob = log(prob)
    elif latent == 1:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        V0 = ASC_PT + \
             BETA_TIME_PT * TimePT_scaled + \
             BETA_WAITING_TIME * WaitingTimePT + \
             BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH + \
             BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther

        V1 = ASC_CAR + \
             BETA_TIME_CAR * TimeCar_scaled + \
             BETA_COST_HWH * CostCarCHF_scaled * PurpHWH + \
             BETA_COST_OTHER * CostCarCHF_scaled * PurpOther

        V2 = ASC_SM + BETA_DIST * distance_km_scaled

        V = {0: V0,
             1: V1,
             2: V2}

        av = {0: 1,
              1: 1,
              2: 1}

        logprob = loglogit(V, av, Choice)

    if 10 <= latent <= 19:
        
        # Create the Biogeme object
        print(f"Nb. of MonteCarlo draws = {R}")
        biogeme = bio.BIOGEME(
            database, logprob, numberOfDraws=R, seed=pandaSeed, numberOfThreads=1
        )
        biogeme.modelName = 'optima_mixed'
        biogeme.saveIterations = False

        # Estimate the parameters
        st_time = time.time()
        results = biogeme.estimate()
        elapsed = time.time() - st_time
        print(f"Estimating Biogeme model takes {elapsed}s")
        pandasResults = results.getEstimatedParameters()
        biog_loglike = round(results.data.logLike, 8)
        print(pandasResults)

        print("")
        print(f"Loglike = {biog_loglike}")

        # Get the results
        biog_beta = list(pandasResults["Value"])

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
        return biog_beta, biog_loglike, None, None, df, None

    biogeme = bio.BIOGEME(database, logprob, numberOfThreads=1)
    timestamp = time.time()
    biogeme.modelName = f'optima_latent_{latent}_{timestamp}'
    start_time = time.time()
    seed_nr = pandaSeed + 1
    random.seed(seed_nr)
    np.random.seed(seed_nr)
    results = biogeme.estimate()

    # Get the results in a pandas table
    pandasResults = results.getEstimatedParameters()
    biog_loglike = round(results.data.logLike, 8)
    print(pandasResults)

    print("")
    print(f"Loglike = {biog_loglike}")
    print("")
    print(f"Estimation time = {time.time() - start_time}s")

    # Get confidence intervals on beta and loglike
    betas = list(pandasResults.index)  # get BetaNames

    # loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
    loglike_conf = [-5, 5]
    beta_confs = dict()
    for k in range(len(betas)):
        beta_confs[k] = [-5, 5]  # we honestly don't care about this now
        # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]

    # Get the results
    biog_beta = list(pandasResults["Value"])
    # beta_stderr = round(pandasResults["Std err"][0], 8)

    signi = False

    if latent == 2:
        denom = np.exp(biog_beta[8]) + 1

        P11 = np.exp(biog_beta[8]) / denom
        P22 = 1 / denom
        print("prob 1 = ", P11)
        print("prob 2 = ", P22)

    if latent == 22:
        denom = np.exp(biog_beta[8]) + 1

        P11 = np.exp(biog_beta[8]) / denom
        P22 = 1 / denom
        print("prob 1 = ", P11)
        print("prob 2 = ", P22)

    if latent == 4:
        denom = np.exp(biog_beta[5]) + np.exp(biog_beta[6]) + np.exp(biog_beta[7]) + 1

        P11 = np.exp(biog_beta[5]) / denom
        P22 = np.exp(biog_beta[6]) / denom
        P33 = np.exp(biog_beta[7]) / denom
        P44 = 1 / denom
        print("prob 1 = ", P11)
        print("prob 2 = ", P22)
        print("prob 3 = ", P33)
        print("prob 4 = ", P44)

    if latent == 5:
        denom = np.exp(biog_beta[5]) + np.exp(biog_beta[6]) + np.exp(biog_beta[7]) + np.exp(biog_beta[8]) + 1

        P11 = np.exp(biog_beta[5]) / denom
        P22 = np.exp(biog_beta[6]) / denom
        P33 = np.exp(biog_beta[7]) / denom
        P44 = np.exp(biog_beta[8]) / denom
        P55 = 1 / denom
        print("prob 1 = ", P11)
        print("prob 2 = ", P22)
        print("prob 3 = ", P33)
        print("prob 4 = ", P44)
        print("prob 5 = ", P55)

    if latent == 3:
        denom = np.exp(biog_beta[8]) + np.exp(biog_beta[9]) + 1

        P11 = np.exp(biog_beta[8]) / denom
        P22 = np.exp(biog_beta[9]) / denom
        P33 = 1 / denom
        print("prob 1 = ", P11)
        print("prob 2 = ", P22)
        print("prob 3 = ", P33)

    return biog_beta, biog_loglike, beta_confs, loglike_conf, df, timestamp


def biogeme_estimate_swissmetro(df, latent, michels_classes, toms_extremists, loadAttr=False, starting_point=None,
                                R=None, pandaSeed=1):
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

    database = db.Database('swissmetro', df)
    if loadAttr:
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

        return None, None, None, None, None, None
    else:
        SM_COST = Variable('SM_COST')
        TRAIN_COST = Variable('TRAIN_COST')

        CAR_AV_SP = Variable('CAR_AV_SP')
        TRAIN_AV_SP = Variable('TRAIN_AV_SP')

        TRAIN_TT_SCALED = Variable('TRAIN_TT_SCALED')
        TRAIN_COST_SCALED = Variable('TRAIN_COST_SCALED')
        SM_TT_SCALED = Variable('SM_TT_SCALED')
        SM_COST_SCALED = Variable('SM_COST_SCALED')
        CAR_TT_SCALED = Variable('CAR_TT_SCALED')
        CAR_CO_SCALED = Variable('CAR_CO_SCALED')
        TRAIN_HE_SCALED = Variable('TRAIN_HE_SCALED')
        SM_HE_SCALED = Variable('SM_HE_SCALED')

    if starting_point is not None:

        # the starting beta is ordered like so:
        # ASC_CAR       -2.850539      0.810487    -3.517067      0.000436
        # ASC_TRAIN     -3.449761      0.879437    -3.922694      0.000088
        # B_COST        -2.275223      0.983560    -2.313252      0.020709
        # B_HE         -44.041174     33.804763    -1.302810      0.192640
        # B_TIME        -6.286555      5.843644    -1.075794      0.282020
        # prob_class_1  -2.189029      1.636809    -1.337376      0.181100
        # prob_class_2 -15.241301

        ASC_CAR = Beta('ASC_CAR', starting_point[0], None, None, 0)
        ASC_TRAIN = Beta('ASC_TRAIN', starting_point[1], None, None, 0)
        ASC_SM = Beta('ASC_SM', 0, None, None, 1)
        B_TIME = Beta('B_TIME', starting_point[4], None, None, 0)
        B_COST = Beta('B_COST', starting_point[2], None, None, 0)
        B_HE = Beta('B_HE', starting_point[3], None, None, 0)
    else:
        ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
        ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
        ASC_SM = Beta('ASC_SM', 0, None, None, 1)
        B_TIME = Beta('B_TIME', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)
        B_HE = Beta('B_HE', 0, None, None, 0)

    if 1020 <= latent <= 1039:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        # ASC_SM = Beta('ASC_SM', 0, None, None, 1)

        # 1 ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
        # 2 ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
        # 3 B_TIME = Beta('B_TIME', 0, None, None, 0)
        # 4 B_COST = Beta('B_COST', 0, None, None, 0)
        # 5 B_HE = Beta('B_HE', 0, None, None, 0)

        mix_inds = None
        if latent == 1030:
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]  # mix time
            class_2_ks = [3, 4, 5, 7, 8]  # sep ASCs
            class_3_ks = [1, 2, 3, 4, 5]  # no car
            prob_inds = [9, 10]
            extra_inds = [[1, 7], [2, 8]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2, 3]
            class_3_av = [1, 2]

            if starting_point is not None:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', starting_point[5], None, None, 0)
            else:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
            B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

            if starting_point is not None:
                Z1_ASC_CAR = Beta('Z1_ASC_CAR', starting_point[6], None, None, 0)
                Z2_ASC_TRAIN = Beta('Z2_ASC_TRAIN', starting_point[7], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
                prob_class_2 = Beta('prob_class_2', starting_point[9], None, None, 0)
            else:
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

            # We integrate over B_TIME_RND using Monte-Carlo
            logprob = log(MonteCarlo(prob))

        if latent == 1031:
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]  # mix time
            class_2_ks = [3, 4, 5, 7, 8]  # sep ASCs
            class_3_ks = [1, 2, 3, 4, 5]  # no SM
            prob_inds = [9, 10]
            extra_inds = [[1, 7], [2, 8]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2, 3]
            class_3_av = [1, 3]

            if starting_point is not None:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', starting_point[5], None, None, 0)
            else:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
            B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

            if starting_point is not None:
                Z1_ASC_CAR = Beta('Z1_ASC_CAR', starting_point[6], None, None, 0)
                Z2_ASC_TRAIN = Beta('Z2_ASC_TRAIN', starting_point[7], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
                prob_class_2 = Beta('prob_class_2', starting_point[9], None, None, 0)
            else:
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

            # We integrate over B_TIME_RND using Monte-Carlo
            logprob = log(MonteCarlo(prob))

        if latent == 1020:
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [3, 4, 5, 7] # new ASCs here and no car
            prob_inds = [8]
            extra_inds = [[2, 7]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2] # no car

            if starting_point is not None:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', starting_point[5], None, None, 0)
            else:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
            B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

            if starting_point is not None:
                Z2_ASC_TRAIN = Beta('Z2_ASC_TRAIN', starting_point[6], None, None, 0)
            else:
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

            # Class membership
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
                # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, CHOICE) +
                    P2 * models.logit(V_class_2, av2, CHOICE)
            )

            # We integrate over B_TIME_RND using Monte-Carlo
            logprob = log(MonteCarlo(prob))

        if latent == 1021:
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6] # mix
            class_2_ks = [3, 4, 5, 7] # sep ASCs, no SM
            prob_inds = [8]
            extra_inds = [[1, 7]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 3]

            if starting_point is not None:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', starting_point[5], None, None, 0)
            else:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
            B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

            if starting_point is not None:
                Z1_ASC_CAR = Beta('Z1_ASC_CAR', starting_point[6], None, None, 0)
            else:
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

            # Class membership
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
                # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, CHOICE) +
                    P2 * models.logit(V_class_2, av2, CHOICE)
            )

            # We integrate over B_TIME_RND using Monte-Carlo
            logprob = log(MonteCarlo(prob))

        if latent == 1022 or latent == 1027:
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6] # mix time
            class_2_ks = [3, 4, 5, 7, 8] # sep ASCs
            prob_inds = [9]
            extra_inds = [[1, 7], [2, 8]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2, 3]

            if starting_point is not None:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', starting_point[5], None, None, 0)
            else:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
            B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

            if starting_point is not None:
                Z02_ASC_SM = Beta('Z02_ASC_SM', 0, None, None, 1)
                Z1_ASC_CAR = Beta('Z1_ASC_CAR', starting_point[6], None, None, 0)
                Z2_ASC_TRAIN = Beta('Z2_ASC_TRAIN', starting_point[7], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
                # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, CHOICE) +
                    P2 * models.logit(V_class_2, av, CHOICE)
            )

            # We integrate over B_TIME_RND using Monte-Carlo
            logprob = log(MonteCarlo(prob))

        if latent == 1023:
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6] # mix time
            class_2_ks = [1, 2, 3, 4, 5] # no car
            prob_inds = [7]
            extra_inds = []
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2]

            if starting_point is not None:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', starting_point[5], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[6], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
                # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, CHOICE) +
                    P2 * models.logit(V_class_2, av2, CHOICE)
            )

            # We integrate over B_TIME_RND using Monte-Carlo
            logprob = log(MonteCarlo(prob))

        if latent == 1024:
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]  # mix time
            class_2_ks = [1, 2, 3, 4, 5]  # no SM
            prob_inds = [7]
            extra_inds = []
            class_1_av = [1, 2, 3]
            class_2_av = [1, 3]

            if starting_point is not None:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', starting_point[5], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[6], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
                # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, CHOICE) +
                    P2 * models.logit(V_class_2, av2, CHOICE)
            )

            # We integrate over B_TIME_RND using Monte-Carlo
            logprob = log(MonteCarlo(prob))

        if latent == 1025:
            # ASC_SM = Beta('ASC_SM', 0, None, None, 1)

            # 1 ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
            # 2 ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
            # 3 B_TIME = Beta('B_TIME', 0, None, None, 0)
            # 4 B_COST = Beta('B_COST', 0, None, None, 0)
            # 5 B_HE = Beta('B_HE', 0, None, None, 0)

            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]  # mix time
            class_2_ks = [1, 2, 3, 4, 7]  # new time
            prob_inds = [8]
            extra_inds = [[5, 7]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2, 3]

            if starting_point is not None:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', starting_point[5], None, None, 0)
            else:
                Z01_B_TIME_S = Beta('Z01_B_TIME_S', 1, None, None, 0)
            B_TIME_RND = B_TIME + Z01_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

            if starting_point is not None:
                Z1_B_TIME_C2 = Beta('Z1_B_TIME_C2', starting_point[6], None, None, 0)
            else:
                Z1_B_TIME_C2 = Beta('Z1_B_TIME_C2', 0, None, None, 0)

            # ASC SM is normalized. Removing Car leaves ASC Train to be estimated, and SM fixed. Works.

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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
                # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, CHOICE) +
                    P2 * models.logit(V_class_2, av, CHOICE)
            )

            # We integrate over B_TIME_RND using Monte-Carlo
            logprob = log(MonteCarlo(prob))
        if latent == 1026:
            # ASC_SM = Beta('ASC_SM', 0, None, None, 1)

            # 1 ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
            # 2 ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
            # 3 B_COST = Beta('B_COST', 0, None, None, 0)
            # 4 B_HE = Beta('B_HE', 0, None, None, 0)
            # 5 B_TIME = Beta('B_TIME', 0, None, None, 0)

            mix_inds = [[3, 6]]  # mix costs in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]  # mix costs
            class_2_ks = [1, 2, 3, 4, 7]  # new time
            prob_inds = [8]
            extra_inds = [[5, 7]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2, 3]

            if starting_point is not None:
                Z01_B_COST_S = Beta('Z01_B_COST_S', starting_point[5], None, None, 0)
            else:
                Z01_B_COST_S = Beta('Z01_B_COST_S', 1, None, None, 0)
            B_COST_RND = B_COST + Z01_B_COST_S * bioDraws('B_COST_RND', 'NORMAL')

            if starting_point is not None:
                Z1_B_TIME_C2 = Beta('Z1_B_TIME_C2', starting_point[6], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
                # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, CHOICE) +
                    P2 * models.logit(V_class_2, av, CHOICE)
            )

            # We integrate over B_TIME_RND using Monte-Carlo
            logprob = log(MonteCarlo(prob))
        if latent == 1028:
            # ASC_SM = Beta('ASC_SM', 0, None, None, 1)

            # 1 ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
            # 2 ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
            # 3 B_COST = Beta('B_COST', 0, None, None, 0)
            # 4 B_HE = Beta('B_HE', 0, None, None, 0)
            # 5 B_TIME = Beta('B_TIME', 0, None, None, 0)

            mix_inds = [[4, 6]]  # mix HE in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]  # mix HE
            class_2_ks = [1, 2, 3, 4, 7]  # new time
            prob_inds = [8]
            extra_inds = [[5, 7]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2, 3]

            if starting_point is not None:
                Z01_B_HE_S = Beta('Z01_B_HE_S', starting_point[5], None, None, 0)
            else:
                Z01_B_HE_S = Beta('Z01_B_HE_S', 1, None, None, 0)
            B_HE_RND = B_HE + Z01_B_HE_S * bioDraws('B_HE_RND', 'NORMAL')

            if starting_point is not None:
                Z1_B_TIME_C2 = Beta('Z1_B_TIME_C2', starting_point[6], None, None, 0)
            else:
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
            if starting_point is not None:
                prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
            else:
                prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
                # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

            denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

            P1 = exp(prob_class_1) / denom
            P2 = 1 / denom

            prob = (
                    P1 * models.logit(V_class_1, av, CHOICE) +
                    P2 * models.logit(V_class_2, av, CHOICE)
            )

            # We integrate over B_TIME_RND using Monte-Carlo
            logprob = log(MonteCarlo(prob))

    if 10 <= latent <= 19:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        mix_inds = None
        if latent == 10:
            mix_inds = [[5, 6]]  # mix just time
        elif latent == 11:
            mix_inds = [[5, 6], [3, 7]]  # mix time and costs
        elif latent == 12:
            mix_inds = [[5, 6], [3, 7], [4, 8]]  # mix time and costs and headway

        firsts = [tupel[0] for tupel in mix_inds]

        betalen = 5

        if 5 in firsts:
            if starting_point is not None:
                Z1_B_TIME_S = Beta('Z1_B_TIME_S', starting_point[betalen], None, None, 0)
            else:
                Z1_B_TIME_S = Beta('Z1_B_TIME_S', 1, None, None, 0)
            B_TIME_RND = B_TIME + Z1_B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')
            betalen += 1
        if 3 in firsts:
            if starting_point is not None:
                Z2_B_COST_S = Beta('Z2_B_COST_S', starting_point[betalen], None, None, 0)
            else:
                Z2_B_COST_S = Beta('Z2_B_COST_S', 1, None, None, 0)
            B_COST_RND = B_COST + Z2_B_COST_S * bioDraws('B_COST_RND', 'NORMAL')
            betalen += 1
        if 4 in firsts:
            if starting_point is not None:
                Z3_B_HE_S = Beta('Z3_B_HE_S', starting_point[betalen], None, None, 0)
            else:
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

        # We integrate over B_TIME_RND using Monte-Carlo
        logprob = log(MonteCarlo(prob))

    if latent == 2:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

        logprob = log(prob)
    if latent == 22:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_TIME * CAR_TT_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

        logprob = log(prob)
    if latent == 23:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

        logprob = log(prob)
    if latent == 24:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

        logprob = log(prob)
    if latent == 29:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z_B_TIME_2 = Beta('Z_B_TIME_2', starting_point[5], None, None, 0)
        else:
            Z_B_TIME_2 = Beta('Z_B_TIME_2', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + Z_B_TIME_2 * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + Z_B_TIME_2 * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + Z_B_TIME_2 * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        # av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        av = {1: 1, 2: 1, 3: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

        logprob = log(prob)

    if latent == 28:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:  # Nun = Fun
            Z1_ASC_CAR = Beta('Z1_ASC_CAR', starting_point[5], None, None, 0)
            Z2_ASC_TRAIN = Beta('Z2_ASC_TRAIN', starting_point[6], None, None, 0)
            Z3_ASC_SM = Beta('Z3_ASC_SM', 0, None, None, 1)
        else:
            Z1_ASC_CAR = Beta('Z1_ASC_CAR', 0, None, None, 0)
            Z2_ASC_TRAIN = Beta('Z2_ASC_TRAIN', 0, None, None, 0)
            Z3_ASC_SM = Beta('Z3_ASC_SM', 0, None, None, 1)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = Z2_ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = Z3_ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = Z1_ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

        logprob = log(prob)

    if latent == 282:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z1_ASC_CAR = Beta('Z1_ASC_CAR', starting_point[5], None, None, 0)
        else:
            Z1_ASC_CAR = Beta('Z1_ASC_CAR', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = Z1_ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

        logprob = log(prob)

    if latent == 229:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z_B_COST_2 = Beta('Z_B_COST_2', starting_point[5], None, None, 0)
            Z_B_TIME_2 = Beta('Z_B_TIME_2', starting_point[6], None, None, 0)
        else:
            Z_B_COST_2 = Beta('Z_B_COST_2', 0, None, None, 0)
            Z_B_TIME_2 = Beta('Z_B_TIME_2', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + Z_B_TIME_2 * TRAIN_TT_SCALED + Z_B_COST_2 * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + Z_B_TIME_2 * SM_TT_SCALED + Z_B_COST_2 * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + Z_B_TIME_2 * CAR_TT_SCALED + Z_B_COST_2 * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

        logprob = log(prob)

    if latent == 2292:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z_B_COST_2 = Beta('Z_B_COST_2', starting_point[5], None, None, 0)
        else:
            Z_B_COST_2 = Beta('Z_B_COST_2', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + Z_B_COST_2 * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + Z_B_COST_2 * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_TIME * CAR_TT_SCALED + Z_B_COST_2 * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

        logprob = log(prob)

    if latent == 239:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z_B_HE_2 = Beta('Z_B_HE_2', starting_point[5], None, None, 0)
        else:
            Z_B_HE_2 = Beta('Z_B_HE_2', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + Z_B_HE_2 * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + Z_B_HE_2 * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

        logprob = log(prob)
    if latent == 249:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z1_B_TIME_2 = Beta('Z1_B_TIME_2', starting_point[5], None, None, 0)
            Z2_B_COST_2 = Beta('Z2_B_COST_2', starting_point[6], None, None, 0)
            Z3_B_HE_2 = Beta('Z3_B_HE_2', starting_point[7], None, None, 0)
        else:
            Z1_B_TIME_2 = Beta('Z1_B_TIME_2', 0, None, None, 0)
            Z2_B_COST_2 = Beta('Z2_B_COST_2', 0, None, None, 0)
            Z3_B_HE_2 = Beta('Z3_B_HE_2', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + Z1_B_TIME_2 * TRAIN_TT_SCALED + Z2_B_COST_2 * TRAIN_COST_SCALED + Z3_B_HE_2 * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + Z1_B_TIME_2 * SM_TT_SCALED + Z2_B_COST_2 * SM_COST_SCALED + Z3_B_HE_2 * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + Z1_B_TIME_2 * CAR_TT_SCALED + Z2_B_COST_2 * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

        logprob = log(prob)
    if latent == 259:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z1_B_TIME_2 = Beta('Z1_B_TIME_2', starting_point[5], None, None, 0)
            Z2_B_COST_2 = Beta('Z2_B_COST_2', starting_point[6], None, None, 0)
        else:
            Z1_B_TIME_2 = Beta('Z1_B_TIME_2', 0, None, None, 0)
            Z2_B_COST_2 = Beta('Z2_B_COST_2', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + Z1_B_TIME_2 * TRAIN_TT_SCALED + Z2_B_COST_2 * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + Z1_B_TIME_2 * SM_TT_SCALED + Z2_B_COST_2 * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + Z1_B_TIME_2 * CAR_TT_SCALED + Z2_B_COST_2 * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            # P1 = Beta('prob_class_1', 0.5, 0, 1, 0)

        denom = exp(prob_class_1) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE)
        )

        logprob = log(prob)

    if latent == 3:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_3 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_3 = ASC_SM + B_TIME * SM_TT_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_COST * CAR_CO_SCALED
        CAR_class_3 = ASC_CAR + B_TIME * CAR_TT_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        # av = {1: 1, 2: 1, 3: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE) +
                P3 * models.logit(V_class_3, av, CHOICE)
        )
        logprob = log(prob)

    if latent == 39:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z1_B_TIME_2 = Beta('Z1_B_TIME_2', starting_point[5], None, None, 0)
            Z2_B_COST_2 = Beta('Z2_B_COST_2', starting_point[6], None, None, 0)
        else:
            Z1_B_TIME_2 = Beta('Z1_B_TIME_2', 0, None, None, 0)
            Z2_B_COST_2 = Beta('Z2_B_COST_2', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + Z1_B_TIME_2 * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_3 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + Z2_B_COST_2 * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + Z1_B_TIME_2 * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_3 = ASC_SM + B_TIME * SM_TT_SCALED + Z2_B_COST_2 * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + Z1_B_TIME_2 * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_3 = ASC_CAR + B_TIME * CAR_TT_SCALED + Z2_B_COST_2 * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        # av = {1: 1, 2: 1, 3: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[8], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE) +
                P3 * models.logit(V_class_3, av, CHOICE)
        )
        logprob = log(prob)

    if latent == 31:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_3 = ASC_TRAIN

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_3 = ASC_SM

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_COST * CAR_CO_SCALED
        CAR_class_3 = ASC_CAR

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        # av = {1: 1, 2: 1, 3: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE) +
                P3 * models.logit(V_class_3, av, CHOICE)
        )
        logprob = log(prob)
    if latent == 319:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        if starting_point is not None:
            Z1_B_TIME_2 = Beta('Z1_B_TIME_2', starting_point[5], None, None, 0)
            Z2_B_TIME_3 = Beta('Z2_B_TIME_3', starting_point[6], None, None, 0)
            Z3_B_COST_2 = Beta('Z3_B_COST_2', starting_point[7], None, None, 0)
        else:
            Z1_B_TIME_2 = Beta('Z1_B_TIME_2', 0, None, None, 0)
            Z2_B_TIME_3 = Beta('Z2_B_TIME_3', 0, None, None, 0)
            Z3_B_COST_2 = Beta('Z3_B_COST_2', 0, None, None, 0)

        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + Z1_B_TIME_2 * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_3 = ASC_TRAIN + Z2_B_TIME_3 * TRAIN_TT_SCALED + Z3_B_COST_2 * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + Z1_B_TIME_2 * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_3 = ASC_SM + Z2_B_TIME_3 * SM_TT_SCALED + Z3_B_COST_2 * SM_COST_SCALED + B_HE * SM_HE_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + Z1_B_TIME_2 * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_3 = ASC_CAR + Z2_B_TIME_3 * CAR_TT_SCALED + Z3_B_COST_2 * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        # av = {1: 1, 2: 1, 3: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[8], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[9], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE) +
                P3 * models.logit(V_class_3, av, CHOICE)
        )
        logprob = log(prob)
    if latent == 32:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_3 = ASC_TRAIN

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + B_HE * SM_HE_SCALED
        SM_class_3 = ASC_SM

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_TIME * CAR_TT_SCALED
        CAR_class_3 = ASC_CAR

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        # av = {1: 1, 2: 1, 3: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE) +
                P3 * models.logit(V_class_3, av, CHOICE)
        )
        logprob = log(prob)
    if latent == 33:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
        TRAIN_class_3 = ASC_TRAIN

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
        SM_class_3 = ASC_SM

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_3 = ASC_CAR

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        # av = {1: 1, 2: 1, 3: 1}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
            prob_class_2 = Beta('prob_class_2', 0, None, None, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE) +
                P3 * models.logit(V_class_3, av, CHOICE)
        )
        logprob = log(prob)

        # elif toms_extremists:
        #     # Add parameters to be estimated
        #     B_TIME_C2 = Beta('B_TIME_C2', starting_point[5], None, None, 0)
        #     B_COST_C3 = Beta('B_COST_C3', starting_point[6], None, None, 0)
        #
        #     TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        #     TRAIN_class_2 = ASC_TRAIN + B_TIME_C2 * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        #     TRAIN_class_3 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST_C3 * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        #
        #     SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        #     SM_class_2 = ASC_SM + B_TIME_C2 * SM_TT_SCALED + B_HE * SM_HE_SCALED
        #     SM_class_3 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST_C3 * SM_COST_SCALED + B_HE * SM_HE_SCALED
        #
        #     CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        #     CAR_class_2 = ASC_CAR + B_TIME_C2 * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        #     CAR_class_3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST_C3 * CAR_CO_SCALED
        #
        #     # Associate utility functions with the numbering of alternatives
        #     V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        #     V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        #     V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}
        #
        #     # Associate the availability conditions with the alternatives
        #     # av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        #     av = {1: 1, 2: 1, 3: 1}
        #
        #     # Class membership
        #     if starting_point is not None:
        #         prob_class_1 = Beta('prob_class_1', starting_point[7], None, None, 0)
        #         prob_class_2 = Beta('prob_class_2', starting_point[8], None, None, 0)
        #     else:
        #         prob_class_1 = Beta('prob_class_1', 0, None, None, 0)
        #         prob_class_2 = Beta('prob_class_2', 0, None, None, 0)
        #
        #     denom = exp(prob_class_1) + exp(prob_class_2) + 1  # exp(0) because one is fixed (like ASC)
        #
        #     P1 = exp(prob_class_1) / denom
        #     P2 = exp(prob_class_2) / denom
        #     P3 = 1 / denom
        #
        #     prob = (
        #             P1 * models.logit(V_class_1, av, CHOICE) +
        #             P2 * models.logit(V_class_2, av, CHOICE) +
        #             P3 * models.logit(V_class_3, av, CHOICE)
        #     )
        #     logprob = log(prob)

    if latent == 4:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_3 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_4 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_3 = ASC_SM + B_TIME * SM_TT_SCALED + B_HE * SM_HE_SCALED
        SM_class_4 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_COST * CAR_CO_SCALED
        CAR_class_3 = ASC_CAR + B_TIME * CAR_TT_SCALED
        CAR_class_4 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}
        V_class_4 = {1: TRAIN_class_4, 2: SM_class_4, 3: CAR_class_4}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
            prob_class_3 = Beta('prob_class_3', starting_point[7], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, 0, 1, 0)
            prob_class_2 = Beta('prob_class_2', 0, 0, 1, 0)
            prob_class_3 = Beta('prob_class_3', 0, 0, 1, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + exp(prob_class_3) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = exp(prob_class_3) / denom
        P4 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE) +
                P3 * models.logit(V_class_3, av, CHOICE) +
                P4 * models.logit(V_class_4, av, CHOICE)
        )

        logprob = log(prob)
    if latent == 5:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        TRAIN_class_1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_2 = ASC_TRAIN + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_3 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_HE * TRAIN_HE_SCALED
        TRAIN_class_4 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
        TRAIN_class_5 = ASC_TRAIN

        SM_class_1 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_2 = ASC_SM + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        SM_class_3 = ASC_SM + B_TIME * SM_TT_SCALED + B_HE * SM_HE_SCALED
        SM_class_4 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
        SM_class_5 = ASC_SM

        CAR_class_1 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_2 = ASC_CAR + B_COST * CAR_CO_SCALED
        CAR_class_3 = ASC_CAR + B_TIME * CAR_TT_SCALED
        CAR_class_4 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED
        CAR_class_5 = ASC_CAR

        # Associate utility functions with the numbering of alternatives
        V_class_1 = {1: TRAIN_class_1, 2: SM_class_1, 3: CAR_class_1}
        V_class_2 = {1: TRAIN_class_2, 2: SM_class_2, 3: CAR_class_2}
        V_class_3 = {1: TRAIN_class_3, 2: SM_class_3, 3: CAR_class_3}
        V_class_4 = {1: TRAIN_class_4, 2: SM_class_4, 3: CAR_class_4}
        V_class_5 = {1: TRAIN_class_5, 2: SM_class_5, 3: CAR_class_5}

        # Associate the availability conditions with the alternatives
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

        # Class membership
        if starting_point is not None:
            prob_class_1 = Beta('prob_class_1', starting_point[5], None, None, 0)
            prob_class_2 = Beta('prob_class_2', starting_point[6], None, None, 0)
            prob_class_3 = Beta('prob_class_3', starting_point[7], None, None, 0)
            prob_class_4 = Beta('prob_class_4', starting_point[8], None, None, 0)
        else:
            prob_class_1 = Beta('prob_class_1', 0, 0, 1, 0)
            prob_class_2 = Beta('prob_class_2', 0, 0, 1, 0)
            prob_class_3 = Beta('prob_class_3', 0, 0, 1, 0)
            prob_class_4 = Beta('prob_class_4', 0, 0, 1, 0)

        denom = exp(prob_class_1) + exp(prob_class_2) + exp(prob_class_3) + exp(
            prob_class_4) + 1  # exp(0) because one is fixed (like ASC)

        P1 = exp(prob_class_1) / denom
        P2 = exp(prob_class_2) / denom
        P3 = exp(prob_class_3) / denom
        P4 = exp(prob_class_4) / denom
        P5 = 1 / denom

        prob = (
                P1 * models.logit(V_class_1, av, CHOICE) +
                P2 * models.logit(V_class_2, av, CHOICE) +
                P3 * models.logit(V_class_3, av, CHOICE) +
                P4 * models.logit(V_class_4, av, CHOICE) +
                P5 * models.logit(V_class_5, av, CHOICE)
        )

        logprob = log(prob)
    elif latent == 0:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        TRAIN = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + B_HE * TRAIN_HE_SCALED
        SM = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + B_HE * SM_HE_SCALED
        CAR = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

        V = {1: TRAIN, 2: SM, 3: CAR}
        # av = {1: 1, 2: 1, 3: 1}
        av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
        logprob = loglogit(V, av, CHOICE)

    if 1020 <= latent <= 1039:
        seed_nr = pandaSeed + 1
        random.seed(seed_nr)
        np.random.seed(seed_nr)
        # Create the Biogeme object
        print(f"Nb. of MonteCarlo draws = {R}")
        biogeme = bio.BIOGEME(
            database, logprob, numberOfDraws=R, seed=seed_nr, numberOfThreads=1
        )
        biogeme.modelName = 'swissmetro_mixed_latent'

        # Estimate the parameters
        st_time = time.time()
        results = biogeme.estimate()
        elapsed = time.time() - st_time
        print(f"Estimating Biogeme model takes {elapsed}s")
        pandasResults = results.getEstimatedParameters()
        biog_loglike = round(results.data.logLike, 8)
        print(pandasResults)

        print("")
        print(f"Loglike = {biog_loglike}")

        # Get the results
        biog_beta = list(pandasResults["Value"])

        # Remove output files
        # os.remove("swissmetro_mixed.html")
        # os.remove("swissmetro_mixed.pickle")
        # os.remove("__swissmetro_mixed.iter")

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
        return biog_beta, biog_loglike, None, None, df, None

    if 10 <= latent <= 19:
        # Create the Biogeme object
        print(f"Nb. of MonteCarlo draws = {R}")
        biogeme = bio.BIOGEME(
            database, logprob, numberOfDraws=R, seed=pandaSeed, numberOfThreads=1
        )
        biogeme.modelName = 'swissmetro_mixed'

        # Estimate the parameters
        st_time = time.time()
        results = biogeme.estimate()
        elapsed = time.time() - st_time
        print(f"Estimating Biogeme model takes {elapsed}s")
        pandasResults = results.getEstimatedParameters()
        biog_loglike = round(results.data.logLike, 8)
        print(pandasResults)

        print("")
        print(f"Loglike = {biog_loglike}")

        # Get the results
        biog_beta = list(pandasResults["Value"])

        # Remove output files
        # os.remove("swissmetro_mixed.html")
        # os.remove("swissmetro_mixed.pickle")
        # os.remove("__swissmetro_mixed.iter")

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
        return biog_beta, biog_loglike, None, None, df, None

    # print("available threads = ", mp.cpu_count())
    # biogeme = bio.BIOGEME(database, logprob, numberOfThreads=1)
    biogeme = bio.BIOGEME(database, logprob, numberOfThreads=1)
    timestamp = time.time()
    biogeme.modelName = f'swissmetro_latent_{timestamp}'
    start_time = time.time()
    seed_nr = pandaSeed + 1
    random.seed(seed_nr)
    np.random.seed(seed_nr)
    results = biogeme.estimate()

    # Get the results in a pandas table
    pandasResults = results.getEstimatedParameters()
    biog_loglike = round(results.data.logLike, 8)
    print(pandasResults)

    print("")
    print(f"Loglike = {biog_loglike}")
    print("")
    print(f"Estimation time = {time.time() - start_time}s")

    # Get confidence intervals on beta and loglike
    betas = list(pandasResults.index)  # get BetaNames

    # loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
    loglike_conf = [-5, 5]
    beta_confs = dict()
    for k in range(len(betas)):
        beta_confs[k] = [-5, 5]  # we honestly don't care about this now
        # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]

    # Get the results
    biog_beta = list(pandasResults["Value"])
    # beta_stderr = round(pandasResults["Std err"][0], 8)

    signi = False

    if latent == 2:
        denom = np.exp(biog_beta[5]) + 1

        P11 = np.exp(biog_beta[5]) / denom
        P22 = 1 / denom
        print("prob 1 = ", P11)
        print("prob 2 = ", P22)

    if latent == 4:
        denom = np.exp(biog_beta[5]) + np.exp(biog_beta[6]) + np.exp(biog_beta[7]) + 1

        P11 = np.exp(biog_beta[5]) / denom
        P22 = np.exp(biog_beta[6]) / denom
        P33 = np.exp(biog_beta[7]) / denom
        P44 = 1 / denom
        print("prob 1 = ", P11)
        print("prob 2 = ", P22)
        print("prob 3 = ", P33)
        print("prob 4 = ", P44)

    if latent == 5:
        denom = np.exp(biog_beta[5]) + np.exp(biog_beta[6]) + np.exp(biog_beta[7]) + np.exp(biog_beta[8]) + 1

        P11 = np.exp(biog_beta[5]) / denom
        P22 = np.exp(biog_beta[6]) / denom
        P33 = np.exp(biog_beta[7]) / denom
        P44 = np.exp(biog_beta[8]) / denom
        P55 = 1 / denom
        print("prob 1 = ", P11)
        print("prob 2 = ", P22)
        print("prob 3 = ", P33)
        print("prob 4 = ", P44)
        print("prob 5 = ", P55)

    if latent == 3:
        if michels_classes:
            denom = np.exp(biog_beta[5]) + np.exp(biog_beta[6]) + 1

            P11 = np.exp(biog_beta[5]) / denom
            P22 = np.exp(biog_beta[6]) / denom
            P33 = 1 / denom
            print("prob 1 = ", P11)
            print("prob 2 = ", P22)
            print("prob 3 = ", P33)
        elif toms_extremists:
            denom = np.exp(biog_beta[7]) + np.exp(biog_beta[8]) + 1

            P11 = np.exp(biog_beta[7]) / denom
            P22 = np.exp(biog_beta[8]) / denom
            P33 = 1 / denom
            print("prob 1 = ", P11)
            print("prob 2 = ", P22)
            print("prob 3 = ", P33)
        else:
            denom = np.exp(biog_beta[6]) + np.exp(biog_beta[7]) + 1

            P11 = np.exp(biog_beta[6]) / denom
            P22 = np.exp(biog_beta[7]) / denom
            P33 = 1 / denom
            print("prob 1 = ", P11)
            print("prob 2 = ", P22)
            print("prob 3 = ", P33)

            pvals = list(pandasResults["Rob. p-value"])
            firstp = pvals[6]
            secp = pvals[7]

            a_level = 0.05

            if firstp < a_level and secp < a_level and P11 > 0.1 and P22 > 0.1 and P33 > 0.1:
                signi = True

    return biog_beta, biog_loglike, beta_confs, loglike_conf, df, timestamp


def biogeme_estimate_swissmetro_mixed(df, mix_inds, R, loadAttr=False, pandaSeed=1):
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

    database = db.Database('swissmetro', df)
    if loadAttr:
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

        return None
    else:
        SM_COST = Variable('SM_COST')
        TRAIN_COST = Variable('TRAIN_COST')

        CAR_AV_SP = Variable('CAR_AV_SP')
        TRAIN_AV_SP = Variable('TRAIN_AV_SP')

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

    # We integrate over B_TIME_RND using Monte-Carlo
    logprob = log(MonteCarlo(prob))

    # Create the Biogeme object
    print(f"Nb. of MonteCarlo draws = {R}")
    biogeme = bio.BIOGEME(
        database, logprob, numberOfDraws=R, seed=pandaSeed, numberOfThreads=1
    )
    biogeme.modelName = 'swissmetro_mixed'

    # Estimate the parameters
    st_time = time.time()
    results = biogeme.estimate()
    elapsed = time.time() - st_time
    print(f"Estimating Biogeme model takes {elapsed}s")
    pandasResults = results.getEstimatedParameters()
    biog_loglike = round(results.data.logLike, 8)
    print(pandasResults)

    print("")
    print(f"Loglike = {biog_loglike}")

    # Get the results
    biog_beta = list(pandasResults["Value"])

    # Remove output files
    # os.remove("swissmetro_mixed.html")
    # os.remove("swissmetro_mixed.pickle")
    # os.remove("__swissmetro_mixed.iter")

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

    return biog_beta, biog_loglike, None, None, elapsed


def biogeme_estimate_swissmetro_nested(df, R):
    database = db.Database('swissmetro', df)

    PURPOSE = Variable('PURPOSE')
    CHOICE = Variable('CHOICE')
    GA = Variable('GA')
    TRAIN_CO = Variable('TRAIN_CO')
    CAR_AV = Variable('CAR_AV')
    SP = Variable('SP')
    TRAIN_AV = Variable('TRAIN_AV')
    TRAIN_TT = Variable('TRAIN_TT')
    SM_TT = Variable('SM_TT')
    CAR_TT = Variable('CAR_TT')
    CAR_CO = Variable('CAR_CO')
    SM_CO = Variable('SM_CO')
    SM_AV = Variable('SM_AV')

    # Here we use the "biogeme" way for backward compatibility
    exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
    database.remove(exclude)

    # Parameters to be estimated
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
    ASC_SM = Beta('ASC_SM', 0, None, None, 1)
    B_TIME = Beta('B_TIME', 0, None, None, 0)
    B_COST = Beta('B_COST', 0, None, None, 0)

    MU_EXISTING = Beta('MU_EXISTING', 1, 1, None, 0)
    MU_PUBLIC = Beta('MU_PUBLIC', 1, 1, None, 0)
    ALPHA_EXISTING = Beta('ALPHA_EXISTING', 0.5, 0, 1, 1)
    ALPHA_PUBLIC = 1 - ALPHA_EXISTING

    # Definition of new variables
    SM_COST = SM_CO * (GA == 0)
    TRAIN_COST = TRAIN_CO * (GA == 0)

    # Definition of new variables: adding columns to the database
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

    # Definition of the utility functions
    V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
    V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
    V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

    # Associate utility functions with the numbering of alternatives
    V = {1: V1, 2: V2, 3: V3}

    # Associate the availability conditions with the alternatives
    av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

    # Definition of nests
    # Nest membership parameters
    alpha_existing = {1: ALPHA_EXISTING, 2: 0.0, 3: 1.0}

    alpha_public = {1: ALPHA_PUBLIC, 2: 1.0, 3: 0.0}

    nest_existing = MU_EXISTING, alpha_existing
    nest_public = MU_PUBLIC, alpha_public
    nests = nest_existing, nest_public

    # The choice model is a cross-nested logit, with availability conditions
    logprob = models.logcnl_avail(V, av, nests, CHOICE)

    # Define level of verbosity
    logger = msg.bioMessage()
    # logger.setSilent()
    # logger.setWarning()
    logger.setGeneral()
    # logger.setDetailed()

    # Create the Biogeme object
    biogeme = bio.BIOGEME(database, logprob, numberOfThreads=1)
    biogeme.modelName = '11cnl'

    # Estimate the parameters
    print(f"Solving Biogeme model")
    st_time = time.time()
    results = biogeme.estimate()
    elapsed = time.time() - st_time
    print(f"Solving Biogeme model takes {elapsed}s")
    print("Extract Biogeme results")
    st_time = time.time()
    pandasResults = results.getEstimatedParameters()
    biog_loglike = round(results.data.logLike, 8)
    print(pandasResults)

    print("")
    print(f"Loglike = {biog_loglike}")
    # print("")
    # print(f"Estimation time = {time.time() - start_time}s")

    # Get confidence intervals on beta and loglike
    betas = list(pandasResults.index)  # get BetaNames
    # b = results.getBetasForSensitivityAnalysis(betas, useBootstrap=False, size=100)
    # # simulatedValues = biogeme.simulate(results.getBetaValues())
    # left, right = biogeme.confidenceIntervals(b, 0.9)

    # loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
    loglike_conf = [-5, 5]
    beta_confs = dict()
    for k in range(len(betas)):
        beta_confs[k] = [-5, 5]  # we honestly don't care about this now
        # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]

    # Get the results
    biog_beta = list(pandasResults["Value"])
    # beta_stderr = round(pandasResults["Std err"][0], 8)

    # Remove output files
    # os.remove("11cnl.html")
    # os.remove("11cnl.pickle")
    # os.remove("__11cnl.iter")

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

    elapsed = time.time() - st_time
    print(f"Extracting Biogeme results takes {elapsed}s")

    return biog_beta, biog_loglike, beta_confs, loglike_conf, df
