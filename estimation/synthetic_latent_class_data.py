# so the idea is to create synthetic indivudals that follow a certain distribution for latent classes
# ideally we give an N, how many classes, and maybe the dataset, and then we get a random / non-random set of
# individuals
# wait so the procedure is this:

# lets just consider netherlands data for now
# first estimate the full model for all individuals I guess with BioGeme => get intercept, beta_times, beta_cost
# now randomly assign people to a class, so that we have 25% chance for each one
# then compute utilities based on their class, and assign them their chosen alternative
# I guess for the assignment of utilites we should use the same error terms as we would for the estimation afterwards?
# maybe it does not play too much of a role

# the endproduct should be a set of inputs x[i, n, k], y[i, n], maybe epsilon[i, n, r]?

# ok so first lets estimate the netherlands model with the three attributes
import pandas as pd
import random
import os
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta, DefineVariable, bioMax, bioNormalCdf, Elem, log
import biogeme.models as models
from biogeme.models import loglogit

import numpy as np
import time
from biogeme.expressions import (
    Beta,
    Variable,
    DefineVariable,
    exp,
    log,
)


def generate_synthetic_latent_observed_choices(df_full, N, R, epsilon, latent, latent_indices):
    x, y, J, K, biog_beta, biog_loglike, beta_confs, loglike_conf = choice_data_netherlands(N, df_full,
                                                                                            latent=False,
                                                                                            intercept=True,
                                                                                            cost=True)

    # create custom error terms
    # R = 1000
    # epsilon = np.random.gumbel(loc=0, scale=1, size=(J, N, R))

    # now we need a dictionary in individuals that assigns a class randomly
    ind_class = dict()
    for n in range(N):
        ind_class[n] = random.randint(0, latent - 1)
    for c in range(latent):
        print(f"percentage of class {c} = {(sum(1 for v in ind_class.values() if v == c)/N)*100}%")
    # now assign y based on highest utilities
    y_new = dict()
    for n in range(N):
        Uinr = dict()
        y_counter = {i: 0 for i in range(J)}
        for r in range(R):
            for i in range(J):
                Uinr[i, r] = assign_utility(biog_beta, K, x, epsilon, n, i, r, latent, latent_indices, ind_class)
            max_util = np.max([Uinr[i, r] for i in range(J)])
            for i in range(J):
                if Uinr[i, r] == max_util:
                    y_counter[i] += 1
        max_counter = np.max([y_counter[i] for i in range(J)])
        for i in range(J):
            if y_counter[i] == max_counter:
                y_new[i, n] = 1
            else:
                y_new[i, n] = 0
    return y_new


def assign_utility(beta, K, x, epsilon, n, i, r, latent, latent_indices, ind_class):
    nonlatent = [k for k in range(K) if k not in latent_indices]
    util = None
    if len(latent_indices) == 1:
        if ind_class[n] == 0:
            util = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                         + beta[latent_indices[0]] * x[i, n, latent_indices[0]] \
                         + epsilon[i, n, r]
        elif ind_class[n] == 1:
            util = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                         + epsilon[i, n, r]

    elif len(latent_indices) == 2 and latent == 3:
        if ind_class[n] == 0:
            util = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                         + beta[latent_indices[0]] * x[i, n, latent_indices[0]] \
                         + epsilon[i, n, r]
        elif ind_class[n] == 1:
            util = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                         + beta[latent_indices[1]] * x[i, n, latent_indices[1]] \
                         + epsilon[i, n, r]
        elif ind_class[n] == 2:
            util = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                         + beta[latent_indices[0]] * x[i, n, latent_indices[0]] \
                         + beta[latent_indices[1]] * x[i, n, latent_indices[1]] \
                         + epsilon[i, n, r]
    elif len(latent_indices) == 2 and latent == 4:
        if ind_class[n] == 0:
            util = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                         + beta[latent_indices[0]] * x[i, n, latent_indices[0]] \
                         + epsilon[i, n, r]
        elif ind_class[n] == 1:
            util = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                         + beta[latent_indices[1]] * x[i, n, latent_indices[1]] \
                         + epsilon[i, n, r]
        elif ind_class[n] == 2:
            util = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                         + beta[latent_indices[0]] * x[i, n, latent_indices[0]] \
                         + beta[latent_indices[1]] * x[i, n, latent_indices[1]] \
                         + epsilon[i, n, r]
        elif ind_class[n] == 3:
            util = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                         + epsilon[i, n, r]
    return util


def choice_data_netherlands(size, df_cut, latent, intercept=True, cost=True, probit=False):
    # df = pd.read_csv('netherlands.dat', '\t')
    # df = df[df['rp'] == 1]
    # df['rail_time'] = df.rail_ivtt + df.rail_acc_time + df.rail_egr_time
    # df['car_time'] = df.car_ivtt + df.car_walk_time
    # df_cut = df.sample(size)
    # df_cut = df_full.head(size)

    # Michel messed up the code
    start_time_biog = time.time()
    # Max N = 31 954
    try:
        biog_beta, biog_loglike, beta_confs, loglike_conf = biogeme_estimate_beta_nether(df_cut,
                                                                                         latent=latent,
                                                                                         intercept=intercept,
                                                                                         cost=cost,
                                                                                         probit=probit)
    except RuntimeError:
        print("BioGeme crashed")
        biog_beta = [5, 5, 5] + [5 for l in range(len(latent - 1))]
        biog_loglike = 5
        loglike_conf = [-5, 5]
        beta_confs = dict()
        for k in range(len(biog_beta)):
            beta_confs[k] = [-5, 5]
    time_biog = time.time() - start_time_biog
    # biog_beta = [20]
    # biog_loglike = [20]
    # beta_confs = [[20, 20], [20, 20], [20, 20], [20, 20], [20, 20]]
    # loglike_conf = [[20, 20], [20, 20], [20, 20], [20, 20], [20, 20]]
    print(f"Biogeme estimated biog_beta = {biog_beta} with biog_loglike = {biog_loglike} in {time_biog} seconds")

    J = 2
    if intercept:
        if cost:
            K = 3
            x = dict()
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
        else:
            K = 2
            x = dict()
            for n in range(size):
                # intercept for rail
                x[0, n, 0] = 0
                x[1, n, 0] = 1
                # beta time
                x[0, n, 1] = df_cut['car_time'].values[n]
                x[1, n, 1] = df_cut['rail_time'].values[n]
    else:
        if cost:
            K = 2
            x = dict()
            for n in range(size):
                # beta cost
                x[0, n, 0] = df_cut['car_cost'].values[n]
                x[1, n, 0] = df_cut['rail_cost'].values[n]
                # beta time
                x[0, n, 1] = df_cut['car_time'].values[n]
                x[1, n, 1] = df_cut['rail_time'].values[n]
        else:
            K = 1
            x = dict()
            for n in range(size):
                # beta time
                x[0, n, 0] = df_cut['car_time'].values[n]
                x[1, n, 0] = df_cut['rail_time'].values[n]

    y = np.array([df_cut.choice == i for i in range(J)]).astype(int)

    return x, y, J, K, biog_beta, biog_loglike, beta_confs, loglike_conf


def biogeme_estimate_beta_nether(df, latent, intercept=False, cost=False, probit=False):
    database = db.Database('netherlands', df)
    globals().update(database.variables)

    # Parameters to be estimated
    # Arguments:
    #   1  Name for report. Typically, the same as the variable
    #   2  Starting value
    #   3  Lower bound
    #   4  Upper bound
    #   5  0: estimate the parameter, 1: keep it fixed

    if not latent >= 1:
        K = 0
        BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
        K += 1
        if intercept:
            K += 1
            ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
        if cost:
            K += 1
            BETA_COST = Beta('BETA_COST', 0, None, None, 0)

        # Utilities
        if cost:
            Car = BETA_TIME * car_time + BETA_COST * car_cost
        else:
            Car = BETA_TIME * car_time
        if intercept:
            if cost:
                Rail = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
            else:
                Rail = ASC_RAIL + BETA_TIME * rail_time
        else:
            if cost:
                Rail = BETA_TIME * rail_time + BETA_COST * rail_cost
            else:
                Rail = BETA_TIME * rail_time

        V = {0: Car, 1: Rail}  # deterministic part: use utilities depending on mode indicator
        av = {0: 1, 1: 1}  # both are available to everyone

        # specify likelihood function
        if probit:
            # Associate choice probability with the numbering of alternatives
            P = {0: bioNormalCdf(V[0] - V[1]), 1: bioNormalCdf(V[1] - V[0])}

            # Definition of the model. This is the contribution of each
            # observation to the log likelihood function.
            logprob = log(Elem(P, choice))
        else:
            # The choice model is a logit, with availability conditions
            logprob = loglogit(V, av, choice)

        # specify formulas for simulation
        if intercept:
            if cost:
                formulas = {
                    "BETA_TIME": BETA_TIME,
                    "BETA_COST": BETA_COST,
                    "ASC_RAIL": ASC_RAIL,
                    "loglike": logprob
                }
            else:
                formulas = {
                    "BETA_TIME": BETA_TIME,
                    "ASC_RAIL": ASC_RAIL,
                    "loglike": logprob
                }
        else:
            if cost:
                formulas = {
                    "BETA_TIME": BETA_TIME,
                    "BETA_COST": BETA_COST,
                    "loglike": logprob
                }
            else:
                formulas = {
                    "BETA_TIME": BETA_TIME,
                    "loglike": logprob
                }

        biogeme = bio.BIOGEME(database, formulas)
        biogeme.modelName = "binary_netherlands"

        start_time = time.time()
        if K > 1:
            results = biogeme.estimate(bootstrap=100)
        else:
            results = biogeme.estimate()

        biog_loglike = round(results.data.logLike, 8)
        pandasResults = results.getEstimatedParameters()
        betas = list(pandasResults.index)
        biog_beta = list(pandasResults["Value"])

        # Get confidence intervals on beta and loglike
        if K > 1:
            b = results.getBetasForSensitivityAnalysis(betas, useBootstrap=False, size=100)
        else:
            b = results.getBetasForSensitivityAnalysis(betas, useBootstrap=False, size=100)
        # simulatedValues = biogeme.simulate(results.getBetaValues())

        # results.getEstimatedParameters()

        left, right = biogeme.confidenceIntervals(b, 0.9)

        loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
        beta_confs = dict()
        for i in range(K):
            beta_confs[i] = [left[betas[i]].iloc[0], right[betas[i]].iloc[0]]

        # print(pandasResults)
        #
        # print("")
        # print(f"Loglike = {biog_loglike}")
        # print("")
        # print(f"Estimation time = {time.time() - start_time}s")

    elif latent == 2:
        BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
        ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
        BETA_COST = Beta('BETA_COST', 0, None, None, 0)

        # Utilities
        Car_class_1 = BETA_TIME * car_time + BETA_COST * car_cost
        Car_class_2 = BETA_COST * car_cost

        Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
        Rail_class_2 = ASC_RAIL + BETA_COST * rail_cost

        V_class_1 = {0: Car_class_1, 1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
        V_class_2 = {0: Car_class_2, 1: Rail_class_2}

        av = {0: 1, 1: 1}  # both are available to everyone

        # specify likelihood function
        prob_class_1 = Beta('prob_class_1', 0.5, 0, 1, 0)
        prob_class_2 = 1 - prob_class_1

        prob = (
                prob_class_1 * models.logit(V_class_1, av, choice) +
                prob_class_2 * models.logit(V_class_2, av, choice)
        )

        logprob = log(prob)
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.modelName = "binary_netherlands"
        start_time = time.time()
        results = biogeme.estimate()

        biog_loglike = round(results.data.logLike, 8)
        pandasResults = results.getEstimatedParameters()
        biog_beta = list(pandasResults["Value"])
        betas = list(pandasResults.index)
        # print(pandasResults)
        #
        # print("")
        # print(f"Loglike = {biog_loglike}")
        # print("")
        # print(f"Estimation time = {time.time() - start_time}s")

        # loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
        loglike_conf = [-5, 5]
        beta_confs = dict()
        for k in range(len(betas)):
            beta_confs[k] = [-5, 5]  # we honestly don't care about this now
            # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]

    elif latent == 3:
        BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
        ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
        BETA_COST = Beta('BETA_COST', 0, None, None, 0)

        # Utilities
        Car_class_1 = BETA_TIME * car_time
        Car_class_2 = BETA_COST * car_cost
        Car_class_3 = BETA_TIME * car_time + BETA_COST * car_cost

        Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time
        Rail_class_2 = ASC_RAIL + BETA_COST * rail_cost
        Rail_class_3 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost

        V_class_1 = {0: Car_class_1, 1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
        V_class_2 = {0: Car_class_2, 1: Rail_class_2}
        V_class_3 = {0: Car_class_3, 1: Rail_class_3}

        av = {0: 1, 1: 1}  # both are available to everyone

        # specify likelihood function
        prob_class_1 = Beta('prob_class_1', 0.3, 0, 1, 0)
        prob_class_2 = Beta('prob_class_2', 0.3, 0, 1, 0)
        prob_class_3 = 1 - prob_class_1 - prob_class_2

        # Michels suggestion
        # vp1 = Beta("vp1", 0, None, None, 0)
        # vp2 = Beta("vp2", 0, None, None, 0)
        # vp3 = 0
        #
        # vp = {1: vp1, 2: vp2, 3: vp3}
        #
        # prob_class_1 = models.logit(vp, None, 1)
        # prob_class_2 = models.logit(vp, None, 2)
        # prob_class_3 = models.logit(vp, None, 3)

        prob = (
                prob_class_1 * models.logit(V_class_1, av, choice) +
                prob_class_2 * models.logit(V_class_2, av, choice) +
                prob_class_3 * models.logit(V_class_3, av, choice)
        )

        logprob = log(prob)
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.modelName = "binary_netherlands"
        start_time = time.time()
        results = biogeme.estimate()

        biog_loglike = round(results.data.logLike, 8)
        pandasResults = results.getEstimatedParameters()
        biog_beta = list(pandasResults["Value"])
        vpi = biog_beta[-2:]
        betas = list(pandasResults.index)
        # print(pandasResults)
        #
        # print("")
        # print(f"Loglike = {biog_loglike}")
        # print("Estimated class parameters:")
        # # p1 = np.exp(vpi[0]) / (np.exp(vpi[0]) + np.exp(vpi[1]) + np.exp(0))
        # # p2 = np.exp(vpi[1]) / (np.exp(vpi[0]) + np.exp(vpi[1]) + np.exp(0))
        # # p3 = 1 - p1 - p2
        # # p1 = round(p1, 8)
        # # p2 = round(p2, 8)
        # # p3 = round(p3, 8)
        # print([vpi[0], vpi[1], 1-vpi[0]-vpi[1]])
        # print("")
        # print(f"Estimation time = {time.time() - start_time}s")

        # biog_beta = biog_beta[0:len(biog_beta) - 2]
        # biog_beta = biog_beta + [p1, p2]

        # loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
        loglike_conf = [-5, 5]
        beta_confs = dict()
        for k in range(len(betas)):
            beta_confs[k] = [-5, 5]  # we honestly don't care about this now
            # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]

    elif latent == 4:
        BETA_TIME = Beta('BETA_TIME', 0, None, None, 0)
        ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 0)
        BETA_COST = Beta('BETA_COST', 0, None, None, 0)

        # Utilities
        Car_class_1 = BETA_TIME * car_time
        Car_class_2 = BETA_COST * car_cost
        Car_class_3 = BETA_TIME * car_time + BETA_COST * car_cost
        Car_class_4 = 0

        Rail_class_1 = ASC_RAIL + BETA_TIME * rail_time
        Rail_class_2 = ASC_RAIL + BETA_COST * rail_cost
        Rail_class_3 = ASC_RAIL + BETA_TIME * rail_time + BETA_COST * rail_cost
        Rail_class_4 = ASC_RAIL

        V_class_1 = {0: Car_class_1, 1: Rail_class_1}  # deterministic part: use utilities depending on mode indicator
        V_class_2 = {0: Car_class_2, 1: Rail_class_2}
        V_class_3 = {0: Car_class_3, 1: Rail_class_3}
        V_class_4 = {0: Car_class_4, 1: Rail_class_4}

        av = {0: 1, 1: 1}  # both are available to everyone

        # specify likelihood function
        # prob_class_1 = Beta('prob_class_1', 0.25, 0, 1, 0)
        # prob_class_2 = Beta('prob_class_2', 0.25, 0, 1, 0)
        # prob_class_3 = Beta('prob_class_3', 0.25, 0, 1, 0)
        # prob_class_4 = 1 - prob_class_1 - prob_class_2 - prob_class_3

        # Michels suggestion
        vp1 = Beta("vp1", 0, None, None, 0)
        vp2 = Beta("vp2", 0, None, None, 0)
        vp3 = Beta("vp3", 0, None, None, 0)
        vp4 = 0

        vp = {1: vp1, 2: vp2, 3: vp3, 4: vp4}

        prob_class_1 = models.logit(vp, None, 1)
        prob_class_2 = models.logit(vp, None, 2)
        prob_class_3 = models.logit(vp, None, 3)
        prob_class_4 = models.logit(vp, None, 4)

        prob = (
                prob_class_1 * models.logit(V_class_1, av, choice) +
                prob_class_2 * models.logit(V_class_2, av, choice) +
                prob_class_3 * models.logit(V_class_3, av, choice) +
                prob_class_4 * models.logit(V_class_4, av, choice)
        )

        logprob = log(prob)
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.modelName = "binary_netherlands"
        start_time = time.time()
        results = biogeme.estimate()

        biog_loglike = round(results.data.logLike, 8)
        pandasResults = results.getEstimatedParameters()
        biog_beta = list(pandasResults["Value"])
        vpi = biog_beta[-3:]
        betas = list(pandasResults.index)
        # print(pandasResults)
        #
        # print("")
        # print(f"Loglike = {biog_loglike}")
        # print("Estimated class parameters:")
        # p1 = np.exp(vpi[0]) / (np.exp(vpi[0]) + np.exp(vpi[1]) + np.exp(vpi[2]) + np.exp(0))
        # p2 = np.exp(vpi[1]) / (np.exp(vpi[0]) + np.exp(vpi[1]) + np.exp(vpi[2]) + np.exp(0))
        # p3 = np.exp(vpi[2]) / (np.exp(vpi[0]) + np.exp(vpi[1]) + np.exp(vpi[2]) + np.exp(0))
        # p4 = 1 - p1 - p2 - p3
        # p1 = round(p1, 8)
        # p2 = round(p2, 8)
        # p3 = round(p3, 8)
        # p4 = round(p4, 8)
        # print([p1, p2, p3, p4])
        # print("")
        # print(f"Estimation time = {time.time() - start_time}s")

        biog_beta = biog_beta[0:len(biog_beta) - 3]
        biog_beta = biog_beta + [p1, p2, p3]

        # loglike_conf = [left["loglike"].sum(), right["loglike"].sum()]
        loglike_conf = [-5, 5]
        beta_confs = dict()
        for k in range(len(betas)):
            beta_confs[k] = [-5, 5]  # we honestly don't care about this now
            # beta_confs[k] = [left[betas[k]].iloc[0], right[betas[k]].iloc[0]]

    # Remove output files
    os.remove("binary_netherlands.html")
    os.remove("binary_netherlands.pickle")
    os.remove("__binary_netherlands.iter")

    return biog_beta, biog_loglike, beta_confs, loglike_conf
