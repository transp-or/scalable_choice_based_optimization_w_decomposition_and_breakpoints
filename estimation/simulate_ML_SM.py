import numpy as np
import pandas as pd
import random
import sys
import os
import contextlib
import copy
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta, Variable, bioDraws, log, MonteCarlo, exp


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
    seed_nr = pandaSeed + 2
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

    seed_nr = pandaSeed + 2
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

    seed_nr = pandaSeed + 2
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

    seed_nr = pandaSeed + 2
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
    biosim = bio.BIOGEME(database, simulate, numberOfDraws=R, seed=pandaSeed)
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


def simulate_likelihood_mixed_optima(N, pandaSeed, beta, latent, R=1000):
    df_full = pd.read_csv("optima.dat", sep='\t')
    # Exclude observations such that the chosen alternative is -1
    df_full = df_full[df_full["Choice"] != -1]

    if not N == 0:
        df_full = df_full.sample(N, random_state=pandaSeed)
    else:
        df_full = df_full

    database = db.Database("optima", df)

    TimePT = Variable('TimePT')
    Choice = Variable('Choice')
    TimeCar = Variable('TimeCar')
    MarginalCostPT = Variable('MarginalCostPT')
    CostCarCHF = Variable('CostCarCHF')
    distance_km = Variable('distance_km')
    TripPurpose = Variable('TripPurpose')
    WaitingTimePT = Variable('WaitingTimePT')

    ASC_PT = Beta('ASC_PT', 0.0, None, None, 1)

    ASC_CAR = Beta('ASC_CAR', 0.0, None, None, 0)
    ASC_SM = Beta('ASC_SM', 0.0, None, None, 0)
    BETA_COST_HWH = Beta('BETA_COST_HWH', 0.0, None, None, 0)
    BETA_COST_OTHER = Beta('BETA_COST_OTHER', 0.0, None, None, 0)
    BETA_DIST = Beta('BETA_DIST', 0.0, None, None, 0)
    BETA_TIME_CAR = Beta('BETA_TIME_CAR', 0.0, None, None, 0)
    BETA_TIME_PT = Beta('BETA_TIME_PT', 0.0, None, None, 0)
    BETA_WAITING_TIME = Beta('BETA_WAITING_TIME', 0.0, None, None, 0)

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

    if 10 <= latent <= 19:
        seed_nr = pandaSeed + 2
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

        elif latent == 11:
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12]]  # mix time and costs
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

    # numberOfDraws = 100000
    integral = MonteCarlo(prob)
    simulate = {
        'Integral': integral,
    }
    # Create the Biogeme object
    biosim = bio.BIOGEME(database, simulate, numberOfDraws=R, seed=pandaSeed)
    biosim.modelName = "optima_mixed_simul"

    betas = {
        'ASC_CAR': beta[0],
        'ASC_SM': beta[1],
        'BETA_COST_HWH': beta[2],
        'BETA_COST_OTHER': beta[3],
        'BETA_DIST': beta[4],
        'BETA_TIME_CAR': beta[5],
        'BETA_TIME_PT': beta[6],
        'BETA_WAITING_TIME': beta[7]
    }

    betaidx = 8

    if latent == 10:
        betas['Z1_BETA_TIME_PT_S'] = beta[8]
        betas['Z2_BETA_TIME_CAR_S'] = beta[9]
    elif latent == 11:
        betas['Z1_BETA_TIME_PT_S'] = beta[8]
        betas['Z2_BETA_TIME_CAR_S'] = beta[9]
        betas['Z3_BETA_COST_HWH_S'] = beta[10]
        betas['Z4_BETA_TIME_CAR_S'] = beta[11]
    elif latent == 12:
        betas['Z1_BETA_TIME_PT_S'] = beta[8]
        betas['Z2_BETA_TIME_CAR_S'] = beta[9]
        betas['Z3_BETA_COST_HWH_S'] = beta[10]
        betas['Z4_BETA_TIME_CAR_S'] = beta[11]
        betas['Z5_BETA_DIST_S'] = beta[12]
    elif latent == 13:
        betas['Z1_BETA_TIME_PT_S'] = beta[8]
        betas['Z2_BETA_TIME_CAR_S'] = beta[9]
        betas['Z5_BETA_DIST_S'] = beta[10]
    elif latent == 14:
        betas['Z1_BETA_TIME_PT_S'] = beta[8]
        betas['Z2_BETA_TIME_CAR_S'] = beta[9]
        betas['Z3_BETA_COST_HWH_S'] = beta[10]
        betas['Z4_BETA_TIME_CAR_S'] = beta[11]
        betas['Z5_BETA_DIST_S'] = beta[12]
        betas['Z6_ASC_CAR_S'] = beta[13]
        betas['Z7_ASC_SM_S'] = beta[14]
    elif latent == 15:
        betas['Z1_BETA_TIME_PT_S'] = beta[8]
        betas['Z2_BETA_TIME_CAR_S'] = beta[9]
        betas['Z3_BETA_COST_HWH_S'] = beta[10]
        betas['Z4_BETA_TIME_CAR_S'] = beta[11]
        betas['Z6_ASC_CAR_S'] = beta[12]
        betas['Z7_ASC_SM_S'] = beta[13]
    elif latent == 16:
        betas['Z1_BETA_TIME_PT_S'] = beta[8]
        betas['Z2_BETA_TIME_CAR_S'] = beta[9]
        betas['Z6_ASC_CAR_S'] = beta[10]
        betas['Z7_ASC_SM_S'] = beta[11]
    elif latent == 17:
        betas['Z6_ASC_CAR_S'] = beta[8]
        betas['Z7_ASC_SM_S'] = beta[9]

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

    with suppress_output():
        # Call the function and print the result
        if 1020 <= latent <= 1039:
            logLikelihood = simulate_likelihood_mixed_latent_swissmetro(N, pandaSeed, beta, latent, R)
        else:
            if latent == 10:
                mix_inds = [[5, 6]]  # mix just time
            elif latent == 11:
                mix_inds = [[5, 6], [3, 7]]  # mix time and costs
            elif latent == 12:
                mix_inds = [[5, 6], [3, 7], [4, 8]]  # mix time and costs and headway
            else:
                mix_inds = None
            logLikelihood = simulate_likelihood_mixed_swissmetro(N, pandaSeed, beta, mix_inds, R)
    print(logLikelihood)
