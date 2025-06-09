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

    seed_nr = pandaSeed + 2
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

    seed_nr = pandaSeed + 2
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
        seed_nr = pandaSeed + 2
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
            Z1_Beta_time_S = Beta('Z1_Beta_time_S', 1, None, None, 0)
            Beta_time_RND = Beta_time + Z1_Beta_time_S * bioDraws('Beta_time_RND', 'NORMAL')

            Walk = ASC_Walk + Walk_TT * Beta_time_RND
            Bike = ASC_Bike + Bike_TT * Beta_time_RND
            PB = ASC_PB + PB_TT * Beta_time_RND + PB_cost * Beta_cost
            Car = ASC_Car + Car_TT * Beta_time_RND + Car_cost * Beta_cost

        if latent == 11:
            mix_inds = [[5, 6], [4, 7]]  # mix time and costs
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
            Z1_Beta_cost_S = Beta('Z1_Beta_cost_S', 1, None, None, 0)
            Beta_cost_RND = Beta_cost + Z1_Beta_cost_S * bioDraws('Beta_cost_RND', 'NORMAL')

            Walk = ASC_Walk + Walk_TT * Beta_time
            Bike = ASC_Bike + Bike_TT * Beta_time
            PB = ASC_PB + PB_TT * Beta_time + PB_cost * Beta_cost_RND
            Car = ASC_Car + Car_TT * Beta_time + Car_cost * Beta_cost_RND

        if latent == 18:
            mix_inds = [[2, 6]]  # mix ASC car
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
    biosim = bio.BIOGEME(database, simulate, numberOfDraws=R, seed=pandaSeed)
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
        betas['Z2_Beta_cost_S'] = beta[6]
    elif latent == 12:
        betas['Z1_Beta_cost_S'] = beta[5]
    elif latent == 18:
        betas['Z1_ASC_Car_S'] = beta[5]

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
            logLikelihood = simulate_likelihood_mixed_latent_lpmc(N, pandaSeed, beta, latent, R)
        else:
            logLikelihood = simulate_likelihood_mixed_lpmc(N, pandaSeed, beta, latent, R)
    print(logLikelihood)
