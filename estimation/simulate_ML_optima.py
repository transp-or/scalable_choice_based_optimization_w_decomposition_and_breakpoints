import numpy as np
import pandas as pd
import random
import sys
import ast
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta, Variable, bioDraws, log, MonteCarlo, exp


def simulate_likelihood_mixed_swissmetro(N, pandaSeed, beta, mix_inds, R=1000):
    df_full = pd.read_csv('swissmetro.dat', sep='\t')
    df_full = df_full.loc[
        ~((df_full["PURPOSE"] != 1) & (df_full["PURPOSE"] != 3) | (df_full["CHOICE"] == 0) > 0)]

    if not N == 0:
        df_full = df_full.sample(N, random_state=pandaSeed)
    else:
        df_full = df_full

    database = db.Database('swissmetro', df_full)

    seed_nr = 192
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


def simulate_likelihood_mixed_optima(N, pandaSeed, beta, latent, R=1000):
    df_full = pd.read_csv("optima.dat", sep='\t')
    # Exclude observations such that the chosen alternative is -1
    df_full = df_full[df_full["Choice"] != -1]

    if not N == 0:
        df_full = df_full.sample(N, random_state=pandaSeed)
    else:
        df_full = df_full

    database = db.Database("optima", df_full)

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
        seed_nr = 192
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
        elif latent == 18:
            mix_inds = [[1, 9]]  # mix ASCs only
            Z6_ASC_CAR_S = Beta('Z6_ASC_CAR_S', 1, None, None, 0)
            ASC_CAR_RND = ASC_CAR + Z6_ASC_CAR_S * bioDraws('ASC_CAR_RND', 'NORMAL')
            
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
    biosim = bio.BIOGEME(database, simulate, numberOfDraws=R)
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
    elif latent == 18:
        betas['Z6_ASC_CAR_S'] = beta[8]

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
    logLikelihood = simulate_likelihood_mixed_optima(N, pandaSeed, beta, latent, R)
    print(logLikelihood)
