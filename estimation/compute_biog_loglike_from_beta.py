import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, bioDraws, MonteCarlo


def simulate_likelihood_mixed_swissmetro(df, beta, numberOfDraws=1000):
    # Read the data
    # df = pd.read_csv('swissmetro.dat', sep='\t')

    # N = 300

    # df = df.sample(N)

    # df_export = df
    # df_export.to_csv("N_sample_SM_300.csv", sep=";", index=False)

    # df = pd.read_csv("N_sample_SM_300.csv", sep=";")

    database = db.Database('swissmetro', df)

    # The Pandas data structure is available as database.data. Use all the
    # Pandas functions to investigate the database. For example:
    # print(database.data.describe())

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

    # Removing some observations can be done directly using pandas.
    # remove = (((database.data.PURPOSE != 1) &
    #           (database.data.PURPOSE != 3)) |
    #          (database.data.CHOICE == 0))
    # database.data.drop(database.data[remove].index,inplace=True)

    # Here we use the "biogeme" way for backward compatibility
    exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
    database.remove(exclude)

    # Parameters to be estimated
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
    ASC_SM = Beta('ASC_SM', 0, None, None, 1)
    B_COST = Beta('B_COST', 0, None, None, 0)

    # Define a random parameter, normally distributed, designed to be used
    # for Monte-Carlo simulation
    B_TIME = Beta('B_TIME', 0, None, None, 0)

    # It is advised not to use 0 as starting value for the following parameter.
    B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
    B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

    # Definition of new variables
    # SM_COST = SM_CO * (GA == 0)
    # TRAIN_COST = TRAIN_CO * (GA == 0)

    # Definition of new variables: adding columns to the database

    # it seems like its not necessary to do this because biogeme changes the actual input during estimation

    CAR_AV_SP = Variable('CAR_AV_SP')
    TRAIN_AV_SP = Variable('TRAIN_AV_SP')
    TRAIN_TT_SCALED = Variable('TRAIN_TT_SCALED')
    TRAIN_COST_SCALED = Variable('TRAIN_COST_SCALED')
    SM_TT_SCALED = Variable('SM_TT_SCALED')
    SM_COST_SCALED = Variable('SM_COST_SCALED')
    CAR_TT_SCALED = Variable('CAR_TT_SCALED')
    CAR_CO_SCALED = Variable('CAR_CO_SCALED')

    # Definition of the utility functions
    V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
    V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
    V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

    # Associate utility functions with the numbering of alternatives
    V = {1: V1, 2: V2, 3: V3}

    # Associate the availability conditions with the alternatives
    av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

    # Conditional to B_TIME_RND, we have a logit model (called the kernel)
    prob = models.logit(V, av, CHOICE)

    # We calculate the integration error. Note that this formula assumes
    # independent draws, and is not valid for Haltom or antithetic draws.

    # numberOfDraws = 100000
    integral = MonteCarlo(prob)
    integralSquare = MonteCarlo(prob * prob)
    variance = integralSquare - integral * integral
    error = (variance / 2.0) ** 0.5

    # And the value of the individual parameters
    numerator = MonteCarlo(B_TIME_RND * prob)
    denominator = integral

    simulate = {
        # 'Numerator': numerator,
        # 'Denominator': denominator,
        'Integral': integral,
    }

    # Create the Biogeme object
    biosim = bio.BIOGEME(database, simulate, numberOfDraws=numberOfDraws)
    biosim.modelName = "swissmetro_mixed_simul"

    # Simulate the requested quantities. The output is a Pandas data frame

    # The estimation results are read from the pickle file

    # try:
    #     results = res.bioResults(pickleFile='05normalMixture.pickle')
    # except biogemeError:
    #     print(
    #         'Run first the script 05normalMixture.py in order to generate the '
    #         'file 05normalMixture.pickle.'
    #     )
    #     sys.exit()

    betas = {
        'ASC_CAR': beta[0],
        'ASC_TRAIN': beta[1],
        'B_COST': beta[2],
        'B_TIME': beta[3],
        'B_TIME_S': beta[4],
    }

    simresults = biosim.simulate(betas)
    # simresults = biosim.simulate(results.getBetaValues())

    # 95% confidence interval on the log likelihood.
    # simresults['left'] = np.log(
    #     simresults['Integral'] - 1.96 * simresults['Integration error']
    # )
    # simresults['right'] = np.log(
    #     simresults['Integral'] + 1.96 * simresults['Integration error']
    # )

    logLikelihood = np.log(simresults["Integral"]).sum()

    # print(
    #     f'Integration error for {numberOfDraws} draws: '
    #     f'{simresults["Integration error"].sum()}'
    # )
    # print(f'In average {simresults["Integration error"].mean()} per observation.')
    # print(
    #     f'95% confidence interval: [{simresults["left"].sum()}-'
    #     f'{simresults["right"].sum()}]'
    # )

    # # Post processing to obtain the individual parameters
    # simresults['beta'] = simresults['Numerator'] / simresults['Denominator']

    # # Plot the histogram of individual parameters
    # if plot:
    #     simresults['beta'].plot(kind='hist', density=True, bins=20)

    # # Plot the general distribution of beta
    # def normalpdf(v, mu=0.0, s=1.0):
    #     """
    #     Calculate the pdf of the normal distribution, for plotting purposes.

    #     """
    #     d = -(v - mu) * (v - mu)
    #     n = 2.0 * s * s
    #     a = d / n
    #     num = np.exp(a)
    #     den = s * 2.506628275
    #     p = num / den
    #     return p

    # x = np.arange(simresults['beta'].min(), simresults['beta'].max(), 0.01)
    # if plot:
    #     plt.plot(x, normalpdf(x, betas['B_TIME'], betas['B_TIME_S']), '-')
    #     plt.show()

    return logLikelihood


def simulate_likelihood_mixed_nether(df, beta, numberOfDraws=1000):
    database = db.Database('netherlands', df)

    CHOICE = Variable("choice")

    CAR_TIME_SCALED = Variable('CAR_TIME_SCALED')
    RAIL_TIME_SCALED = Variable('RAIL_TIME_SCALED')
    CAR_COST_SCALED = Variable('CAR_COST_SCALED')
    RAIL_COST_SCALED = Variable('RAIL_COST_SCALED')

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

    # numberOfDraws = 100000
    integral = MonteCarlo(prob)

    simulate = {
        'Integral': integral,
    }

    # Create the Biogeme object
    biosim = bio.BIOGEME(database, simulate, numberOfDraws=numberOfDraws)
    biosim.modelName = "nether_mixed_simul"

    betas = {
        'ASC_RAIL': beta[0],
        'BETA_COST': beta[1],
        'B_TIME': beta[2],
        'B_TIME_S': beta[3],
    }

    simresults = biosim.simulate(betas)

    logLikelihood = np.log(simresults["Integral"]).sum()

    return logLikelihood


def compute_biog_loglike(x, y, beta, mixed=None):
    beta = [b for b in beta if b is not None]
    K = len(beta)
    N = len(y[0])
    J = len(y)

    Vin = dict()
    Pin = dict()

    if not mixed:
        for i in range(J):
            for n in range(N):
                Vin[i, n] = sum(beta[k] * x[i, n, k] for k in range(K))

        for i in range(J):
            for n in range(N):
                Pin[i, n] = np.exp(Vin[i, n]) / sum(np.exp(Vin[j, n]) for j in range(J))
    else:
        mixed_index = mixed
        not_mixed_params = [k for k in range(K) if not (k == mixed_index or k == mixed_index + 1)]
        for i in range(J):
            for n in range(N):
                Vin[i, n] = sum(beta[k] * x[i, n, k] for k in not_mixed_params) \
                            + beta[mixed_index] * x[i, n, mixed_index] \
                            + beta[mixed_index + 1] * x[i, n, mixed_index + 1]

        for i in range(J):
            for n in range(N):
                Pin[i, n] = np.exp(Vin[i, n]) / sum(np.exp(Vin[j, n]) for j in range(J))

    biog_obj = sum(y[i, n] * np.log(Pin[i, n]) for i in range(J) for n in range(N))

    return biog_obj


def compute_biog_loglike_latent(x, y, av, latent, latent_indices, beta):
    if latent == 2:
        pi = beta[-1]
        beta = beta[0:len(beta) - 1]
    if latent == 3:
        pi = beta[-2:]
        beta = beta[0:len(beta) - 2]
    if latent == 4:
        pi = beta[-3:]
        beta = beta[0:len(beta) - 3]
    # we assume that pi is simply the last beta
    K = len(beta)
    N = len(y[0])
    J = len(y)

    Vin = dict()
    Pin = dict()

    nonlatent = [k for k in range(K) if k not in latent_indices]
    available_alt = dict()
    for n in range(N):
        available_alt[n] = [i for i in range(J) if av[i][n] == 1]

    if latent == 2:
        for n in range(N):
            for i in available_alt[n]:
                Vin[i, n, 0] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                               + beta[latent_indices[0]] * x[i, n, latent_indices[0]]

                Vin[i, n, 1] = sum(beta[k] * x[i, n, k] for k in nonlatent)

        for n in range(N):
            for i in available_alt[n]:
                Pin[i, n, 0] = np.exp(Vin[i, n, 0]) / sum(np.exp(Vin[j, n, 0])
                                                          for j in available_alt[n])
                Pin[i, n, 1] = np.exp(Vin[i, n, 1]) / sum(np.exp(Vin[j, n, 1])
                                                          for j in available_alt[n])

        biog_obj = sum(y[i, n] * np.log(pi * Pin[i, n, 0] + (1 - pi) * Pin[i, n, 1])
                       for n in range(N) for i in available_alt[n])
    if latent == 3:
        for n in range(N):
            for i in available_alt[n]:
                Vin[i, n, 0] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                               + beta[latent_indices[0]] * x[i, n, latent_indices[0]]

                Vin[i, n, 1] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                               + beta[latent_indices[1]] * x[i, n, latent_indices[1]]

                Vin[i, n, 2] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                               + beta[latent_indices[0]] * x[i, n, latent_indices[0]] \
                               + beta[latent_indices[1]] * x[i, n, latent_indices[1]]

        for n in range(N):
            for i in available_alt[n]:
                Pin[i, n, 0] = np.exp(Vin[i, n, 0]) / sum(np.exp(Vin[j, n, 0])
                                                          for j in available_alt[n])
                Pin[i, n, 1] = np.exp(Vin[i, n, 1]) / sum(np.exp(Vin[j, n, 1])
                                                          for j in available_alt[n])
                Pin[i, n, 2] = np.exp(Vin[i, n, 2]) / sum(np.exp(Vin[j, n, 2])
                                                          for j in available_alt[n])

        biog_obj = sum(y[i, n] * np.log(pi[0] * Pin[i, n, 0] + pi[1] * Pin[i, n, 1]
                                        + (1 - pi[0] - pi[1]) * Pin[i, n, 2])
                       for n in range(N) for i in available_alt[n])
    if latent == 4:
        for n in range(N):
            for i in available_alt[n]:
                Vin[i, n, 0] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                               + beta[latent_indices[0]] * x[i, n, latent_indices[0]]

                Vin[i, n, 1] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                               + beta[latent_indices[1]] * x[i, n, latent_indices[1]]

                Vin[i, n, 2] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                               + beta[latent_indices[0]] * x[i, n, latent_indices[0]] \
                               + beta[latent_indices[1]] * x[i, n, latent_indices[1]]
                Vin[i, n, 3] = sum(beta[k] * x[i, n, k] for k in nonlatent)

        for n in range(N):
            for i in available_alt[n]:
                Pin[i, n, 0] = np.exp(Vin[i, n, 0]) / sum(np.exp(Vin[j, n, 0])
                                                          for j in available_alt[n])
                Pin[i, n, 1] = np.exp(Vin[i, n, 1]) / sum(np.exp(Vin[j, n, 1])
                                                          for j in available_alt[n])
                Pin[i, n, 2] = np.exp(Vin[i, n, 2]) / sum(np.exp(Vin[j, n, 2])
                                                          for j in available_alt[n])
                Pin[i, n, 3] = np.exp(Vin[i, n, 3]) / sum(np.exp(Vin[j, n, 3])
                                                          for j in available_alt[n])

        biog_obj = sum(y[i, n] * np.log(pi[0] * Pin[i, n, 0] + pi[1] * Pin[i, n, 1] + pi[2] * Pin[i, n, 2]
                                        + (1 - pi[0] - pi[1] - pi[2]) * Pin[i, n, 3])
                       for n in range(N) for i in available_alt[n])
    return biog_obj
