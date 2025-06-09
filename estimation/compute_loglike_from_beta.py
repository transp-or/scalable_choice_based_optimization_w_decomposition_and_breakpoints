import numpy as np

import sys
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
import biogeme.results as res
from biogeme.exceptions import biogemeError
from biogeme.expressions import Beta, Variable, bioDraws, MonteCarlo


def compute_nb_of_successes(beta, x, y, epsilon):
    beta = [b for b in beta if b is not None]
    K = len(beta)
    N = len(y[0])
    J = len(y)

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    epsJ, epsN, R = epsilon.shape

    Uinr = dict()
    omega = dict()

    for n in range(N):
        for r in range(R):
            for i in range(J):
                Uinr[i, n, r] = sum(beta[k] * x[i, n, k] for k in range(K)) + epsilon[i, n, r]
                if not i == y_index[n]:
                    Uinr[i, n, r] = Uinr[i, n, r]
            max_util = np.max([Uinr[i, n, r] for i in range(J)])
            for i in range(J):
                if Uinr[i, n, r] == max_util:
                    omega[i, n, r] = 1
                else:
                    omega[i, n, r] = 0

    # compute objective value
    s_dict = dict()
    for n in range(N):
        for i in range(J):
            s_dict[i, n] = sum(omega[i, n, r] for r in range(R))

    nb_of_succs = sum(s_dict[y_index[n], n] for n in range(N))

    return nb_of_succs


def compute_simulated_expected_loss(beta, x, y, epsilon):
    beta = [b for b in beta if b is not None]
    K = len(beta)
    N = len(y[0])
    J = len(y)

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    epsJ, epsN, R = epsilon.shape

    Uinr = dict()
    d_nr = dict()

    for n in range(N):
        for r in range(R):
            for i in range(J):
                Uinr[i, n, r] = sum(beta[k] * x[i, n, k] for k in range(K)) + epsilon[i, n, r]
                if not i == y_index[n]:
                    Uinr[i, n, r] = Uinr[i, n, r]
            max_util = np.max([Uinr[i, n, r] for i in range(J)])
            d_nr[n, r] = max_util - Uinr[y_index[n], n, r]

    # compute loss
    loss = sum(d_nr[n, r] for n in range(N) for r in range(R))

    return loss / (N * R)


def compute_loglike(x, y, epsilon, beta, logOfZero, linFix=0):
    beta = [b for b in beta if b is not None]
    K = len(beta)
    N = len(y[0])
    J = len(y)

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    epsJ, epsN, R = epsilon.shape

    if linFix:
        sep = 1e-12
    else:
        sep = 0

    Uinr = dict()
    omega = dict()

    for n in range(N):
        for r in range(R):
            for i in range(J):
                Uinr[i, n, r] = sum(beta[k] * x[i, n, k] for k in range(K)) + epsilon[i, n, r]
                if not i == y_index[n]:
                    Uinr[i, n, r] = Uinr[i, n, r] - sep  # this makes sure that if (for lineMLE) GRB thinks it
                    # dominated an alternative, it actually does dominate it
            max_util = np.max([Uinr[i, n, r] for i in range(J)])
            for i in range(J):
                if Uinr[i, n, r] == max_util:
                    omega[i, n, r] = 1
                else:
                    omega[i, n, r] = 0

    # compute objective value
    s_dict = dict()
    z_dict = dict()
    obj_dict = dict()
    for n in range(N):
        for i in range(J):
            s_dict[i, n] = sum(omega[i, n, r] for r in range(R))
            if s_dict[i, n] > 0:
                z_dict[i, n] = np.log(s_dict[i, n])
            else:
                z_dict[i, n] = logOfZero
        obj_dict[n] = sum(-y[i, n] * z_dict[i, n] for i in range(J))
    total_obj = sum(obj_dict[n] for n in range(N)) + N * np.log(R)
    # print(f"For Beta = {beta} we got obj = {total_obj}")

    return -total_obj


def compute_loglike_nested(x, y, epsilon, beta, ex_epsilon, pu_epsilon):
    N = len(y[0])
    J = len(y)
    R = int(np.shape(epsilon)[1] / N)
    K = 4
    mu_ex = beta[4]
    mu_pu = beta[5]

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    Uinr = dict()
    omega = dict()

    for n in range(N):
        for r in range(R):
            Uinr[0, n, r] = sum(beta[k] * x[0, n, k] for k in range(K)) + epsilon[0, n * N + r] \
                                 + mu_ex * ex_epsilon[n * N + r] \
                                 + mu_pu * pu_epsilon[n * N + r]
            Uinr[1, n, r] = sum(beta[k] * x[1, n, k] for k in range(K)) + epsilon[1, n * N + r] \
                            + mu_pu * pu_epsilon[n * N + r]
            Uinr[2, n, r] = sum(beta[k] * x[2, n, k] for k in range(K)) + epsilon[2, n * N + r] \
                            + mu_ex * ex_epsilon[n * N + r]

            max_util = np.max([Uinr[i, n, r] for i in range(J)])
            for i in range(J):
                if Uinr[i, n, r] == max_util:
                    omega[i, n, r] = 1
                else:
                    omega[i, n, r] = 0

    # compute objective value
    s_dict = dict()
    z_dict = dict()
    obj_dict = dict()
    for n in range(N):
        for i in range(J):
            s_dict[i, n] = sum(omega[i, n, r] for r in range(R))
            if s_dict[i, n] > 0:
                z_dict[i, n] = np.log(s_dict[i, n])
            else:
                z_dict[i, n] = -100
        obj_dict[n] = sum(-y[i, n] * z_dict[i, n] for i in range(J))
    total_obj = sum(obj_dict[n] for n in range(N)) + N * np.log(R)

    return -total_obj


def compute_recreation_rates_mixed(x, y, mixed_index, av, epsilon, normal_epsilon, beta, logOfZero, R=None):
    beta = [b for b in beta if b is not None]
    K = len(beta) - 1
    N = len(y[0])
    J = len(y)

    if R is not None:
        epsilon = np.random.gumbel(loc=0, scale=1, size=(J, N, R))
        normal_epsilon = np.random.normal(loc=0, scale=1, size=(N, R))

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    not_mixed_params = [k for k in range(K) if not (k == mixed_index or k == mixed_index + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    R = int(epsilon.shape[1] / N)

    Uinr = dict()
    recr = dict()
    alt_0 = dict()
    alt_1 = dict()
    alt_2 = dict()

    for n in range(N):
        for r in range(R):
            for i in av_alts[n]:
                Uinr[i, n, r] = sum(beta[k] * x[i, n, k] for k in not_mixed_params) \
                                + beta[mixed_index] * x[i, n, mixed_index] \
                                + beta[mixed_index + 1] * x[i, n, mixed_index] * normal_epsilon[n * N + r] \
                                + epsilon[i, n * N + r]
            max_util = np.max([Uinr[i, n, r] for i in av_alts[n]])
            if Uinr[y_index[n], n, r] == max_util:
                recr[n, r] = 1
            else:
                recr[n, r] = 0
            if Uinr[0, n, r] == max_util:
                alt_0[n, r] = 1
            else:
                alt_0[n, r] = 0
            if Uinr[1, n, r] == max_util:
                alt_1[n, r] = 1
            else:
                alt_1[n, r] = 0
            if 2 in av_alts[n]:
                if Uinr[2, n, r] == max_util:
                    alt_2[n, r] = 1
                else:
                    alt_2[n, r] = 0

    # compute average recreation rate per individual
    ind_rate = {n: (1/R) * sum(recr[n, r] for r in range(R)) for n in range(N)}
    ind_prob_0 = {n: (1 / R) * sum(alt_0[n, r] for r in range(R)) for n in range(N)}
    ind_prob_1 = {n: (1 / R) * sum(alt_1[n, r] for r in range(R)) for n in range(N)}
    ind_prob_2 = {n: (1 / R) * sum(alt_2[n, r] for r in range(R)) for n in [n for n in range(N) if 2 in av_alts[n]]}
    ind_prob_rel_diff = dict()
    for n in range(N):
        if y_index[n] == 0:
            diff_1 = ind_prob_1[n] - ind_prob_0[n]
            if 2 in av_alts[n]:
                diff_2 = ind_prob_2[n] - ind_prob_0[n]
            else:
                diff_2 = -222222
            diff = max(diff_1, diff_2, 0)
            rel_diff = diff / ind_prob_0[n]
            ind_prob_rel_diff[n] = rel_diff
        if y_index[n] == 1:
            diff_0 = ind_prob_0[n] - ind_prob_1[n]
            if 2 in av_alts[n]:
                diff_2 = ind_prob_2[n] - ind_prob_1[n]
            else:
                diff_2 = -22222
            diff = max(diff_0, diff_2, 0)
            rel_diff = diff / ind_prob_1[n]
            ind_prob_rel_diff[n] = rel_diff
        if 2 in av_alts[n]:
            if y_index[n] == 2:
                diff_0 = ind_prob_0[n] - ind_prob_2[n]
                diff_1 = ind_prob_1[n] - ind_prob_2[n]
                diff = max(diff_0, diff_1, 0)
                rel_diff = diff / ind_prob_2[n]
                ind_prob_rel_diff[n] = rel_diff

    avg_rel_prob_diff = (1 / N) * sum(ind_prob_rel_diff[n] for n in range(N))
    avg_prob_0 = (1 / (N * R)) * sum(alt_0[n, r] for r in range(R) for n in range(N))
    avg_prob_1 = (1 / (N * R)) * sum(alt_1[n, r] for r in range(R) for n in range(N))
    avg_prob_2 = (1 / (N * R)) * sum(alt_2[n, r] for r in range(R) for n in [n for n in range(N) if 2 in av_alts[n]])

    probs = [avg_prob_0, avg_prob_1, avg_prob_2]

    min_ind_rate = min(ind_rate.values())
    max_ind_rate = max(ind_rate.values())
    median_ind_rate = np.median(list(ind_rate.values()))

    average_ind_rate = (1 / (N * R)) * sum(recr[n, r] for r in range(R) for n in range(N))

    return min_ind_rate, max_ind_rate, median_ind_rate, average_ind_rate, probs, avg_rel_prob_diff


def compute_sEL_mixed(x, y, mixed_index, av, epsilon, normal_epsilon, beta, logOfZero, R=None):
    beta = [b for b in beta if b is not None]
    K = len(beta) - 1
    N = len(y[0])
    J = len(y)
    R = int(epsilon.shape[1] / N)

    if R is not None:
        epsilon = np.random.gumbel(loc=0, scale=1, size=(J, N * R))
        normal_epsilon = np.random.normal(loc=0, scale=1, size=(N * R))

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    not_mixed_params = [k for k in range(K) if not (k == mixed_index or k == mixed_index + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    Uinr = dict()
    d_nr = dict()

    for n in range(N):
        for r in range(R):
            for i in av_alts[n]:
                Uinr[i, n, r] = sum(beta[k] * x[i, n, k] for k in not_mixed_params) \
                                + beta[mixed_index] * x[i, n, mixed_index] \
                                + beta[mixed_index + 1] * x[i, n, mixed_index] * normal_epsilon[n * N + r] \
                                + epsilon[i, n * N + r]
            max_util = np.max([Uinr[i, n, r] for i in av_alts[n]])
            d_nr[n, r] = max_util - Uinr[y_index[n], n, r]

    # compute loss
    loss = sum(d_nr[n, r] for n in range(N) for r in range(R))

    return loss / R


def compute_nb_of_correct_minima(x, y, mixed_index, av, epsilon, normal_epsilon, beta, dnr_sol, R=None):
    beta = [b for b in beta if b is not None]
    K = len(beta) - 1
    N = len(y[0])
    J = len(y)

    if R is not None:
        epsilon = np.random.gumbel(loc=0, scale=1, size=(J, N, R))
        normal_epsilon = np.random.normal(loc=0, scale=1, size=(N, R))

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    not_mixed_params = [k for k in range(K) if not (k == mixed_index or k == mixed_index + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    R = int(epsilon.shape[1] / N)

    Uinr = dict()
    minima_diff = dict()

    minima_counter = 0
    for n in range(N):
        for r in range(R):
            for i in av_alts[n]:
                Uinr[i, n, r] = sum(beta[k] * x[i, n, k] for k in not_mixed_params) \
                                + beta[mixed_index] * x[i, n, mixed_index] \
                                + beta[mixed_index + 1] * x[i, n, mixed_index] * normal_epsilon[n * N + r] \
                                + epsilon[i, n * N + r]
            max_util = np.max([Uinr[i, n, r] for i in av_alts[n]])
            minima_diff[n, r] = dnr_sol[n, r] - (max_util - Uinr[y_index[n], n, r])
            if minima_diff[n, r] <= 1e-7:
                minima_counter += 1

    return minima_counter, minima_diff


def compute_sLL_mixed(x, y, mixed_index, av, epsilon, normal_epsilon, beta, logOfZero, R=None):
    beta = [b for b in beta if b is not None]
    K = len(beta) - 1
    N = len(y[0])
    J = len(y)
    epsJ, epsNR = epsilon.shape
    R = int(epsNR / N)

    if R is not None:
        epsilon = np.random.gumbel(loc=0, scale=1, size=(J, N * R))

        # multi = np.random.multivariate_normal(mean=np.zeros(N),
        #                                       cov=np.eye(N),
        #                                       size=(J, R))
        # normal_epsilon = np.empty(shape=(J, N, R))
        # for i in range(J):
        #     for n in range(N):
        #         for r in range(R):
        #             normal_epsilon[i, n, r] = multi[i, r, n]

        normal_epsilon = np.random.normal(loc=0, scale=1, size=N * R)

        # multi = np.random.multivariate_normal(mean=np.zeros(J),
        #                                       cov=np.eye(J),
        #                                       size=(N, R))
        # # numpy says: Because each sample is N-dimensional, the output shape is (N, R, J)
        # normal_epsilon = np.empty(shape=(J, N, R))
        # for i in range(J):
        #     for n in range(N):
        #         for r in range(R):
        #             normal_epsilon[i, n, r] = multi[n, r, i]

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    not_mixed_params = [k for k in range(K) if not (k == mixed_index or k == mixed_index + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    Uinr = dict()
    omega = dict()

    for n in range(N):
        for r in range(R):
            for i in av_alts[n]:
                Uinr[i, n, r] = sum(beta[k] * x[i, n, k] for k in not_mixed_params) \
                                + beta[mixed_index] * x[i, n, mixed_index] \
                                + beta[mixed_index + 1] * x[i, n, mixed_index + 1] * normal_epsilon[n * N + r] \
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


def compute_sLL_latent(x, y, av, latent, latent_indices, epsilon, beta, logOfZero, linFix=1):
    if latent == 2:
        pi = beta[-1]
        beta = beta[0:len(beta) - 1]
    if latent == 3:
        pi = beta[-2:]
        beta = beta[0:len(beta) - 2]
    if latent == 4:
        pi = beta[-3:]
        beta = beta[0:len(beta) - 3]
    K = len(beta)
    N = len(y[0])
    J = len(y)

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    epsJ, epsN, R = epsilon.shape

    if linFix:
        sep = 1e-12
    else:
        sep = 0

    Uinr = dict()
    omega = dict()

    nonlatent = [k for k in range(K) if k not in latent_indices]
    available_alt = dict()
    for n in range(N):
        available_alt[n] = [i for i in range(J) if av[i][n] == 1]

    if latent == 2:
        for n in range(N):
            for r in range(R):
                for i in available_alt[n]:
                    Uinr[i, n, r, 0] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                                       + beta[latent_indices[0]] * x[i, n, latent_indices[0]] \
                                       + epsilon[i, n, r]

                    Uinr[i, n, r, 1] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                                       + epsilon[i, n, r]

                    if not i == y_index[n]:
                        Uinr[i, n, r, 0] = Uinr[i, n, r, 0] - sep  # this makes sure that if (for lineMLE) GRB thinks it
                        # dominated an alternative, it actually does dominate it
                        Uinr[i, n, r, 1] = Uinr[i, n, r, 1] - sep
                max_util_0 = np.max([Uinr[i, n, r, 0] for i in available_alt[n]])
                max_util_1 = np.max([Uinr[i, n, r, 1] for i in available_alt[n]])
                for i in [i for i in range(J) if av[i][n] == 1]:
                    if Uinr[i, n, r, 0] == max_util_0:
                        omega[i, n, r, 0] = 1
                    else:
                        omega[i, n, r, 0] = 0
                    if Uinr[i, n, r, 1] == max_util_1:
                        omega[i, n, r, 1] = 1
                    else:
                        omega[i, n, r, 1] = 0

        # compute objective value
        s_dict = dict()
        z_dict = dict()
        obj_dict = dict()
        for n in range(N):
            for i in available_alt[n]:
                s_dict[i, n] = sum(pi * omega[i, n, r, 0] + (1 - pi) * omega[i, n, r, 1] for r in range(R))
                if s_dict[i, n] > 0:
                    z_dict[i, n] = np.log(s_dict[i, n])
                else:
                    z_dict[i, n] = logOfZero
            obj_dict[n] = sum(-y[i, n] * z_dict[i, n] for i in available_alt[n])
        total_obj = sum(obj_dict[n] for n in range(N)) + N * np.log(R)
    if latent == 3:
        for n in range(N):
            for r in range(R):
                for i in available_alt[n]:
                    Uinr[i, n, r, 0] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                                       + beta[latent_indices[0]] * x[i, n, latent_indices[0]] \
                                       + epsilon[i, n, r]

                    Uinr[i, n, r, 1] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                                       + beta[latent_indices[1]] * x[i, n, latent_indices[1]] \
                                       + epsilon[i, n, r]

                    Uinr[i, n, r, 2] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                                       + beta[latent_indices[0]] * x[i, n, latent_indices[0]] \
                                       + beta[latent_indices[1]] * x[i, n, latent_indices[1]] \
                                       + epsilon[i, n, r]

                    if not i == y_index[n]:
                        Uinr[i, n, r, 0] = Uinr[i, n, r, 0] - sep  # this makes sure that if (for lineMLE) GRB thinks it
                        # dominated an alternative, it actually does dominate it
                        Uinr[i, n, r, 1] = Uinr[i, n, r, 1] - sep
                        Uinr[i, n, r, 2] = Uinr[i, n, r, 2] - sep
                max_util_0 = np.max([Uinr[i, n, r, 0] for i in available_alt[n]])
                max_util_1 = np.max([Uinr[i, n, r, 1] for i in available_alt[n]])
                max_util_2 = np.max([Uinr[i, n, r, 2] for i in available_alt[n]])
                for i in [i for i in range(J) if av[i][n] == 1]:
                    if Uinr[i, n, r, 0] == max_util_0:
                        omega[i, n, r, 0] = 1
                    else:
                        omega[i, n, r, 0] = 0
                    if Uinr[i, n, r, 1] == max_util_1:
                        omega[i, n, r, 1] = 1
                    else:
                        omega[i, n, r, 1] = 0
                    if Uinr[i, n, r, 2] == max_util_2:
                        omega[i, n, r, 2] = 1
                    else:
                        omega[i, n, r, 2] = 0

        # compute objective value
        s_dict = dict()
        z_dict = dict()
        obj_dict = dict()
        for n in range(N):
            for i in available_alt[n]:
                s_dict[i, n] = sum(pi[0] * omega[i, n, r, 0] + pi[1] * omega[i, n, r, 1]
                                   + (1 - pi[0] - pi[1]) * omega[i, n, r, 2] for r in range(R))
                if s_dict[i, n] > 0:
                    z_dict[i, n] = np.log(s_dict[i, n])
                else:
                    z_dict[i, n] = logOfZero
            obj_dict[n] = sum(-y[i, n] * z_dict[i, n] for i in available_alt[n])
        total_obj = sum(obj_dict[n] for n in range(N)) + N * np.log(R)
    if latent == 4:
        for n in range(N):
            for r in range(R):
                for i in available_alt[n]:
                    Uinr[i, n, r, 0] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                                       + beta[latent_indices[0]] * x[i, n, latent_indices[0]] \
                                       + epsilon[i, n, r]

                    Uinr[i, n, r, 1] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                                       + beta[latent_indices[1]] * x[i, n, latent_indices[1]] \
                                       + epsilon[i, n, r]

                    Uinr[i, n, r, 2] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                                       + beta[latent_indices[0]] * x[i, n, latent_indices[0]] \
                                       + beta[latent_indices[1]] * x[i, n, latent_indices[1]] \
                                       + epsilon[i, n, r]
                    Uinr[i, n, r, 3] = sum(beta[k] * x[i, n, k] for k in nonlatent) \
                                       + epsilon[i, n, r]

                    if not i == y_index[n]:
                        Uinr[i, n, r, 0] = Uinr[i, n, r, 0] - sep  # this makes sure that if (for lineMLE) GRB thinks it
                        # dominated an alternative, it actually does dominate it
                        Uinr[i, n, r, 1] = Uinr[i, n, r, 1] - sep
                        Uinr[i, n, r, 2] = Uinr[i, n, r, 2] - sep
                        Uinr[i, n, r, 3] = Uinr[i, n, r, 3] - sep
                max_util_0 = np.max([Uinr[i, n, r, 0] for i in available_alt[n]])
                max_util_1 = np.max([Uinr[i, n, r, 1] for i in available_alt[n]])
                max_util_2 = np.max([Uinr[i, n, r, 2] for i in available_alt[n]])
                max_util_3 = np.max([Uinr[i, n, r, 3] for i in available_alt[n]])
                for i in [i for i in range(J) if av[i][n] == 1]:
                    if Uinr[i, n, r, 0] == max_util_0:
                        omega[i, n, r, 0] = 1
                    else:
                        omega[i, n, r, 0] = 0
                    if Uinr[i, n, r, 1] == max_util_1:
                        omega[i, n, r, 1] = 1
                    else:
                        omega[i, n, r, 1] = 0
                    if Uinr[i, n, r, 2] == max_util_2:
                        omega[i, n, r, 2] = 1
                    else:
                        omega[i, n, r, 2] = 0
                    if Uinr[i, n, r, 3] == max_util_3:
                        omega[i, n, r, 3] = 1
                    else:
                        omega[i, n, r, 3] = 0

        # compute objective value
        s_dict = dict()
        z_dict = dict()
        obj_dict = dict()
        for n in range(N):
            for i in available_alt[n]:
                s_dict[i, n] = sum(pi[0] * omega[i, n, r, 0] + pi[1] * omega[i, n, r, 1] + pi[2] * omega[i, n, r, 2]
                                   + (1 - pi[0] - pi[1] - pi[2]) * omega[i, n, r, 3] for r in range(R))
                if s_dict[i, n] > 0:
                    z_dict[i, n] = np.log(s_dict[i, n])
                else:
                    z_dict[i, n] = logOfZero
            obj_dict[n] = sum(-y[i, n] * z_dict[i, n] for i in available_alt[n])
        total_obj = sum(obj_dict[n] for n in range(N)) + N * np.log(R)
    # print(f"For Beta = {beta} we got obj = {total_obj}")

    return -total_obj


def compute_loglike_ind(x, y, epsilon, beta, logOfZero):
    beta = [b for b in beta if b is not None]
    K = len(beta)
    N = len(y[0])
    J = int(len(x.keys()) / (N * K))

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    epsJ, epsN, R = epsilon.shape

    Uinr = dict()
    omega = dict()

    for n in range(N):
        for r in range(R):
            for i in range(J):
                Uinr[i, n, r] = sum(beta[k] * x[i, n, k] for k in range(K)) + epsilon[i, n, r]
                if not i == y_index[n]:
                    Uinr[i, n, r] = Uinr[i, n, r] - 1e-12  # this makes sure that if GRB thinks it dominated an
                    # alternative, it actually does dominate it
            max_util = np.max([Uinr[i, n, r] for i in range(J)])
            for i in range(J):
                if Uinr[i, n, r] == max_util:
                    omega[i, n, r] = 1
                else:
                    omega[i, n, r] = 0

    # compute objective value
    s_dict = dict()
    z_dict = dict()
    obj_dict = dict()
    for n in range(N):
        for i in range(J):
            s_dict[i, n] = sum(omega[i, n, r] for r in range(R))
            if s_dict[i, n] > 0:
                z_dict[i, n] = np.log(s_dict[i, n])
            else:
                z_dict[i, n] = logOfZero
        obj_dict[n] = sum(-y[i, n] * z_dict[i, n] for i in range(J))
    return obj_dict
