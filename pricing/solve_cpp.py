import numpy as np
import sys
import time
import h5py
import subprocess
import os
import re
import copy
import warnings
import contextlib
from scipy.stats import norm, gumbel_r, multivariate_normal
from scipy.integrate import quad, dblquad, nquad
from cubature import cubature

import solve_full_MILP_bigM_cpp as fullMILP_bigM
import solve_full_MILP_BnB_cpp as fullMILP_BnB
from data_parking import get_input_data_parking, get_input_data_parking_logit

N = int(sys.argv[1])  # IF N = 10 we're using the 10 customer classes from the CoBiT paper
R = int(sys.argv[2])
J_PSP = int(sys.argv[3])
J_PUP = int(sys.argv[4])
onep_input = (J_PSP == 0) and (J_PUP == 1)
timelim = int(sys.argv[5])

if timelim == 0:
    timelim = 3600
meth_input = [int(el) for el in sys.argv[6].split(",")]

pres = 0
try:
    pl_VIs = bool(int(sys.argv[7]))
except IndexError:
    pl_VIs = False

try:
    breakbounds = bool(int(sys.argv[8]))
except IndexError:
    breakbounds = False

try:
    do_vis = bool(int(sys.argv[9]))
except IndexError:
    do_vis = False

try:
    directrun = int(sys.argv[10])
except IndexError:
    directrun = False

try:
    minutes = int(sys.argv[11])
except IndexError:
    minutes = 0

try:
    addeta = int(sys.argv[12])  # 0, 1 or 4
except IndexError:
    addeta = 0

try:
    optvalid = int(sys.argv[13])  # 0 or 2
except IndexError:
    optvalid = 0

try:
    mcm2 = bool(int(sys.argv[14]))
except IndexError:
    mcm2 = False

try:
    timetricks = bool(int(sys.argv[15]))
except IndexError:
    timetricks = False

guided_branch = False
forced_improvement = False
threads = 1

fixed_choices = 1
cheapest_alternative = None  # realistically we can't make any assumptions on that a priori
enum = 0  # 0 means normal, 1 means inverse, 2 means opt-outs enforced
heur = 1
perfect_start = 0


def monte_carlo_revenue(p, num_samples=1000000):
    np.random.seed(0)  # For reproducibility
    covariance_fee_AT = -12.8
    μ = np.array([-0.788, -32.328])  # Mean of the random variable
    Σ = np.array([[1.064 ** 2, covariance_fee_AT], [covariance_fee_AT, 14.168 ** 2]])  # Covariance matrix of the random variable

    # Stratified sampling
    samples = np.random.multivariate_normal(μ, Σ, num_samples)
    total_revenue = 0

    for β in samples:
        Beta_parameter, q_parameter, NUM_POINTS, N = Mixed_Logit_10(β)
        for n in range(N):
            utilities = np.exp(Beta_parameter[:, n] * p + q_parameter[:, n])
            probabilities = utilities / utilities.sum()
            for i in range(NUM_POINTS):
                total_revenue += probabilities[i] * p[i]

    expected_revenue = total_revenue / num_samples
    return expected_revenue


def compute_cpp_from_p_parking_HighR_probs(J, N, Rbig, p, J_PSP, J_PUP):
    # we did 100k and 1m
    R_obj = Rbig

    parking_data = get_input_data_parking(N, R_obj, J_PSP, J_PUP, False, minutes)
    J = parking_data["J_tot"]
    # reshape the inputs so that in a way that Julia needs it
    exo_utility = parking_data['exo_utility'].reshape((J, N * R_obj))
    endo_coef = parking_data['endo_coef'].reshape((J, N * R_obj))

    # loaded_data = np.load(f'compObj{R_obj}.npz')
    # exo_utility = loaded_data['exo_utility']
    # endo_coef = loaded_data['endo_coef']

    p.append(0)  # this is for tie breaks
    obj = 0
    for n in range(N):
        for r in range(R_obj):
            util_list = [exo_utility[0, n * R_obj + r]]
            for i in range(J_PSP):
                util_list.append(exo_utility[i + 1, n * R_obj + r] + endo_coef[i + 1, n * R_obj + r] * p[i])
            for i in range(J_PUP):
                util_list.append(exo_utility[i + J_PSP + 1, n * R_obj + r]
                                 + endo_coef[i + J_PSP + 1, n * R_obj + r] * p[i + J_PSP])

            exp_utils = np.exp(util_list)
            probabilities = exp_utils / exp_utils.sum()

            for i in range(J-1):
                obj += (1 / R_obj) * probabilities[i + 1] * p[i]
    del p[-1]
    return obj


def Mixed_Logit_10(β):
    NUM_POINTS = 3  # alternatives
    N = 10  # customers

    # Parameters choice model
    ASC_PSP = 32
    ASC_PUP = 34
    Beta_TD = -0.612
    Beta_Origin = -5.762
    Beta_Age_Veh = 4.037
    Beta_FEE_INC_PSP = -10.995
    Beta_FEE_RES_PSP = -11.440
    Beta_FEE_INC_PUP = -13.729
    Beta_FEE_RES_PUP = -10.668

    # Variables choice model
    AT_FSP = 10
    TD_FSP = 10
    AT_PSP = 10
    TD_PSP = 10
    AT_PUP = 5
    TD_PUP = 10
    Origin = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0])
    Age_veh = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    Low_inc = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    Res = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1])

    q_parameter = np.ones((NUM_POINTS, N))
    for n in range(N):
        q_parameter[0, n] = β[0] * AT_FSP + Beta_TD * TD_FSP + Beta_Origin * Origin[n]
        q_parameter[1, n] = ASC_PSP + β[0] * AT_PSP + Beta_TD * TD_PSP
        q_parameter[2, n] = ASC_PUP + β[0] * AT_PUP + Beta_TD * TD_PUP + Beta_Age_Veh * Age_veh[n]

    Beta_parameter = np.ones((NUM_POINTS, N))
    for n in range(N):
        Beta_parameter[0, n] = 0
        Beta_parameter[1, n] = β[1] + Beta_FEE_INC_PSP * Low_inc[n] + Beta_FEE_RES_PSP * Res[n]
        Beta_parameter[2, n] = β[1] + Beta_FEE_INC_PUP * Low_inc[n] + Beta_FEE_RES_PUP * Res[n]

    return Beta_parameter, q_parameter, NUM_POINTS, N

def Mixed_Logit_distribution(p, β, i):
    Beta_parameter, q_parameter, NUM_POINTS, N = Mixed_Logit_10(β)
    μ = np.array([-0.788, -32.328])  # mean of the random variable
    Σ = np.array([[1.064**2, -12.8], [-12.8, 14.168**2]])  # covariance matrix of the random variable

    f = np.zeros(N)
    for n in range(N):
        denominator = sum(np.exp(Beta_parameter[j, n] * p[j] + q_parameter[j, n]) for j in range(NUM_POINTS))
        if denominator == 0:
            print(f"Warning: Zero denominator encountered for customer {n}")
            denominator = 1e-10
        f[n] = np.exp(Beta_parameter[i, n] * p[i] + q_parameter[i, n]) / denominator

    pdf_value = multivariate_normal.pdf(β, mean=μ, cov=Σ)
    if pdf_value == 0:
        print(f"Warning: Zero PDF value encountered for β {β}")
        pdf_value = 1e-10

    return np.sum(f) * pdf_value

def Mixed_logit_function_i(p, i):
    a = np.array([-5.992703338, -101.6327339])  # lower bounds on the integral
    b = np.array([4.416703338, 36.97673392])  # upper bounds on the integral

    def integrand(x):
        β = a + x * (b - a)  # Scale x to the correct bounds
        result = Mixed_Logit_distribution(p, β, i)
        return result

    result, error = cubature(integrand, 2, 1, np.zeros(2), np.ones(2), vectorized=False)
    # print(f"Integration result: {result}, error: {error}")
    return 1 / (np.prod(b - a) * result[0])  # Apply scaling similar to Julia

def Mixed_logit_function(p):
    NUM_POINTS = 3
    if len(p) != NUM_POINTS:
        raise ValueError(f"Prices vector length {len(p)} does not match NUM_POINTS {NUM_POINTS}")
    results = []
    for i in range(NUM_POINTS):
        try:
            # print(f"Calculating Mixed_logit_function_i for point {i}")
            result = Mixed_logit_function_i(p, i)
            # print(f"Result for point {i}: {result}")
            results.append(result)
        except Exception as e:
            print(f"Error calculating Mixed_logit_function_i for point {i}: {e}")
            results.append(np.inf)  # To avoid division by zero or other issues
    return np.sum([p[i] / results[i] for i in range(NUM_POINTS)])

def compute_expected_revenue_simulated(p, num_simulations=100000):
    datadict = dict()
    # Alternative Specific Coefficients
    datadict['ASC_FSP'] = 0.0  # Free Street Parking
    datadict['ASC_PSP'] = 32.0  # Paid Street Parking
    datadict['ASC_PUP'] = 34.0  # Paid Underground Parking
    # Beta coefficients
    datadict['Beta_TD'] = -0.612
    datadict['Beta_Origin'] = -5.762
    datadict['Beta_Age_Veh'] = 4.037
    datadict['Beta_FEE_INC_PSP'] = -10.995
    datadict['Beta_FEE_RES_PSP'] = -11.440
    datadict['Beta_FEE_INC_PUP'] = -13.729
    datadict['Beta_FEE_RES_PUP'] = -10.668
    # Access times to parking (AT)
    datadict['AT_FSP'] = 10
    datadict['AT_PSP'] = 10
    datadict['AT_PUP'] = 5
    # Access time to final destination from the parking space (TD)
    datadict['TD_FSP'] = 10
    datadict['TD_PSP'] = 10
    datadict['TD_PUP'] = 10
    # Number of PSP alternatives in the choice set
    datadict['J_PSP'] = 1
    # Number of PUP alternatives in the choice set
    datadict['J_PUP'] = 1
    # Number of opt-out alternatives in the choice set
    datadict['J_opt_out'] = 1
    # Size of the universal choice set
    datadict['J_tot'] = datadict['J_PSP'] + datadict['J_PUP'] + datadict['J_opt_out']

    datadict['Pop'] = 10
    datadict['Origin'] = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0])
    datadict['Age_veh'] = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    datadict['Low_inc'] = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    datadict['Res'] = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1])

    N = datadict['Pop']
    J = datadict['J_tot']

    p.append(0)  # this is for tie breaks
    total_revenue = 0

    for _ in range(num_simulations):
        Beta_AT = np.random.normal(-0.788, 1.064, N)
        Beta_Fee = np.random.normal(-32.328, 14.168, N)
        xi = np.random.gumbel(size=(J, N))

        obj = 0
        for n in range(N):
            util_list = [(Beta_AT[n] * datadict['AT_FSP'] +
                         datadict['Beta_TD'] * datadict['TD_FSP'] +
                         datadict['Beta_Origin'] * datadict['Origin'][n] +
                         xi[0, n])]

            for i in range(datadict['J_PSP']):
                util_list.append((datadict['ASC_PSP'] +
                                  Beta_AT[n] * datadict['AT_PSP'] +
                                  datadict['Beta_TD'] * datadict['TD_PSP'] + xi[1, n] +
                                  (Beta_Fee[n]
                                   + datadict['Beta_FEE_INC_PSP'] * datadict['Low_inc'][n]
                                   + datadict['Beta_FEE_RES_PSP'] * datadict['Res'][n])
                                  * p[i]))
            for i in range(datadict['J_PUP']):
                util_list.append((datadict['ASC_PUP'] +
                                  Beta_AT[n] * datadict['AT_PUP'] +
                                  datadict['Beta_TD'] * datadict['TD_PUP'] +
                                  datadict['Beta_Age_Veh'] * datadict['Age_veh'][n] +
                                  xi[datadict['J_PSP'] + 1, n] +
                                  (Beta_Fee[n]
                                   + datadict['Beta_FEE_INC_PUP'] * datadict['Low_inc'][n]
                                   + datadict['Beta_FEE_RES_PUP'] * datadict['Res'][n])
                                  * p[i + datadict['J_PSP']]))

            # get best alternative, with tie-breaks based on price values
            max_value = max(util_list)
            max_util = util_list.index(max_value)

            if 1 <= max_util:
                obj += p[max_util - 1]

        total_revenue += obj

    del p[-1]
    expected_revenue = total_revenue / num_simulations
    return expected_revenue


def compute_cpp_from_p_parking_continuous(p):
    datadict = dict()
    # Alternative Specific Coefficients
    datadict['ASC_FSP'] = 0.0  # Free Street Parking
    datadict['ASC_PSP'] = 32.0  # Paid Street Parking
    datadict['ASC_PUP'] = 34.0  # Paid Underground Parking
    # Beta coefficients
    datadict['Beta_TD'] = -0.612
    datadict['Beta_Origin'] = -5.762
    datadict['Beta_Age_Veh'] = 4.037
    datadict['Beta_FEE_INC_PSP'] = -10.995
    datadict['Beta_FEE_RES_PSP'] = -11.440
    datadict['Beta_FEE_INC_PUP'] = -13.729
    datadict['Beta_FEE_RES_PUP'] = -10.668
    # Access times to parking (AT)
    datadict['AT_FSP'] = 10
    datadict['AT_PSP'] = 10
    datadict['AT_PUP'] = 5
    # Access time to final destination from the parking space (TD)
    datadict['TD_FSP'] = 10
    datadict['TD_PSP'] = 10
    datadict['TD_PUP'] = 10
    # Number of PSP alternatives in the choice set
    datadict['J_PSP'] = 1
    # Number of PUP alternatives in the choice set
    datadict['J_PUP'] = 1
    # Number of opt-out alternatives in the choice set
    datadict['J_opt_out'] = 1
    # Size of the universal choice set
    datadict['J_tot'] = datadict['J_PSP'] + datadict['J_PUP'] + datadict['J_opt_out']

    datadict['Pop'] = 10
    datadict['Origin'] = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0])
    datadict['Age_veh'] = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    datadict['Low_inc'] = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    datadict['Res'] = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1])

    N = datadict['Pop']
    J = datadict['J_tot']

    datadict['Beta_AT'] = [-0.788] * N
    datadict['Beta_Fee'] = [-32.328] * N
    datadict['xi'] = np.zeros(shape=(J, N))

    p.append(0)  # this is for tie breaks
    revenue = 0
    for n in range(N):
        util_list = [(datadict['Beta_AT'][n] * datadict['AT_FSP'] +
                     datadict['Beta_TD'] * datadict['TD_FSP'] +
                     datadict['Beta_Origin'] * datadict['Origin'][n] +
                     datadict["xi"][0, n])]

        for i in range(J_PSP):
            util_list.append((datadict['ASC_PSP'] +
                              datadict['Beta_AT'][n] * datadict['AT_PSP'] +
                              datadict['Beta_TD'] * datadict['TD_PSP'] + datadict["xi"][1, n] +
                              (datadict['Beta_Fee'][n]
                               + datadict['Beta_FEE_INC_PSP'] * datadict['Low_inc'][n]
                               + datadict['Beta_FEE_RES_PSP'] * datadict['Res'][n])
                              * p[i]))
        for i in range(J_PUP):
            util_list.append((datadict['ASC_PUP'] +
                              datadict['Beta_AT'][n] * datadict['AT_PUP'] +
                              datadict['Beta_TD'] * datadict['TD_PUP'] +
                              datadict['Beta_Age_Veh'] * datadict['Age_veh'][n] +
                              datadict["xi"][J_PSP + 1, n] +
                              (datadict['Beta_Fee'][n]
                               + datadict['Beta_FEE_INC_PUP'] * datadict['Low_inc'][n]
                               + datadict['Beta_FEE_RES_PUP'] * datadict['Res'][n])
                              * p[i + J_PSP]))

        # get best alternative, with tie-breaks based on price values
        max_value = max(util_list)
        max_util = util_list.index(max_value)

        if 1 <= max_util:
            revenue += p[max_util - 1]
    del p[-1]
    return revenue


def compute_cpp_from_p_parking_HighR(J, N, Rbig, p, J_PSP, J_PUP):
    # we did 100k and 1m
    R_obj = Rbig

    parking_data = get_input_data_parking(N, R_obj, J_PSP, J_PUP, False, minutes)
    J = parking_data["J_tot"]
    # reshape the inputs so that in a way that Julia needs it
    exo_utility = parking_data['exo_utility'].reshape((J, N * R_obj))
    endo_coef = parking_data['endo_coef'].reshape((J, N * R_obj))

    p.append(0)  # this is for tie breaks
    obj = 0
    for n in range(N):
        for r in range(R_obj):
            util_list = [exo_utility[0, n * R_obj + r]]
            for i in range(J_PSP):
                util_list.append(exo_utility[i + 1, n * R_obj + r] + endo_coef[i + 1, n * R_obj + r] * p[i])
            for i in range(J_PUP):
                util_list.append(exo_utility[i + J_PSP + 1, n * R_obj + r]
                                 + endo_coef[i + J_PSP + 1, n * R_obj + r] * p[i + J_PSP])

            # get best alternative, with tie-breaks based on price values
            max_value = max(util_list)
            candidate_indices = [i for i, val in enumerate(util_list) if abs(val - max_value) < 1e-9]
            max_util = max(candidate_indices, key=lambda idx: p[idx - 1])
            if len(candidate_indices) > 1:
                print(f"n = {n}, r = {r} had best candidates = {candidate_indices}")
                print(f"prices are {p}")
                print(f"thus the boy chose {max_util}")

            if 1 <= max_util:
                obj -= (1 / R_obj) * p[max_util - 1]
    del p[-1]
    return -obj


def compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef, p, onep, J_PSP, J_PUP):
    if onep:
        p = p[0]
        obj = 0
        omega = np.zeros((N * R))
        eta = np.zeros((N * R))
        for n in range(N):
            for r in range(R):
                max_util = int(
                    np.argmax([exo_utility[0, n * R + r], exo_utility[1, n * R + r] + endo_coef[n * R + r] * p]))
                if max_util == 1:
                    obj -= (1 / R) * p
                    omega[n * R + r] = 1
                    eta[n * R + r] = p
    else:
        p.append(0)  # this is for tie breaks
        obj = 0
        omega = np.zeros((J, N * R))
        eta = {(i, n * R + r): 0 for i in range(1, J) for n in range(N) for r in range(R)}
        for n in range(N):
            for r in range(R):
                util_list = [exo_utility[0, n * R + r]]
                for i in range(J_PSP):
                    util_list.append(exo_utility[i + 1, n * R + r] + endo_coef[i + 1, n * R + r] * p[i])
                for i in range(J_PUP):
                    util_list.append(exo_utility[i + J_PSP + 1, n * R + r]
                                     + endo_coef[i + J_PSP + 1, n * R + r] * p[i + J_PSP])

                # get best alternative, with tie-breaks based on price values
                max_value = max(util_list)
                candidate_indices = [i for i, val in enumerate(util_list) if abs(val - max_value) < 1e-9]
                max_util = max(candidate_indices, key=lambda idx: p[idx - 1])
                omega[max_util, n * R + r] = 1

                if 1 <= max_util:
                    obj -= (1 / R) * p[max_util - 1]
                    eta[max_util, n * R + r] = p[max_util - 1]
        del p[-1]
    return obj, omega, eta


def recompute_exo_util_onep(N, R, exo_utility, endo_coef, PSP_price, alt=False):
    if not alt:
        exo_utility_new = np.zeros((2, N * R))
        endo_coef_new = np.zeros((N * R))
        for n in range(N):
            for r in range(R):
                U_0 = exo_utility[0, n * R + r]
                U_1 = exo_utility[1, n * R + r] + endo_coef[1, n * R + r] * PSP_price
                if U_0 > U_1:
                    exo_utility_new[0, n * R + r] = U_0
                else:
                    exo_utility_new[0, n * R + r] = U_1
                exo_utility_new[1, n * R + r] = exo_utility[2, n * R + r]
                endo_coef_new[n * R + r] = endo_coef[2][n * R + r]
    else:
        exo_utility_new = np.zeros((2, N, R))
        endo_coef_new = np.zeros((2, N, R))
        for n in range(N):
            for r in range(R):
                U_0 = exo_utility[0, n * R + r]
                U_1 = exo_utility[1, n * R + r] + endo_coef[1, n * R + r] * PSP_price
                if U_0 > U_1:
                    exo_utility_new[0, n, r] = U_0
                else:
                    exo_utility_new[0, n, r] = U_1
                exo_utility_new[1, n, r] = exo_utility[2, n * R + r]
                endo_coef_new[0, n, r] = 1
                endo_coef_new[1, n, r] = endo_coef[2][n * R + r]
    return exo_utility_new, endo_coef_new


def compute_breakpoints_exo(exo_utility, endo_coef, cheapest_alternative):
    j_c = cheapest_alternative
    U_0 = exo_utility[0]
    U_j_c = exo_utility[j_c]
    breakpoints = (U_0 - U_j_c) / endo_coef[j_c]
    return breakpoints


def print_results(N, R, J, total_time, iteration, best_price, best_obj, gap):
    print("N = ", N, ", R = ", R, ", J = ", J, ", Total time = ", total_time, ", Iterations = ", iteration,
          ", Best Price = ",
          best_price, ", objective = ", best_obj, f"Gap = {gap}%")


def compute_heuristic_initial_solution(p_L, p_U, parking_data, exo_utility, endo_coef, onep_input, J_PSP, J_PUP):
    parking_data_alt = dict()
    parking_data_alt['N'] = 5
    parking_data_alt['R'] = 1
    parking_data_alt['J_tot'] = J_PSP + J_PUP + 1
    heur_start = [(p_L[i] + p_U[i]) / 2 for i in range(1, parking_data['J_tot'])]
    _, heur_price, _, _, _, _ = fullMILP_bigM.cpp_QCLP(100, 1, p_L, p_U, parking_data_alt, exo_utility, endo_coef,
                                                       0, onep_input, None, J_PSP, J_PUP, heur_start)
    # in case one product is ignored by customers, the price will take the upper bound due to the profit maximization
    # However, this is likely not going to be the best price once we add more customers. Thus for those products
    # where this occurs, we replace the price by the standard basic heuristic of the average between the bounds
    new_heur_price = []
    for i, price in enumerate(heur_price):
        if price == p_U[i + 1]:
            new_heur_price.append((p_L[i + 1] + p_U[i + 1]) / 2)
        else:
            new_heur_price.append(price)
    heur_price = new_heur_price
    return heur_price


def compute_last_price_with_polytime(prices, p_L, p_U, exo_utility, endo_coef, J_PSP, J_PUP):
    J = J_PSP + J_PUP + 1
    exo = exo_utility.reshape(J, -1)
    endo = endo_coef.reshape(J, -1)
    pr_values = prices
    # Initialize an array to store intermediate U arrays
    U_arrays = []
    U_0 = exo[0]
    U_arrays.append(U_0)
    # Calculate U_j for each j from 1 to J-1
    for j in range(1, J - 1):
        U_j = exo[j] + pr_values[j - 1] * endo[j]
        U_arrays.append(U_j)
    U_tuple = tuple(U_arrays)
    # Calculate element-wise maximum over all U arrays
    U_n = np.maximum.reduce(U_tuple)
    # Compute breakpoints
    breakpoints = (U_n - exo[J - 1]) / endo[J - 1]
    breakpoints = np.sort(breakpoints)

    sze = len(breakpoints)
    profitss = np.zeros(sze)
    for p_ind in range(sze):
        profitss[p_ind] = breakpoints[p_ind] * (sze - p_ind) * (1 / R)
    best_ind = np.argmax(profitss)
    best_price = breakpoints[best_ind]

    if best_price > p_U[J - 1]:
        best_price = p_U[J - 1]
    elif best_price < p_L[J - 1]:
        best_price = p_L[J - 1]

    return best_price


def compute_breakbounds(N, R, J, p_l, p_u, exo_utility, endo_coef, pl_VIs):
    exo_utility = exo_utility.reshape((J, N * R))
    endo_coef = endo_coef.reshape((J, N * R))
    if pl_VIs:
        smallest_breakbounds_l = {i: 100 for i in range(1, J)}
        highest_breakbounds_u = {i: -100 for i in range(1, J)}
        for n in range(N):
            for r in range(R):
                for opt_choice in range(1, J):
                    low_utils = [exo_utility[0, n * R + r]]
                    high_utils = [exo_utility[0, n * R + r]]
                    for i in range(1, J):
                        low_utils.append(exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u[i])
                        high_utils.append(exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_l[i])
                    low_util_max = max(low_utils)
                    high_util_max = max(high_utils)
                    bp_low = (high_util_max - exo_utility[opt_choice, n * R + r]) / endo_coef[opt_choice, n * R + r]
                    bp_high = (low_util_max - exo_utility[opt_choice, n * R + r]) / endo_coef[opt_choice, n * R + r]
                    if bp_low < smallest_breakbounds_l[opt_choice]:
                        smallest_breakbounds_l[opt_choice] = bp_low
                    if bp_high > highest_breakbounds_u[opt_choice]:
                        highest_breakbounds_u[opt_choice] = bp_high
    else:
        smallest_breakbounds_l = {i: -100 for i in range(1, J)}  # just to be returned. Will never be > p_l
        highest_breakbounds_u = {i: -100 for i in range(1, J)}
        for n in range(N):
            for r in range(R):
                for opt_choice in range(1, J):
                    low_utils = [exo_utility[0, n * R + r]]
                    for i in range(1, J):
                        low_utils.append(exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u[i])
                    low_util_max = max(low_utils)
                    bp_high = (low_util_max - exo_utility[opt_choice, n * R + r]) / endo_coef[opt_choice, n * R + r]
                    if bp_high > highest_breakbounds_u[opt_choice]:
                        highest_breakbounds_u[opt_choice] = bp_high
    return smallest_breakbounds_l, highest_breakbounds_u


def cpp_benders(N, R, J_PSP, J_PUP, p_L, p_U, time_limit=10800, startpr=None, guidedbranch=False, validcuts=False,
                gapcuts=False, guidedenum=False, optcut=False, mcm2=False, objcut=False, cap=False, more_classes=False,
                pl_VIs=False, forced_improvement=False, breakbounds=False, do_vis=False, heuro=False, minutes=0,
                addeta=0, optvalid=0):
    if meth_input in [[7], [8], [9], [10], [11], [82]]:
        heuro = True
    
    forced_improvement = False
    behanocap = False
    behacap = False
    BEAcap = False
    ILScap = False
    ILSnocap = False
    if meth_input == [8]:
        behanocap = True
    elif meth_input == [9]:
        BEAcap = True
    elif meth_input == [10]:
        behacap = True
    elif meth_input == [11]:
        ILScap = True
    elif meth_input == [82]:
        ILSnocap = True

    if onep_input:
        parking_data = get_input_data_parking(N, R, J_PSP=1, J_PUP=1)  # we gain one-price setting from two-price
        exo_utility = parking_data['exo_utility'].reshape((3, N * R))  # flattening
        endo_coef = parking_data['endo_coef'].reshape((3, N * R))
        J = 3
    else:
        logit = False
        create_data = False
        if logit:
            parking_data = get_input_data_parking_logit(N, R, J_PSP, J_PUP)
        else:
            parking_data = get_input_data_parking(N, R, J_PSP, J_PUP, more_classes,
                                                  minutes)  # this is where we get data
        J = parking_data["J_tot"]

        if heuro:  # save data for heur and run heuristic
            J = parking_data["J_tot"]
            # reshape the inputs so that in a way that Julia needs it
            exo_utility = parking_data['exo_utility'].reshape((J, R, N))
            endo_coef = parking_data['endo_coef'].reshape((J, R, N))

            # switch the opt out to be the last entry, as Robin does it
            new_exo_utility = np.empty_like(exo_utility)
            new_endo_coef = np.empty_like(endo_coef)
            new_exo_utility[:-1, :, :] = exo_utility[1:, :, :]
            new_exo_utility[-1, :, :] = exo_utility[0, :, :]
            new_endo_coef[:-1, :, :] = endo_coef[1:, :, :]
            new_endo_coef[-1, :, :] = endo_coef[0, :, :]
            exo_utility = new_exo_utility
            endo_coef = new_endo_coef
            full_data = np.empty((2, J, R, N))
            # Assign values to the first and second slices along the first axis
            full_data[0] = exo_utility
            full_data[1] = endo_coef
            np.savez(f'data_test_{N}_{R}_{J_PSP}_{J_PUP}.npz', arr=full_data)

            # print(f'data_test_{N}_{R}_{J_PSP}_{J_PUP}.npz created')
            # exit()

            if behanocap:
                command = f"julia heur_0.jl {N} {R} {J_PSP} {J_PUP}"
            elif ILSnocap:
                command = f"julia heur_4.jl {N} {R} {J_PSP} {J_PUP}"
            elif behacap:
                command = f"julia bhac.jl {N} {R} {J_PSP} {J_PUP}"
            elif ILScap:
                command = f"julia ilscap.jl {N} {R} {J_PSP} {J_PUP}"
            elif BEAcap:
                command = f"julia beac.jl {N} {R} {J_PSP} {J_PUP}"
            else:
                command = f"julia bea.jl {N} {R} {J_PSP} {J_PUP}"

            # Execute the command
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            output = result.stdout

            #print(output)

            if meth_input in [[7], [8], [10], [11], [82]]:
                # Extract time_heur values and select the second one
                time_heur = float(re.findall(r"time_heur\s*=\s*([0-9.e+-]+)", output)[1])  # Second occurrence
                # Extract best_obj values and select the second one
                best_obj = float(re.findall(r"best_obj\s*=\s*([0-9.]+)", output)[1])  # Second occurrence
                # Extract best_prices lists and select the second one
                best_prices_str = re.findall(r"best_prices\s*=\s*\[([0-9.,\s]+)\]", output)[1]  # Second occurrence
            else:
                # Extract time_heur values and select the second one
                time_heur = float(re.findall(r"time_poly\s*=\s*([0-9.e+-]+)", output)[1])  # Second occurrence
                # Extract best_obj values and select the second one
                best_obj = float(re.findall(r"Z_poly\s*=\s*([0-9.]+)", output)[1])  # Second occurrence
                # Extract best_prices lists and select the second one
                best_prices_str = re.findall(r"x_poly\s*=\s*\[([0-9.,\s]+)\]", output)[1]  # Second occurrence

            # Convert the best_prices string to an actual list of floats
            best_prices = [float(price.strip()) for price in best_prices_str.split(',')]


            os.remove(f'data_test_{N}_{R}_{J_PSP}_{J_PUP}.npz')
            return best_prices, time_heur, best_obj, 0, None



        if create_data:
            # reshape the inputs so that in a way that Julia needs it
            exo_utility = parking_data['exo_utility'].reshape((J, R, N))
            endo_coef = parking_data['endo_coef'].reshape((J, R, N))

            # switch the opt out to be the last entry, as Robin does it
            new_exo_utility = np.empty_like(exo_utility)
            new_endo_coef = np.empty_like(endo_coef)
            new_exo_utility[:-1, :, :] = exo_utility[1:, :, :]
            new_exo_utility[-1, :, :] = exo_utility[0, :, :]
            new_endo_coef[:-1, :, :] = endo_coef[1:, :, :]
            new_endo_coef[-1, :, :] = endo_coef[0, :, :]
            exo_utility = new_exo_utility
            endo_coef = new_endo_coef

            print(f"Created instances {N} {R} {J_PSP} {J_PUP}")
            print(type(exo_utility))
            print(type(endo_coef))
            print(exo_utility.shape)
            print(endo_coef.shape)

            # Assuming you have two NumPy arrays: array1 and array2
            if logit:
                with h5py.File(f'data_test_logit_{N}_{R}_{J_PSP}_{J_PUP}.h5', 'w') as hf:
                    hf.create_dataset('exo_utility_test', data=exo_utility)
                    hf.create_dataset('endo_coef_test', data=endo_coef)
                print(f"created data_test_logit_{N}_{R}_{J_PSP}_{J_PUP}.h5")
            else:
                full_data = np.empty((2, J, R, N))

                # Assign values to the first and second slices along the first axis
                full_data[0] = exo_utility
                full_data[1] = endo_coef

                np.savez(f'data_test_{N}_{R}_{J_PSP}_{J_PUP}.npz', arr=full_data)
            exit()

        exo_utility = parking_data['exo_utility'].reshape((J, N * R))
        endo_coef = parking_data['endo_coef'].reshape((J, N * R))
        # print("data created")

        caps = None

        if cap:
            D = J - 1
            S = R
            # Meris capacities
            if D == 2:
                if N == 50:
                    caps = [20, 20]
                elif N == 197:
                    caps = [80, 80]
                else:
                    caps = [np.floor(N * 0.4) for i in range(1, J)]
            elif D == 4:
                if N == 5:
                    caps = [2, 2, 2, 2]
                elif N == 50:
                    caps = [15, 15, 15, 15]
                else:
                    caps = [np.floor(N * 0.3) for i in range(1, J)]
            else:
                caps = [np.floor(N * 0.3) for i in range(1, J)]

        run_logit = False

        if run_logit:
            exo_utility = exo_utility[:, ::R] / 100 + 1
            endo_coef = endo_coef[:, ::R] / 100

            # print("")
            # print("Run GRB convex logit CPP (linearized)")
            pPSP_L = 0
            pPSP_U = 2
            pPUP_L = 0
            pPUP_U = 2
            p_L = dict()
            p_U = dict()
            for i in range(J_PSP):
                p_L[i + 1] = pPSP_L
                p_U[i + 1] = pPSP_U
            for i in range(J_PUP):
                p_L[i + J_PSP + 1] = pPUP_L
                p_U[i + J_PSP + 1] = pPUP_U

            total_time_conv, bestprice_conv, best_obj_conv, best_lowerbound_conv, nodes_conv = \
                fullMILP_bigM.cpp_MILP_convex(
                    time_limit,
                    threads,
                    p_L,
                    p_U,
                    parking_data,
                    exo_utility,
                    endo_coef)
            if total_time_conv > time_limit:
                total_time_conv = time_limit
            print_results(N, R, J, total_time_conv, nodes_conv, bestprice_conv, best_obj_conv, 0)
            exit()

    # print(f"Instance: N = {N}, R = {R}, J_PSP = {J_PSP}, J_PUP = {J_PUP}, Method = {meth_input}")
    # print("")

    run_cpp_MILP = 0
    run_cpp_QCQP = 0
    run_cpp_QCLP = 0
    run_cpp_bnb = 0
    run_cpp_bnb_benders_disagg = 0

    viol = 0

    if 2 in meth_input:
        run_cpp_MILP = 1
    if 3 in meth_input:
        run_cpp_QCQP = 1
    if 4 in meth_input:
        run_cpp_QCLP = 1
    if 5 in meth_input:
        run_cpp_bnb = 1
    if 6 in meth_input:
        run_cpp_bnb_benders_disagg = 1
    if 7 in meth_input:
        run_cpp_bnb_benders_disagg = 1
        viol = 1  # only add benders cut if its violated by the current optimal solution

    branching = "eta"

    if guidedbranch:
        branching = "guided"

    if cap:
        branching = "longestEdge"
    # branching = "longestEdge"

    # branching = "guided"

    if p_L is None and p_U is None:
        pPSP_L = 0.5
        pPSP_U = 0.7
        pPUP_L = 0.65
        pPUP_U = 0.85

        # pPSP_L = 0
        # pPSP_U = 2
        # pPUP_L = 0
        # pPUP_U = 2

        if onep_input:
            pPSP_L = 0.6
            pPSP_U = 0.6

        p_L = dict()
        p_U = dict()
        if onep_input:
            p_L[1] = pPSP_L
            p_U[1] = pPSP_U
            p_L[2] = pPUP_L
            p_U[2] = pPUP_U
        else:
            for i in range(J_PSP):
                p_L[i + 1] = pPSP_L
                p_U[i + 1] = pPSP_U
            for i in range(J_PUP):
                p_L[i + J_PSP + 1] = pPUP_L
                p_U[i + J_PSP + 1] = pPUP_U

    bp_cheapest = 0

    if onep_input:
        # so here we basically recompute the exo_utility so that we have only two alternatives.
        # the zero alternative will be the best one given a fixed price for PSP
        PSP_price = 0.6
        exo_utility, endo_coef = recompute_exo_util_onep(N, R, exo_utility, endo_coef, PSP_price)
        if fixed_choices:
            # for the bp we need to get the coefficients into the same form as for multiple prices
            bp_cheapest = compute_breakpoints_exo(exo_utility, np.vstack((np.zeros_like(endo_coef), endo_coef)), 1)

    if cheapest_alternative is not None:
        if onep_input:  # onep makes things incredibly complicated since it reduces twop to onep but then keeps
            # product 2 as the available one
            # compute the new starting bounds:
            sorted_bp = np.sort(bp_cheapest)
            closest_bp_pL = fullMILP_BnB.find_closest(sorted_bp, p_L[2], "up")
            closest_bp_pU = fullMILP_BnB.find_closest(sorted_bp, p_U[2], "down")
            p_L[2] = closest_bp_pL
            p_U[2] = closest_bp_pU
        else:
            bp_cheapest = compute_breakpoints_exo(exo_utility, endo_coef, cheapest_alternative)
            # compute the new starting bounds:
            sorted_bp = np.sort(bp_cheapest)
            closest_bp_pL = fullMILP_BnB.find_closest(sorted_bp, p_L[cheapest_alternative], "up")
            closest_bp_pU = fullMILP_BnB.find_closest(sorted_bp, p_U[cheapest_alternative], "down")
            p_L[cheapest_alternative] = closest_bp_pL
            p_U[cheapest_alternative] = closest_bp_pU

    if not onep_input:  # bp_cheapest are not used if we have more than one price
        bp_cheapest = 0

    stimee = time.time()

    reduce_dim_1 = 0

    heur_reduce_dim_1 = 0
    if reduce_dim_1 == 0:
        heur_reduce_dim_1 = 0

    if onep_input:
        reduce_dim_1 = 0
    if heur:
        if heur_reduce_dim_1:
            start_price = compute_heuristic_initial_solution(p_L, p_U, parking_data, exo_utility, endo_coef, onep_input,
                                                             J_PSP, J_PUP - 1)
            last_price = compute_last_price_with_polytime(start_price, p_L, p_U, exo_utility, endo_coef, J_PSP, J_PUP)
            start_price = start_price + [last_price]
            obj, omega, eta = compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef, start_price, onep_input,
                                                         J_PSP, J_PUP)
            # print(start_price, obj)
        else:
            if False:  # J_PSP + J_PUP == 2:
                start_price = [0 for i in range(J - 1)]
                # start_price = [0.638862129955429, 0.5842902566303168, 0.6722968074831345, 0.6867455301834305]
                # start_price = [0.5971821646534721, 0.6503569889585201, 0.6532581685417445, 0.6671875] # 23.472123861312866, Iterations = 893
                # start_price = [0, 0, 0, 0] # Total time = 24.96291184425354, Iterations = 926
                # start_price = [0.5650119234531322, 0.6617442186399446]
                # start_price = [0, 0] # -> Total time = 29.33857297897339, Iterations = 277
                # start_price = [0.6004077541661854, 0.6159068569677679]
                # start_price = [0.5456515836090531, 0.679451032279653]
                # start_price = [0.5798001532575636, 0.5757018729582583, 0.6634532447511973, 0.6635794818743744]
                # start_price = [0.586907129415623, 0.5535144790260481, 0.5306805152020806, 0.734386349532225, 0.704392435483885, 0.672672813459359]
                # start_price = [0.565813004340579, 0.6152547990598985, 0.6507812500000001, 0.6636668929419397]

                # this is with the opt as sstart
                # N = 40, R = 40, J = 5, Total
                # time = 3180.9881007671356, Iterations = 14063, Best
                # Price = [0.6024475969179449, 0.5752542861152058, 0.6707223426825221,
                #          0.650764847802606], objective = 21.053982614148534, Gap = 0.009984374088963055 %

                # N = 30, R = 30, J = 5, Total
                # time = 1097.8827707767487, Iterations = 9529, Best
                # Price = [0.565813004340579, 0.6152547990598985, 0.6507812500000001,
                #          0.6636668929419397], objective = 15.63786048115592, Gap = 0.009975285613986043 %

                # N = 10, R = 10, J = 7, Total
                # time = 111.8249819278717, Iterations = 6489, Best
                # Price = [0.586907129415623, 0.5535144790260481, 0.5306805152020806, 0.734386349532225,
                #          0.704392435483885,
                #          0.672672813459359], objective = 5.1704399530280005, Gap = 0.009921704096585734 %

                # N = 20, R = 20, J = 5, Total
                # time = 96.3353979587555, Iterations = 1814, Best
                # Price = [0.5798001532575636, 0.5757018729582583, 0.6634532447511973,
                #          0.6635794818743744], objective = 10.485260111038789,
            else:
                if startpr is None:
                    # start_price = [0 for i in range(J - 1)]
                    start_price = [(p_L[i] + p_U[i]) / 2 for i in range(1, J)]
                    # start_price = compute_heuristic_initial_solution(p_L, p_U, parking_data, exo_utility, endo_coef,
                    #                                                  onep_input, J_PSP, J_PUP)
                else:
                    start_price = startpr
            obj, omega, eta = compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef, start_price, onep_input,
                                                         J_PSP, J_PUP)
            # print("")
            # print("start = ", start_price, -obj)
            # print("")
    else:
        if reduce_dim_1:
            start_price = [(p_L[i] + p_U[i]) / 2 for i in range(1, J)]
        else:
            start_price = [(p_L[i] + p_U[i]) / 2 for i in range(1, J)]

    if run_cpp_MILP:
        # print("")
        # print("Run GRB MILP (linearized)")
        total_time_MILP, bestprice_MILP, best_obj_MILP, best_lowerbound_MILP, gap_MILP, nodes_MILP = \
            fullMILP_bigM.cpp_MILP(
                time_limit,
                threads,
                p_L,
                p_U,
                parking_data,
                exo_utility,
                endo_coef,
                bp_cheapest,
                onep_input,
                fixed_choices,
                J_PSP,
                J_PUP,
                start_price)
        if total_time_MILP > time_limit:
            total_time_MILP = time_limit
        print_results(N, R, J, total_time_MILP, nodes_MILP, bestprice_MILP, best_obj_MILP, gap_MILP)

    if run_cpp_QCQP:
        # print("")
        # print("Run GRB QCQP")
        total_time_nonlinNH, bestprice_nonlinNH, best_obj_nonlinNH, best_lowerbound_nonlinNH, gap_nonlinNH, \
        nodes_nonlinNH = \
            fullMILP_bigM.cpp_QCQP(
                time_limit,
                threads,
                p_L,
                p_U,
                parking_data,
                exo_utility,
                endo_coef,
                bp_cheapest,
                onep_input,
                fixed_choices,
                J_PSP,
                J_PUP,
                start_price)
        if total_time_nonlinNH > time_limit:
            total_time_nonlinNH = time_limit
        print_results(N, R, J, total_time_nonlinNH, nodes_nonlinNH, bestprice_nonlinNH, best_obj_nonlinNH,
                      gap_nonlinNH)

    if run_cpp_QCLP:
        # print("")
        # print("Run GRB QCLP")
        total_time_nonlin, bestprice_nonlin, best_obj_nonlin, best_lowerbound_nonlin, gap_nonlin, nodes_nonlin = \
            fullMILP_bigM.cpp_QCLP(
                time_limit,
                threads,
                p_L,
                p_U,
                parking_data,
                exo_utility,
                endo_coef,
                bp_cheapest,
                onep_input,
                fixed_choices,
                J_PSP,
                J_PUP,
                start_price)
        if total_time_nonlin > time_limit:
            total_time_nonlin = time_limit
        print_results(N, R, J, total_time_nonlin, nodes_nonlin, bestprice_nonlin, best_obj_nonlin, gap_nonlin)
    if run_cpp_bnb:
        # print("")
        # print("Run BnB w/out Benders")
        total_time_BnB, bestprice_BnB, best_obj_BnB, best_lowerbound_BnB, node_count, gap_BnB, occs_BnB = \
            fullMILP_BnB.cpp_bnb(N, R,
                                 exo_utility,
                                 endo_coef,
                                 time_limit,
                                 threads,
                                 p_L,
                                 p_U,
                                 branching,  # branching ("eta", "omega", "longestEdge")
                                 bendering=0,
                                 lwbing=1,
                                 pres=0,
                                 breakpoints=bp_cheapest,
                                 onep=onep_input,
                                 fixed_choices=fixed_choices,
                                 J_PSP=J_PSP,
                                 J_PUP=J_PUP,
                                 enum=enum,
                                 start_price=start_price,
                                 timetricks=0,
                                 viol=viol,
                                 validcuts=validcuts,
                                 gapcuts=gapcuts,
                                 guidedenum=guidedenum,
                                 optcut=optcut,
                                 objcut=objcut,
                                 mcm2=mcm2,
                                 caps=caps,
                                 pl_VIs=pl_VIs,
                                 forced_improvement=forced_improvement,
                                 breakbounds=breakbounds,
                                 do_vis=do_vis,
                                 addeta=addeta,
                                 optvalid=optvalid)
        if total_time_BnB > time_limit:
            total_time_BnB = time_limit
        # print_results(N, R, J, total_time_BnB, node_count, bestprice_BnB, best_obj_BnB, gap_BnB)

    if run_cpp_bnb_benders_disagg:
        # print("")
        # print("Run BnB with Benders")
        total_time_BnBBd, bestprice_BnBBd, best_obj_BnBBd, best_lowerbound_BnBBd, node_countBd, gap_BnBBd, occs_BnBBD = \
            fullMILP_BnB.cpp_bnb(N, R,
                                 exo_utility,
                                 endo_coef,
                                 time_limit,
                                 threads,
                                 p_L,
                                 p_U,
                                 branching,  # branching ("eta", "omega", "longestEdge")
                                 bendering=1,
                                 lwbing=1,
                                 pres=0,
                                 breakpoints=bp_cheapest,
                                 onep=onep_input,
                                 fixed_choices=fixed_choices,
                                 J_PSP=J_PSP,
                                 J_PUP=J_PUP,
                                 enum=enum,
                                 start_price=start_price,
                                 timetricks=timetricks,
                                 viol=viol,
                                 validcuts=validcuts,
                                 gapcuts=gapcuts,
                                 guidedenum=guidedenum,
                                 optcut=optcut,
                                 objcut=objcut,
                                 mcm2=mcm2,
                                 caps=caps,
                                 pl_VIs=pl_VIs,
                                 forced_improvement=forced_improvement,
                                 breakbounds=breakbounds,
                                 do_vis=do_vis,
                                 addeta=addeta,
                                 optvalid=optvalid
                                 )
        if total_time_BnBBd > time_limit:
            total_time_BnBBd = time_limit
        # print_results(N, R, J, total_time_BnBBd, node_countBd, bestprice_BnBBd, best_obj_BnBBd, gap_BnBBd)
    if run_cpp_MILP: #MILP
        # obj, _, _ = compute_cpp_from_p_parking(J_PSP + J_PUP + 1, N, R, exo_utility, endo_coef, bestprice_BnB, False,
        #                                        J_PSP, J_PUP)
        # print("Evaluated with fun gives revenue = ", obj)
        return bestprice_MILP, total_time_MILP, best_obj_MILP, None, None
    if run_cpp_QCQP: # nonlinNH
        # obj, _, _ = compute_cpp_from_p_parking(J_PSP + J_PUP + 1, N, R, exo_utility, endo_coef, bestprice_BnB, False,
        #                                        J_PSP, J_PUP)
        # print("Evaluated with fun gives revenue = ", obj)
        return bestprice_nonlinNH, total_time_nonlinNH, best_obj_nonlinNH, None, None
    if run_cpp_QCLP: # nonlin
        # obj, _, _ = compute_cpp_from_p_parking_PSP + J_PUP + 1, N, R, exo_utility, endo_coef, bestprice_BnBBd, False,
        #                                        J_PSP, J_PUP)
        # print("Evaluated with fun gives revenue = ", obj)
        return bestprice_nonlin, total_time_nonlin, best_obj_nonlin, None, None
    if run_cpp_bnb:
        # obj, _, _ = compute_cpp_from_p_parking(J_PSP + J_PUP + 1, N, R, exo_utility, endo_coef, bestprice_BnB, False,
        #                                        J_PSP, J_PUP)
        # print("Evaluated with fun gives revenue = ", obj)
        return bestprice_BnB, total_time_BnB, best_obj_BnB, node_count, occs_BnB
    if run_cpp_bnb_benders_disagg:
        # obj, _, _ = compute_cpp_from_p_parking(J_PSP + J_PUP + 1, N, R, exo_utility, endo_coef, bestprice_BnBBd, False,
        #                                        J_PSP, J_PUP)
        # print("Evaluated with fun gives revenue = ", obj)
        return bestprice_BnBBd, total_time_BnBBd, best_obj_BnBBd, node_countBd, occs_BnBBD


# Meris bounds (cool)
# boundsL = [0.5, 0.5, 0.65, 0.65]
# boundsU = [0.7, 0.7, 0.85, 0.85]


# write it nicer:
# Lower and upper bounds for PSP
psp_lower, psp_upper = 0.5, 0.7
# Lower and upper bounds for PUP
pup_lower, pup_upper = 0.65, 0.85

# Create boundsL and boundsU using list comprehension and repetition
boundsL = [psp_lower] * J_PSP + [pup_lower] * J_PUP
boundsU = [psp_upper] * J_PSP + [pup_upper] * J_PUP

# reasonable bounds (boring)
# boundsL = [0.25, 0.25, 0.25, 0.25]
# boundsU = [1.25, 1.25, 1.25, 1.25]

p_L = {i: boundsL[i - 1] for i in range(1, J_PUP + J_PSP + 1)}
p_U = {i: boundsU[i - 1] for i in range(1, J_PUP + J_PSP + 1)}

prices_norm = None

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='biogeme')
    
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


if meth_input == [2]:
    with suppress_output():
        prices_norm, time_norm, obj_norm, node_norm, occs_norm = cpp_benders(N=N,
                                                                             R=R,
                                                                             J_PSP=J_PSP,
                                                                             J_PUP=J_PUP,
                                                                             p_L=p_L,
                                                                             p_U=p_U,
                                                                             minutes=minutes,
                                                                             time_limit=timelim,
                                                                             mcm2=mcm2)
    print(f"R={R}, MILP gives", prices_norm, obj_norm, "in", time_norm, "s,", node_norm, "nodes")
elif meth_input == [3]:
    with suppress_output():
        prices_norm, time_norm, obj_norm, node_norm, occs_norm = cpp_benders(N=N,
                                                                             R=R,
                                                                             J_PSP=J_PSP,
                                                                             J_PUP=J_PUP,
                                                                             p_L=p_L,
                                                                             p_U=p_U,
                                                                             minutes=minutes,
                                                                             time_limit=timelim,
                                                                             mcm2=mcm2)
    print(f"R={R}, QCQP gives", prices_norm, obj_norm, "in", time_norm, "s,", node_norm, "nodes")
elif meth_input == [4]:
    with suppress_output():
        prices_norm, time_norm, obj_norm, node_norm, occs_norm = cpp_benders(N=N,
                                                                             R=R,
                                                                             J_PSP=J_PSP,
                                                                             J_PUP=J_PUP,
                                                                             p_L=p_L,
                                                                             p_U=p_U,
                                                                             minutes=minutes,
                                                                             time_limit=timelim,
                                                                             mcm2=mcm2)
    print(f"R={R}, QCQP-L gives", prices_norm, obj_norm, "in", time_norm, "s,", node_norm, "nodes")
elif meth_input == [5]:
    prices_norm, time_norm, obj_norm, node_norm, occs_norm = cpp_benders(N=N,
                                                                         R=R,
                                                                         J_PSP=J_PSP,
                                                                         J_PUP=J_PUP,
                                                                         p_L=p_L,
                                                                         p_U=p_U,
                                                                         minutes=minutes,
                                                                         time_limit=timelim,
                                                                         mcm2=mcm2)
    print(f"R={R}, BnB gives", prices_norm, obj_norm, "in", time_norm, "s,", node_norm, "nodes")
elif meth_input == [6]:
    prices_norm, time_norm, obj_norm, node_norm, occs_norm = cpp_benders(N=N,
                                                                         R=R,
                                                                         J_PSP=J_PSP,
                                                                         J_PUP=J_PUP,
                                                                         p_L=p_L,
                                                                         p_U=p_U,
                                                                         minutes=minutes,
                                                                         time_limit=timelim,
                                                                         mcm2=mcm2)
    print(f"R={R}, BnBD gives", prices_norm, obj_norm, "in", time_norm, "s,", node_norm, "nodes")
elif meth_input == [7]:
    prices_norm, time_norm, obj_norm, node_norm, occs_norm = cpp_benders(N=N,
                                                                         R=R,
                                                                         J_PSP=J_PSP,
                                                                         J_PUP=J_PUP,
                                                                         p_L=p_L,
                                                                         p_U=p_U,
                                                                         minutes=minutes,
                                                                         time_limit=timelim,
                                                                         mcm2=mcm2)
    print(f"R={R}, BEA gives", prices_norm, obj_norm, "in", time_norm, "s")
elif meth_input == [8]:
    prices_norm, time_norm, obj_norm, node_norm, occs_norm = cpp_benders(N=N,
                                                                         R=R,
                                                                         J_PSP=J_PSP,
                                                                         J_PUP=J_PUP,
                                                                         p_L=p_L,
                                                                         p_U=p_U,
                                                                         minutes=minutes,
                                                                         time_limit=timelim,
                                                                         mcm2=mcm2)
    print(f"R={R}, BHA gives", prices_norm, obj_norm, "in", time_norm, "s")
elif meth_input == [82]:
    prices_norm, time_norm, obj_norm, node_norm, occs_norm = cpp_benders(N=N,
                                                                         R=R,
                                                                         J_PSP=J_PSP,
                                                                         J_PUP=J_PUP,
                                                                         p_L=p_L,
                                                                         p_U=p_U,
                                                                         minutes=minutes,
                                                                         time_limit=timelim,
                                                                         mcm2=mcm2)
    print(f"R={R}, ILS gives", prices_norm, obj_norm, "in", time_norm, "s")
elif meth_input == [9]:
    prices_norm, time_norm, obj_norm, node_norm, occs_norm = cpp_benders(N=N,
                                                                         R=R,
                                                                         J_PSP=J_PSP,
                                                                         J_PUP=J_PUP,
                                                                         p_L=p_L,
                                                                         p_U=p_U,
                                                                         minutes=minutes,
                                                                         time_limit=timelim,
                                                                         mcm2=mcm2)
    print(f"R={R}, BEAC gives", prices_norm, obj_norm, "in", time_norm, "s")
elif meth_input == [10]:
    prices_norm, time_norm, obj_norm, node_norm, occs_norm = cpp_benders(N=N,
                                                                         R=R,
                                                                         J_PSP=J_PSP,
                                                                         J_PUP=J_PUP,
                                                                         p_L=p_L,
                                                                         p_U=p_U,
                                                                         minutes=minutes,
                                                                         time_limit=timelim,
                                                                         mcm2=mcm2)
    print(f"R={R}, BHAC gives", prices_norm, obj_norm, "in", time_norm, "s")
elif meth_input == [11]:
    prices_norm, time_norm, obj_norm, node_norm, occs_norm = cpp_benders(N=N,
                                                                         R=R,
                                                                         J_PSP=J_PSP,
                                                                         J_PUP=J_PUP,
                                                                         p_L=p_L,
                                                                         p_U=p_U,
                                                                         minutes=minutes,
                                                                         time_limit=timelim,
                                                                         mcm2=mcm2)
    print(f"R={R}, ILSC gives", prices_norm, obj_norm, "in", time_norm, "s")
else:
    print("Method is not 2, 3, 4, 5, 6, 7, 8, 9, 10, or 11")

