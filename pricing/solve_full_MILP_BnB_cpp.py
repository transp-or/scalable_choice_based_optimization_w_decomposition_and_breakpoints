import numpy as np
import time
import logging
import datetime
import copy

import gurobipy as gp
from gurobipy import GRB
from McCormick import initialize_McCormick, solve_MILP_cap, compute_obj_value_caps, cpp_MILP_cap, cpp_QCLP_cap

import cpp_bnb_benders

console = False

presolve = 0
# 0 = none, -1 = automatic, 1 = conservative, 2 = aggressive
numeric_focus = 1


class Node:
    def __init__(self, lower_bound, p_lower_bound, p_upper_bound, fixed_omega, optOuts):
        self.lb = lower_bound
        self.plb = p_lower_bound
        self.pub = p_upper_bound
        self.fixed = fixed_omega
        self.optOuts = optOuts


class CppModel:
    def __init__(self, model, MCconstraints, p_vars, omega_vars=None, eta_vars=None):
        self.model = model
        self.MCconstraints = MCconstraints
        self.p_vars = p_vars
        self.omega_vars = omega_vars
        self.eta_vars = eta_vars


def update_relaxation_bounds(N, R, J, mdl, p_lb, p_ub, onep, mcm2):
    if onep:
        mdl.p_vars[2].lb = p_lb[2]
        mdl.p_vars[2].ub = p_ub[2]

        for n in range(N):
            for r in range(R):
                mdl.model.chgCoeff(mdl.MCconstraints[0, n, r, 2], mdl.omega_vars[n * R + r], - p_lb[2])

                mdl.model.chgCoeff(mdl.MCconstraints[1, n, r, 2], mdl.omega_vars[n * R + r], - p_ub[2])
                mdl.MCconstraints[1, n, r, 2].rhs = - p_ub[2]

                mdl.model.chgCoeff(mdl.MCconstraints[2, n, r, 2], mdl.omega_vars[n * R + r], - p_lb[2])
                mdl.MCconstraints[2, n, r, 2].rhs = - p_lb[2]

                mdl.model.chgCoeff(mdl.MCconstraints[3, n, r, 2], mdl.omega_vars[n * R + r], - p_ub[2])
    else:
        for i in range(1, J):
            mdl.p_vars[i].lb = p_lb[i]
            mdl.p_vars[i].ub = p_ub[i]

        # update McCormick
        for n in range(N):
            for r in range(R):
                for i in range(1, J):
                    if not mcm2:
                        mdl.model.chgCoeff(mdl.MCconstraints[0, n, r, i], mdl.omega_vars[i, n * R + r], - p_lb[i])

                        mdl.model.chgCoeff(mdl.MCconstraints[1, n, r, i], mdl.omega_vars[i, n * R + r], - p_ub[i])
                        mdl.MCconstraints[1, n, r, i].rhs = - p_ub[i]

                        mdl.model.chgCoeff(mdl.MCconstraints[2, n, r, i], mdl.omega_vars[i, n * R + r], - p_lb[i])
                        mdl.MCconstraints[2, n, r, i].rhs = - p_lb[i]

                        mdl.model.chgCoeff(mdl.MCconstraints[3, n, r, i], mdl.omega_vars[i, n * R + r], - p_ub[i])
                    else:
                        mdl.model.chgCoeff(mdl.MCconstraints[1, n, r, i], mdl.omega_vars[i, n * R + r], - p_ub[i])
                        mdl.MCconstraints[1, n, r, i].rhs = - p_ub[i]

                        mdl.model.chgCoeff(mdl.MCconstraints[2, n, r, i], mdl.omega_vars[i, n * R + r], - p_ub[i])


def remove_obj_fixed_onep(N, R, mdl, fixed_tuples):
    for n in range(N):
        for r in range(R):
            if not fixed_tuples[n, r] == 3:
                mdl.eta_vars[n * R + r].Obj = 0
            else:
                mdl.eta_vars[n * R + r].Obj = - (1 / R)


def remove_obj_fixed(N, R, J, mdl, fixed, p_l, p_u):
    inf = GRB.INFINITY
    for n in range(N):
        for r in range(R):
            if sum(fixed[i, n, r] for i in range(1, J)) == 1:
                # if the choice is fixed, remove the etas, GRB should then ignore all of their constraints
                for i in range(1, J):
                    mdl.eta_vars[i, n * R + r].Obj = 0
                    # if fixed[i, n, r] == 1:
                    #     mdl.omega_vars[i, n * R + r].lb = 1
                    # else:
                    #     mdl.omega_vars[i, n * R + r].ub = 0
                    #     mdl.eta_vars[i, n * R + r].ub = 0
            else:
                # otherwise keep the etas, but do what we can
                for i in range(1, J):
                    mdl.eta_vars[i, n * R + r].Obj = - (1 / R)

                # those alternatives that are dominated get fixed to 0
                for i in range(J):
                    if fixed[i, n, r] == 0:
                        mdl.omega_vars[i, n * R + r].ub = 0
                        # if i >= 1:
                        #     mdl.eta_vars[i, n * R + r].ub = 0
                    else:
                        mdl.omega_vars[i, n * R + r].ub = inf
                        # if i >= 1:
                        #     mdl.eta_vars[i, n * R + r].ub = inf
    return


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
                candidate_indices = [i for i, val in enumerate(util_list) if
                                     abs(round(val, 3) - round(max_value, 3)) == 0]
                max_util = max(candidate_indices, key=lambda idx: p[idx - 1])
                omega[max_util, n * R + r] = 1

                # if n == 2 and r == 4:
                #     print("util list = ", util_list)
                #     print("max_value = ", max_value)
                #     print("candidate_indices = ", candidate_indices)
                #     print("prices = ", p)
                #     print("max_util_index = ", max_util)

                if 1 <= max_util:
                    obj -= (1 / R) * p[max_util - 1]
                    eta[max_util, n * R + r] = p[max_util - 1]
        del p[-1]
    return obj, omega, eta


def relative_optimality_gap(BIG_omega, best_upper_bound):
    return ((min([node.lb for node in BIG_omega]) - best_upper_bound) / best_upper_bound) * 100


def eta_branching_price(N, R, J, p_value, omega_values, eta_values):
    # Calculate eta_difference
    eta_difference = np.zeros((J - 1, N * R))
    for i in range(1, J):
        for n in range(N):
            for r in range(R):
                idx = n * R + r
                eta_difference[i - 1, idx] = abs(omega_values[i, idx] * p_value[i - 1] - eta_values[i, idx])

    # Calculate average violations and find branch_price
    average_violations = np.mean(eta_difference, axis=1)

    # deterministic
    branch_price = np.argmax(average_violations)
    return branch_price + 1


def guided_branching_price_new2(N, R, J, p_value, start_price, start_choices, start_etas, p_l, p_u, omega_values,
                                eta_values):
    # branch on the price where the etas have largest absolute? difference to start_etas (or start_omegas?)

    differences = np.zeros((J - 1, N * R))
    for i in range(1, J):
        for n in range(N):
            for r in range(R):
                idx = n * R + r
                # eta
                differences[i - 1, idx] = abs(start_etas[i, idx] - eta_values[i, idx])
                # omega
                # differences[i - 1, idx] = abs(start_choices[i, idx] - omega_values[i, idx])

    # Calculate average violations and find branch_price
    average_violations = np.mean(differences, axis=1)

    # deterministic
    branch_price = np.argmax(average_violations)
    return branch_price + 1


def guided_branching_price_new(N, R, J, p_value, start_price, p_l, p_u, omega_values, eta_values):
    opt_indices = []
    for i in range(1, J):
        if p_l[i] <= start_price[i - 1] <= p_u[i]:
            opt_indices.append(i)
    if len(opt_indices) == 1:
        print("opt_indices = ", opt_indices)
        print("p_l = ", p_l)
        print("p_u = ", p_u)
        branch_price = opt_indices[0]
    elif len(opt_indices) == 0:
        print("opt_indices = ", opt_indices)
        print("p_l = ", p_l)
        print("p_u = ", p_u)
        # do eta branching
        # Calculate eta_difference
        eta_difference = np.zeros((J - 1, N * R))
        for i in range(1, J):
            for n in range(N):
                for r in range(R):
                    idx = n * R + r
                    eta_difference[i - 1, idx] = abs(omega_values[i, idx] * p_value[i - 1] - eta_values[i, idx])

        # Calculate average violations and find branch_price
        average_violations = np.mean(eta_difference, axis=1)

        # deterministic
        branch_price = np.argmax(average_violations) + 1  # since the argmax gives python indexing
    else:
        print("opt_indices = ", opt_indices)
        print("p_l = ", p_l)
        print("p_u = ", p_u)
        # Initialize eta_difference for selected indices
        eta_difference_opt = np.zeros((len(opt_indices), N * R))

        # Calculate eta_difference only for selected indices
        for counter, i in enumerate(opt_indices):  # Using enumerate to keep track of new index
            for n in range(N):
                for r in range(R):
                    idx = n * R + r
                    # Adjust index for zero-based indexing of eta_values and omega_values
                    eta_difference_opt[counter, idx] = abs(omega_values[i, idx] * p_value[i - 1] - eta_values[i, idx])

        # Calculate average violations for the selected indices
        average_violations_opt = np.mean(eta_difference_opt, axis=1)

        # Find the index within opt_indices that has the maximum average violation
        max_violation_index = np.argmax(average_violations_opt)

        # Map back to original indices
        branch_price = opt_indices[max_violation_index]
    return branch_price


def guided_branching_price(J, p_value, start_price, p_l, p_u):
    # Calculate difference to start price
    # print("current price = ", p_value)
    # print("start price = ", start_price)
    # guide_difference = [abs(start_price[i - 1] - p_value[i - 1]) for i in range(1, J)]
    # guide_difference = [(start_price[i - 1] - p_u[i])**2 for i in range(1, J)]
    guide_difference = [min(start_price[i - 1] - p_l[i], p_u[i] - start_price[i - 1]) for i in range(1, J)]

    for i in range(1, J):
        if guide_difference[i - 1] < 0:
            guide_difference[i - 1] = 100

    # guided_difference = []
    # for i in range(1, J):
    #     if all(start_price[j - 1] >= p_l[j] for j in range(1, J)) and all(
    #             start_price[j - 1] <= p_u[j] for j in range(1, J)):
    #         guided_difference.append(abs(start_price[i - 1] - p_value[i - 1]))
    #         # guided_difference.append(p_u[i] - p_l[i] for i in range(1, J))
    #     else:
    #         guided_difference.append(1000)

    # branch_price = np.argmax(guide_difference)
    # branch_price = np.argmin(guide_difference)
    sorted_indices = [index for index, value in sorted(enumerate(guide_difference), key=lambda pair: pair[1])]
    branch_price = sorted_indices[0] + 1
    if p_u[branch_price] - p_l[branch_price] <= 1e-5:
        branch_price = 0
    # print("biggest difference at ", branch_price)
    return branch_price


def marketShare_branching_price(J, integer_omega):
    # we take second largest / smallest instead of largest to not get stuck
    shares = np.array([sum(integer_omega[i, :]) for i in range(J)])
    # remove opt out share
    shares = shares[1:]

    # if we branch on second smallest:

    # Deterministic:

    # # Find the indices of the two smallest values in the shares array
    # min_indices = np.argpartition(shares, 2)[:2]
    # # Find the second smallest value and its index in the original shares array
    # second_min_value = shares[min_indices[1]]
    # branch_price = np.where(shares == second_min_value)[0][0]

    # Probabilistic:
    # Compute the inverted probabilities based on the relative sizes of the elements
    inverted_probabilities = 1 - (shares / np.sum(shares))

    # Normalize the inverted probabilities to make them sum to 1
    normalized_probabilities = inverted_probabilities / np.sum(inverted_probabilities)

    # Sample an index using the normalized inverted probabilities
    branch_price = np.random.choice(range(len(shares)), p=normalized_probabilities)

    # elif we branch on second largest:

    # Deterministic:

    # # Find the indices of the two largest values in the shares array
    # max_indices = np.argpartition(shares, -2)[-2:]
    #
    # # Find the second largest value and its index in the original shares array
    # second_max_value = shares[max_indices[0]]
    # branch_price = np.where(shares == second_max_value)[0][0]

    # Probabilistic:

    # # Compute the probabilities based on the relative sizes of the elements
    # probabilities = shares / np.sum(shares)
    #
    # # Sample an index using the computed probabilities
    # branch_price = np.random.choice(range(len(shares)), p=probabilities)

    # print(f"We branch on {branch_price + 1}")

    return branch_price + 1


def profit_branching_price(N, R, J, p_value, omega_values, eta_values, integer_omega, integer_eta):
    profits = dict()
    for i in range(1, J):
        prof_i = 0
        for n in range(N):
            for r in range(R):
                # prof_i += integer_omega[i, n * R + r] * p_value[i - 1] # integer profits with omega
                # prof_i += integer_eta[i, n * R + r]  # integer profits with eta
                # prof_i += omega_values[i, n * R + r] * p_value[i - 1]  # fractional profit with omega
                prof_i += eta_values[i, n * R + r]  # fractional profit with eta
        profits[i] = prof_i

    branch_price = min(profits, key=profits.get)
    return branch_price


def longestEdge_branching_price(p_l, p_u):
    J = len(p_l) + 1
    bounds_difference = [p_u[i] - p_l[i] for i in range(1, J)]
    branch_price = np.argmax(bounds_difference)
    return branch_price + 1


def compute_fixed_tuples_onep(N, R, p_2_lb, p_2_ub, breakpoints, fixed):
    fixed_tuples = dict()
    for n in range(N):
        for r in range(R):
            if not fixed[n, r] == 3:
                fixed_tuples[n, r] = fixed[n, r]
            else:
                if p_2_ub <= breakpoints[n * R + r]:
                    fixed_tuples[n, r] = 2
                elif p_2_lb > breakpoints[n * R + r]:
                    fixed_tuples[n, r] = 0
                else:
                    fixed_tuples[n, r] = 3
    return fixed_tuples


def compute_fixed_tuples(N, R, p_L, p_U, exo_utility, endo_coeff, J_PSP, J_PUP, fixed,
                         forced_improvement=False, start_choices=None):
    # for each alternative, we check is it worse than all others, given its p_L and all others p_U
    # as well as checking if it is better than all others, given its p_U and all others p_L
    # also, if they are already fixed, keep them fixed, because as we saw changing the bounds cant change these outcomes
    J = 1 + J_PSP + J_PUP
    fixed_tuples = 2 * np.ones((J, N, R))
    utils_UB = np.zeros(J)
    utils_LB = np.zeros(J)
    dom_buffer = 1e-12
    # trigger = False
    for n in range(N):
        for r in range(R):
            # compute utilities
            utils_UB[0] = exo_utility[0, n * R + r]
            utils_LB[0] = exo_utility[0, n * R + r]
            for i in range(J_PSP):
                utils_UB[i + 1] = exo_utility[i + 1, n * R + r] + endo_coeff[i + 1, n * R + r] * p_L[i + 1]
                utils_LB[i + 1] = exo_utility[i + 1, n * R + r] + endo_coeff[i + 1, n * R + r] * p_U[i + 1]
            for i in range(J_PUP):
                utils_UB[i + J_PSP + 1] = exo_utility[i + J_PSP + 1, n * R + r] \
                                          + endo_coeff[i + J_PSP + 1, n * R + r] * p_L[i + J_PSP + 1]
                utils_LB[i + J_PSP + 1] = exo_utility[i + J_PSP + 1, n * R + r] \
                                          + endo_coeff[i + J_PSP + 1, n * R + r] * p_U[i + J_PSP + 1]
            # opt-out
            if not fixed[0, n, r] == 2:
                fixed_tuples[0, n, r] = fixed[0, n, r]
            else:
                # check if dominated or dominating
                if any(utils_UB[0] <= utils_LB[j] - dom_buffer for j in range(J)):
                    fixed_tuples[0, n, r] = 0
                if all(utils_LB[0] >= utils_UB[j] for j in range(J)):
                    fixed_tuples[0, n, r] = 1
            # PSPs
            for i in range(J_PSP):
                if not fixed[i + 1, n, r] == 2:
                    fixed_tuples[i + 1, n, r] = fixed[i + 1, n, r]
                else:
                    # check if dominated or dominating
                    if any(utils_UB[i + 1] <= utils_LB[j] - dom_buffer for j in range(J)):
                        fixed_tuples[i + 1, n, r] = 0
                    if all(utils_LB[i + 1] >= utils_UB[j] for j in range(J)):
                        fixed_tuples[i + 1, n, r] = 1
            # PUPs
            for i in range(J_PUP):
                if not fixed[i + J_PSP + 1, n, r] == 2:
                    fixed_tuples[i + J_PSP + 1, n, r] = fixed[i + J_PSP + 1, n, r]
                else:
                    # check if dominated or dominating
                    if any(utils_UB[i + J_PSP + 1] <= utils_LB[j] - dom_buffer for j in range(J)):
                        fixed_tuples[i + J_PSP + 1, n, r] = 0
                    if all(utils_LB[i + J_PSP + 1] >= utils_UB[j] for j in range(J)):
                        fixed_tuples[i + J_PSP + 1, n, r] = 1

            # set choices to 0 if they are dominated
            for i in range(J):
                if fixed_tuples[i, n, r] == 1:
                    for j in [j for j in range(J) if not j == i]:
                        fixed_tuples[j, n, r] = 0
                # if forced_improvement:
                #     if fixed_tuples[i, n, r] != start_choices[i, n * R + r]:
                #         trigger = True

            # # set choice to 1 if all other choices are dominated
            # if not sum(fixed_tuples[i, n, r] for i in range(J)) == 1:
            #     zero_count = 0
            #     nonzero_index = 0
            #     for j in range(J):
            #         if fixed_tuples[j, n, r] == 0:
            #             zero_count += 1
            #         else:
            #             nonzero_index = j
            #     if zero_count == J - 1:
            #         fixed_tuples[nonzero_index, n, r] = 1

    return fixed_tuples


def compute_marketShares(N, R, J, fixedTuples):
    marketShares = dict()
    for i in range(J):
        mS_i = 0
        for n in range(N):
            for r in range(R):
                mS_i += int(fixedTuples[i, n, r] == 1)
        marketShares[i] = mS_i
    return marketShares


def compute_optOuts(N, R, J, fixedTuples):
    optOuts = np.sum(fixedTuples[0, :, :])
    marketShares = dict()
    for i in range(J):
        mS_i = 0
        for n in range(N):
            for r in range(R):
                mS_i += int(fixedTuples[i, n, r] == 1)
        marketShares[i] = mS_i
    return marketShares


def compute_profits(J, p_L, p_U, marketShares):
    max_profits = dict()
    min_profits = dict()
    for i in range(1, J):
        max_profits[i] = marketShares[i] * p_U[i]
        min_profits[i] = marketShares[i] * p_L[i]
    return max_profits, min_profits


def compute_valid_tuples_onep(p_L_2, p_U_2, breakpoints, valid_tuples):
    new_valid_tuples = np.array(valid_tuples)  # Make a copy of valid_tuples

    # Create a condition mask using broadcasting
    condition_mask = (p_L_2 < breakpoints[:, 0]) & (breakpoints[:, 0] < p_U_2)
    condition_mask = condition_mask.reshape(-1)

    # Set values to zero where condition is not satisfied
    new_valid_tuples[~condition_mask] = 0

    return new_valid_tuples


def find_closest(sorted_bp, x, dir):
    # Check if index is at the beginning or end of the array
    index = np.searchsorted(sorted_bp, x)
    if index == 0:
        closest = sorted_bp[0]
    elif index == len(sorted_bp):
        closest = sorted_bp[-1]
    else:
        closest = sorted_bp[index - 1] if dir == "down" else sorted_bp[index]
    return closest


def count_contained_values(node, start_point):
    # Counts how many values of start_point are within the node's bounds
    return sum(plb <= sp <= pub for plb, sp, pub in zip(node.plb, start_point, node.pub))


def check_for_more_bounds_analytically(N, R, J, best_upper_bound, fixed, p_l, p_u, exo_utility, endo_coef):
    remove_this_node = False
    p_l[0] = 0
    p_u[0] = 0
    for n_bar in range(N):
        for r_bar in range(R):
            if sum(fixed[h, n_bar, r_bar] for h in range(J)) >= 2:
                fix_to_zero = {h: False for h in range(J)}
                for h in range(J):
                    if fixed[h, n_bar, r_bar] == 0:
                        # if its known to be dominated there is no need to act
                        fix_to_zero[h] = True
                    else:
                        # print("")
                        # print("")
                        # print(f"fixing choice of ({n_bar}, {r_bar}) to {h}")
                        # print("")
                        # first compute p_bar[h] and p_bar[i] for i != h
                        p_bar = {i: 0 for i in range(1, J)}
                        if h >= 1:
                            p_bar[h] = min((exo_utility[i, n_bar * R + r_bar] + endo_coef[i, n_bar * R + r_bar] * p_u[i]
                                            - exo_utility[h, n_bar * R + r_bar]) / endo_coef[h, n_bar * R + r_bar]
                                           for i in range(1, J))
                        for i in range(1, J):
                            if i != h:
                                p_bar[i] = (exo_utility[h, n_bar * R + r_bar]
                                            + endo_coef[h, n_bar * R + r_bar] * p_l[h]
                                            - exo_utility[i, n_bar * R + r_bar]) / endo_coef[i, n_bar * R + r_bar]
                        # now compute bounds on eta
                        eta_bounds = {(i, n * R + r): p_u[i] for i in range(1, J) for n in range(N)
                                      for r in range(R)}
                        # start with bounds on eta[:, n_bar * R + r_bar]
                        if h >= 1:
                            eta_bounds[h, n_bar * R + r_bar] = p_bar[h]
                        for i in range(1, J):
                            if i != h:
                                eta_bounds[i, n_bar * R + r_bar] = 0
                        # print("n_bar, r_bar bounds:")
                        # for i in range(1, J):
                        #     print(f"eta_bounds[{i}, {n_bar}, {r_bar}] = {eta_bounds[i, n_bar * R + r_bar]}")
                        # now lets go to the general bounds. Always only update when the bound is better than before.
                        for n in range(N):
                            for r in range(R):
                                # if n == 1 and r == 1:
                                #     print("(1, 1) bounds:")
                                #     for f in range(J):
                                #         print(f"fixed[{f}, {n}, {r}] = {fixed[f, n, r]}")
                                # start with choice h, as we know a bound already
                                if h >= 1:
                                    if fixed[h, n, r] == 0:
                                        eta_bounds[h, n * R + r] = 0
                                    else:
                                        eta_bounds[h, n * R + r] = p_bar[h]
                                # now continue to the other choices
                                for i in range(1, J):
                                    if fixed[i, n, r] == 0:
                                        eta_bounds[i, n * R + r] = 0
                                    else:
                                        # define U_bar and c_bar
                                        U_bar = {j: 0 for j in range(J)}
                                        c_bar = {j: 0 for j in range(J)}
                                        U_bar[0] = exo_utility[0, n * R + r]
                                        if exo_utility[0, n * R + r] >= 0:
                                            c_bar[0] = exo_utility[0, n * R + r]
                                        else:
                                            c_bar[0] = 0
                                        for j in range(1, J):
                                            U_bar[j] = exo_utility[j, n * R + r] + endo_coef[j, n * R + r] * p_bar[j]
                                            if exo_utility[j, n * R + r] >= 0:
                                                c_bar[j] = exo_utility[j, n * R + r]
                                            else:
                                                c_bar[j] = 0
                                        # compute the new bound for eta_inr
                                        new_bound = min((U_bar[j] - sum(c_bar[k] for k in range(J)))
                                                        / endo_coef[i, n * R + r]
                                                        for j in range(J))
                                        if new_bound < eta_bounds[i, n * R + r]:
                                            # print(f"found better bounds for ({n}, {r}), "
                                            #       f"{new_bound} < {eta_bounds[i, n * R + r]}")
                                            eta_bounds[i, n * R + r] = new_bound
                                        # else:
                                        #     if n == 1 and r == 1:
                                        #         print("new_bound = ", new_bound)
                                        #         print("old bound = ", p_u[i])
                                # if n == 1 and r == 1:
                                #     for i in range(1, J):
                                #         print(f"p_u[{i}] = {p_u[i]}")
                                #         print(f"eta_bounds[{i}, {n}, {r}] = {eta_bounds[i, n * R + r]}")
                                #     exit()
                        obj_bound = sum(-eta_bounds[i, n * R + r]
                                        for i in range(1, J) for n in range(N) for r in range(R))
                        if obj_bound > best_upper_bound:
                            fix_to_zero[h] = True
                        # else:
                        #     print(f"NOT found bounded obj! {obj_bound} < {best_upper_bound}")

                if all(fix_to_zero[h] for h in range(J)):
                    remove_this_node = True
                    return remove_this_node, fixed
                else:
                    for h in range(J):
                        if fix_to_zero[h]:
                            fixed[h, n_bar, r_bar] = 0
    return remove_this_node, fixed


def compute_analytical_eta_bounds(N, R, J, eta_bounds_matrix, mdl, fixed, p_l, p_u, exo_utility, endo_coef):
    p_l[0] = 0
    p_u[0] = 0
    eta_violation_matrix = np.zeros(shape=(J, N, R, J, N, R))
    for n_bar in range(N):
        for r_bar in range(R):
            # if an individual is fixed, we hope that we take care of it already
            if sum(fixed[h, n_bar, r_bar] for h in range(J)) >= 2:
                for h in range(J):
                    if fixed[h, n_bar, r_bar] != 0:
                        # first compute p_bar[h] and p_bar[i] for i != h
                        p_bar = {i: 0 for i in range(1, J)}
                        if h >= 1:
                            p_bar[h] = min((exo_utility[i, n_bar * R + r_bar] + endo_coef[i, n_bar * R + r_bar] * p_u[i]
                                            - exo_utility[h, n_bar * R + r_bar]) / endo_coef[h, n_bar * R + r_bar]
                                           for i in range(1, J))
                        for i in range(1, J):
                            if i != h:
                                p_bar[i] = (exo_utility[h, n_bar * R + r_bar]
                                            + endo_coef[h, n_bar * R + r_bar] * p_l[h]
                                            - exo_utility[i, n_bar * R + r_bar]) / endo_coef[i, n_bar * R + r_bar]
                        # start with bounds on eta[:, n_bar * R + r_bar]
                        if h >= 1:
                            eta_bounds_matrix[h, n_bar, r_bar, h, n_bar, r_bar] = min(p_bar[h], p_u[h],
                                                                                      eta_bounds_matrix[
                                                                                          h, n_bar, r_bar, h, n_bar,
                                                                                          r_bar])
                            if mdl.eta_vars[h, n_bar * R + r_bar].x > eta_bounds_matrix[
                                h, n_bar, r_bar, h, n_bar, r_bar]:
                                eta_violation_matrix[h, n_bar, r_bar, h, n_bar, r_bar] = 1
                        for i in range(1, J):
                            if i != h:
                                eta_bounds_matrix[i, n_bar, r_bar, h, n_bar, r_bar] = 0
                                if mdl.eta_vars[i, n_bar * R + r_bar].x > eta_bounds_matrix[
                                    i, n_bar, r_bar, h, n_bar, r_bar]:
                                    eta_violation_matrix[i, n_bar, r_bar, h, n_bar, r_bar] = 1
                        for n in range(N):
                            for r in range(R):
                                # start with choice h, as we know a bound already
                                if h >= 1:
                                    if fixed[h, n, r] == 0:
                                        eta_bounds_matrix[h, n, r, h, n_bar, r_bar] = 0
                                    else:
                                        eta_bounds_matrix[h, n, r, h, n_bar, r_bar] = min(p_bar[h], p_u[h],
                                                                                          eta_bounds_matrix[
                                                                                              h, n, r, h, n_bar, r_bar])
                                    if mdl.eta_vars[h, n * R + r].x > eta_bounds_matrix[h, n, r, h, n_bar, r_bar]:
                                        eta_violation_matrix[h, n, r, h, n_bar, r_bar] = 1
                                # now continue to the other choices
                                for i in range(1, J):
                                    if fixed[i, n, r] == 0:
                                        eta_bounds_matrix[i, n, r, h, n_bar, r_bar] = 0
                                    else:
                                        # define U_bar and c_bar
                                        U_bar = {j: 0 for j in range(J)}
                                        c_bar = {j: 0 for j in range(J)}
                                        U_bar[0] = exo_utility[0, n * R + r]
                                        if exo_utility[0, n * R + r] >= 0:
                                            c_bar[0] = exo_utility[0, n * R + r]
                                        else:
                                            c_bar[0] = 0
                                        for j in range(1, J):
                                            U_bar[j] = exo_utility[j, n * R + r] + endo_coef[j, n * R + r] * p_bar[j]
                                            if exo_utility[j, n * R + r] >= 0:
                                                c_bar[j] = exo_utility[j, n * R + r]
                                            else:
                                                c_bar[j] = 0
                                        # compute the new bound for eta_inr
                                        new_bound = min((U_bar[j] - sum(c_bar[k] for k in range(J)))
                                                        / endo_coef[i, n * R + r]
                                                        for j in range(J))
                                        eta_bounds_matrix[i, n, r, h, n_bar, r_bar] = min(new_bound, p_u[i],
                                                                                          eta_bounds_matrix[
                                                                                              i, n, r, h, n_bar, r_bar])
                                    if mdl.eta_vars[i, n * R + r].x > eta_bounds_matrix[i, n, r, h, n_bar, r_bar]:
                                        eta_violation_matrix[i, n, r, h, n_bar, r_bar] = 1
    return eta_bounds_matrix, eta_violation_matrix


def check_for_more_bounds(mc_cormick_model, N, R, J, best_upper_bound, fixed, p_l, p_u, exo_utility, endo_coef):
    inf = GRB.INFINITY
    remove_this_node = False
    for n in range(N):
        for r in range(R):
            if sum(fixed[j, n, r] for j in range(J)) >= 2:
                fix_to_zero = {i: False for i in range(J)}
                for i in range(J):
                    if fixed[i, n, r] == 0:
                        # if its known to be dominated there is no need to act
                        fix_to_zero[i] = True
                    else:
                        # test if setting it to one causes problems

                        # using inequalities

                        # if i >= 1:
                        #     mc_cormick_model.model.addConstr(float(exo_utility[0, n * R + r])
                        #                                      <= exo_utility[i, n * R + r] + endo_coef[i, n * R + r] *
                        #                                      mc_cormick_model.p_vars[i], f"fix_choice_{0}")
                        #     for k in [j for j in range(1, J) if j != i]:
                        #         mc_cormick_model.model.addConstr(exo_utility[k, n * R + r]
                        #                                          + endo_coef[k, n * R + r] *
                        #                                          mc_cormick_model.p_vars[k]
                        #                                          <= exo_utility[i, n * R + r] + endo_coef[i, n * R + r] *
                        #                                          mc_cormick_model.p_vars[i], f"fix_choice_{k}")
                        # else:
                        #     mc_cormick_model.model.addConstr(1 <= 1, f"fix_choice_{0}")
                        #     for k in [j for j in range(1, J) if j != i]:
                        #         mc_cormick_model.model.addConstr(exo_utility[k, n * R + r]
                        #                                          + endo_coef[k, n * R + r] *
                        #                                          mc_cormick_model.p_vars[k]
                        #                                          <= exo_utility[0, n * R + r], f"fix_choice_{k}")

                        # using variable bounds
                        mc_cormick_model.omega_vars[i, n * R + r].lb = 1
                        for j in range(J):
                            if j != i:
                                # reset other lower bounds to be 0 so we can set choice to 0
                                mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                # set to zeros
                                mc_cormick_model.omega_vars[j, n * R + r].ub = 0
                                if j >= 1:
                                    mc_cormick_model.eta_vars[j, n * R + r].ub = 0

                        # then solve the relaxation
                        mc_cormick_model.model.optimize()
                        try:
                            new_obj_val = mc_cormick_model.model.ObjVal
                            # compare to best known bound
                            buff = 1e-12
                            if new_obj_val >= best_upper_bound + buff:
                                fix_to_zero[i] = True
                        except AttributeError:
                            # this can happen since we consider all people at once here
                            # vs. computing extremes for individuals utilities
                            fix_to_zero[i] = True
                        finally:
                            # remove fixing constraint again

                            # c = mc_cormick_model.model.getConstrByName(f"fix_choice_{0}")
                            # mc_cormick_model.model.remove(c)
                            # for k in [j for j in range(1, J) if j != i]:
                            #     c = mc_cormick_model.model.getConstrByName(f"fix_choice_{k}")
                            #     mc_cormick_model.model.remove(c)

                            # reset bounds
                            for j in range(J):
                                # reset all bounds
                                mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                mc_cormick_model.omega_vars[j, n * R + r].ub = inf
                                if j >= 1:
                                    mc_cormick_model.eta_vars[j, n * R + r].ub = inf
                if all(fix_to_zero[i] for i in range(J)):
                    # if all choices are infeasible or inferior, remove the node
                    remove_this_node = True
                    return remove_this_node, fixed
                else:
                    for i in range(J):
                        if fix_to_zero[i]:
                            fixed[i, n, r] = 0

                    # set choice to 1 if all other choices are dominated
                    # if not sum(fixed[i, n, r] for i in range(J)) == 1:
                    #     zero_count = 0
                    #     nonzero_index = 0
                    #     for j in range(J):
                    #         if fixed[j, n, r] == 0:
                    #             zero_count += 1
                    #         else:
                    #             nonzero_index = j
                    #     if zero_count == J - 1:
                    #         fixed[nonzero_index, n, r] = 1
    return remove_this_node, fixed


def compute_breakpoint_intervals(N, R, J, p_l, p_u, exo_utility, endo_coef):
    improv_count = 0
    for n in range(N):
        for r in range(R):
            for opt_choice in range(1, J):  # ben for opt-out we cant improve the bounds can we
                # so first we set all the utilities of the alternatives to their highest
                # i.e. with the p_l, thus p_h would have to be low as well to make us switch
                # thus this is the lower bound of the bp interval
                # if on the other hand we set all utils to their lowest, with p_u, it will take less
                # incentive to switch, so the breakpoint will be at a higher price already
                # thus this is the upper bound of the bp interval
                high_utils = []
                low_utils = []
                high_utils.append(exo_utility[0, n * R + r])
                low_utils.append(exo_utility[0, n * R + r])
                for i in range(1, J):
                    high_utils.append(exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_l[i])
                    low_utils.append(exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u[i])
                # now eeh get max values of each list
                high_util_max = max(high_utils)
                low_util_max = max(low_utils)
                # now eeh compute the breakpoint with each
                bp_low = (high_util_max - exo_utility[opt_choice, n * R + r]) / endo_coef[opt_choice, n * R + r]
                bp_high = (low_util_max - exo_utility[opt_choice, n * R + r]) / endo_coef[opt_choice, n * R + r]
                # ok and now check if these bounds are better, at least for this tupel, than p_l, p_u
                if bp_high < p_u[opt_choice] or bp_low > p_l[opt_choice]:
                    print(f"for n,r = {n},{r}")
                    print(f"bp_high[{opt_choice}] = {bp_high} <? {p_u[opt_choice]} = p_u[{opt_choice}]")
                    print(f"bp_low[{opt_choice}] = {bp_low} >? {p_l[opt_choice]} = p_l[{opt_choice}]")
                    improv_count += 1
                # so bp means if price <= than that we buy. But, always? No.
                # so what it really means is if price > than that then we FOR SURE dont buy. OK.
                # in any case, its clear that any breakpoint gives a decision making interval already, 0 to bp
                # thus in our case, only the higher bp is the relevant one. Its the highest possible price at which
                # we buy the prodcut, for all other possible combinations of prices

                # so for now forget about the lower one. We use just the highest bp.

                # 1.)
                # we can at each relaxation, compute these upper bp. And then if this upper bp would be below the lb
                # we can set the fixed to 0. (maybe, likely, this is already covered in fixed computation)

                # 2.)
                # we can also have it as inequalities though
                # if p_i is > bp[n, r][i] then omega_inr = 0:

                # p_vars[i] <= bp[n, r][i] + (p_u[i]-bp[n, r][i]) * (1 − omega[i, n * R + r])
                # if omega = 0 then anything allowed, if omega = 1 then price has to be less than bp

                # these are NR(J-1) inequalities, where the coefficients of omega and the RHS change each relaxation
                # breakpoint_constraints = dict()

                # p_vars[i] <= bp[n, r][i] + (p_u[i]-bp[n, r][i]) * (1 − omega_vars[i, n * R + r])
                # p_vars[i] + (p_u[i]-bp[n, r][i]) * omega_vars[i, n * R + r] <= bp[n, r][i] + (p_u[i] - bp[n, r][i])
                # p_vars[i] + (p_u[i] - bp[n, r][i]) * omega_vars[i, n * R + r] <= p_u[i]

                # one last thing to consider: Can we compute stronger bounds by repeatedly applying this? Hmm no.
                # Because asserting these bps as real bounds would imply we want each alterantive to be chosen?
                # Simultaneoulsy? makes no sense

                # 3.) once, for any n, r, an omega is different from the optimal solution, we can add the constraint
                # that the total profit has to be at least the opt sol. That should cut away stuff... although it
                # wont't really, since the relaxation is so weak. But eeh it might help a little bit
                # ===> actually we can just always add this constraint. Nothing new here.

                # in general, lets say the opt is not in the bounds.

    print(f"{improv_count} improvements out of {N * R * (J - 1)}")
    return


def compute_fixed_based_on_bp(N, R, J, p_l, p_u, exo_utility, endo_coef, fixed):
    # literally not a single improvement over fixed
    fixed = copy.copy(fixed)
    for n in range(N):
        for r in range(R):
            for opt_choice in range(1, J):
                low_utils = [exo_utility[0, n * R + r]]
                for i in range(1, J):
                    low_utils.append(exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u[i])
                low_util_max = max(low_utils)
                bp_high = (low_util_max - exo_utility[opt_choice, n * R + r]) / endo_coef[opt_choice, n * R + r]
                if bp_high < p_l[opt_choice]:
                    if fixed[opt_choice, n, r] != 0:
                        fixed[opt_choice, n, r] = 0
    return fixed


def selected_check_for_more_bounds_new(N, R, J, fixed, p_l, p_u, exo_utility, endo_coef,
                                       omega_values, start_choices, ineq_matrix,
                                       alt=False, fixed_a=None, fixed_b=None,
                                       p_l_a=None, p_u_a=None, p_l_b=None, p_u_b=None):
    if omega_values is None:  # this is just for the root node
        omega_values = np.zeros(shape=(J, N * R))

    if ineq_matrix is None:  # also for the root node
        ineq_matrix = np.zeros(shape=(N, R, J, J))

    omega_thresh = 1

    if not alt:
        p_u[0] = 0
        p_l[0] = 0
        ineq_matrix = copy.copy(ineq_matrix)
        for n in range(N):
            for r in range(R):
                if sum(fixed[j, n, r] for j in range(J)) >= 2:  # seems fine to keep
                    for opt_choice in range(J):
                        if omega_values[opt_choice, n * R + r] <= omega_thresh:  # not sure if we need this
                            for i in range(J):
                                if opt_choice >= 1:
                                    # assume that U_h > U_i, implying the following upper bound on p_h
                                    p_h_i = (exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u[i] -
                                             exo_utility[opt_choice, n * R + r]) / endo_coef[opt_choice, n * R + r]
                                else:
                                    p_h_i = 100

                                # as well as the following lower bound on p_i
                                if i >= 1:
                                    p_i_h = (exo_utility[opt_choice, n * R + r] +
                                             endo_coef[opt_choice, n * R + r] * p_l[opt_choice] - exo_utility[
                                                 i, n * R + r]) \
                                            / endo_coef[i, n * R + r]
                                else:
                                    p_i_h = -100
                                if (p_h_i < p_l[opt_choice]) or (p_i_h > p_u[i]):
                                    # add inequalities w_h >= w_i, eta_h >= eta_i, U_h >= U_i to the relaxation
                                    ineq_matrix[n, r, opt_choice, i] = 1
    else:
        p_u_a[0] = 0
        p_u_b[0] = 0
        p_l_a[0] = 0
        p_l_b[0] = 0
        ineq_matrix_a = copy.copy(ineq_matrix)
        ineq_matrix_b = copy.copy(ineq_matrix)
        # in alt we do everything twice kinda
        for n in range(N):
            for r in range(R):
                if sum(fixed_a[j, n, r] for j in range(J)) >= 2:
                    opt_choice = 0
                    for i in range(J):
                        if start_choices[i, n * R + r] >= 1 - 1e9:
                            opt_choice = i
                    if omega_values[opt_choice, n * R + r] <= omega_thresh:  # not sure if we need this
                        for i in range(J):
                            # assume that U_h > U_i, implying the following upper bound on p_h
                            p_h_i = (exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u_a[i] -
                                     exo_utility[opt_choice, n * R + r]) / endo_coef[opt_choice, n * R + r]
                            # as well as the following lower bound on p_i
                            if i >= 1:
                                p_i_h = (exo_utility[opt_choice, n * R + r] +
                                         endo_coef[opt_choice, n * R + r] * p_l_a[opt_choice] - exo_utility[
                                             i, n * R + r]) \
                                        / endo_coef[i, n * R + r]
                            else:
                                p_i_h = -100
                            if (p_h_i < p_l_a[opt_choice]) or (p_i_h > p_u_a[i]):
                                # add inequalities w_h >= w_i, eta_h >= eta_i, U_h >= U_i to the relaxation
                                ineq_matrix_a[n, r, i] = 1
                # now do it for b
                if sum(fixed_b[j, n, r] for j in range(J)) >= 2:
                    opt_choice = 0
                    for i in range(J):
                        if start_choices[i, n * R + r] >= 1 - 1e9:
                            opt_choice = i
                    if omega_values[opt_choice, n * R + r] <= omega_thresh:  # not sure if we need this
                        for i in range(J):
                            # assume that U_h > U_i, implying the following upper bound on p_h
                            p_h_i = (exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u_b[i] -
                                     exo_utility[opt_choice, n * R + r]) / endo_coef[opt_choice, n * R + r]
                            # as well as the following upper bound on p_i
                            if i >= 1:
                                p_i_h = (exo_utility[opt_choice, n * R + r] +
                                         endo_coef[opt_choice, n * R + r] * p_l_b[opt_choice] - exo_utility[
                                             i, n * R + r]) \
                                        / endo_coef[i, n * R + r]
                            else:
                                p_i_h = -100
                            if (p_h_i < p_l_b[opt_choice]) or (p_i_h > p_u_b[i]):
                                # add inequalities w_h >= w_i, eta_h >= eta_i, U_h >= U_i to the relaxation
                                ineq_matrix_b[n, r, i] = 1

    if alt:
        return ineq_matrix_a, ineq_matrix_b
    else:
        return ineq_matrix


def selected_check_for_more_bounds(mc_cormick_model, N, R, J, best_upper_bound, fixed, p_l, p_u, exo_utility,
                                   endo_coef, omega_values, start_choices, vers1, vers2, vers3, vers4, vers5,
                                   alt=False, fixed_a=None, fixed_b=None,
                                   p_l_a=None, p_u_a=None, p_l_b=None, p_u_b=None):
    newfixed = copy.copy(fixed)
    omega_thresh = 1
    if vers4 or vers5:
        if omega_values is None:  # this is just for the root node
            omega_values = np.zeros(shape=(J, N * R))

    # Normal BnB gives [0.5881140254923818, 0.6665039062500001, 0.7084363869112116] 7.80186969595417 in 8.006178140640259 s, 500 nodes
    # Best Bound BnB gives [0.5881140254923818, 0.6665039062500001, 0.7084363869112116] 7.80186969595417 in 7.69995903968811 s, 491 nodes

    # thresh = 1 - 1/J
    # BB validcuts BnB gives [0.5881140254923818, 0.6665039062500001, 0.7084363869112116] 7.80186969595417 in 6.376427888870239 s, 339 nodes

    # thresh = 1 (meaning you apply it for each n, r since omega_val[opt_choice, :] <= 1 is always satisfied)
    # BB validcuts BnB gives [0.5881140254923818, 0.6665039062500001, 0.7084363869112116] 7.80186969595417 in 4.983373165130615 s, 307 nodes

    # but ok, even there, what you is for ALL n, r you see what happens if we force the opt choice, and if its infeasible,
    # instead we force it to be zero. I think this helps in proving the lower bound because you get to the solutions
    # that are "on the opposite side of the search space" quicker

    # now what if FOR ALL people (OR MAYBE ONLY FOR THRESH)
    # we try to set all to 1 and get infeasible? does it help even more or less?
    # call this vers5 and again with thresh you can switch between criteria

    # vers5 with thresh = 1 - 1/J
    # BB validcuts BnB gives [0.5881140254923818, 0.6665039062500001, 0.7084363869112116] 7.80186969595417 in 2.9236679077148438 s, 169 nodes

    # vers5 with thresh = 1
    # BB validcuts BnB gives [0.5881140254923818, 0.6665039062500001, 0.7084363869112116] 7.80186969595417 in 2.682610273361206 s, 127 nodes

    # and dont forget to play with alt (outside of this scope otherwise you mess up things)

    simple_feas = False
    inf = GRB.INFINITY
    remove_this_node = False
    if vers1:
        # version 1: for ALL tuples, set optimal choice to 0
        for n in range(N):
            for r in range(R):
                if sum(fixed[j, n, r] for j in range(J)) >= 2:
                    fix_opt_to_one = False
                    opt_choice = 0
                    for i in range(J):
                        if start_choices[i, n * R + r] >= 1 - 1e12:
                            opt_choice = i

                    if fixed[opt_choice, n, r] != 0:  # if its known to be dominated there is no need to act
                        # using variable bounds
                        mc_cormick_model.omega_vars[opt_choice, n * R + r].lb = 0
                        mc_cormick_model.omega_vars[opt_choice, n * R + r].ub = 0
                        for j in range(J):
                            if j != opt_choice:
                                # reset bounds so they could be 1
                                mc_cormick_model.omega_vars[j, n * R + r].ub = 1
                                mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                if j >= 1:
                                    mc_cormick_model.eta_vars[j, n * R + r].lb = 0

                        # then solve the relaxation
                        mc_cormick_model.model.optimize()
                        try:
                            new_obj_val = mc_cormick_model.model.ObjVal
                            # compare to best known bound
                            buff = 1e-12
                            if new_obj_val >= best_upper_bound + buff:
                                # print(f"for n, r = ({n}, {r}), opt_choice = {opt_choice}")
                                # print(f"fixing that to 0 gave new obj val = {new_obj_val} > {best_upper_bound}")
                                fix_opt_to_one = True
                        except AttributeError:
                            # this can happen since we consider all people at once here
                            # vs. computing extremes for individuals utilities
                            fix_opt_to_one = True
                        finally:
                            # remove fixing constraint again
                            # reset bounds
                            for j in range(J):
                                # reset all bounds
                                mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                mc_cormick_model.omega_vars[j, n * R + r].ub = inf
                                if j >= 1:
                                    mc_cormick_model.eta_vars[j, n * R + r].lb = 0
                                    mc_cormick_model.eta_vars[j, n * R + r].ub = inf
                        if fix_opt_to_one:
                            # fixed[opt_choice, n, r] = 1
                            for j in range(J):
                                if j != opt_choice:
                                    fixed[j, n, r] = 0
    elif vers2:
        # version 2: select those tuples where omega[opt_choice, :] is close to 0, and try to set it to 1
        if not alt:
            infeascuts = 0
            boundcuts = 0
            for n in range(N):
                for r in range(R):
                    if sum(fixed[j, n, r] for j in range(J)) >= 2:
                        opt_choice = 0
                        for i in range(J):
                            if start_choices[i, n * R + r] >= 1 - 1e9:
                                opt_choice = i
                        # if True:  # maybe we should check for everyone what happens if we set opt choice = 0?
                        if omega_values[opt_choice, n * R + r] <= omega_thresh:  # maybe this value should be smaller
                            fix_opt_to_zero = False
                            if fixed[opt_choice, n, r] != 0:  # if its known to be dominated there is no need to act
                                # test if setting it to one causes bad objective or even infeasibility, if yes, set it to 0
                                # set opt choice omega to 1
                                mc_cormick_model.omega_vars[opt_choice, n * R + r].lb = 1
                                for j in range(J):
                                    if j != opt_choice:
                                        # reset other lower bounds to be 0 so we can set choice to 0
                                        mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                        # set to zeros
                                        mc_cormick_model.omega_vars[j, n * R + r].ub = 0
                                        if j >= 1:
                                            mc_cormick_model.eta_vars[j, n * R + r].ub = 0

                                # then solve the relaxation
                                mc_cormick_model.model.optimize()
                                try:
                                    new_obj_val = mc_cormick_model.model.ObjVal
                                    # compare to best known bound
                                    buff = 1e-12
                                    if new_obj_val >= best_upper_bound + buff:
                                        fix_opt_to_zero = True
                                        boundcuts += 1
                                except AttributeError:  # implies infeasibility of the model
                                    # this can happen since we consider all people at once here
                                    # vs. computing extremes for individuals utilities
                                    fix_opt_to_zero = True
                                    infeascuts += 1
                                finally:
                                    # reset bounds
                                    for j in range(J):
                                        # reset all bounds
                                        mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                        if fixed[j, n, r] == 0:
                                            mc_cormick_model.omega_vars[j, n * R + r].ub = 0
                                        else:
                                            mc_cormick_model.omega_vars[j, n * R + r].ub = inf
                                        if j >= 1:
                                            mc_cormick_model.eta_vars[j, n * R + r].ub = inf
                            if fix_opt_to_zero:
                                fixed[opt_choice, n, r] = 0
        else:
            if simple_feas:
                # Set parameters for speedup
                mc_cormick_model.model.Params.presolve = 2  # Try just presolve or enhance its aggressiveness
                # mc_cormick_model.model.Params.FeasibilityTol = 1e-9  # Adjust feasibility tolerance
                # mc_cormick_model.model.Params.MIPGap = 0.9  # Set a high optimality gap, for MILP
                # mc_cormick_model.model.Params.Heuristics = 1

                # mc_cormick_model.model.Params.TimeLimit = 0.5  # Limit solve time
                # mc_cormick_model.model.Params.IterationLimit = 100  # Limit to 1000 simplex iterations

                # Set objective to a constant value
                # mc_cormick_model.model.setObjective(0, GRB.MINIMIZE)

                # infeascuts = 0
                # boundcuts = 0
                for n in range(N):
                    for r in range(R):
                        if sum(fixed_a[j, n, r] for j in range(J)) >= 2:
                            opt_choice = 0
                            for i in range(J):
                                if start_choices[i, n * R + r] >= 1 - 1e9:
                                    opt_choice = i
                            # if True:  # maybe we should check for everyone what happens if we set opt choice = 0?
                            if omega_values[
                                opt_choice, n * R + r] <= omega_thresh:  # maybe this value should be smaller
                                fix_opt_to_zero = False
                                if fixed_a[
                                    opt_choice, n, r] != 0:  # if its known to be dominated there is no need to act
                                    # test if setting it to one causes bad objective or even infeasibility, if yes, set it to 0
                                    # set opt choice omega to 1
                                    mc_cormick_model.omega_vars[opt_choice, n * R + r].lb = 1
                                    for j in range(J):
                                        if j != opt_choice:
                                            # reset other lower bounds to be 0 so we can set choice to 0
                                            mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                            # set to zeros
                                            mc_cormick_model.omega_vars[j, n * R + r].ub = 0
                                            if j >= 1:
                                                mc_cormick_model.eta_vars[j, n * R + r].ub = 0

                                    # then solve the relaxation
                                    mc_cormick_model.model.optimize()
                                    try:
                                        new_obj_val = mc_cormick_model.model.ObjVal
                                        # compare to best known bound
                                        buff = 1e-12
                                        if new_obj_val >= best_upper_bound + buff:
                                            fix_opt_to_zero = True
                                    except AttributeError:  # implies infeasibility of the model
                                        # this can happen since we consider all people at once here
                                        # vs. computing extremes for individuals utilities
                                        fix_opt_to_zero = True
                                    finally:
                                        # reset bounds
                                        for j in range(J):
                                            # reset all bounds
                                            mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                            if fixed[j, n, r] == 0:
                                                mc_cormick_model.omega_vars[j, n * R + r].ub = 0
                                            else:
                                                mc_cormick_model.omega_vars[j, n * R + r].ub = inf
                                            if j >= 1:
                                                mc_cormick_model.eta_vars[j, n * R + r].ub = inf
                                if fix_opt_to_zero:
                                    fixed_a[opt_choice, n, r] = 0
                        # now do it for b
                        if sum(fixed_b[j, n, r] for j in range(J)) >= 2:
                            opt_choice = 0
                            for i in range(J):
                                if start_choices[i, n * R + r] >= 1 - 1e9:
                                    opt_choice = i
                            # if True:  # maybe we should check for everyone what happens if we set opt choice = 0?
                            if omega_values[
                                opt_choice, n * R + r] <= omega_thresh:  # maybe this value should be smaller
                                fix_opt_to_zero = False
                                if fixed_b[
                                    opt_choice, n, r] != 0:  # if its known to be dominated there is no need to act
                                    # test if setting it to one causes bad objective or even infeasibility, if yes, set it to 0
                                    # set opt choice omega to 1
                                    mc_cormick_model.omega_vars[opt_choice, n * R + r].lb = 1
                                    for j in range(J):
                                        if j != opt_choice:
                                            # reset other lower bounds to be 0 so we can set choice to 0
                                            mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                            # set to zeros
                                            mc_cormick_model.omega_vars[j, n * R + r].ub = 0
                                            if j >= 1:
                                                mc_cormick_model.eta_vars[j, n * R + r].ub = 0

                                    # then solve the relaxation
                                    mc_cormick_model.model.optimize()
                                    try:
                                        new_obj_val = mc_cormick_model.model.ObjVal
                                        # compare to best known bound

                                        # cant do this if we remove objective

                                        buff = 1e-12
                                        if new_obj_val >= best_upper_bound + buff:
                                            fix_opt_to_zero = True

                                            # boundcuts += 1
                                    except AttributeError:  # implies infeasibility of the model
                                        # this can happen since we consider all people at once here
                                        # vs. computing extremes for individuals utilities
                                        fix_opt_to_zero = True
                                        # infeascuts += 1
                                    finally:
                                        # reset bounds
                                        for j in range(J):
                                            # reset all bounds
                                            mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                            if fixed[j, n, r] == 0:
                                                mc_cormick_model.omega_vars[j, n * R + r].ub = 0
                                            else:
                                                mc_cormick_model.omega_vars[j, n * R + r].ub = inf
                                            if j >= 1:
                                                mc_cormick_model.eta_vars[j, n * R + r].ub = inf
                                if fix_opt_to_zero:
                                    fixed_b[opt_choice, n, r] = 0
                # print("infeascuts = ", infeascuts)
                # print("boundcuts = ", boundcuts)

                # reset parameters
                mc_cormick_model.model.Params.presolve = -1  # Default is -1 (automatic)
                # mc_cormick_model.model.Params.FeasibilityTol = 1e-6  # Default feasibility tolerance
                # mc_cormick_model.model.Params.MIPGap = 1e-4  # Default optimality gap for MIP models
                # mc_cormick_model.model.Params.Heuristics = 0.05  # Default heuristic effort
                #
                # mc_cormick_model.model.Params.TimeLimit = GRB.INFINITY  # Remove any time limit
                # mc_cormick_model.model.Params.IterationLimit = GRB.INFINITY  # Remove any iteration limit

                # reset objective
                # objective = - (1 / R) * gp.quicksum(mc_cormick_model.eta_vars[i, n * R + r] for i in range(1, J)
                #                                     for n in range(N) for r in range(R))
                # mc_cormick_model.model.setObjective(objective, GRB.MINIMIZE)
            else:
                infeascuts = 0
                boundcuts = 0
                for n in range(N):
                    for r in range(R):
                        if sum(fixed_a[j, n, r] for j in range(J)) >= 2:
                            opt_choice = 0
                            for i in range(J):
                                if start_choices[i, n * R + r] >= 1 - 1e9:
                                    opt_choice = i
                            # if True:  # maybe we should check for everyone what happens if we set opt choice = 0?
                            if omega_values[
                                opt_choice, n * R + r] <= omega_thresh:  # maybe this value should be smaller
                                fix_opt_to_zero = False
                                if fixed_a[
                                    opt_choice, n, r] != 0:  # if its known to be dominated there is no need to act
                                    # test if setting it to one causes bad objective or even infeasibility, if yes, set it to 0
                                    # set opt choice omega to 1
                                    mc_cormick_model.omega_vars[opt_choice, n * R + r].lb = 1
                                    for j in range(J):
                                        if j != opt_choice:
                                            # reset other lower bounds to be 0 so we can set choice to 0
                                            mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                            # set to zeros
                                            mc_cormick_model.omega_vars[j, n * R + r].ub = 0
                                            if j >= 1:
                                                mc_cormick_model.eta_vars[j, n * R + r].ub = 0

                                    # then solve the relaxation
                                    mc_cormick_model.model.optimize()
                                    try:
                                        new_obj_val = mc_cormick_model.model.ObjVal
                                        # compare to best known bound
                                        buff = 1e-12
                                        if new_obj_val >= best_upper_bound + buff:
                                            fix_opt_to_zero = True
                                    except AttributeError:  # implies infeasibility of the model
                                        # this can happen since we consider all people at once here
                                        # vs. computing extremes for individuals utilities
                                        fix_opt_to_zero = True
                                    finally:
                                        # reset bounds
                                        for j in range(J):
                                            # reset all bounds
                                            mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                            if fixed[j, n, r] == 0:
                                                mc_cormick_model.omega_vars[j, n * R + r].ub = 0
                                            else:
                                                mc_cormick_model.omega_vars[j, n * R + r].ub = inf
                                            if j >= 1:
                                                mc_cormick_model.eta_vars[j, n * R + r].ub = inf
                                if fix_opt_to_zero:
                                    fixed_a[opt_choice, n, r] = 0
                        # now do it for b
                        if sum(fixed_b[j, n, r] for j in range(J)) >= 2:
                            opt_choice = 0
                            for i in range(J):
                                if start_choices[i, n * R + r] >= 1 - 1e9:
                                    opt_choice = i
                            # if True:  # maybe we should check for everyone what happens if we set opt choice = 0?
                            if omega_values[
                                opt_choice, n * R + r] <= omega_thresh:  # maybe this value should be smaller
                                fix_opt_to_zero = False
                                if fixed_b[
                                    opt_choice, n, r] != 0:  # if its known to be dominated there is no need to act
                                    # test if setting it to one causes bad objective or even infeasibility, if yes, set it to 0
                                    # set opt choice omega to 1
                                    mc_cormick_model.omega_vars[opt_choice, n * R + r].lb = 1
                                    for j in range(J):
                                        if j != opt_choice:
                                            # reset other lower bounds to be 0 so we can set choice to 0
                                            mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                            # set to zeros
                                            mc_cormick_model.omega_vars[j, n * R + r].ub = 0
                                            if j >= 1:
                                                mc_cormick_model.eta_vars[j, n * R + r].ub = 0

                                    # then solve the relaxation
                                    mc_cormick_model.model.optimize()
                                    try:
                                        new_obj_val = mc_cormick_model.model.ObjVal
                                        # compare to best known bound
                                        buff = 1e-12
                                        if new_obj_val >= best_upper_bound + buff:
                                            fix_opt_to_zero = True
                                            boundcuts += 1
                                    except AttributeError:  # implies infeasibility of the model
                                        # this can happen since we consider all people at once here
                                        # vs. computing extremes for individuals utilities
                                        fix_opt_to_zero = True
                                        infeascuts += 1
                                    finally:
                                        # reset bounds
                                        for j in range(J):
                                            # reset all bounds
                                            mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                            if fixed[j, n, r] == 0:
                                                mc_cormick_model.omega_vars[j, n * R + r].ub = 0
                                            else:
                                                mc_cormick_model.omega_vars[j, n * R + r].ub = inf
                                            if j >= 1:
                                                mc_cormick_model.eta_vars[j, n * R + r].ub = inf
                                if fix_opt_to_zero:
                                    fixed_b[opt_choice, n, r] = 0
                # print("infeascuts = ", infeascuts)
                # print("boundcuts = ", boundcuts)
    elif vers3:
        # version 1: for ALL tuples, set optimal choice to 0
        for n in range(N):
            for r in range(R):
                if sum(fixed[j, n, r] for j in range(J)) >= 2:
                    fix_opt_to_one = False
                    opt_choice = 0
                    for i in range(J):
                        if start_choices[i, n * R + r] >= 1 - 1e12:
                            opt_choice = i
                    if omega_values[opt_choice, n * R + r] <= omega_thresh:  # maybe this value should be smaller
                        if fixed[opt_choice, n, r] != 0:  # if its known to be dominated there is no need to act
                            # using variable bounds
                            mc_cormick_model.omega_vars[opt_choice, n * R + r].lb = 0
                            mc_cormick_model.omega_vars[opt_choice, n * R + r].ub = 0
                            for j in range(J):
                                if j != opt_choice:
                                    # reset bounds so they could be 1
                                    mc_cormick_model.omega_vars[j, n * R + r].ub = 1
                                    mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                    if j >= 1:
                                        mc_cormick_model.eta_vars[j, n * R + r].lb = 0

                            # then solve the relaxation
                            mc_cormick_model.model.optimize()
                            try:
                                new_obj_val = mc_cormick_model.model.ObjVal
                                # compare to best known bound
                                buff = 1e-12
                                if new_obj_val >= best_upper_bound + buff:
                                    # print(f"for n, r = ({n}, {r}), opt_choice = {opt_choice}")
                                    # print(f"fixing that to 0 gave new obj val = {new_obj_val} > {best_upper_bound}")
                                    fix_opt_to_one = True
                            except AttributeError:
                                # this can happen since we consider all people at once here
                                # vs. computing extremes for individuals utilities
                                fix_opt_to_one = True
                            finally:
                                # remove fixing constraint again
                                # reset bounds
                                for j in range(J):
                                    # reset all bounds
                                    mc_cormick_model.omega_vars[j, n * R + r].lb = 0
                                    mc_cormick_model.omega_vars[j, n * R + r].ub = inf
                                    if j >= 1:
                                        mc_cormick_model.eta_vars[j, n * R + r].lb = 0
                                        mc_cormick_model.eta_vars[j, n * R + r].ub = inf
                            if fix_opt_to_one:
                                # fixed[opt_choice, n, r] = 1
                                for j in range(J):
                                    if j != opt_choice:
                                        fixed[j, n, r] = 0

    elif vers4:
        p_u[0] = 0
        p_l[0] = 0
        count = 0
        # version 4: analytical: compute p_h bar and p_i bar resulting from it, if its out of bounds eh add the cut
        if not alt:
            # infeascuts = 0
            # boundcuts = 0
            for n in range(N):
                for r in range(R):
                    if sum(fixed[j, n, r] for j in range(J)) >= 2:
                        # opt_choice = 0
                        # for i in range(J):
                        #     if start_choices[i, n * R + r] >= 1 - 1e9:
                        #         opt_choice = i
                        # set_to_one = {i: False for i in range(J)}
                        for opt_choice in range(J):
                            # if True:  # maybe we should check for everyone what happens if we set opt choice = 0?
                            if omega_values[
                                opt_choice, n * R + r] <= omega_thresh:  # maybe this value should be smaller

                                # fix_opt_to_zero = False

                                fix_opt_to_one = False
                                # set_to_one[opt_choice] = False

                                if fixed[opt_choice, n, r] != 0:  # if its known to be dominated there is no need to act
                                    # see if setting it to one causes infeasibility, if yes, set it to 0

                                    # p_opt_c_upper_bound = min((exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u[i]
                                    #                           - exo_utility[opt_choice, n * R + r]) /
                                    #                           endo_coef[opt_choice, n * R + r]
                                    #                           for i in range(J))

                                    # n, r = 0, 2
                                    # {0: False, 1: True, 2: True, 3: True}

                                    if opt_choice >= 1:
                                        # p_opt_c_lower_bound = min((exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_l[i]
                                        #                            - exo_utility[opt_choice, n * R + r]) /
                                        #                           endo_coef[opt_choice, n * R + r]
                                        #                           for i in range(J))

                                        p_opt_c_lower_bound, min_index = min(
                                            (
                                                ((exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_l[i] -
                                                  exo_utility[opt_choice, n * R + r]) / endo_coef[
                                                     opt_choice, n * R + r], i)
                                                for i in range(J)
                                            ),
                                            key=lambda x: x[0]
                                        )

                                        # if p_opt_c_upper_bound < p_l[opt_choice]:
                                        # print(f"p_opt_c_lower_bound = {p_opt_c_lower_bound} > {p_u[opt_choice]} = p_u[opt_choice]")

                                        if p_opt_c_lower_bound > p_u[opt_choice]:
                                            # fix_opt_to_zero = True

                                            fix_opt_to_one = True
                                            count += 1
                                            # set_to_one[opt_choice] = True
                                            # print(f"p_opt_c_lower_bound = {p_opt_c_lower_bound} > {p_u[opt_choice]} = p_u[opt_choice]")

                                        # infeascuts += 1
                                        # for i in range(J):
                                        for i in [min_index]:
                                            if i != opt_choice and i >= 1:
                                                # p_i_lower_bound = (exo_utility[opt_choice, n * R + r] +
                                                #                    endo_coef[opt_choice, n * R + r] * p_l[opt_choice]
                                                #                    - exo_utility[i, n * R + r]) \
                                                #                   / endo_coef[i, n * R + r]

                                                p_i_upper_bound = (exo_utility[opt_choice, n * R + r] +
                                                                   endo_coef[opt_choice, n * R + r] * p_u[opt_choice]
                                                                   - exo_utility[i, n * R + r]) \
                                                                  / endo_coef[i, n * R + r]

                                                # if p_i_lower_bound > p_u[i]:
                                                if p_i_upper_bound < p_l[i]:
                                                    # fix_opt_to_zero = True
                                                    # print(
                                                    #     f"p_i_upper_bound = {p_i_upper_bound} < {p_l[i]} = p_l[i]")
                                                    fix_opt_to_one = True
                                                    # set_to_one[opt_choice] = True
                                                    # infeascuts += 1
                                                    count += 1

                                # if fix_opt_to_zero:
                                # if sum(int(set_to_one[i]) for i in range(J)) >= 3:
                                #     print(f"n, r = {n}, {r}")
                                #     print(set_to_one)
                                if fix_opt_to_one:
                                    # fixed[opt_choice, n, r] = 0
                                    for i in range(J):
                                        # fixed[opt_choice, n, r] = 1
                                        if i != opt_choice:
                                            newfixed[i, n, r] = 0
            if count > 0:
                print(f"We had {count} cuts")
            # print("infeascuts = ", infeascuts)
            # print("boundcuts = ", boundcuts)
        else:
            p_u_a[0] = 0
            p_u_b[0] = 0
            p_l_a[0] = 0
            p_l_b[0] = 0
            count = 0
            # in alt we do everything twice kinda
            # infeascuts = 0
            # boundcuts = 0
            for n in range(N):
                for r in range(R):
                    if sum(fixed_a[j, n, r] for j in range(J)) >= 2:
                        for opt_choice in range(J):
                            # if True:  # maybe we should check for everyone what happens if we set opt choice = 0?
                            if omega_values[
                                opt_choice, n * R + r] <= omega_thresh:  # maybe this value should be smaller

                                # fix_opt_to_zero = False

                                fix_opt_to_one = False
                                # set_to_one[opt_choice] = False

                                if fixed[opt_choice, n, r] != 0:  # if its known to be dominated there is no need to act
                                    # see if setting it to one causes infeasibility, if yes, set it to 0

                                    # p_opt_c_upper_bound = min((exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u[i]
                                    #                           - exo_utility[opt_choice, n * R + r]) /
                                    #                           endo_coef[opt_choice, n * R + r]
                                    #                           for i in range(J))

                                    # n, r = 0, 2
                                    # {0: False, 1: True, 2: True, 3: True}

                                    if opt_choice >= 1:
                                        # p_opt_c_lower_bound = min((exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_l[i]
                                        #                            - exo_utility[opt_choice, n * R + r]) /
                                        #                           endo_coef[opt_choice, n * R + r]
                                        #                           for i in range(J))

                                        p_opt_c_lower_bound, min_index = min(
                                            (
                                                ((exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_l_a[i] -
                                                  exo_utility[opt_choice, n * R + r]) / endo_coef[
                                                     opt_choice, n * R + r], i)
                                                for i in range(J)
                                            ),
                                            key=lambda x: x[0]
                                        )

                                        # if p_opt_c_upper_bound < p_l[opt_choice]:
                                        # print(f"p_opt_c_lower_bound = {p_opt_c_lower_bound} > {p_u[opt_choice]} = p_u[opt_choice]")

                                        if p_opt_c_lower_bound > p_u_a[opt_choice]:
                                            # fix_opt_to_zero = True

                                            fix_opt_to_one = True
                                            count += 1
                                            # set_to_one[opt_choice] = True
                                            # print(f"p_opt_c_lower_bound = {p_opt_c_lower_bound} > {p_u[opt_choice]} = p_u[opt_choice]")

                                        # infeascuts += 1
                                        # for i in range(J):
                                        for i in [min_index]:
                                            if i != opt_choice and i >= 1:
                                                # p_i_lower_bound = (exo_utility[opt_choice, n * R + r] +
                                                #                    endo_coef[opt_choice, n * R + r] * p_l[opt_choice]
                                                #                    - exo_utility[i, n * R + r]) \
                                                #                   / endo_coef[i, n * R + r]

                                                p_i_upper_bound = (exo_utility[opt_choice, n * R + r] +
                                                                   endo_coef[opt_choice, n * R + r] * p_u_a[opt_choice]
                                                                   - exo_utility[i, n * R + r]) \
                                                                  / endo_coef[i, n * R + r]

                                                # if p_i_lower_bound > p_u[i]:
                                                if p_i_upper_bound < p_l_a[i]:
                                                    # fix_opt_to_zero = True
                                                    # print(
                                                    #     f"p_i_upper_bound = {p_i_upper_bound} < {p_l[i]} = p_l[i]")
                                                    fix_opt_to_one = True
                                                    # set_to_one[opt_choice] = True
                                                    # infeascuts += 1
                                                    count += 1
                                if fix_opt_to_one:
                                    # fixed[opt_choice, n, r] = 0
                                    for i in range(J):
                                        # fixed[opt_choice, n, r] = 1
                                        if i != opt_choice:
                                            newfixed[i, n, r] = 0
                    # now do it for b
                    if sum(fixed_b[j, n, r] for j in range(J)) >= 2:
                        for opt_choice in range(J):
                            # if True:  # maybe we should check for everyone what happens if we set opt choice = 0?
                            if omega_values[
                                opt_choice, n * R + r] <= omega_thresh:  # maybe this value should be smaller

                                # fix_opt_to_zero = False

                                fix_opt_to_one = False
                                # set_to_one[opt_choice] = False

                                if fixed[opt_choice, n, r] != 0:  # if its known to be dominated there is no need to act
                                    # see if setting it to one causes infeasibility, if yes, set it to 0

                                    # p_opt_c_upper_bound = min((exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u[i]
                                    #                           - exo_utility[opt_choice, n * R + r]) /
                                    #                           endo_coef[opt_choice, n * R + r]
                                    #                           for i in range(J))

                                    # n, r = 0, 2
                                    # {0: False, 1: True, 2: True, 3: True}

                                    if opt_choice >= 1:
                                        # p_opt_c_lower_bound = min((exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_l[i]
                                        #                            - exo_utility[opt_choice, n * R + r]) /
                                        #                           endo_coef[opt_choice, n * R + r]
                                        #                           for i in range(J))

                                        p_opt_c_lower_bound, min_index = min(
                                            (
                                                ((exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_l_b[i] -
                                                  exo_utility[opt_choice, n * R + r]) / endo_coef[
                                                     opt_choice, n * R + r], i)
                                                for i in range(J)
                                            ),
                                            key=lambda x: x[0]
                                        )

                                        # if p_opt_c_upper_bound < p_l[opt_choice]:
                                        # print(f"p_opt_c_lower_bound = {p_opt_c_lower_bound} > {p_u[opt_choice]} = p_u[opt_choice]")

                                        if p_opt_c_lower_bound > p_u_b[opt_choice]:
                                            # fix_opt_to_zero = True

                                            fix_opt_to_one = True
                                            count += 1
                                            # set_to_one[opt_choice] = True
                                            # print(f"p_opt_c_lower_bound = {p_opt_c_lower_bound} > {p_u[opt_choice]} = p_u[opt_choice]")

                                        # infeascuts += 1
                                        # for i in range(J):
                                        for i in [min_index]:
                                            if i != opt_choice and i >= 1:
                                                # p_i_lower_bound = (exo_utility[opt_choice, n * R + r] +
                                                #                    endo_coef[opt_choice, n * R + r] * p_l[opt_choice]
                                                #                    - exo_utility[i, n * R + r]) \
                                                #                   / endo_coef[i, n * R + r]

                                                p_i_upper_bound = (exo_utility[opt_choice, n * R + r] +
                                                                   endo_coef[opt_choice, n * R + r] * p_u_b[opt_choice]
                                                                   - exo_utility[i, n * R + r]) \
                                                                  / endo_coef[i, n * R + r]

                                                # if p_i_lower_bound > p_u[i]:
                                                if p_i_upper_bound < p_l_b[i]:
                                                    # fix_opt_to_zero = True
                                                    # print(
                                                    #     f"p_i_upper_bound = {p_i_upper_bound} < {p_l[i]} = p_l[i]")
                                                    fix_opt_to_one = True
                                                    # set_to_one[opt_choice] = True
                                                    # infeascuts += 1
                                                    count += 1
                                if fix_opt_to_one:
                                    # fixed[opt_choice, n, r] = 0
                                    for i in range(J):
                                        # fixed[opt_choice, n, r] = 1
                                        if i != opt_choice:
                                            newfixed[i, n, r] = 0
            if count > 0:
                print(f"We had {count} cuts")
            # print("infeascuts = ", infeascuts)
            # print("boundcuts = ", boundcuts)
    elif vers5:
        p_u[0] = 0
        p_l[0] = 0
        # version 5: do stuff for all choices, not only for opt
        if not alt:
            for n in range(N):
                for r in range(R):
                    if sum(fixed[j, n, r] for j in range(J)) >= 2:
                        for opt_choice in range(J):
                            if omega_values[
                                opt_choice, n * R + r] <= omega_thresh:  # maybe this value should be smaller
                                fix_opt_to_zero = False
                                if fixed[opt_choice, n, r] != 0:
                                    if opt_choice >= 1:
                                        p_opt_c_upper_bound = min(
                                            (exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u[i]
                                             - exo_utility[opt_choice, n * R + r]) /
                                            endo_coef[opt_choice, n * R + r]
                                            for i in range(J))
                                        # print(
                                        #     f"p_opt_c_upper_bound = {p_opt_c_upper_bound}, {p_l[opt_choice]} = p_l[opt_choice]")
                                        if p_opt_c_upper_bound < p_l[opt_choice]:
                                            fix_opt_to_zero = True
                                        for i in range(J):
                                            if i != opt_choice and i >= 1:
                                                p_i_lower_bound = (exo_utility[opt_choice, n * R + r] +
                                                                   endo_coef[opt_choice, n * R + r] * p_l[opt_choice]
                                                                   - exo_utility[i, n * R + r]) \
                                                                  / endo_coef[i, n * R + r]
                                                # print(
                                                #     f"p_i_lower_bound = {p_i_lower_bound}, {p_u[i]} = p_u[i]")
                                                if p_i_lower_bound > p_u[i]:
                                                    fix_opt_to_zero = True
                                    else:
                                        for i in range(J):
                                            if i != opt_choice and i >= 1:
                                                p_i_lower_bound = (exo_utility[opt_choice, n * R + r]
                                                                   - exo_utility[i, n * R + r]) \
                                                                  / endo_coef[i, n * R + r]
                                                if p_i_lower_bound > p_u[i]:
                                                    fix_opt_to_zero = True
                                if fix_opt_to_zero:
                                    fixed[opt_choice, n, r] = 0
        else:
            p_u_a[0] = 0
            p_u_b[0] = 0
            p_l_a[0] = 0
            p_l_b[0] = 0
            # in alt we do everything twice kinda
            for n in range(N):
                for r in range(R):
                    if sum(fixed_a[j, n, r] for j in range(J)) >= 2:
                        for opt_choice in range(J):
                            if omega_values[
                                opt_choice, n * R + r] <= omega_thresh:  # maybe this value should be smaller
                                fix_opt_to_zero = False
                                if fixed_a[opt_choice, n, r] != 0:
                                    if opt_choice >= 1:
                                        p_opt_c_upper_bound = min(
                                            (exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u_a[i]
                                             - exo_utility[opt_choice, n * R + r]) /
                                            endo_coef[opt_choice, n * R + r]
                                            for i in range(J))
                                        if p_opt_c_upper_bound < p_l_a[opt_choice]:
                                            fix_opt_to_zero = True
                                        for i in range(J):
                                            if i != opt_choice and i >= 1:
                                                p_i_lower_bound = (exo_utility[opt_choice, n * R + r] +
                                                                   endo_coef[opt_choice, n * R + r] * p_l_a[opt_choice]
                                                                   - exo_utility[i, n * R + r]) \
                                                                  / endo_coef[i, n * R + r]
                                                if p_i_lower_bound > p_u_a[i]:
                                                    fix_opt_to_zero = True
                                    else:
                                        for i in range(J):
                                            if i != opt_choice and i >= 1:
                                                p_i_lower_bound = (exo_utility[opt_choice, n * R + r]
                                                                   - exo_utility[i, n * R + r]) \
                                                                  / endo_coef[i, n * R + r]
                                                if p_i_lower_bound > p_u_a[i]:
                                                    fix_opt_to_zero = True
                                                    # infeascuts += 1
                                if fix_opt_to_zero:
                                    fixed_a[opt_choice, n, r] = 0
                    # now do it for b
                    if sum(fixed_b[j, n, r] for j in range(J)) >= 2:
                        for opt_choice in range(J):
                            if opt_choice >= 1:
                                if omega_values[
                                    opt_choice, n * R + r] <= omega_thresh:  # maybe this value should be smaller
                                    fix_opt_to_zero = False
                                    if fixed_b[opt_choice, n, r] != 0:
                                        if opt_choice >= 1:
                                            p_opt_c_upper_bound = min(
                                                (exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_u_b[i]
                                                 - exo_utility[opt_choice, n * R + r]) /
                                                endo_coef[opt_choice, n * R + r]
                                                for i in range(J))
                                            if p_opt_c_upper_bound < p_l_b[opt_choice]:
                                                fix_opt_to_zero = True
                                            for i in range(J):
                                                if i != opt_choice and i >= 1:
                                                    p_i_lower_bound = (exo_utility[opt_choice, n * R + r] +
                                                                       endo_coef[opt_choice, n * R + r] * p_l_b[
                                                                           opt_choice]
                                                                       - exo_utility[i, n * R + r]) \
                                                                      / endo_coef[i, n * R + r]
                                                    if p_i_lower_bound > p_u_b[i]:
                                                        fix_opt_to_zero = True
                                        else:
                                            for i in range(J):
                                                if i != opt_choice and i >= 1:
                                                    p_i_lower_bound = (exo_utility[opt_choice, n * R + r]
                                                                       - exo_utility[i, n * R + r]) \
                                                                      / endo_coef[i, n * R + r]
                                                    if p_i_lower_bound > p_u_b[i]:
                                                        fix_opt_to_zero = True
                                    if fix_opt_to_zero:
                                        fixed_b[opt_choice, n, r] = 0
    if alt:
        return fixed_a, fixed_b
    else:
        return remove_this_node, newfixed


def update_eta_bounds_in_relaxation(eta_bounds_matrix, mdl, p_lb, p_ub, eta_viol_matrix):
    if eta_bounds_matrix is not None:
        shape = eta_bounds_matrix.shape
        J, N, R = shape[0], shape[1], shape[2]
        for h in range(1, J):
            for n_bar in range(N):
                for r_bar in range(R):
                    for i in range(1, J):
                        for n in range(N):
                            for r in range(R):
                                if eta_viol_matrix[i, n, r, h, n_bar, r_bar] == 1:
                                    # update coeff
                                    mdl.model.chgCoeff(mdl.eta_bounds_constraints[i, n, r, h, n_bar, r_bar],
                                                       mdl.omega_vars[h, n_bar * R + r_bar],
                                                       (p_ub[i] - eta_bounds_matrix[i, n, r, h, n_bar, r_bar]))
                                    # update rhs
                                    mdl.eta_bounds_constraints[i, n, r, h, n_bar, r_bar].rhs = p_ub[i]
                                else:
                                    # update coeff
                                    mdl.model.chgCoeff(mdl.eta_bounds_constraints[i, n, r, h, n_bar, r_bar],
                                                       mdl.omega_vars[h, n_bar * R + r_bar], 0)
                                    # update rhs
                                    mdl.eta_bounds_constraints[i, n, r, h, n_bar, r_bar].rhs = 10
    return


def update_breakpoint_constraints(N, R, J, p_l, p_u, mcm, exo_utility, endo_coef, pl_VIs, do_vis, addeta, optvalid,
                                  start_prices, start_choices):
    if pl_VIs:
        smallest_breakbounds_l = {i: 100 for i in range(1, J)}
        highest_breakbounds_u = {i: -100 for i in range(1, J)}
        for n in range(N):
            for r in range(R):
                # opt_choice = 0
                # for k in range(J):
                #     if start_choices[k, n * R + r] >= 1 - 1e9:
                #         opt_choice = k
                # if opt_choice >= 1:
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
                    # if p_opt_choice is > bp[n, r][opt_choice] then omega[opt_choice, n, r] = 0
                    # p_vars[i] + (p_u[i] - bp[n, r][i]) * omega_vars[i, n * R + r] <= p_u[i]
                    if do_vis:
                        # mcm.bp_constr[opt_choice, n, r, 0].rhs = bp_high
                        # mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 0], mcm.omega_vars[opt_choice, n * R + r],
                        #                    (bp_high - p_u[opt_choice]))
                        mcm.bp_constr[opt_choice, n, r, 0].rhs = p_u[opt_choice]
                        mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 0], mcm.omega_vars[opt_choice, n * R + r],
                                           (p_u[opt_choice] - bp_high))
                        mcm.bp_constr[opt_choice, n, r, 1].rhs = bp_low
                        mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 1], mcm.omega_vars[opt_choice, n * R + r],
                                           (bp_low - p_l[opt_choice]))
                        if addeta >= 1:
                            # mcm.bp_constr[opt_choice, n, r, 2].rhs = bp_high
                            # mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 2],
                            #                    mcm.omega_vars[opt_choice, n * R + r],
                            #                    (bp_high - p_u[opt_choice]))
                            mcm.bp_constr[opt_choice, n, r, 2].rhs = p_u[opt_choice]
                            if addeta == 4:
                                mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 2],
                                                   mcm.omega_vars[opt_choice, n * R + r],
                                                   (p_u[opt_choice] - bp_high))
                            elif addeta == 1:
                                mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 2],
                                                   mcm.eta_vars[opt_choice, n * R + r],
                                                   (p_u[opt_choice] - bp_high) / bp_high)
                            if addeta == 2:
                                mcm.bp_constr[opt_choice, n, r, 3].rhs = bp_low
                                mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 3],
                                                   mcm.eta_vars[opt_choice, n * R + r],
                                                   (bp_low - p_l[opt_choice]) / bp_low)
                            elif addeta == 3:
                                mcm.bp_constr[opt_choice, n, r, 3].rhs = bp_low
                                mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 3],
                                                   mcm.eta_vars[opt_choice, n * R + r],
                                                   (bp_low - p_l[opt_choice]) / p_u[opt_choice])

                    if bp_low < smallest_breakbounds_l[opt_choice]:
                        smallest_breakbounds_l[opt_choice] = bp_low
                    if bp_high > highest_breakbounds_u[opt_choice]:
                        highest_breakbounds_u[opt_choice] = bp_high

                    # if optvalid == 1:
                    #     mcm.model.chgCoeff(mcm.optvalid_constr[opt_choice, n, r],
                    #                        mcm.omega_vars[opt_choice, n * R + r],
                    #                        (start_prices[opt_choice - 1] - p_l[opt_choice] + sum(
                    #                            p_u[j] - start_prices[j - 1] for j in
                    #                            [j for j in range(1, J) if j != opt_choice])))
                    if optvalid == 2:
                        if start_choices[opt_choice, n * R + r] == 1:
                            mcm.model.chgCoeff(mcm.optvalid_constr[opt_choice, n, r],
                                               mcm.opt_z_vars[opt_choice, n * R + r],
                                               (start_prices[opt_choice - 1] - p_l[opt_choice]))
                            for j in range(1, J):
                                if j != opt_choice:
                                    mcm.model.chgCoeff(mcm.optvalid_constr[j, n, r],
                                                       mcm.opt_z_vars[j, n * R + r],
                                                       (p_u[j] - start_prices[j - 1]))
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
                    if do_vis:
                        # mcm.bp_constr[opt_choice, n, r, 0].rhs = bp_high
                        # mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 0], mcm.omega_vars[opt_choice, n * R + r],
                        #                    (bp_high - p_u[opt_choice]))
                        mcm.bp_constr[opt_choice, n, r, 0].rhs = p_u[opt_choice]
                        mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 0], mcm.omega_vars[opt_choice, n * R + r],
                                           (p_u[opt_choice] - bp_high))
                        if addeta >= 1:
                            # mcm.bp_constr[opt_choice, n, r, 2].rhs = bp_high
                            # mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 2],
                            #                    mcm.omega_vars[opt_choice, n * R + r],
                            #                    (bp_high - p_u[opt_choice]))
                            mcm.bp_constr[opt_choice, n, r, 2].rhs = p_u[opt_choice]
                            if addeta == 4:
                                mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 2],
                                                   mcm.omega_vars[opt_choice, n * R + r],
                                                   (p_u[opt_choice] - bp_high))
                            elif addeta == 1:
                                mcm.model.chgCoeff(mcm.bp_constr[opt_choice, n, r, 2],
                                                   mcm.eta_vars[opt_choice, n * R + r],
                                                   (p_u[opt_choice] - bp_high) / bp_high)
                    if bp_high > highest_breakbounds_u[opt_choice]:
                        highest_breakbounds_u[opt_choice] = bp_high
                    if optvalid == 1:
                        mcm.model.chgCoeff(mcm.optvalid_constr[opt_choice, n, r, 1],
                                           mcm.omega_vars[opt_choice, n * R + r],
                                           (start_prices[opt_choice - 1] - p_l[opt_choice] + sum(
                                               p_u[j] - start_prices[j - 1] for j in
                                               [j for j in range(1, J) if j != opt_choice])))
                    if optvalid == 2:
                        if start_choices[opt_choice, n * R + r] == 1:
                            mcm.model.chgCoeff(mcm.optvalid_constr[opt_choice, n, r],
                                               mcm.opt_z_vars[opt_choice, n * R + r],
                                               (start_prices[opt_choice - 1] - p_l[opt_choice]))
                            for j in range(1, J):
                                if j != opt_choice:
                                    mcm.model.chgCoeff(mcm.optvalid_constr[j, n, r],
                                                       mcm.opt_z_vars[j, n * R + r],
                                                       (p_u[j] - start_prices[j - 1]))
    return smallest_breakbounds_l, highest_breakbounds_u


def update_utility_ineqs(N, R, J, mcm, ineq_matrix, exo_utility, start_choices):
    # if we only do it for eeh the opt choice of each tuple, then we really only need NRJ entries
    for n in range(N):
        for r in range(R):
            for opt_choice in range(J):
                for i in range(J):
                    if i != opt_choice:
                        if ineq_matrix[n, r, opt_choice, i] == 1:
                            # lets say this implies that we should add the inequality U_h >= U_i etc to the model
                            mcm.om_v_constr[n, r, opt_choice, i].rhs = 0  # this constr is ""_h - ""_i >= -bigM
                            mcm.ut_v_constr[n, r, opt_choice, i].rhs = exo_utility[i, n * R + r] \
                                                                       - exo_utility[opt_choice, n * R + r]
                            if i >= 1 and opt_choice >= 1:
                                mcm.et_v_constr[n, r, opt_choice, i].rhs = 0
                        else:
                            mcm.om_v_constr[n, r, opt_choice, i].rhs = 100
                            mcm.ut_v_constr[n, r, opt_choice, i].rhs = 100
                            if i >= 1 and opt_choice >= 1:
                                mcm.et_v_constr[n, r, opt_choice, i].rhs = 100
    return


def compute_breakbounds(N, R, J, p_l, p_u, exo_utility, endo_coef, pl_VIs):
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


def cpp_bnb(N, R, exo_utility, endo_coef, time_limit, threads, price_lower, price_upper, branch, bendering,
            lwbing, pres, breakpoints, onep, fixed_choices, J_PSP, J_PUP, enum, start_price, timetricks, viol,
            validcuts, gapcuts, guidedenum, optcut, objcut, mcm2, caps, pl_VIs, forced_improvement, breakbounds,
            do_vis, addeta, optvalid):
    J = J_PSP + J_PUP + 1
    dates = str(datetime.datetime.now())

    dirrun = 1
    if 0.6 not in start_price:
        dirrun += 2
    if validcuts:
        dirrun += 1

    if not console:
        if bendering:
            filenamee = f"newLog_{N}_{R}_{J_PSP}_{J_PUP}_6_{dirrun}_{dates}.txt"
        else:
            filenamee = f"newLog_{N}_{R}_{J_PSP}_{J_PUP}_5_{dirrun}_{dates}.txt"
        logging.basicConfig(filename=filenamee, level=logging.INFO, format='%(message)s')
    else:
        filenamee = None

    if bendering:
        if console:
            print(f'Starting BnB Benders with: Threads = {threads}, Timetricks = {timetricks}')
        else:
            logging.info(f'Starting BnB Benders with: Threads = {threads}, Timetricks = {timetricks}')
    else:
        if console:
            print(f'Starting BnB with: Threads = {threads}')
        else:
            logging.info(f'Starting BnB with: Threads = {threads}')

    # say hello to gurobi to trigger license text (purely aesthetic)
    test = gp.Model()
    test.dispose()
    # print("initialized GRB")
    if console:
        print("initialized GRB")
    else:
        logging.info("initialized GRB")

    # [0.6004077541661854, 0.6159068569677679] - 26.83019767869515
    # Initial
    # guess
    # for price =[0.6004077541661854, 0.6159068569677679] obtained in 0.7902119159698486s

    start_time = time.time()
    if pres:
        opt_gap_percentage_tol = 1
    else:
        opt_gap_percentage_tol = 0.01
    # print("set optimality gap")
    if console:
        print("set optimality gap")
    else:
        logging.info("set optimality gap")

    branching = branch

    p_L = price_lower
    p_U = price_upper

    if onep:
        lower_bound = - N * p_U[2]
    else:
        lower_bound = - N * max(p_U[i] for i in range(1, J))

    first_guess_price = start_price  # start_price will be with all prices
    # Best Bound BnB w/ caps gives [0.675253404796914, 0.8192252466783915] 2.83791167620071 in 27.334763050079346 s, 117 nodes
    # With occs =  [13.0, 9.0, 8.0]

    if caps:
        # timeo = time.time()
        start_choices, first_guess_obj = cpp_QCLP_cap(N, R, J, exo_utility, endo_coef, time_limit, threads,
                                                      {i: start_price[i - 1] for i in range(1, J)},
                                                      {i: start_price[i - 1] for i in range(1, J)},
                                                      caps)
        print("first guess = ", first_guess_price)
        print("first guess obj ", first_guess_obj)
        occss = [sum(start_choices[i, n * R + r] for n in range(N) for r in range(R)) for i in
                 range(J_PSP + J_PUP + 1)]
        print("first guess occs = ", occss)
        # print("MC this toook ", time.time()-timeo, "seconds")
    else:
        first_guess_obj, start_choices, start_etas = compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef,
                                                                                first_guess_price,
                                                                                onep, J_PSP, J_PUP)
    best_upper_bound = first_guess_obj

    # print("set first price and first upper bound: ", first_guess_price, first_guess_obj)
    if console:
        print(f"set first price and first upper bound: {first_guess_price}, {first_guess_obj}")
    else:
        logging.info(f"set first price and first upper bound: {first_guess_price}, {first_guess_obj}")

    if not bendering:
        # mc_cormick_model = initialize_cpp_McCormick_parking(N, R, J, exo_utility, endo_coef, time_limit, threads,
        #                                                     p_U, p_L, onep)
        if optcut or objcut:
            mc_cormick_model = initialize_McCormick(N, R, J, exo_utility, endo_coef, time_limit, threads, p_U, p_L,
                                                    onep,
                                                    gapcuts, mcm2, caps, optcut, objcut, validcuts, start_choices,
                                                    first_guess_obj, pl_VIs, do_vis, addeta, optvalid, start_price,
                                                    start_etas)
        else:
            mc_cormick_model = initialize_McCormick(N, R, J, exo_utility, endo_coef, time_limit, threads, p_U, p_L,
                                                    onep,
                                                    gapcuts, mcm2, caps, optcut, objcut, validcuts, start_choices)

    BIG_omega = list()

    if caps is not None:
        fixed_choices = False
        validcuts = False

    # print("fixed Choices = ", fixed_choices)

    if fixed_choices:
        # compute first fixed tuples
        if not onep:
            fixed = {(i, n, r): 2 for i in range(J) for n in range(N) for r in range(R)}
            fixedTuples = compute_fixed_tuples(N, R, p_L, p_U, exo_utility, endo_coef, J_PSP, J_PUP, fixed,
                                               forced_improvement, start_choices)
        else:
            fixed = {(n, r): 3 for n in range(N) for r in range(R)}
            fixedTuples = compute_fixed_tuples_onep(N, R, p_L[2], p_U[2], breakpoints, fixed)
    else:
        fixedTuples = None
    # print("computed fixed choices for given bounds")
    if console:
        print("computed fixed choices for given bounds")
    else:
        logging.info("computed fixed choices for given bounds")

    # marketShares = compute_marketShares(N, R, J, fixedTuples)
    # max_profits, min_profits = compute_profits(J, p_L, p_U, marketShares)
    # profits = [max_profits, min_profits]
    # profits = sum(min_profits[i] for i in range(1, J)) / (J - 1)
    # profits = None

    if enum == 2:
        optOuts = np.count_nonzero(fixedTuples[0, :, :] == 1)
    else:
        optOuts = 0

    cutsatroot = False

    # do the cuts for the root node already
    if not bendering:
        # the hope here is that maybe its enough to do it at the root node.
        # or the first few nodes. Otherwise, its a bit ridiculous the whole thing
        # although actually, I guess you could just add a benders whatever option to it
        # aand in the benders subproblem you'd have to enforce the choice fixing
        if cutsatroot:
            # if rootcuts:
            # _, fixedTuples = check_for_more_bounds(mc_cormick_model, N, R, J,
            #                                        best_upper_bound,
            #                                        fixedTuples, p_L, p_U,
            #                                        exo_utility,
            #                                        endo_coef)
            vers1 = False
            vers2 = False
            vers3 = False
            vers4 = True
            vers5 = False
            if vers1 or vers4 or vers5:
                _, fixedTuples = selected_check_for_more_bounds(mc_cormick_model, N, R, J,
                                                                best_upper_bound,
                                                                fixedTuples, p_L,
                                                                p_U,
                                                                exo_utility,
                                                                endo_coef,
                                                                None, start_choices,
                                                                vers1, vers2, vers3, vers4, vers5
                                                                )

    if gapcuts:
        eta_bounds_matrix = np.zeros(shape=(J, N, R, J, N, R))
        # eta_bounds_matrix = compute_analytical_eta_bounds(N, R, J, eta_bounds_matrix, fixedTuples, p_L, p_U,
        #                                                   exo_utility,
        #                                                   endo_coef)
        eta_violation_matrix = np.zeros(shape=(J, N, R, J, N, R))
    else:
        eta_bounds_matrix = None
        eta_violation_matrix = None

    # if validcuts:
    #     ineq_matrix = selected_check_for_more_bounds_new(N, R, J, fixedTuples, p_L, p_U, exo_utility, endo_coef,
    #                                                      None, start_choices, None)
    # else:
    #     ineq_matrix = None

    # ineq_matrix = None

    BIG_omega.append(Node(lower_bound, p_L, p_U, fixedTuples, optOuts))

    # print("added root node to tree")
    if console:
        print("added root node to tree")
    else:
        logging.info("added root node to tree")

    node_counter = 0  # counts traversed nodes only
    rel_opt_gap = 100000
    opt_p_value = first_guess_price

    long_print = True
    gap_print = False
    stuck_counter = 0
    # death_count = 0 # death count is irrelevant, since changing the bounds can cause infeasibility when fixed has been
    # precomputed
    # print("intialized parameters")
    if console:
        print("intialized parameters")
    else:
        logging.info("intialized parameters")

    if long_print:
        if console:
            print('\033[1m' + "\n nodes  |      incumbent     |  rel. optimality gap  |  time " + '\033[0m')
        else:
            logging.info('\033[1m' + "\n nodes  |      incumbent     |  rel. optimality gap  |  time " + '\033[0m')

    cutto = False

    while ((time.time() - start_time) < time_limit and
           len(BIG_omega) > 0 and rel_opt_gap > opt_gap_percentage_tol):
        # select node with the lowest lower bound (best-first-search)
        # at the moment enumeration happens by index, so if two nodes have the same lower bound we explore them in order
        # of their index. In our case this means we explore lower prices first.
        if enum <= 1:
            if guidedenum:
                min_lb = min(node.lb for node in BIG_omega)  # Find the minimum lower bound
                nodes_with_min_lb = [node for node in BIG_omega if node.lb == min_lb]
                # Choose the node with the maximum number of start_point values within its bounds
                node = max(nodes_with_min_lb, key=lambda node: count_contained_values(node, start_price))
            else:
                node = min(BIG_omega, key=lambda _: _.lb)
        else:
            # we might want to explore alternative enumeration strategies

            # here we would, at a tie, choose the node with the lowest number of guaranteed opt-outs to branch on.
            node = min(BIG_omega, key=lambda _: (_.lb, _.optOuts))
            # node = min(BIG_omega, key=lambda _: (_.lb, _.marketShares[0]))

            # here we would, at a tie, choose the node with the highest number of guaranteed opt-outs to branch on.
            # node = min(BIG_omega, key=lambda _: (_.lb, -_.marketShares[0]))
            # node = min(BIG_omega, key=lambda _: (_.lb, -_.optOuts))

            # here we would, at a tie, choose the node with the lowest average minimal profits across all products.
            # node = min(BIG_omega, key=lambda _: (_.lb, _.profits))

            # here we would, at a tie, choose the node with the highest average minimal profits across all products.
            # node = min(BIG_omega, key=lambda _: (_.lb, -_.profits))

        # print("Treat node with lower bound = ", node.lb)

        BIG_omega.remove(node)
        node_counter += 1
        p_l = node.plb
        p_u = node.pub
        fixed = node.fixed

        # p_l = {1: 0.5364863254452219, 2: 0.7005072522987446}
        # p_u = {1: 0.5364863264452219, 2: 0.7005072622987446}

        # p_l_orig = copy.copy(p_l)
        # p_u_orig = copy.copy(p_u)

        # print(f"current node = {node_counter}, has bound {node.lb}")
        # print(f"p_l = {p_l}")
        # print(f"p_u = {p_u}")

        if False:
            eta = (branching == "eta")
            if lwbing:  # lwbing is set to 1 standard
                lwbound = node.lb
            else:
                lwbound = - N * max(p_u[i] for i in range(1, J))
            outsource = False  # (node_counter == 2) or (node_counter == 3)

            if objcut or optcut:
                subobjcut = False  # subcut can never work actually, since you do what we cant: you cant enforce
                # FOR EACH TUPLE that their revenue has to be at least this good, as there might be
                # better solutions that are distributed differently
                masobjcut = True  # up for to investigation
            else:
                subobjcut = False
                masobjcut = False

            outsidebreakbounds = True  # "True" should be easier to understand first.
            optBDstart = False  # up for to investigation -> seems to not cause damage

            mcm2 = mcm2  # up for investigation?

            if not validcuts:
                do_vis = False  # eh that makes sense

            timetricks = False  # maybe up for investigation? -> seems to not cause damage

            iteration_o, best_p_o, best_omega_o, best_eta_o, best_profit_o, gapB_o, new_p_l_o, new_p_u_o, feaso_o = cpp_bnb_benders.bnb_benders(
                N, R,
                exo_utility,
                endo_coef,
                p_l, p_u,
                eta,
                lwbound,
                pres,
                threads,
                fixed,
                onep,
                outsource,
                fixed_choices,
                J_PSP,
                J_PUP,
                filenamee,
                timetricks,
                viol,
                pl_VIs,
                subobjcut,
                masobjcut,
                best_upper_bound,
                validcuts,
                mcm2,
                breakbounds,
                do_vis,
                addeta,
                optvalid,
                start_price,
                start_choices,
                start_etas,
                outsidebreakbounds,
                optBDstart)

            # print(f"solved node {node_counter}")
            # if feaso_o:
            #     print(f"node {node_counter}, bounds:")
            #     print(new_p_l_o)
            #     print(new_p_u_o)
            #     print("BnD obj = ", best_profit_o)
            #     print("len Omega = ", len(BIG_omega))
            # else:
            #     print("")
            #     print("INF")
            #     print("")

            # smallest_breakbounds_l_o, highest_breakbounds_u_o = compute_breakbounds(N, R, J, p_L, p_U, exo_utility,
            #                                                                         endo_coef,
            #                                                                         pl_VIs)
            # print(f"BnB smallest.. = {[smallest_breakbounds_l[i] for i in range(1, J)]}")
            # print(f"BnB largest.. = {[highest_breakbounds_u[i] for i in range(1, J)]}")
            # print(f"BnD smallest.. = {[smallest_breakbounds_l_o[i] for i in range(1, J)]}")
            # print(f"BnD largest.. = {[highest_breakbounds_u_o[i] for i in range(1, J)]}")
            #
            # p_l_o = copy.copy(p_l)
            # p_u_o = copy.copy(p_u)
            # print(f"start p_l_o = {p_l_o}")
            # print(f"start p_u_o = {p_u_o}")
            # for i in range(1, J):
            #     if smallest_breakbounds_l_o[i] > p_l_o[i]:
            #         p_l_o[i] = smallest_breakbounds_l_o[i]
            #     if highest_breakbounds_u_o[i] < p_u_o[i]:
            #         p_u_o[i] = highest_breakbounds_u_o[i]
            # print(f"start p_l = {p_l}")
            # print(f"start p_u = {p_u}")
            # for i in range(1, J):
            #     if smallest_breakbounds_l[i] > p_l[i]:
            #         p_l[i] = smallest_breakbounds_l[i]
            #     if highest_breakbounds_u[i] < p_u[i]:
            #         p_u[i] = highest_breakbounds_u[i]

            # print("Breakbounds mismatch!")
            # print("BnB:")
            # print(p_l)
            # print(p_u)
            # print("BnD:")
            # print(new_p_l_o)
            # print(new_p_u_o)
            # exit()

        # if node_counter == 86 and validcuts:
        #     integer_opt_value_o, _, _ = compute_cpp_from_p_parking(J, N, R,
        #                                                            exo_utility,
        #                                                            endo_coef,
        #                                                            best_p_o, onep,
        #                                                            J_PSP,
        #                                                            J_PUP)
        #     print("")
        #     print("")
        #     print("BnD price = ", best_p_o)
        #     print("BnB int opt value = ", integer_opt_value_o)
        #     print("")
        #     print("")

        # if node_counter == 5:
        #     print("BD price = ", best_p_o)
        #     print("BD_etas:")
        #     for n in range(N):
        #         for r in range(R):
        #             for i in range(1, J):
        #                 print(f"profit[{i},{n},{r}] = ", best_eta_o[i, n * R + r])

        if bendering:
            eta = (branching == "eta")
            if lwbing:  # lwbing is set to 1 standard
                lwbound = node.lb
            else:
                lwbound = - N * max(p_u[i] for i in range(1, J))
            outsource = False  # (node_counter == 2) or (node_counter == 3)

            if objcut or optcut:
                subobjcut = False  # subcut can never work actually, since you do what we cant: you cant enforce
                # FOR EACH TUPLE that their revenue has to be at least this good, as there might be
                # better solutions that are distributed differently
                masobjcut = True  # up for to investigation
            else:
                subobjcut = False
                masobjcut = False

            outsidebreakbounds = True  # if its False, then we solve the first dual problem only after initializing the
            # the MP hmmm
            optBDstart = False  # up for to investigation -> seems to not cause damage

            mcm2 = mcm2  # up for investigation?

            if not validcuts:
                do_vis = False  # eh that makes sense

            timetricks = timetricks  # maybe up for investigation? -> seems to not cause damage

            iteration, best_p, best_omega, best_eta, best_profit, gapB, new_p_l, new_p_u, feaso = cpp_bnb_benders.bnb_benders(
                N, R,
                exo_utility,
                endo_coef,
                p_l, p_u,
                eta,
                lwbound,
                pres,
                threads,
                fixed,
                onep,
                outsource,
                fixed_choices,
                J_PSP,
                J_PUP,
                filenamee,
                timetricks,
                viol,
                pl_VIs,
                subobjcut,
                masobjcut,
                best_upper_bound,
                validcuts,
                mcm2,
                breakbounds,
                do_vis,
                addeta,
                optvalid,
                start_price,
                start_choices,
                start_etas,
                outsidebreakbounds,
                optBDstart)
            # print(f"solved node {node_counter}")
            if feaso:
                p_value = best_p
                obj_value = best_profit
                omega_values = best_omega
                eta_values = best_eta
                p_l = new_p_l
                p_u = new_p_u
                integer_opt_value, integer_omega, integer_eta = compute_cpp_from_p_parking(J, N, R,
                                                                                           exo_utility,
                                                                                           endo_coef,
                                                                                           p_value, onep,
                                                                                           J_PSP,
                                                                                           J_PUP)
                # print(f"node {node_counter}, bounds:")
                # print(p_l)
                # print(p_u)
                # print("BnD (real) obj = ", best_profit_o)
                # print("len Omega = ", len(BIG_omega))
            else:
                # if feaso_o:
                #     print("fake benders thinks its feasible. real benders does not")
                #     exit()
                integer_opt_value = 1e9
                obj_value = 1e9
                lower_bound = 1e9
                # print("")
                # print("INF")
                # print("")
        else:
            update_relaxation_bounds(N, R, J, mc_cormick_model, p_l, p_u, onep, mcm2)
            checko = False

            if validcuts:
                # update_utility_ineqs(N, R, J, mc_cormick_model, ineq_matrix, exo_utility, start_choices)
                smallest_breakbounds_l, highest_breakbounds_u = update_breakpoint_constraints(N, R, J, p_l, p_u,
                                                                                              mc_cormick_model,
                                                                                              exo_utility,
                                                                                              endo_coef,
                                                                                              pl_VIs,
                                                                                              do_vis,
                                                                                              addeta,
                                                                                              optvalid,
                                                                                              start_price,
                                                                                              start_choices)

                # smallest_breakbounds_l_o, highest_breakbounds_u_o = compute_breakbounds(N, R, J, p_L, p_U, exo_utility,
                #                                                                         endo_coef,
                #                                                                         pl_VIs)
                # print(f"BnB smallest.. = {[smallest_breakbounds_l[i] for i in range(1, J)]}")
                # print(f"BnB largest.. = {[highest_breakbounds_u[i] for i in range(1, J)]}")
                # print(f"BnD smallest.. = {[smallest_breakbounds_l_o[i] for i in range(1, J)]}")
                # print(f"BnD largest.. = {[highest_breakbounds_u_o[i] for i in range(1, J)]}")
                #
                # p_l_o = copy.copy(p_l)
                # p_u_o = copy.copy(p_u)
                # print(f"start p_l_o = {p_l_o}")
                # print(f"start p_u_o = {p_u_o}")
                # for i in range(1, J):
                #     if smallest_breakbounds_l_o[i] > p_l_o[i]:
                #         p_l_o[i] = smallest_breakbounds_l_o[i]
                #     if highest_breakbounds_u_o[i] < p_u_o[i]:
                #         p_u_o[i] = highest_breakbounds_u_o[i]
                # print(f"start p_l = {p_l}")
                # print(f"start p_u = {p_u}")
                # for i in range(1, J):
                #     if smallest_breakbounds_l[i] > p_l[i]:
                #         p_l[i] = smallest_breakbounds_l[i]
                #     if highest_breakbounds_u[i] < p_u[i]:
                #         p_u[i] = highest_breakbounds_u[i]
                # print("Breakbounds mismatch!")
                # print("BnB:")
                # print(p_l)
                # print(p_u)
                # print("BnD:")
                # print(p_l_o)
                # print(p_u_o)
                # exit()

                if breakbounds:
                    # p_L = copy.copy(p_l)
                    # p_U = copy.copy(p_u)
                    # p_L_orig = copy.copy(p_L)
                    # p_U_orig = copy.copy(p_U)
                    improvo = False
                    for i in range(1, J):
                        if smallest_breakbounds_l[i] > p_l[i]:
                            print(f"changed p_l[{i}] from {p_l[i]} to {smallest_breakbounds_l[i]}")
                            exit()
                            p_l[i] = smallest_breakbounds_l[i]
                        if highest_breakbounds_u[i] < p_u[i]:
                            p_u[i] = highest_breakbounds_u[i]
                            improvo = True
                    # if improvo:
                    #     infosable = False
                    #     for i in range(1, J_PSP + J_PUP + 1):
                    #         if p_l[i] > p_u[i]:
                    #             infosable = True
                    # if infosable:
                    #     print("Ashley,look at me (in the BnB)")
                    #     print(p_l)
                    #     print(p_u)
                    # exit()
                    if improvo:
                        update_relaxation_bounds(N, R, J, mc_cormick_model, p_l, p_u, onep, mcm2)

                    # smallest_breakbounds_l, highest_breakbounds_u = compute_breakbounds(N, R, J, p_L, p_U, exo_utility,
                    #                                                                     endo_coef,
                    #                                                                     pl_VIs)
                    # # outsidebreakbounds = False
                    # # then update the p_l and p_u
                    # improvo = False
                    # feaso = True
                    # checko = False
                    # for i in range(1, J):
                    #     if smallest_breakbounds_l[i] > p_L[i]:
                    #         print(f"changed p_l[{i}] from {p_L[i]} to {smallest_breakbounds_l[i]}")
                    #         p_L[i] = smallest_breakbounds_l[i]
                    #         improvo = True
                    #         if p_L[i] > p_U[i]:
                    #             feaso = False
                    #     if highest_breakbounds_u[i] < p_U[i]:
                    #         p_U[i] = highest_breakbounds_u[i]
                    #         improvo = True
                    #         if p_U[i] < p_L[i]:
                    #             feaso = False
                    # if improvo:
                    #     if not feaso:
                    #         print("p_L orig = ", p_L_orig)
                    #         print("p_U orig = ", p_U_orig)
                    #         print("after updates")
                    #         print("p_l = ", p_l)
                    #         print("p_u = ", p_u)
                    #         print("p_L = ", p_L)
                    #         print("p_U = ", p_U)
                    #         checko = True
            # if trigger:
            #     mc_cormick_model.objConstr.rhs = best_upper_bound - 0.001

            if fixed is not None:
                if not onep:
                    PSPs = np.zeros(J_PSP)
                    for i in range(J_PSP):
                        PSPs[i] = np.count_nonzero(fixed[i + 1, :, :] == 1)
                    PUPs = np.zeros(J_PUP)
                    for i in range(J_PUP):
                        PUPs[i] = np.count_nonzero(fixed[i + J_PSP + 1, :, :] == 1)

                    remove_obj_fixed(N, R, J, mc_cormick_model, fixed, p_l, p_u)
                else:
                    fixvals = list(fixed.values())
                    PUPs = fixvals.count(2)
                    remove_obj_fixed_onep(N, R, mc_cormick_model, fixed)

                if onep:
                    mc_cormick_model.p_vars[2].Obj = - (1 / R) * PUPs
                else:
                    for i in range(J_PSP):
                        mc_cormick_model.p_vars[i + 1].Obj = - (1 / R) * PSPs[i]
                    for i in range(J_PUP):
                        mc_cormick_model.p_vars[i + J_PSP + 1].Obj = - (1 / R) * PUPs[i]

            if gapcuts:  # and rel_opt_gap < 1:
                mc_cormick_model.model.update()  # just do this to see if it helps
                update_eta_bounds_in_relaxation(eta_bounds_matrix, mc_cormick_model, p_l, p_u, eta_violation_matrix)

            # Normal BnB gives[0.5765498676517845, 0.5353900105120707, 0.6741021082164244, 0.65]
            # 5.058596885688009 in 16.43583106994629s, 47 nodes
            # Guided BnB gives[0.5765498676517845, 0.5353900105120707, 0.6741021082164244, 0.65]
            # 5.058596885688009 in 16.05015993118286s, 47 nodes

            # without setting etas to 0:
            # Normal BnB gives[0.638862129955429, 0.5842902566303168, 0.6722968074831345, 0.6867455301834305]
            # 5.163898972697436 in 123.88431406021118s, 635 nodes
            # Guided BnB gives[0.638862129955429, 0.5842902566303168, 0.6722968074831345, 0.6867455301834305]
            # 5.163898972697436 in 83.95534610748291 s, 355 nodes

            remove_this_node = False

            # Normal BnB gives[0.638862129955429, 0.5842902566303168, 0.6722968074831345, 0.6867455301834305]
            # 5.163898972697436 in 8.873416900634766 s, 822 nodes
            # Guided BnB gives[0.638862129955429, 0.5842902566303168, 0.6722968074831345, 0.6867455301834305]
            # 5.163898972697436 in 8.217888116836548 s, 787 nodes

            if not remove_this_node:
                # timeo = time.time()
                mc_cormick_model.model.optimize()
                # print("this toook ", time.time() - timeo, "seconds")
                if onep:
                    p_value = [mc_cormick_model.p_vars[2].x]
                    obj_value = mc_cormick_model.model.ObjVal
                else:
                    if mc_cormick_model.model.status == 2:
                        # if not feaso_o:
                        #     print("hmm Benders said its not feasible, but BnB says it is")
                        #     print("original bounds: ")
                        #     print(p_l_orig)
                        #     print(p_u_orig)
                        #     print("BnB new bounds")
                        #     print(p_l)
                        #     print(p_u)
                        #     print("BnB sais it got obj value = ", mc_cormick_model.model.ObjVal)
                        #     print("Benders sais it got ", best_profit_o)
                        #     exit()
                        # if abs(mc_cormick_model.model.ObjVal - best_profit_o) >= 1e-2:
                        #     print("node", node_counter)
                        #     print("original bounds: ")
                        #     print(p_l_orig)
                        #     print(p_u_orig)
                        #     print("BnB new bounds")
                        #     print(p_l)
                        #     print(p_u)
                        #     print("BnB sais it got obj value = ", mc_cormick_model.model.ObjVal)
                        #     print("price = ", [mc_cormick_model.p_vars[i].x for i in range(1, J)])
                        #     print("Benders sais it got ", best_profit_o)
                        #     print("price = ", best_p_o)
                        #     print("fixing GRB relaxation to this price gives")
                        #     for i in range(1, J):
                        #         mc_cormick_model.model.addConstr(mc_cormick_model.p_vars[i] == best_p_o[i - 1])
                        #     # print("relaxing 0 fixes")
                        #     # for n in range(N):
                        #     #     for r in range(R):
                        #     #         for i in range(1, J):
                        #     #             mc_cormick_model.omega_vars[i, n * R + r].ub = 1
                        #     mc_cormick_model.model.optimize()
                        #     print("status = ", mc_cormick_model.model.status)
                        #     print("objval = ", mc_cormick_model.model.ObjVal)
                        #     # for n in range(N):
                        #     #     for r in range(R):
                        #     #         if any(fixed[i, n, r] == 0 for i in range(1, J)):
                        #     #             # print(f"{n}, {r} is fixed")
                        #     #             pass
                        #     #         else:
                        #     #             pass
                        #     #             # print(f"{n}, {r} is not fixed")
                        #     #         print(f"GRBprofit[{n},{r}] = ",
                        #     #               sum(mc_cormick_model.eta_vars[i, n * R + r].x for i in range(1, J)))
                        #     print("bestPrice = ", [mc_cormick_model.p_vars[i].x for i in range(1, J)])
                        #     for n in range(N):
                        #         for r in range(R):
                        #             for i in range(1, J):
                        #                 if fixed[i, n, r] == 0:
                        #                     print(f"GRBprofit[{i},{n},{r}] = ",
                        #                           mc_cormick_model.eta_vars[i, n * R + r].x)
                        #                     print(f"GRBomega[{i},{n},{r}] = ",
                        #                           mc_cormick_model.omega_vars[i, n * R + r].x, "(fixed=0)")
                        #                 else:
                        #                     print(f"GRBprofit[{i},{n},{r}] = ",
                        #                           mc_cormick_model.eta_vars[i, n * R + r].x)
                        #                     print(f"GRBomega[{i},{n},{r}] = ",
                        #                           mc_cormick_model.omega_vars[i, n * R + r].x)
                        #     print("total = ", mc_cormick_model.model.ObjVal)
                        #
                        #     exit()
                        p_value = [mc_cormick_model.p_vars[i].x for i in range(1, J)]
                        omega_values = {(i, n * R + r): mc_cormick_model.omega_vars[i, n * R + r].x
                                        for i in range(J) for n in range(N) for r in range(R)}
                        eta_values = {(i, n * R + r): mc_cormick_model.eta_vars[i, n * R + r].x
                                      for i in range(1, J) for n in range(N) for r in range(R)}
                        obj_value = mc_cormick_model.model.ObjVal

                        # print("solved node with bounds")
                        # print(p_l)
                        # print(p_u)
                        # print("new price = ", p_value)
                        # print("new price obj ", obj_value)
                        # occss = [sum(omega_values[i, n * R + r] for n in range(N) for r in range(R)) for i in
                        #          range(J_PSP + J_PUP + 1)]
                        # print("new price occss = ", occss)
                        # print("omegas: ", omega_values)

                        # print(f"node {node_counter}, bounds:")
                        # print(p_l)
                        # print(p_u)
                        # print("BnB obj = ", obj_value)
                        # print("len Omega = ", len(BIG_omega))

                        # if node_counter == 5:
                        #     print("BnB price = ", p_value)
                        #     print("BnB_etas:")
                        #     for n in range(N):
                        #         for r in range(R):
                        #             for i in range(1, J):
                        #                 print(f"profit[{i},{n},{r}] = ", eta_values[i, n * R + r])

                        # p_value_o = best_p_o
                        # obj_value_o = best_profit_o

                        # if p_value != p_value_o:
                        #     print("BnB p value = ", p_value)
                        #     print("BD p value = ", p_value_o)
                        #     exit()
                        # if obj_value != obj_value_o:
                        #     print("BnB obj_value = ", obj_value)
                        #     print("BD obj_value = ", obj_value_o)
                        #     exit()

                        # integero = True
                        # for i in range(J):
                        #     for n in range(N):
                        #         for r in range(R):
                        #             print(f"omega_values[{i}, {n * R + r}] = ", omega_values[i, n * R + r])
                        #             if 1e-9 < omega_values[i, n * R + r] < 1 - 1e-9:
                        #                 integero = False
                        #                 break
                        #
                        # print("rel sol is integer = ", integero)

                        # print("solving the relax gives occs =",
                        #       [sum(omega_values[i, n * R + r] for n in range(N) for r in range(R)) for i in range(J_PSP + J_PUP + 1)])
                        #
                        if caps is None:
                            integer_opt_value, integer_omega, integer_eta = compute_cpp_from_p_parking(J, N, R,
                                                                                                       exo_utility,
                                                                                                       endo_coef,
                                                                                                       p_value, onep,
                                                                                                       J_PSP,
                                                                                                       J_PUP)
                            # if node_counter == 86 and validcuts:
                            #     print("")
                            #     print("")
                            #     print("BnB price = ", p_value)
                            #     print("BnB int opt value = ", integer_opt_value)
                            #     print("")
                            #     print("")
                            #     exit()
                        else:
                            # _, integer_opt_value = compute_obj_value_caps(J, N, R, exo_utility, endo_coef, p_value,
                            #                                               J_PSP, J_PUP,
                            #                                               caps)

                            # print("solve the MILP with bounds ")
                            # p_ll = {1: 0.6999999988824128, 2: 0.766543280892074}
                            # p_uu = {1: 0.6999999988824128, 2: 0.766543280892074}
                            # # p_uu = {1: 0.6999999989755451, 2: 0.7665432810783386}
                            # print(p_ll)
                            # print(p_uu)
                            # this_om, this = solve_MILP_cap(N, R, J, exo_utility, endo_coef, time_limit,
                            #                                threads, p_value, p_ll, p_uu,
                            #                                caps, binar=False)
                            # integero = True
                            # for i in range(J):
                            #     for n in range(N):
                            #         for r in range(R):
                            #             print(f"omega_values[{i}, {n * R + r}] = ", this_om[i, n * R + r])
                            #             if 1e-9 < this_om[i, n * R + r] < 1 - 1e-9:
                            #                 integero = False
                            #                 break
                            #
                            # print("rel sol is integer = ", integero)
                            # exit()

                            # opt_or, integer_opt_value = solve_MILP_cap(N, R, J, exo_utility, endo_coef, time_limit,
                            #                                            threads, p_value, None, None,
                            #                                            caps, binar=True)
                            # opt_or_cont, integer_opt_value_cont = solve_MILP_cap(N, R, J, exo_utility, endo_coef,
                            #                                                      time_limit,
                            #                                                      threads, p_value, None, None,
                            #                                                      caps, binar=False)
                            # print(opt_or_cont)
                            # timeo = time.time()
                            int_omegas, integer_opt_value = cpp_QCLP_cap(N, R, J, exo_utility, endo_coef,
                                                                         time_limit, threads,
                                                                         {i: p_value[i-1] for i in range(1, J)},
                                                                         {i: p_value[i-1] for i in range(1, J)},
                                                                         caps)
                            # print("new price obj integer", obj_value)
                            # occss = [sum(start_choices[i, n * R + r] for n in range(N) for r in range(R)) for i in
                            #          range(J_PSP + J_PUP + 1)]
                            # print("new price integer occss = ", occss)
                            # print("MC this toook ", time.time() - timeo, "seconds")
                            # print("QCLP omegas")
                            # print(opt_or_qclp)
                            # occss = [sum(opt_or[i, n * R + r] for n in range(N) for r in range(R)) for i in
                            #          range(J_PSP + J_PUP + 1)]
                            # print("MCM int occs = ", occss)
                            # print("with obj value = ", integer_opt_value)
                            # print("solvnig using cpp_milp_cap")
                            # cpp_MILP_cap(time_limit, threads, p_L, p_U, N, R, J, exo_utility, endo_coef,
                            #              J_PSP, J_PUP, caps, p_value)

                            # print("current price bounds:")
                            # print(p_l)
                            # print(p_u)
                            # print("current rel opt value = ", obj_value)
                            # print("current int opt value = ", integer_opt_value)
                            # print("current int opt value w relaxaed Omega = ", integer_opt_value_cont)
                            # print("current QCLP opt value = ", -integer_opt_value_qclp)
                            # print("gap", ((obj_value - integer_opt_value) / obj_value) * 100, "%")
                            # print("current best int opt value = ", best_upper_bound)
                    else:
                        # print("")
                        # print("INF")
                        # print("")
                        # if infosable:
                        #     print("because of bounds! wsh")
                        #     exit()
                        if checko:
                            print("eeh status is ", mc_cormick_model.model.status)
                            print("we just ignored it")
                        integer_opt_value = 1e9
                        obj_value = 1e9
                        lower_bound = 1e9
            else:
                integer_opt_value = 1e9
                obj_value = 1e9
                lower_bound = 1e9

        lower_bound = obj_value

        # consider this: this pruning only happens every time we eeh find a new best feasible solution
        # this will thus never really occurr will it, if the first feasible sol is the optimum
        if integer_opt_value < best_upper_bound:
            opt_p_value = p_value
            best_upper_bound = integer_opt_value
            BIG_omega = [node for node in BIG_omega if not node.lb >= best_upper_bound]
            if objcut and not bendering:
                mc_cormick_model.objConstr.rhs = best_upper_bound

        # the true pruning is happening here I guess, where we only keep branching if the obj value is better than the
        # the best bound. So IT IS happening in the right way. Hm.
        if obj_value < best_upper_bound:
            if not onep:
                if branching == "eta":
                    branching_price = eta_branching_price(N, R, J, p_value, omega_values, eta_values)
                    # iteration_o, best_p_o, best_omega_o, best_eta_o, best_profit_o, gapB_o, new_p_l_o, new_p_u_o, feaso_o
                    # branching_price_o = eta_branching_price(N, R, J, best_p_o, best_omega_o, best_eta_o)
                    # if branching_price != branching_price_o:
                    #     print("Branching mismatch!")
                    #     print("BnB")
                    #     print(f"p = {p_value}")
                    #     # print(f"omega = {omega_values}")
                    #     # print(f"eta = {eta_values}")
                    #     print(f"best profit = {obj_value}")
                    #     print(f"Branch = {branching_price}")
                    #     print("BnD")
                    #     print(f"p = {best_p_o}")
                    #     # print(best_omega_o)
                    #     # print(best_eta_o)
                    #     print(f"best profit = {best_profit_o}")
                    #     print(f"Branch = {branching_price_o}")
                    #
                    #     exit()
                    # print(f"branching on {branching_price}")
                elif branching == "guided":
                    # branching_price = guided_branching_price_new2(N, R, J, p_value, start_price, start_choices,
                    #                                               start_etas, p_l, p_u, omega_values, eta_values)
                    branching_price = guided_branching_price(J, p_value, start_price, p_l, p_u)
                    # print("start_price = ", start_price)
                    # print("bounds:")
                    # for i in range(1, J):
                    #     print(p_l[i], p_u[i])
                    # print("thus we branch on ", branching_price)
                    if branching_price == 0:
                        branching_price = eta_branching_price(N, R, J, p_value, omega_values, eta_values)
                elif branching == "longestEdge":
                    branching_price = longestEdge_branching_price(p_l, p_u)
                elif branching == "marketShare":
                    branching_price = marketShare_branching_price(J, integer_omega)
                elif branching == "profit":
                    branching_price = profit_branching_price(N, R, J, p_value, omega_values, eta_values,
                                                             integer_omega, integer_eta)

            # if node_counter == 5:
            # print(f"branching on {branching_price}")

            p_l_new_a = dict()
            p_u_new_a = dict()
            p_l_new_b = dict()
            p_u_new_b = dict()

            if onep:
                p_l_new_a = {1: 0.6, 2: p_l[2]}
                p_u_new_a = {1: 0.6, 2: (p_l[2] + p_u[2]) / 2}
                p_l_new_b = {1: 0.6, 2: (p_l[2] + p_u[2]) / 2}
                p_u_new_b = {1: 0.6, 2: p_u[2]}
            else:
                for i in range(1, J):
                    if i == branching_price:  # branching price must be in 1, ..., J indexing
                        if enum == 1:
                            # changing the order
                            p_l_new_a[i] = (p_l[i] + p_u[i]) / 2
                            p_u_new_a[i] = p_u[i]
                            p_l_new_b[i] = p_l[i]
                            p_u_new_b[i] = (p_l[i] + p_u[i]) / 2
                        else:
                            p_l_new_a[i] = p_l[i]
                            p_u_new_a[i] = (p_l[i] + p_u[i]) / 2
                            p_l_new_b[i] = (p_l[i] + p_u[i]) / 2
                            p_u_new_b[i] = p_u[i]

                    else:
                        p_l_new_a[i] = p_l[i]
                        p_u_new_a[i] = p_u[i]
                        p_l_new_b[i] = p_l[i]
                        p_u_new_b[i] = p_u[i]

            remove_node_a = False
            remove_node_b = False
            if onep:
                if fixed_choices:
                    fixedTuples_a = compute_fixed_tuples_onep(N, R, p_l_new_a[2], p_u_new_a[2], breakpoints, fixed)
                    fixedTuples_b = compute_fixed_tuples_onep(N, R, p_l_new_b[2], p_u_new_b[2], breakpoints, fixed)
                else:
                    fixedTuples_a = fixed
                    fixedTuples_b = fixed
            else:
                if fixed_choices:
                    # if not validcuts:
                    #     fixedTuples_a = compute_fixed_tuples(N, R, p_l_new_a, p_u_new_a, exo_utility, endo_coef,
                    #                                          J_PSP, J_PUP, fixed)
                    #     fixedTuples_b = compute_fixed_tuples(N, R, p_l_new_b, p_u_new_b, exo_utility, endo_coef,
                    #                                          J_PSP, J_PUP, fixed)
                    # else:
                    #     fixedTuples_a = fixed
                    #     fixedTuples_b = fixed

                    fixedTuples_a = compute_fixed_tuples(N, R, p_l_new_a, p_u_new_a, exo_utility, endo_coef,
                                                         J_PSP, J_PUP, fixed, forced_improvement,
                                                         start_choices)
                    fixedTuples_b = compute_fixed_tuples(N, R, p_l_new_b, p_u_new_b, exo_utility, endo_coef,
                                                         J_PSP, J_PUP, fixed, forced_improvement,
                                                         start_choices)

                    # if validcuts:
                    #     cutto = True
                    # if not cutto:
                    #     if rel_opt_gap < 1:
                    #         cutto = True
                    cutto = False
                    #
                    # lev = 1
                    # node_thresh = sum(2 ** k for k in range(lev + 1))

                    # if max(p_u_new_a[i] - p_l_new_a[i] for i in range(1, J)) >= thresh:
                    # if node_counter <= node_thresh:
                    # if cutto or (rootcuts and node_counter <= node_thresh):
                    if cutto:
                        vers1 = False  # maybe the other fixed things.. make this impossible? hmm
                        vers2 = False
                        vers3 = False
                        vers4 = False
                        vers5 = False
                        alt = False

                        if not alt:
                            # if not (vers4 or vers5):
                            #     update_relaxation_bounds(N, R, J, mc_cormick_model, p_l_new_a, p_u_new_a, onep, mcm2)
                            # remove_node_a, fixedTuples_a = check_for_more_bounds(mc_cormick_model, N, R, J,
                            #                                                      best_upper_bound,
                            #                                                      fixedTuples_a, p_l_new_a, p_u_new_a,
                            #                                                      exo_utility,
                            #                                                      endo_coef)
                            # remove_node_a, fixedTuples_a = selected_check_for_more_bounds(mc_cormick_model, N, R, J,
                            #                                                               best_upper_bound,
                            #                                                               fixedTuples_a, p_l_new_a,
                            #                                                               p_u_new_a,
                            #                                                               exo_utility,
                            #                                                               endo_coef,
                            #                                                               omega_values, start_choices,
                            #                                                               vers1, vers2, vers3, vers4,
                            #                                                               vers5)

                            # ineq_matrix_a = selected_check_for_more_bounds_new(N, R, J, fixedTuples_a, p_l_new_a,
                            #                                                    p_u_new_a, exo_utility, endo_coef,
                            #                                                    omega_values, start_choices, ineq_matrix,
                            #                                                    alt=False, fixed_a=None, fixed_b=None,
                            #                                                    p_l_a=None, p_u_a=None, p_l_b=None,
                            #                                                    p_u_b=None)

                            # if not (vers4 or vers5):
                            #     update_relaxation_bounds(N, R, J, mc_cormick_model, p_l_new_b, p_u_new_b, onep, mcm2)
                            # remove_node_b, fixedTuples_b = check_for_more_bounds(mc_cormick_model, N, R, J,
                            #                                                      best_upper_bound,
                            #                                                      fixedTuples_b, p_l_new_b, p_u_new_b,
                            #                                                      exo_utility,
                            #                                                      endo_coef)
                            # remove_node_b, fixedTuples_b = selected_check_for_more_bounds(mc_cormick_model, N, R, J,
                            #                                                               best_upper_bound,
                            #                                                               fixedTuples_b, p_l_new_b,
                            #                                                               p_u_new_b,
                            #                                                               exo_utility,
                            #                                                               endo_coef,
                            #                                                               omega_values, start_choices,
                            #                                                               vers1, vers2, vers3, vers4,
                            #                                                               vers5)

                            # ineq_matrix_b = selected_check_for_more_bounds_new(N, R, J, fixedTuples_b, p_l_new_b,
                            #                                                    p_u_new_b, exo_utility, endo_coef,
                            #                                                    omega_values, start_choices, ineq_matrix,
                            #                                                    alt=False, fixed_a=None, fixed_b=None,
                            #                                                    p_l_a=None, p_u_a=None, p_l_b=None,
                            #                                                    p_u_b=None)

                            # fixedTuples_a = compute_fixed_based_on_bp(N, R, J, p_l_new_a, p_u_new_a, exo_utility,
                            #                                           endo_coef, fixedTuples_a)
                            # fixedTuples_b = compute_fixed_based_on_bp(N, R, J, p_l_new_b, p_u_new_b, exo_utility,
                            #                                           endo_coef, fixedTuples_b)
                            pass

                        # ALT:
                        else:
                            # fixedTuples_a, fixedTuples_b = selected_check_for_more_bounds(mc_cormick_model, N, R, J,
                            #                                                               best_upper_bound,
                            #                                                               fixedTuples_a, p_l_new_a,
                            #                                                               p_u_new_a,
                            #                                                               exo_utility,
                            #                                                               endo_coef,
                            #                                                               omega_values,
                            #                                                               start_choices,
                            #                                                               vers1, vers2, vers3, vers4,
                            #                                                               vers5,
                            #                                                               alt=alt,
                            #                                                               fixed_a=fixedTuples_a,
                            #                                                               fixed_b=fixedTuples_b,
                            #                                                               p_l_a=p_l_new_a,
                            #                                                               p_u_a=p_u_new_a,
                            #                                                               p_l_b=p_l_new_b,
                            #                                                               p_u_b=p_u_new_b)

                            # ineq_matrix_a, ineq_matrix_b = selected_check_for_more_bounds_new(N, R, J, fixedTuples_a,
                            #                                                                   p_l_new_a,
                            #                                                                   p_u_new_a, exo_utility,
                            #                                                                   endo_coef,
                            #                                                                   omega_values,
                            #                                                                   start_choices,
                            #                                                                   ineq_matrix,
                            #                                                                   alt=True,
                            #                                                                   fixed_a=fixedTuples_a,
                            #                                                                   fixed_b=fixedTuples_b,
                            #                                                                   p_l_a=p_l_new_a,
                            #                                                                   p_u_a=p_u_new_a,
                            #                                                                   p_l_b=p_l_new_b,
                            #                                                                   p_u_b=p_u_new_b)
                            pass
                    remove_node_a = False
                    remove_node_b = False

                    # analytical bounds
                    # if cutto or (rootcuts and node_counter <= node_thresh):
                    if gapcuts:
                        # you cant really do that, since you're manipulating the same relaxation with different
                        # bound assumptions
                        # instead I guess we should return the bounds, check which are violated, and then add these
                        # cuts to the relaxation.

                        eta_bounds_matrix, eta_violation_matrix = compute_analytical_eta_bounds(N, R, J,
                                                                                                eta_bounds_matrix,
                                                                                                mc_cormick_model,
                                                                                                fixedTuples_a,
                                                                                                p_l_new_a,
                                                                                                p_u_new_a,
                                                                                                exo_utility, endo_coef)

                        # Or for now just add all cuts?
                        # eta_bounds_matrix_a = compute_analytical_eta_bounds(N, R, J, eta_bounds_matrix, fixedTuples_a,
                        #                                                     p_l_new_a,
                        #                                                     p_u_new_a,
                        #                                                     exo_utility, endo_coef)
                        #
                        # eta_bounds_matrix_b = compute_analytical_eta_bounds(N, R, J, eta_bounds_matrix, fixedTuples_b,
                        #                                                     p_l_new_b,
                        #                                                     p_u_new_b,
                        #                                                     exo_utility, endo_coef)

                        # here both child nodes get the bounds from the parent nodes, together with another
                        # matrix that says wether or not to activate the constraints
                        # eta_bounds_matrix_a = compute_analytical_eta_bounds(N, R, J, eta_bounds_matrix, fixed,
                        #                                                     p_l,
                        #                                                     p_u,
                        #                                                     exo_utility, endo_coef)
                        # eta_bounds_matrix_b = eta_bounds_matrix_a

                        # mc_cormick_model = add_analytical_eta_bounds(N, R, J, mc_cormick_model, fixedTuples_a,
                        #                                                 p_l_new_a,p_u_new_a, exo_utility, endo_coef)
                        # mc_cormick_model = add_analytical_eta_bounds(N, R, J, mc_cormick_model, fixedTuples_b,
                        #                                                 p_l_new_b, p_u_new_b, exo_utility, endo_coef)

                        # remove_node_a, fixedTuples_a = check_for_more_bounds_analytically(N, R, J,
                        #                                                                   best_upper_bound,
                        #                                                                   fixedTuples_a,
                        #                                                                   p_l_new_a,
                        #                                                                   p_u_new_a,
                        #                                                                   exo_utility,
                        #                                                                   endo_coef)
                        # remove_node_b, fixedTuples_b = check_for_more_bounds_analytically(N, R, J,
                        #                                                                   best_upper_bound,
                        #                                                                   fixedTuples_b,
                        #                                                                   p_l_new_b,
                        #                                                                   p_u_new_b,
                        #                                                                   exo_utility,
                        #                                                                   endo_coef)
                    else:
                        eta_bounds_matrix_a = eta_bounds_matrix
                        eta_bounds_matrix_b = eta_bounds_matrix
                else:
                    fixedTuples_a = fixed
                    fixedTuples_b = fixed

            # marketShares_a = compute_marketShares(N, R, J, fixedTuples_a)
            # max_profits_a, min_profits_a = compute_profits(J, p_l_new_a, p_u_new_a, marketShares_a)
            # profits_a = [max_profits_a, min_profits_a]
            # profits_a = sum(min_profits_a[i] for i in range(1, J)) / (J - 1)
            # profits_a = None

            if not remove_node_a:
                if enum == 2:
                    optOuts_a = np.count_nonzero(fixedTuples_a[0, :, :] == 1)
                else:
                    optOuts_a = 0
                if obj_value < integer_opt_value:
                    BIG_omega.append(
                        Node(lower_bound, p_l_new_a, p_u_new_a, fixedTuples_a, optOuts_a))
                    # print("adding Node a ", lower_bound, p_l_new_a, p_u_new_a)

                # if start_price == [0] * (J-1):
                #     BIG_omega.append(Node(lower_bound, p_l_new_a, p_u_new_a, fixedTuples_a, optOuts_a, eta_bounds_matrix,
                #                           eta_violation_matrix))
                # else:
                #     # if we use a BHA start, check if all p_l are higher or all p_u are lower
                #
                #     BIG_omega.append(
                #         Node(lower_bound, p_l_new_a, p_u_new_a, fixedTuples_a, optOuts_a, eta_bounds_matrix,
                #              eta_violation_matrix))

            # marketShares_b = compute_marketShares(N, R, J, fixedTuples_b)
            # max_profits_b, min_profits_b = compute_profits(J, p_l_new_b, p_u_new_b, marketShares_b)
            # profits_b = [max_profits_b, min_profits_b]
            # profits_b = sum(min_profits_b[i] for i in range(1, J)) / (J - 1)
            # profits_b = None

            if not remove_node_b:
                if enum == 2:
                    optOuts_b = np.count_nonzero(fixedTuples_b[0, :, :] == 1)
                else:
                    optOuts_b = 0
                if obj_value < integer_opt_value:
                    BIG_omega.append(
                        Node(lower_bound, p_l_new_b, p_u_new_b, fixedTuples_b, optOuts_b))
                    # print("adding Node b ", lower_bound, p_l_new_b, p_u_new_b)

        if len(BIG_omega) > 0:
            rel_opt_gap_new = relative_optimality_gap(BIG_omega, best_upper_bound)
            # print("new gap = ", rel_opt_gap_new)
            if rel_opt_gap_new < rel_opt_gap:
                rel_opt_gap = rel_opt_gap_new
                stuck_counter = 0
                if gap_print:  # and (node_counter % 20 == 0)
                    if console:
                        print("   %s   |      %s      |          %s         |   %s   " % (
                            node_counter, -round(best_upper_bound, 3), round(rel_opt_gap, 2),
                            round(time.time() - start_time, 2)))
                    else:
                        logging.info("   %s   |      %s      |          %s         |   %s   " % (
                            node_counter, -round(best_upper_bound, 3), round(rel_opt_gap, 2),
                            round(time.time() - start_time, 2)))
            else:
                stuck_counter += 1
                if stuck_counter == 50:
                    if not bendering:
                        mc_cormick_model.model.setParam('NumericFocus', 3)
                        mc_cormick_model.model.setParam('FeasibilityTol', 1e-9)
                        mc_cormick_model.model.setParam('OptimalityTol', 1e-9)
            if long_print:
                if node_counter % 1 == 0:
                    if console:
                        print("   %s   |      %s      |          %s         |   %s   " % (
                            node_counter, -round(best_upper_bound, 3), round(rel_opt_gap, 2),
                            round(time.time() - start_time, 2)))
                    else:
                        logging.info("   %s   |      %s      |          %s         |   %s   " % (
                            node_counter, -round(best_upper_bound, 3), round(rel_opt_gap, 2),
                            round(time.time() - start_time, 2)))

    solve_time = time.time() - start_time
    if len(BIG_omega) > 0:
        gap = relative_optimality_gap(BIG_omega, best_upper_bound)
    else:
        gap = 0
    if onep:
        opt_p_value = [0.6, opt_p_value[0]]

    if console:
        print(f"N = {N}, R = {R}, J = {J}, Total time = {solve_time}, Iterations = {node_counter}, "
              f"Best Price = {opt_p_value}, objective = {-best_upper_bound}, Gap = {gap}%")
    else:
        logging.info(f"N = {N}, R = {R}, J = {J}, Total time = {solve_time}, Iterations = {node_counter}, "
                     f"Best Price = {opt_p_value}, objective = {-best_upper_bound}, Gap = {gap}%")

    if caps is None:
        pass
        # integer_opt_value, integer_opt_omega, integer_opt_eta = compute_cpp_from_p_parking(J, N, R, exo_utility,
        #                                                                                    endo_coef,
        #                                                                                    opt_p_value, onep, J_PSP,
        #                                                                                    J_PUP)
        # occss = [sum(integer_opt_omega[i, n * R + r] for n in range(N) for r in range(R)) for i in
        #          range(J_PSP + J_PUP + 1)]
        # _, best_choices, _ = compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef, opt_p_value, onep,
        #                                                 J_PSP,
        #                                                 J_PUP)
        # occss = [sum(best_choices[i, n * R + r] for n in range(N) for r in range(R)) for i in range(J_PSP + J_PUP + 1)]
        # for i in range(1, J):
        #     print(f"{i} is chosen by {sum(best_choices[i, n * R + r] for n in range(N) for r in range(R))}")
        occss = None
    else:
        # occso, integer_opt_value = compute_obj_value_caps(J, N, R, exo_utility, endo_coef, opt_p_value,
        #                                                   J_PSP, J_PUP,
        #                                                   caps)
        best_choices, integer_opt_value = cpp_QCLP_cap(N, R, J, exo_utility, endo_coef, time_limit, threads,
                                                       {i: opt_p_value[i - 1] for i in range(1, J)},
                                                       {i: opt_p_value[i - 1] for i in range(1, J)},
                                                       caps)
        for i in range(1, J):
            print(f"{i} is chosen by {sum(best_choices[i, n * R + r] for n in range(N) for r in range(R))}")

        occss = [sum(best_choices[i, n * R + r] for n in range(N) for r in range(R)) for i in range(J_PSP + J_PUP + 1)]
        # print(caps)
        # print("QCLP occs = ", occss)
        # print("Computed occs", occso)

        # print("current price bounds:")
        # print(p_l)
        # print(p_u)
        # print("current rel opt value = ", obj_value)
        # print("current int opt value = ", integer_opt_value)
        # print("current best int opt value = ", best_upper_bound)

    return solve_time, opt_p_value, -best_upper_bound, -lower_bound, node_counter, gap, occss
