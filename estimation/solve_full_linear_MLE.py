import numpy as np
import time

import gurobipy as gp
from gurobipy import GRB

presolve = 0
# 0 = none, -1 = automatic, 1 = conservative, 2 = aggressive
numeric_focus = 1

inf = float("inf")
bigM = 200


def compute_sol_from_beta(J, N, R, x, y, epsilon, beta, linFix=0, mixed=None):
    K = len(beta)
    Uinr = dict()
    Unr = dict()
    omega = dict()

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    if mixed is None:

        if linFix:
            sep = 1e-12
        else:
            sep = 0

        for i in range(J):
            for n in range(N):
                for r in range(R):
                    Uinr[i, n, r] = sum(beta[k] * x[i, n, k] for k in range(K)) + epsilon[i, n, r]
                    if not i == y_index[n]:
                        Uinr[i, n, r] = Uinr[i, n, r] - sep  # this makes sure that if GRB thinks it dominated an
                        # alternative, it actually does dominate it

        for n in range(N):
            for r in range(R):
                max_util = int(np.argmax([Uinr[i, n, r] for i in range(J)]))
                for i in range(J):
                    if i == max_util:
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
            obj_dict[n] = sum(y[i, n] * z_dict[i, n] for i in range(J))
        total_obj = sum(obj_dict[n] for n in range(N)) - N * np.log(R)
    else:
        mixed_index = mixed[0]
        normal_epsilon = mixed[1]
        not_mixed_params = [k for k in range(K) if not (k == mixed_index or k == mixed_index + 1)]
        for i in range(J):
            for n in range(N):
                for r in range(R):
                    Uinr[i, n, r] = sum(beta[k] * x[i, n, k] for k in not_mixed_params) \
                                    + beta[mixed_index] * x[i, n, mixed_index] \
                                    + beta[mixed_index + 1] * x[i, n, mixed_index + 1] * normal_epsilon[i, n, r] \
                                    + epsilon[i, n, r]

        for n in range(N):
            for r in range(R):
                max_util = int(np.argmax([Uinr[i, n, r] for i in range(J)]))
                for i in range(J):
                    if i == max_util:
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
            obj_dict[n] = sum(y[i, n] * z_dict[i, n] for i in range(J))
        total_obj = sum(obj_dict[n] for n in range(N)) - N * np.log(R)

    return omega, Unr, Uinr, beta, s_dict, z_dict, total_obj


def maximumlikelihood_linMLE_vers5(y, x, K, R, epsilon, logOfZer, time_limit, warm_start):
    start_time = time.time()
    N = len(y[0])
    J = int(len(x.keys()) / (N * K))

    logOfZero = -100
    logL = np.array([logOfZero] +
                    [(1 + r) * np.log(r) - r * np.log(1 + r) for r in range(1, R + 1)])
    logK = np.array([logOfZero] +
                    [np.log(r) - np.log(1 + r) for r in range(1, R + 1)])

    # initialize model
    m = gp.Model("full MILP")

    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    Uinr_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"Uinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    dinr_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"dinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    enr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"enr_{n}_{r}")
                for n in range(N) for r in range(R)}
    ginr_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"ginr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    omega_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}_{n}_{r}")
                  for i in range(J) for n in range(N) for r in range(R)}
    s_vars = {(i, n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"s_{i}_{n}")
              for i in range(J) for n in range(N)}
    z_vars = {(i, n): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"z_{i}_{n}")
              for i in range(J) for n in range(N)}

    # define Utilities
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(
                    Uinr_vars[i, n, r] - gp.quicksum(beta_vars[k] * x[i, n, k] for k in range(K))
                    == epsilon[i, n, r])

    # define dinr
    for n in range(N):
        for r in range(R):
            m.addConstr(dinr_vars[0, n, r] <= 1 * (Uinr_vars[0, n, r] - Uinr_vars[1, n, r]))
            m.addConstr(dinr_vars[1, n, r] <= 1 * (Uinr_vars[1, n, r] - Uinr_vars[0, n, r]))

    # define enr
    for n in range(N):
        for r in range(R):
            m.addConstr(enr_vars[n, r] >= 1 * (Uinr_vars[0, n, r] - Uinr_vars[1, n, r]))
            m.addConstr(enr_vars[n, r] >= 1 * (Uinr_vars[1, n, r] - Uinr_vars[0, n, r]))

    # define ginr
    for n in range(N):
        for r in range(R):
            for i in range(J):
                m.addConstr(ginr_vars[i, n, r] <= (dinr_vars[i, n, r] + enr_vars[n, r]) / 2)

    # define omega_vars
    for n in range(N):
        for r in range(R):
            m.addConstr(gp.quicksum(omega_vars[i, n, r] for i in range(J)) == 1)

    for n in range(N):
        for r in range(R):
            for i in range(J):
                m.addConstr(omega_vars[i, n, r] >= 0)
                m.addConstr(omega_vars[i, n, r] <= ginr_vars[i, n, r] * 1000)

    # Constraints theta
    for n in range(N):
        for i in range(J):
            m.addConstr(s_vars[i, n] - gp.quicksum(omega_vars[i, n, r] for r in range(R)) == 0)

    # Constraints xi
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(z_vars[i, n] + logK[r] * s_vars[i, n] <= logL[r])

    # Objective function
    objective = 1 * gp.quicksum(-y[i, n] * z_vars[i, n] for i in range(J) for n in range(N)) + \
                4 * gp.quicksum(enr_vars[n, r] for r in range(R) for n in range(N))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    m.setParam('DualReductions', 0)
    m.setParam("TimeLimit", 100)
    # m.setParam("NonConvex", 0)
    m.setParam("NumericFocus", 2)
    # m.setParam("Presolve", presolve)
    # m.setParam("Method", solver_method)
    # m.setParam("PreCrush", 1)
    m.optimize()

    # print("new linMLE Status = ", m.status)

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    diff_problem_counter = 0
    enr_problem_counter = 0

    success_counter = dict()

    smallest_diff = 100
    for n in range(N):
        success_count_ind = 0
        for r in range(R):

            # lets print all of them
            # print("")
            # print(f"linMLE ind {n}, scenario {r}:")
            # print(f"U[0, {n}, {r}] = {Uinr_vars[0, n, r].x}")
            # print(f"U[1, {n}, {r}] = {Uinr_vars[1, n, r].x}")
            # print(f"abs. Difference = {abs(Uinr_vars[0, n, r].x - Uinr_vars[1, n, r].x)}")
            # print(f"e[{n}, {r}] = {enr_vars[n, r].x}")
            # print(f"d[0, {n}, {r}] = {dinr_vars[0, n, r].x}")
            # print(f"d[1, {n}, {r}] = {dinr_vars[1, n, r].x}")
            # print(f"f[0, {n}, {r}] = {(dinr_vars[0, n, r].x + enr_vars[n, r].x) / 2}")
            # print(f"f[1, {n}, {r}] = {(dinr_vars[1, n, r].x + enr_vars[n, r].x) / 2}")
            # print(f"g[0, {n}, {r}] = {ginr_vars[0, n, r].x}")
            # print(f"g[1, {n}, {r}] = {ginr_vars[1, n, r].x}")
            # print(f"omega[0, {n}, {r}] = {omega_vars[0, n, r].x}")
            # print(f"omega[1, {n}, {r}] = {omega_vars[1, n, r].x}")
            # print(f"observed choice = {y_index[n]}")
            # print("")

            diff = abs(Uinr_vars[0, n, r].x - Uinr_vars[1, n, r].x)
            if abs(enr_vars[n, r].x - diff) >= 1e-12:
                enr_problem_counter += 1
                print("")
                print(f"ind {n}, scenario {r} has wrong e_nr:")
                print(f"U[0, {n}, {r}] = {Uinr_vars[0, n, r].x}")
                print(f"U[1, {n}, {r}] = {Uinr_vars[1, n, r].x}")
                print(f"abs. Difference = {abs(Uinr_vars[0, n, r].x - Uinr_vars[1, n, r].x)}")
                print(f"e[{n}, {r}] = {enr_vars[n, r].x}")
                print(f"d[0, {n}, {r}] = {dinr_vars[0, n, r].x}")
                print(f"d[1, {n}, {r}] = {dinr_vars[1, n, r].x}")
                print(f"f[0, {n}, {r}] = {(dinr_vars[0, n, r].x + enr_vars[n, r].x) / 2}")
                print(f"f[1, {n}, {r}] = {(dinr_vars[1, n, r].x + enr_vars[n, r].x) / 2}")
                print(f"g[0, {n}, {r}] = {ginr_vars[0, n, r].x}")
                print(f"g[1, {n}, {r}] = {ginr_vars[1, n, r].x}")
                print(f"omega[0, {n}, {r}] = {omega_vars[0, n, r].x}")
                print(f"omega[1, {n}, {r}] = {omega_vars[1, n, r].x}")
                print(f"observed choice = {y_index[n]}")
                print("")
            if diff < smallest_diff:
                smallest_diff = diff
            if diff <= 1e-12:
                diff_problem_counter += 1
                print("")
                print(f"ind {n}, scenario {r} has small diff:")
                print(f"U[0, {n}, {r}] = {Uinr_vars[0, n, r].x}")
                print(f"U[1, {n}, {r}] = {Uinr_vars[1, n, r].x}")
                print(f"abs. Difference = {abs(Uinr_vars[0, n, r].x - Uinr_vars[1, n, r].x)}")
                print(f"e[{n}, {r}] = {enr_vars[n, r].x}")
                print(f"d[0, {n}, {r}] = {dinr_vars[0, n, r].x}")
                print(f"d[1, {n}, {r}] = {dinr_vars[1, n, r].x}")
                print(f"f[0, {n}, {r}] = {(dinr_vars[0, n, r].x + enr_vars[n, r].x) / 2}")
                print(f"f[1, {n}, {r}] = {(dinr_vars[1, n, r].x + enr_vars[n, r].x) / 2}")
                print(f"g[0, {n}, {r}] = {ginr_vars[0, n, r].x}")
                print(f"g[1, {n}, {r}] = {ginr_vars[1, n, r].x}")
                print(f"omega[0, {n}, {r}] = {omega_vars[0, n, r].x}")
                print(f"omega[1, {n}, {r}] = {omega_vars[1, n, r].x}")
                print(f"observed choice = {y_index[n]}")
                print("")
            if abs(omega_vars[y_index[n], n, r].x - 1) <= 1e-12:
                success_count_ind += 1
        success_counter[n] = success_count_ind
    if diff_problem_counter == 0:
        print(f"No diff problems, smallest diff = {smallest_diff}")
    else:
        print(f"{diff_problem_counter} diff problem(s) encountered. Smallest diff = {smallest_diff}")

    if enr_problem_counter == 0:
        print(f"No enr problems")
    else:
        print(f"{enr_problem_counter} enr problem(s) encountered.")

    print(f"Model recreated observed choice {sum(success_counter[n] for n in range(N))} times.")
    print(f"objective z, y sum = {sum(-y[i, n] * z_vars[i, n].x for i in range(J) for n in range(N))}")
    print(f"We get total sum abs = {sum(enr_vars[n, r].x for r in range(R) for n in range(N))}")
    print("")
    for n in range(N):
        print(f"For ind {n} we succeed {success_counter[n]} times")
        print(f"We get objective value = {sum(-y[i, n] * z_vars[i, n].x for i in range(J))}")
        print(f"We get sum abs = {sum(enr_vars[n, r].x for r in range(R))}")
    print("")

    if m.status == GRB.TIME_LIMIT:
        if m.SolCount >= 1:
            objective_value = m.ObjVal + N * np.log(R)
        else:
            objective_value = 1000
    else:
        objective_value = m.ObjVal + N * np.log(R)

    total_time = time.time() - start_time
    try:
        best_lowerbound = m.ObjBoundC + N * np.log(R)
    except:
        best_lowerbound = -1000
    if m.SolCount >= 1:
        bestbeta = [beta_vars[k].x for k in range(K)]
    else:
        bestbeta = [1000 for k in range(K)]

    best_loglike = -objective_value

    # for n in range(N):
    #     for r in range(R):
    #         for i in range(J):
    #             print(f"U_{i}{n}{r} = {Uinr_vars[i, n, r].x}")
    #             print(f"w_{i}{n}{r} = {omega_vars[i, n, r].x}")

    m.dispose()

    omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta)
    best_loglike = total_obj

    return total_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_vers4(y, x, K, R, epsilon, logOfZer, time_limit, warm_start):
    start_time = time.time()
    N = len(y[0])
    J = int(len(x.keys()) / (N * K))

    logOfZero = -100
    logL = np.array([logOfZero] +
                    [(1 + r) * np.log(r) - r * np.log(1 + r) for r in range(1, R + 1)])
    logK = np.array([logOfZero] +
                    [np.log(r) - np.log(1 + r) for r in range(1, R + 1)])

    # initialize model
    m = gp.Model("full MILP")

    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    Uinr_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"Uinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    Unr_vars = {(n, r): m.addVar(lb=-100, ub=inf, vtype=GRB.CONTINUOUS, name=f"Unr_{n}_{r}")
                for n in range(N) for r in range(R)}
    omega_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}_{n}_{r}")
                  for i in range(J) for n in range(N) for r in range(R)}
    s_vars = {(i, n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"s_{i}_{n}")
              for i in range(J) for n in range(N)}
    z_vars = {(i, n): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"z_{i}_{n}")
              for i in range(J) for n in range(N)}

    # define Utilities
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(
                    Uinr_vars[i, n, r] - gp.quicksum(beta_vars[k] * x[i, n, k] for k in range(K))
                    == epsilon[i, n, r])

    # define Unr
    for n in range(N):
        for r in range(R):
            for i in range(J):
                m.addConstr(Unr_vars[n, r] >= Uinr_vars[i, n, r])

    # define omega_vars
    for n in range(N):
        for r in range(R):
            m.addConstr(gp.quicksum(omega_vars[i, n, r] for i in range(J)) == 1)

    for n in range(N):
        for r in range(R):
            for i in range(J):
                m.addConstr(1 - omega_vars[i, n, r] <= (Unr_vars[n, r] - Uinr_vars[i, n, r]) * 1e12)
                m.addConstr(1 - omega_vars[i, n, r] >= - (Unr_vars[n, r] - Uinr_vars[i, n, r]) * 1e12)

    # Constraints theta
    for n in range(N):
        for i in range(J):
            m.addConstr(s_vars[i, n] - gp.quicksum(omega_vars[i, n, r] for r in range(R)) == 0)

    # Constraints xi
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(z_vars[i, n] + logK[r] * s_vars[i, n] <= logL[r])

    # Objective function
    objective = 1 * gp.quicksum(-y[i, n] * z_vars[i, n] for i in range(J) for n in range(N)) + \
                1000 * gp.quicksum(Unr_vars[n, r] for r in range(R) for n in range(N))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    m.setParam('DualReductions', 0)
    m.setParam("TimeLimit", 100)
    # m.setParam("NonConvex", 0)
    # m.setParam("NumericFocus", numeric_focus)
    # m.setParam("Presolve", presolve)
    # m.setParam("Method", solver_method)
    # m.setParam("PreCrush", 1)
    m.optimize()

    # print("new linMLE Status = ", m.status)

    if m.status == GRB.TIME_LIMIT:
        if m.SolCount >= 1:
            objective_value = m.ObjVal + N * np.log(R)
        else:
            objective_value = 1000
    else:
        objective_value = m.ObjVal + N * np.log(R)

    total_time = time.time() - start_time
    try:
        best_lowerbound = m.ObjBoundC + N * np.log(R)
    except:
        best_lowerbound = -1000
    if m.SolCount >= 1:
        bestbeta = [beta_vars[k].x for k in range(K)]
    else:
        bestbeta = [1000 for k in range(K)]

    best_loglike = -objective_value

    # for n in range(N):
    #     for r in range(R):
    #         for i in range(J):
    #             print(f"U_{i}{n}{r} = {Uinr_vars[i, n, r].x}")
    #             print(f"w_{i}{n}{r} = {omega_vars[i, n, r].x}")

    m.dispose()

    omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta)
    best_loglike = total_obj

    return total_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_vers3(y, x, K, R, epsilon, logOfZer, time_limit, warm_start):
    start_time = time.time()
    N = len(y[0])
    J = int(len(x.keys()) / (N * K))

    logOfZero = -100
    logL = np.array([logOfZero] +
                    [(1 + r) * np.log(r) - r * np.log(1 + r) for r in range(1, R + 1)])
    logK = np.array([logOfZero] +
                    [np.log(r) - np.log(1 + r) for r in range(1, R + 1)])

    # initialize model
    m = gp.Model("full MILP")

    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    Uinr_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"Uinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    dinr_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"dinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    omega_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}_{n}_{r}")
                  for i in range(J) for n in range(N) for r in range(R)}
    s_vars = {(i, n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"s_{i}_{n}")
              for i in range(J) for n in range(N)}
    z_vars = {(i, n): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"z_{i}_{n}")
              for i in range(J) for n in range(N)}

    # define Utilities
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(
                    Uinr_vars[i, n, r] - gp.quicksum(beta_vars[k] * x[i, n, k] for k in range(K))
                    == epsilon[i, n, r])

    # define d_vars
    for n in range(N):
        for r in range(R):
            for i in range(J):
                for j in range(J):
                    m.addConstr(dinr_vars[i, n, r] <= 50 * (Uinr_vars[i, n, r] - Uinr_vars[j, n, r]))

    # define omega_vars
    for n in range(N):
        for r in range(R):
            m.addConstr(gp.quicksum(omega_vars[i, n, r] for i in range(J)) == 1)

    for n in range(N):
        for r in range(R):
            for i in range(J):
                # dinr_vars are <= 0 thus the signs are reversed
                m.addConstr(omega_vars[i, n, r] >= 1 + 1e10 * dinr_vars[i, n, r])
                m.addConstr(omega_vars[i, n, r] <= 1 - 1e10 * dinr_vars[i, n, r])

    # Constraints theta
    for n in range(N):
        for i in range(J):
            m.addConstr(s_vars[i, n] - gp.quicksum(omega_vars[i, n, r] for r in range(R)) == 0)

    # Constraints xi
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(z_vars[i, n] + logK[r] * s_vars[i, n] <= logL[r])

    # Objective function
    objective = gp.quicksum(-y[i, n] * z_vars[i, n] for i in range(J) for n in range(N)) + \
                1000 * gp.quicksum(-dinr_vars[i, n, r] for i in range(J) for r in range(R) for n in range(N))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    m.setParam('DualReductions', 0)
    m.setParam("TimeLimit", 100)
    # m.setParam("NonConvex", 0)
    # m.setParam("NumericFocus", numeric_focus)
    # m.setParam("Presolve", presolve)
    # m.setParam("Method", solver_method)
    # m.setParam("PreCrush", 1)
    m.optimize()

    print("new linMLE Status = ", m.status)

    if m.status == GRB.TIME_LIMIT:
        if m.SolCount >= 1:
            objective_value = m.ObjVal + N * np.log(R)
        else:
            objective_value = 1000
    else:
        objective_value = m.ObjVal + N * np.log(R)

    total_time = time.time() - start_time
    try:
        best_lowerbound = m.ObjBoundC + N * np.log(R)
    except:
        best_lowerbound = -1000
    if m.SolCount >= 1:
        bestbeta = [beta_vars[k].x for k in range(K)]
    else:
        bestbeta = [1000 for k in range(K)]

    best_loglike = -objective_value

    for n in range(N):
        for r in range(R):
            for i in range(J):
                print(f"U_{i}{n}{r} = {Uinr_vars[i, n, r].x}")
                print(f"d_{i}{n}{r} = {dinr_vars[i, n, r].x}")
                print(f"w_{i}{n}{r} = {omega_vars[i, n, r].x}")

    m.dispose()

    omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta)
    best_loglike = total_obj

    return total_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_michelmixed_absolute(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    znr_vars = {(n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"znr_{n}_{r}")
                for n in range(N) for r in range(R)}
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    for n in range(N):
        for r in range(R):
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            )
            m.addConstr(znr_vars[n, r] >= dynr_vars[n, r] - (
                    gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                    + beta_vars[mixed_param] * x[y_index[n], n, mixed_param]))
            m.addConstr(znr_vars[n, r] >= (
                    gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                    + beta_vars[mixed_param] * x[y_index[n], n, mixed_param])
                        - dynr_vars[n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    # objective = gp.quicksum(
    #     gp.quicksum(dynr_vars[n, r] for r in range(R))
    #     - R * (gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
    #             + beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
    #            )
    #     for n in range(N))

    objective = gp.quicksum(znr_vars[n, r] for n in range(N) for r in range(R))
    # objective = gp.quicksum(dynr_vars[n, r] for n in range(N) for r in range(R))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    # m.setParam('DualReductions', 0)
    # m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_michelmixed_abs_nonlinear(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    abs_vars = {n: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"abs_{n}")
                for n in range(N)}
    abss_vars = {n: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"abss_{n}")
                 for n in range(N)}
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    for n in range(N):
        for r in range(R):
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            )
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    for n in range(N):
        m.addConstr(abs_vars[n] == gp.quicksum(dynr_vars[n, r] for r in range(R))
                    - R * (gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                           + beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                           ))
        m.addConstr(abss_vars[n] == gp.abs_(abs_vars[n]))

    objective = gp.quicksum(abss_vars[n] for n in range(N))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    # m.setParam('DualReductions', 0)
    # m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_michelmixed(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    # znr_vars = {(n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"znr_{n}_{r}")
    #             for n in range(N) for r in range(R)}
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    for n in range(N):
        for r in range(R):
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            )
            # m.addConstr(znr_vars[n, r] >= dynr_vars[n, r] - (
            #         gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
            #         + beta_vars[mixed_param] * x[y_index[n], n, mixed_param]))
            # m.addConstr(znr_vars[n, r] >= (
            #         gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
            #         + beta_vars[mixed_param] * x[y_index[n], n, mixed_param])
            #             - dynr_vars[n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = gp.quicksum(
        gp.quicksum(dynr_vars[n, r] for r in range(R))
        - R * (gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
               + beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
               )
        for n in range(N))

    # objective = gp.quicksum(znr_vars[n, r] for n in range(N) for r in range(R))
    # objective = gp.quicksum(dynr_vars[n, r] for n in range(N) for r in range(R))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    # m.setParam('DualReductions', 0)
    # m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_remixed(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    # znr_vars = {(n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"znr_{n}_{r}")
    #             for n in range(N) for r in range(R)}
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    for n in range(N):
        for r in range(R):
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            - (
                                    gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                                    + beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                            )
                            )
            # m.addConstr(znr_vars[n, r] >= dynr_vars[n, r] - R * (
            #         gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
            #         + beta_vars[mixed_param] * x[y_index[n], n, mixed_param]))
            # m.addConstr(znr_vars[n, r] >= R * (
            #         gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
            #         + beta_vars[mixed_param] * x[y_index[n], n, mixed_param])
            #             - dynr_vars[n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    # objective = gp.quicksum(dynr_vars[n, r] - R * (
    #                  gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
    #                  + beta_vars[mixed_param] * x[y_index[n], n, mixed_param])
    #                         for n in range(N) for r in range(R))
    # objective = gp.quicksum(znr_vars[n, r] for n in range(N) for r in range(R))
    objective = gp.quicksum(dynr_vars[n, r] for n in range(N) for r in range(R))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    # m.setParam('DualReductions', 0)
    # m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_mixed_8(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    dyr_vars = {(r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dyr_{r}")
                for r in range(R)}
    dy_vars = m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dy"
                                                                f"")
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    m.addConstr(dy_vars >= gp.quicksum(dyr_vars[r] for r in range(R)))
    for r in range(R):
        m.addConstr(dyr_vars[r] >= gp.quicksum(dynr_vars[n, r] for n in range(N)))
    for n in range(N):
        for r in range(R):
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                            - beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                            - beta_vars[mixed_param + 1] * x[y_index[n], n, mixed_param] * normal_epsilon[n, r]
                            - epsilon[y_index[n], n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = gp.quicksum(dyr_vars[r] for r in range(R))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_mixed_7(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    dyn_vars = {(n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dyn_{n}")
                for n in range(N)}
    dy_vars = m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dy")
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    m.addConstr(dy_vars >= gp.quicksum(dyn_vars[n] for n in range(N)))
    for n in range(N):
        m.addConstr(dyn_vars[n] >= gp.quicksum(dynr_vars[n, r] for r in range(R)))
        for r in range(R):
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                            - beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                            - beta_vars[mixed_param + 1] * x[y_index[n], n, mixed_param] * normal_epsilon[n, r]
                            - epsilon[y_index[n], n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = dy_vars

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_mixed_6(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    dyr_vars = {(r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dyr_{r}")
                for r in range(R)}
    dy_vars = m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dy"
                                                                f"")
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    for r in range(R):
        m.addConstr(dy_vars >= dyr_vars[r])
        m.addConstr(dyr_vars[r] >= gp.quicksum(dynr_vars[n, r] for n in range(N)))
    for n in range(N):
        for r in range(R):
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                            - beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                            - beta_vars[mixed_param + 1] * x[y_index[n], n, mixed_param] * normal_epsilon[n, r]
                            - epsilon[y_index[n], n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = gp.quicksum(dyr_vars[r] for r in range(R))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_mixed_5(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    dyn_vars = {(n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dyn_{n}")
                for n in range(N)}
    dy_vars = m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dy")
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    for n in range(N):
        m.addConstr(dy_vars >= dyn_vars[n])
        m.addConstr(dyn_vars[n] >= gp.quicksum(dynr_vars[n, r] for r in range(R)))
        for r in range(R):
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                            - beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                            - beta_vars[mixed_param + 1] * x[y_index[n], n, mixed_param] * normal_epsilon[n, r]
                            - epsilon[y_index[n], n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = dy_vars

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_mixed_4(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    dyr_vars = {(r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dyr_{r}")
                for r in range(R)}
    dy_vars = m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dy"
                                                                f"")
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    for r in range(R):
        m.addConstr(dy_vars >= dyr_vars[r])
    for n in range(N):
        for r in range(R):
            m.addConstr(dyr_vars[r] >= dynr_vars[n, r])
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                            - beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                            - beta_vars[mixed_param + 1] * x[y_index[n], n, mixed_param] * normal_epsilon[n, r]
                            - epsilon[y_index[n], n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = gp.quicksum(dyr_vars[r] for r in range(R))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_mixed_3(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    dyn_vars = {(n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dyn_{n}")
                for n in range(N)}
    dy_vars = m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dy")
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    for n in range(N):
        m.addConstr(dy_vars >= dyn_vars[n])
        for r in range(R):
            m.addConstr(dyn_vars[n] >= dynr_vars[n, r])
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                            - beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                            - beta_vars[mixed_param + 1] * x[y_index[n], n, mixed_param] * normal_epsilon[n, r]
                            - epsilon[y_index[n], n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = dy_vars

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_mixed_21(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    dyr_vars = {(r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dyr_{r}")
                for r in range(R)}
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    for r in range(R):
        m.addConstr(dyr_vars[r] >= gp.quicksum(dynr_vars[n, r] for n in range(N)))
    for n in range(N):
        for r in range(R):
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                            - beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                            - beta_vars[mixed_param + 1] * x[y_index[n], n, mixed_param] * normal_epsilon[n, r]
                            - epsilon[y_index[n], n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = gp.quicksum(dyr_vars[r] for r in range(R))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_mixed_2(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    dyr_vars = {(r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dyr_{r}")
                for r in range(R)}
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    for n in range(N):
        for r in range(R):
            m.addConstr(dyr_vars[r] >= dynr_vars[n, r])
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                            - beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                            - beta_vars[mixed_param + 1] * x[y_index[n], n, mixed_param] * normal_epsilon[n, r]
                            - epsilon[y_index[n], n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = gp.quicksum(dyr_vars[r] for r in range(R))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_mixed_11(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    dyn_vars = {(n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dyn_{n}")
                for n in range(N)}
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    for n in range(N):
        m.addConstr(dyn_vars[n] >= gp.quicksum(dynr_vars[n, r] for r in range(R)))
        for r in range(R):
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                            - beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                            - beta_vars[mixed_param + 1] * x[y_index[n], n, mixed_param] * normal_epsilon[n, r]
                            - epsilon[y_index[n], n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = gp.quicksum(dyn_vars[n] for n in range(N))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_mixed_1(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = epsilon.shape[2]

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    dyn_vars = {(n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dyn_{n}")
                for n in range(N)}
    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    for n in range(N):
        for r in range(R):
            m.addConstr(dyn_vars[n] >= dynr_vars[n, r])
            for j in av_alts[n]:
                m.addConstr(dynr_vars[n, r] >=
                            gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                            + beta_vars[mixed_param] * x[j, n, mixed_param]
                            + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n, r]
                            + epsilon[j, n, r]
                            - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                            - beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                            - beta_vars[mixed_param + 1] * x[y_index[n], n, mixed_param] * normal_epsilon[n, r]
                            - epsilon[y_index[n], n, r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = gp.quicksum(dyn_vars[n] for n in range(N))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    m.setParam('Presolve', 1)
    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loglike = None

    return total_time, solve_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_mixed_matrix(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = int(np.shape(epsilon)[1] / N)

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    normal_params = [k for k in range(K) if not (k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = m.addMVar(shape=K, lb=-inf, ub=inf, name="beta")
    dynr_vars = m.addMVar(shape=N * R, lb=0, name="Lnr")
    L_var = m.addVar(lb=0, ub=inf, name="L")
    m.setObjective(np.ones(N * R) @ dynr_vars, GRB.MINIMIZE)

    # BioGeme does this so lets also do that
    beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Creating better x and eps")
    st_time = time.time()
    print("type x = ", type(x))
    x = np.array(x)
    print("new type x = ", type(x))
    # create better x and eps
    better_x = np.empty(shape=(J, N, K))
    better_eps = np.empty(shape=(J, N * R))
    for n in range(N):
        for j in range(J):
            if av[j][n] == 1:
                better_x[j, n, :] = x[j, n, :]
            else:
                better_x[j, n, :] = np.zeros(shape=K)
            if av[j][n] == 1:
                for r in range(R):
                    better_eps[j, n * N + r] = epsilon[j, n * N + r]
            else:
                for r in range(R):
                    better_eps[j, n * N + r] = -10000  # hopefully equivalent to deleting the alt
    print(f"Time to create better x and eps: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()

    for j in range(J):
        x_j = np.array([[x[j, n, k] for k in normal_params] + [x[j, n, mixed_param + 1] * normal_epsilon[n * N + r]]
                        for n in range(N) for r in range(R)])
        m.addConstr(dynr_vars >= x_j @ beta_vars + (better_eps[j, :] - better_eps[y, :]))
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    m.setParam('OutputFlag', 1)
    m.setParam('Method', 2)
    m.setParam('ScaleFlag', 0)
    m.setParam('Presolve', 0)

    # print("Tune LP")
    # m.setParam('tuneTrials', 10)
    # m.setParam('tuneTimeLimit', 7200)
    # m.tune()

    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loss = sum(dynr_vars[n * N + r].x for n in range(N) for r in range(R))
    dnr = {(n, r): dynr_vars[n * N + r].x for n in range(N) for r in range(R)}
    m.dispose()

    return total_time, solve_time, bestbeta, best_loss, dnr


def maximumlikelihood_linMLE_mixed_dual(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = int(np.shape(epsilon)[1] / N)

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}
    normal_params = [k for k in range(K) if not (k == mixed_param + 1)]

    # initialize model

    # # chat gpt says this
    # m = gp.Model("linearMLE")
    #
    # # Creating all variables at once
    # alpha_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"alpha_{i}_{n}_{r}")
    #               for n in range(N) for i in av_alts[n] for r in range(R)}
    #
    # # Creating alpha_vars constraints
    # alpha_lhs = [alpha_vars[i, n, r] for n in range(N) for i in av_alts[n] for r in range(R)]
    # alpha_op = [GRB.EQUAL] * len(alpha_lhs)
    # alpha_rhs = [1] * len(alpha_lhs)
    # m.addConstrs(alpha_lhs, alpha_op, alpha_rhs)
    #
    # # Creating dnr_vars constraints
    # dnr_lhs = [gp.quicksum(alpha_vars[i, n, r] for i in av_alts[n]) for n in range(N) for r in range(R)]
    # dnr_op = [GRB.EQUAL] * len(dnr_lhs)
    # dnr_rhs = [1] * len(dnr_lhs)
    # m.addConstrs(dnr_lhs, dnr_op, dnr_rhs)
    #
    # # Creating betas constraints
    # betas_lhs = [gp.quicksum(alpha_vars[i, n, r] * (x[y_index[n], n, k] - x[i, n, k])
    #                          for n in range(N) for i in av_alts[n] for r in range(R)) for k in normal_params] + [
    #                 gp.quicksum(alpha_vars[i, n, r] * (x[y_index[n], n, mixed_param + 1] - x[i, n, mixed_param + 1]) *
    #                             normal_epsilon[n * N + r]
    #                             for n in range(N) for i in av_alts[n] for r in range(R)) for k in mixed_param + 1]
    # betas_op = [GRB.EQUAL] * len(betas_lhs)
    # betas_rhs = [0] * len(betas_lhs)
    # m.addConstrs(betas_lhs, betas_op, betas_rhs)
    #
    # # Optimizing the model
    # m.optimize()
    #
    # # Get dual value of constraint
    # all_constr = m.getConstrs()
    # dual_values = [c.getAttr("Pi") for c in all_constr]

    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    alpha_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"alpha_{i}_{n}_{r}")
                  for n in range(N) for i in av_alts[n] for r in range(R)}
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()
    # define d_vars
    dnr_vars = dict()
    for n in range(N):
        for r in range(R):
            dnr_vars[n, r] = m.addLConstr(gp.quicksum(alpha_vars[i, n, r] for i in av_alts[n]), GRB.EQUAL, 1)
    betas = dict()
    for k in normal_params:
        betas[k] = m.addLConstr(gp.quicksum(alpha_vars[i, n, r] * (x[y_index[n], n, k] - x[i, n, k])
                                            for n in range(N) for i in av_alts[n] for r in range(R)),
                                GRB.EQUAL, 0
                                )
    betas[mixed_param + 1] = m.addLConstr(gp.quicksum(alpha_vars[i, n, r] *
                                                      (x[y_index[n], n, mixed_param + 1] - x[i, n, mixed_param + 1])
                                                      * normal_epsilon[n * N + r]
                                                      for n in range(N) for i in av_alts[n] for r in range(R)),
                                          GRB.EQUAL, 0
                                          )
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = gp.quicksum(alpha_vars[i, n, r] * (epsilon[i, n * N + r] - epsilon[y_index[n], n * N + r])
                            for n in range(N) for i in av_alts[n] for r in range(R))

    m.setObjective(objective, GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 2)
    m.setParam('PreDepRow', 0)

    # m.write("dual_LP.lp")

    # print("Tune LP")
    # m.setParam('tuneTrials', 10)
    # m.setParam('tuneTimeLimit', 7200)
    # m.tune()

    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time

    best_loss = sum(alpha_vars[i, n, r].x * (epsilon[i, n * N + r] - epsilon[y_index[n], n * N + r])
                    for n in range(N) for i in av_alts[n] for r in range(R))
    bestbeta = [betas[k].Pi for k in range(K)]
    dnr = {(n, r): dnr_vars[n, r].Pi for n in range(N) for r in range(R)}

    # print("Printing the alpha vars")
    # for n in range(N):
    #     for i in av_alts[n]:
    #         for r in range(R):
    #             if alpha_vars[i, n, r].x == 1:
    #                 print("")
    #                 print(f"alpha[{i}{n}{r}] = {alpha_vars[i, n, r].x}")
    #                 print("")
    #             else:
    #                 print(f"alpha[{i}{n}{r}] = {alpha_vars[i, n, r].x}")

    m.dispose()

    return total_time, solve_time, bestbeta, best_loss, dnr


def maximumlikelihood_linMLE_crossnested(y, x, av, K, epsilon, ex_epsilon, pu_epsilon):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = int(np.shape(epsilon)[1] / N)
    K = 4

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    mu_ex_var = m.addVar(lb=1, ub=inf, vtype=GRB.CONTINUOUS, name="mu_ex")
    mu_pu_var = m.addVar(lb=1, ub=inf, vtype=GRB.CONTINUOUS, name="mu_pu")
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}

    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()

    # define d_vars
    for n in range(N):
        for r in range(R):
            if 0 in av_alts[n]:
                if y_index[n] == 0:
                    m.addLConstr(dynr_vars[n, r], GRB.GREATER_EQUAL,
                                 gp.quicksum(beta_vars[k] * x[0, n, k] for k in range(K))
                                 + epsilon[0, n * N + r]
                                 + mu_ex_var * ex_epsilon[n * N + r]
                                 + mu_pu_var * pu_epsilon[n * N + r]
                                 - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in range(K))
                                 - epsilon[y_index[n], n * N + r]
                                 - mu_ex_var * ex_epsilon[n * N + r]
                                 - mu_pu_var * pu_epsilon[n * N + r]
                                 )
                elif y_index[n] == 1:
                    m.addLConstr(dynr_vars[n, r], GRB.GREATER_EQUAL,
                                 gp.quicksum(beta_vars[k] * x[0, n, k] for k in range(K))
                                 + epsilon[0, n * N + r]
                                 + mu_ex_var * ex_epsilon[n * N + r]
                                 + mu_pu_var * pu_epsilon[n * N + r]
                                 - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in range(K))
                                 - epsilon[y_index[n], n * N + r]
                                 - mu_pu_var * pu_epsilon[n * N + r]
                                 )
                elif y_index[n] == 2:
                    m.addLConstr(dynr_vars[n, r], GRB.GREATER_EQUAL,
                                 gp.quicksum(beta_vars[k] * x[0, n, k] for k in range(K))
                                 + epsilon[0, n * N + r]
                                 + mu_ex_var * ex_epsilon[n * N + r]
                                 + mu_pu_var * pu_epsilon[n * N + r]
                                 - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in range(K))
                                 - epsilon[y_index[n], n * N + r]
                                 - mu_ex_var * ex_epsilon[n * N + r]
                                 )
            if 1 in av_alts[n]:
                if y_index[n] == 0:
                    m.addLConstr(dynr_vars[n, r], GRB.GREATER_EQUAL,
                                 gp.quicksum(beta_vars[k] * x[1, n, k] for k in range(K))
                                 + epsilon[1, n * N + r]
                                 + mu_pu_var * pu_epsilon[n * N + r]
                                 - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in range(K))
                                 - epsilon[y_index[n], n * N + r]
                                 - mu_ex_var * ex_epsilon[n * N + r]
                                 - mu_pu_var * pu_epsilon[n * N + r]
                                 )
                elif y_index[n] == 1:
                    m.addLConstr(dynr_vars[n, r], GRB.GREATER_EQUAL,
                                 gp.quicksum(beta_vars[k] * x[1, n, k] for k in range(K))
                                 + epsilon[1, n * N + r]
                                 + mu_pu_var * pu_epsilon[n * N + r]
                                 - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in range(K))
                                 - epsilon[y_index[n], n * N + r]
                                 - mu_pu_var * pu_epsilon[n * N + r]
                                 )
                elif y_index[n] == 2:
                    m.addLConstr(dynr_vars[n, r], GRB.GREATER_EQUAL,
                                 gp.quicksum(beta_vars[k] * x[1, n, k] for k in range(K))
                                 + epsilon[1, n * N + r]
                                 + mu_pu_var * pu_epsilon[n * N + r]
                                 - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in range(K))
                                 - epsilon[y_index[n], n * N + r]
                                 - mu_ex_var * ex_epsilon[n * N + r]
                                 )
            if 2 in av_alts[n]:
                if y_index[n] == 0:
                    m.addLConstr(dynr_vars[n, r], GRB.GREATER_EQUAL,
                                 gp.quicksum(beta_vars[k] * x[2, n, k] for k in range(K))
                                 + epsilon[2, n * N + r]
                                 + mu_ex_var * ex_epsilon[n * N + r]
                                 - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in range(K))
                                 - epsilon[y_index[n], n * N + r]
                                 - mu_ex_var * ex_epsilon[n * N + r]
                                 - mu_pu_var * pu_epsilon[n * N + r]
                                 )
                elif y_index[n] == 1:
                    m.addLConstr(dynr_vars[n, r], GRB.GREATER_EQUAL,
                                 gp.quicksum(beta_vars[k] * x[2, n, k] for k in range(K))
                                 + epsilon[2, n * N + r]
                                 + mu_ex_var * ex_epsilon[n * N + r]
                                 - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in range(K))
                                 - epsilon[y_index[n], n * N + r]
                                 - mu_pu_var * pu_epsilon[n * N + r]
                                 )
                elif y_index[n] == 2:
                    m.addLConstr(dynr_vars[n, r], GRB.GREATER_EQUAL,
                                 gp.quicksum(beta_vars[k] * x[2, n, k] for k in range(K))
                                 + epsilon[2, n * N + r]
                                 + mu_ex_var * ex_epsilon[n * N + r]
                                 - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in range(K))
                                 - epsilon[y_index[n], n * N + r]
                                 - mu_ex_var * ex_epsilon[n * N + r]
                                 )
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = gp.quicksum(dynr_vars[n, r] for n in range(N) for r in range(R))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    m.setParam('Method', 2)
    m.setParam('ScaleFlag', 0)
    m.setParam('Presolve', 0)
    #
    # m.setParam("Threads", 1)

    # print("Tune LP")
    # m.setParam('tuneTrials', 10)
    # m.setParam('tuneTimeLimit', 7200)
    # m.tune()

    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)] + [mu_ex_var.x, mu_pu_var.x]
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loss = sum(dynr_vars[n, r].x for n in range(N) for r in range(R))
    dnr = {(n, r): dynr_vars[n, r].x for n in range(N) for r in range(R)}
    m.dispose()

    return total_time, solve_time, bestbeta, best_loss, dnr


def maximumlikelihood_linMLE_mixed(y, x, av, K, epsilon, normal_epsilon, mixed_param):
    start_time = time.time()
    N = len(y[0])
    J = len(y)
    R = int(np.shape(epsilon)[1] / N)

    print("Define y_index dict")
    st_time = time.time()
    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)
    print(f"Takes {time.time() - st_time}s")

    not_mixed_params = [k for k in range(K) if not (k == mixed_param or k == mixed_param + 1)]
    av_alts = {n: [i for i in range(J) if av[i][n] == 1] for n in range(N)}


    # gpt3 says do this: but he assume av_alts has same length for everyone. We should create a better eps that adds
    # minus inf to all that are not available
    # import numpy as np
    # import gurobipy as gp
    #
    # # Create a matrix of constraint coefficients
    # coeffs = np.zeros((N * R * len(av_alts), K))
    #
    # for n in range(N):
    #     for r in range(R):
    #         for j in av_alts[n]:
    #             coeffs[n * R * len(av_alts) + r * len(av_alts) + j, mixed_param] = x[j, n, mixed_param]
    #             coeffs[n * R * len(av_alts) + r * len(av_alts) + j, mixed_param + 1] = x[j, n, mixed_param] * \
    #                                                                                    normal_epsilon[n, r]
    #
    # # Create a vector of constraint values
    # values = epsilon[av_alts, range(N), range(R)].flatten() - epsilon[y_index[range(N)], range(N), range(R)].flatten()
    #
    # # Add the constraints using the matrix of coefficients and vector of values
    # m.addLConstrs((dynr_vars[n, r] >= gp.LinExpr(beta_vars.values(),
    #                                              coeffs[n * R * len(av_alts) + r * len(av_alts) + j, :]) + values[
    #                    n * R * len(av_alts) + r * len(av_alts) + j] for n in range(N) for r in range(R) for j in
    #                av_alts[n]))

    # initialize model
    m = gp.Model("linearMLE")

    print("Initialize LP variables")
    st_time = time.time()
    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    # BioGeme does this so lets also do that
    # beta_vars[mixed_param + 1].start = 1
    print(f"Time to initialize LP variables: {time.time() - st_time}s")

    print("Set up LP constraints")
    st_time = time.time()

    # define d_vars
    for n in range(N):
        for r in range(R):
            for j in av_alts[n]:
                m.addLConstr(dynr_vars[n, r], GRB.GREATER_EQUAL,
                             gp.quicksum(beta_vars[k] * x[j, n, k] for k in not_mixed_params)
                             + beta_vars[mixed_param] * x[j, n, mixed_param]
                             + beta_vars[mixed_param + 1] * x[j, n, mixed_param] * normal_epsilon[n * N + r]
                             + epsilon[j, n * N + r]
                             - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in not_mixed_params)
                             - beta_vars[mixed_param] * x[y_index[n], n, mixed_param]
                             - beta_vars[mixed_param + 1] * x[y_index[n], n, mixed_param] * normal_epsilon[n * N + r]
                             - epsilon[y_index[n], n * N + r])
    print(f"Time to set LP constraints: {time.time() - st_time}s")

    # Objective function
    objective = gp.quicksum(dynr_vars[n, r] for n in range(N) for r in range(R))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 2)
    m.setParam('ScaleFlag', 0)
    m.setParam('Presolve', 0)
    #
    # m.setParam("Threads", 1)

    # print("Tune LP")
    # m.setParam('tuneTrials', 10)
    # m.setParam('tuneTimeLimit', 7200)
    # m.tune()

    print("Solve LP")
    st_time = time.time()
    m.optimize()
    # m.tune()
    solve_time = time.time() - st_time
    print(f"Time to solve LP: {solve_time}s")

    print("linMLE mixed Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    #
    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=0,
    #                                                                           mixed=[mixed_index, normal_epsilon])

    best_loss = sum(dynr_vars[n, r].x for n in range(N) for r in range(R))
    dnr = {(n, r): dynr_vars[n, r].x for n in range(N) for r in range(R)}
    m.dispose()

    return total_time, solve_time, bestbeta, best_loss, dnr


def maximumlikelihood_linMLE_vers2_matrix(y, x, K, R, epsilon, time_limit):
    start_time = time.time()
    N = len(y[0])
    J = len(y)

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    # initialize model
    m = gp.Model("linearMLE_matrix")

    # Create variables
    #     beta_vars = m.addMVar(shape=K, vtype=GRB.CONTINUOUS, name="beta")
    #     d_vars = m.addMVar(shape=(N, R), lb=0, vtype=GRB.CONTINUOUS, name="d")

    all_vars = m.addMVar(shape=K + R * N, vtype=GRB.CONTINUOUS, name="allvars")

    # define constraint matrix A
    # initaize first dummy row of zeros
    A = np.zeros(K + R * N)
    for r_row in range(R):
        for n_row in range(N):
            for i_row in range(J):
                # build row rnj with K + RN entries
                # intialize empty row
                newrow_K = np.zeros(K)
                arr_RN = np.zeros(shape=(R, N))
                # fill first K entries relating to beta_k
                for k in range(K):
                    newrow_K[k] = - (x[i_row, n_row, k] - x[y_index[n_row], n_row, k])
                    # fill next RN entries relating to d_rn
                # its literally always zero except for d_nrow,rrow
                arr_RN[r_row, n_row] = 1
                newrow_RN = arr_RN.reshape(-1)
                newrow = np.concatenate([newrow_K, newrow_RN])
                A = np.vstack((A, newrow))
                # delete dummy row
    A = np.delete(A, 0, axis=0)

    # define rhs
    rhs_arr = np.empty(shape=(R, N, J))
    for r in range(R):
        for n in range(N):
            for i in range(J):
                rhs_arr[r, n, j] = epsilon[i, n, r] - epsilon[y_index[n], n, r]
    rhs = rhs_arr.reshape(-1)

    # Add constraints
    m.addConstr(A @ all_vars <= rhs, name="c")

    # Objective function
    obj = np.concatenate([np.zeros(shape=K), np.ones(shape=N * R)])
    m.setObjective(obj @ all_vars, GRB.MINIMIZE)

    m.setParam('OutputFlag', 0)
    # m.setParam('DualReductions', 0)
    print(f"Total LP setup time = {time.time() - start_time}s")
    st_time = time.time()
    m.optimize()
    print(f"Total LP optimization time = {time.time() - st_time}s")

    total_time = time.time() - start_time
    bestbeta = [all_vars[k].x for k in range(K)]
    obj = m.ObjVal

    m.dispose()

    return total_time, bestbeta, obj


def maximumlikelihood_linMLE_vers2(y, x, K, R, epsilon, time_limit):
    start_time = time.time()
    N = len(y[0])
    J = len(y)

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    # initialize model
    m = gp.Model("linearMLE")

    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}

    # define d_vars
    for n in range(N):
        for r in range(R):
            for j in range(J):
                m.addConstr(dynr_vars[n, r] >= gp.quicksum(beta_vars[k] * x[j, n, k] for k in range(K))
                            - gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in range(K))
                            + epsilon[j, n, r]
                            - epsilon[y_index[n], n, r])

    # Objective function
    objective = gp.quicksum(dynr_vars[n, r] for n in range(N) for r in range(R))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    # m.setParam('DualReductions', 0)
    m.optimize()

    print("linMLE Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()

    omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
                                                                              linFix=0)

    derivative_of_exp_loss = sum(((1 / R) * sum(omega[y_index[n], n, r] for r in range(R)) - 1) for n in range(N))
    print(f"Derivative of expected loss = {derivative_of_exp_loss}")

    # success_counter = dict()
    # abs_diff = dict()
    #
    # for n in range(N):
    #     success_counter_ind = 0
    #     abs_diff_ind = 0
    #     for r in range(R):
    #
    #         # print("")
    #         # print(f"linMLE ind {n}, scenario {r}:")
    #         # print(f"U[0, {n}, {r}] = {Uinr[0, n, r]}")
    #         # print(f"U[1, {n}, {r}] = {Uinr[1, n, r]}")
    #         # print(f"abs. Difference = {abs(Uinr[0, n, r] - Uinr[1, n, r])}")
    #         # print(f"omega[0, {n}, {r}] = {omega[0, n, r]}")
    #         # print(f"omega[1, {n}, {r}] = {omega[1, n, r]}")
    #         # print(f"observed choice = {y_index[n]}")
    #         # print("")
    #
    #         abs_diff_ind += abs(Uinr[0, n, r] - Uinr[1, n, r])
    #
    #         if int(omega[y_index[n], n, r]) == 1:
    #             success_counter_ind += 1
    #     success_counter[n] = success_counter_ind
    #     abs_diff[n] = abs_diff_ind
    #
    # print(f"Model recreated observed choice {sum(success_counter[n] for n in range(N))} times.")
    # print(f"objective z, y sum = {sum(-y[i, n] * z_dict[i, n] for i in range(J) for n in range(N))}")
    # print(f"We get total sum abs = {sum(abs_diff[n] for n in range(N))}")
    # print("")
    # for n in range(N):
    #     print(f"For ind {n} we succeed {success_counter[n]} times")
    #     print(f"We get objective value = {sum(-y[i, n] * z_dict[i, n] for i in range(J))}")
    #     print(f"We get sum abs = {abs_diff[n]}")
    # print("")

    best_loglike = total_obj

    return total_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_vers2test(y, x, K, R, epsilon, time_limit):
    start_time = time.time()
    N = len(y[0])
    J = len(y)

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    # initialize model
    m = gp.Model("linearMLE")

    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}

    largeM = 1e12

    # define d_vars
    for n in range(N):
        for r in range(R):
            for j in range(J):
                m.addConstr(dynr_vars[n, r] >= largeM * (gp.quicksum(beta_vars[k] * x[j, n, k] for k in range(K))
                                                         - gp.quicksum(
                            beta_vars[k] * x[y_index[n], n, k] for k in range(K))
                                                         + epsilon[j, n, r]
                                                         - epsilon[y_index[n], n, r])
                            )

    # Objective function
    objective = gp.quicksum(dynr_vars[n, r] for n in range(N) for r in range(R))

    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    # m.setParam('DualReductions', 0)
    m.optimize()

    print("linMLE Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]
    m.dispose()

    omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
                                                                              linFix=1)
    #
    # success_counter = dict()
    # abs_diff = dict()
    #
    # for n in range(N):
    #     success_counter_ind = 0
    #     abs_diff_ind = 0
    #     for r in range(R):
    #
    #         # print("")
    #         # print(f"linMLE ind {n}, scenario {r}:")
    #         # print(f"U[0, {n}, {r}] = {Uinr[0, n, r]}")
    #         # print(f"U[1, {n}, {r}] = {Uinr[1, n, r]}")
    #         # print(f"abs. Difference = {abs(Uinr[0, n, r] - Uinr[1, n, r])}")
    #         # print(f"omega[0, {n}, {r}] = {omega[0, n, r]}")
    #         # print(f"omega[1, {n}, {r}] = {omega[1, n, r]}")
    #         # print(f"observed choice = {y_index[n]}")
    #         # print("")
    #
    #         abs_diff_ind += abs(Uinr[0, n, r] - Uinr[1, n, r])
    #
    #         if int(omega[y_index[n], n, r]) == 1:
    #             success_counter_ind += 1
    #     success_counter[n] = success_counter_ind
    #     abs_diff[n] = abs_diff_ind
    #
    # print(f"Model recreated observed choice {sum(success_counter[n] for n in range(N))} times.")
    # print(f"objective z, y sum = {sum(-y[i, n] * z_dict[i, n] for i in range(J) for n in range(N))}")
    # print(f"We get total sum abs = {sum(abs_diff[n] for n in range(N))}")
    # print("")
    # for n in range(N):
    #     print(f"For ind {n} we succeed {success_counter[n]} times")
    #     print(f"We get objective value = {sum(-y[i, n] * z_dict[i, n] for i in range(J))}")
    #     print(f"We get sum abs = {abs_diff[n]}")
    # print("")

    best_loglike = total_obj

    return total_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_vers9(y, x, K, R, epsilon, time_limit):
    start_time = time.time()
    N = len(y[0])
    J = len(y)

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    not_obs_alts = {n: [i for i in range(J) if y[i, n] == 0] for n in range(N)}

    # initialize model
    m = gp.Model("linearMLE")

    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dyn_vars = {n: m.addVar(lb=-inf, ub=0, vtype=GRB.CONTINUOUS, name=f"dyn_{n}")
                for n in range(N)}

    # define d_vars
    for n in range(N):
        for j in not_obs_alts[n]:
            m.addConstr(dyn_vars[n] <= (1 / R) * gp.quicksum(
                gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in range(K))
                - gp.quicksum(beta_vars[k] * x[j, n, k] for k in range(K))
                + epsilon[y_index[n], n, r]
                - epsilon[j, n, r]
                for r in range(R)))

    # Objective function
    objective = gp.quicksum(dyn_vars[n] for n in range(N))

    m.setObjective(objective, GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    m.setParam('DualReductions', 0)
    m.optimize()

    print("linMLE Status = ", m.status)

    total_time = time.time() - start_time
    bestbeta = [beta_vars[k].x for k in range(K)]

    m.dispose()

    omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
                                                                              linFix=1)
    best_loglike = total_obj

    return total_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_latent_quad1(y, x, av, K, R, epsilon, logOfZer, latent, latent_index, time_limit):
    start_time = time.time()
    N = len(y[0])
    J = len(y)

    logOfZero = logOfZer

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    # initialize model
    m = gp.Model("linearMLE")

    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}")
                 for n in range(N) for r in range(R)}
    class_vars = {c: m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"class_{c}")
                  for c in range(latent)}

    # avoid local optima
    for c in range(latent):
        class_vars[c].start = 1 / (latent + 1)

    # define d_vars

    if latent == 1:
        nonlatent = [k for k in range(K) if not k == latent_index]
        for n in range(N):
            for r in range(R):
                for j in [i for i in range(J) if av[i][n] == 1]:
                    m.addConstr(dynr_vars[n, r] <= gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in nonlatent)
                                + class_vars[0] * beta_vars[latent_index] * x[y_index[n], n, latent_index]
                                + epsilon[y_index[n], n, r]
                                - (gp.quicksum(beta_vars[k] * x[j, n, k] for k in nonlatent)
                                   + class_vars[0] * beta_vars[latent_index] * x[j, n, latent_index]
                                   + epsilon[j, n, r]))

    # Objective function
    objective = gp.quicksum(dynr_vars[n, r] for r in range(R) for n in range(N))
    m.setObjective(objective, GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    # m.setParam('DualReductions', 0)
    # m.setParam("TimeLimit", 3600)
    m.setParam("NonConvex", 2)
    # m.setParam("NumericFocus", 3)
    # m.setParam("Presolve", presolve)
    # m.setParam("Method", solver_method)
    # m.setParam("PreCrush", 1)
    m.optimize()

    total_time = time.time() - start_time

    print("")
    print("linMLE quad 1 Status = ", m.status)
    print("Estimated beta parameters:")
    print([beta_vars[k].x for k in range(K)])
    print("Estimated class parameters:")
    print([class_vars[c].x for c in range(latent)])
    print("Time: ", time.time() - start_time)

    bestbeta = [beta_vars[k].x for k in range(K)]
    bestbeta.append(class_vars[0].x)

    m.dispose()

    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=1)
    best_loglike = 25

    return total_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_latent_quad2(y, x, av, K, R, epsilon, logOfZer, latent, latent_indices, time_limit):
    start_time = time.time()
    N = len(y[0])
    J = len(y)

    logOfZero = logOfZer

    y_index = dict()
    for n in range(N):
        for i in range(J):
            if y[i, n] == 1:
                y_index[n] = int(i)

    nonlatent = [k for k in range(K) if k not in latent_indices]

    # initialize model
    m = gp.Model("linearMLE")

    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    dynr_vars = {(n, r, l): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"dynr_{n}_{r}_{l}")
                 for n in range(N) for r in range(R) for l in range(latent)}
    pi_vars = {c: m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"class_{c}")
               for c in range(latent - 1)}

    # avoid local optima
    for c in range(latent - 1):
        pi_vars[c].start = 1 / latent

    # I think this is necessary
    m.addConstr(gp.quicksum(pi_vars[c] for c in range(len(latent_indices))) <= 1)

    # convention: (this holds for all numbers of latent classes)
    # class_0 takes into account indices[0]
    # class_1 takes into account indices[1]
    # class_3 takes into account indices[0] & indices[1]
    # class_4 has none

    # define d_vars
    for n in range(N):
        for r in range(R):
            for j in [i for i in range(J) if av[i][n] == 1]:
                if len(latent_indices) == 1:
                    m.addConstr(dynr_vars[n, r, 0] <= gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in nonlatent)
                                + beta_vars[latent_indices[0]] * x[y_index[n], n, latent_indices[0]]
                                + epsilon[y_index[n], n, r]
                                - (gp.quicksum(beta_vars[k] * x[j, n, k] for k in nonlatent)
                                   + beta_vars[latent_indices[0]] * x[j, n, latent_indices[0]]
                                   + epsilon[j, n, r]))

                    m.addConstr(dynr_vars[n, r, 1] <= gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in nonlatent)
                                + epsilon[y_index[n], n, r]
                                - (gp.quicksum(beta_vars[k] * x[j, n, k] for k in nonlatent)
                                   + epsilon[j, n, r]))
                elif len(latent_indices) == 2 and latent == 3:
                    m.addConstr(dynr_vars[n, r, 0] <= gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in nonlatent)
                                + beta_vars[latent_indices[0]] * x[y_index[n], n, latent_indices[0]]
                                + epsilon[y_index[n], n, r]
                                - (gp.quicksum(beta_vars[k] * x[j, n, k] for k in nonlatent)
                                   + beta_vars[latent_indices[0]] * x[j, n, latent_indices[0]]
                                   + epsilon[j, n, r]))
                    m.addConstr(dynr_vars[n, r, 1] <= gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in nonlatent)
                                + beta_vars[latent_indices[1]] * x[y_index[n], n, latent_indices[1]]
                                + epsilon[y_index[n], n, r]
                                - (gp.quicksum(beta_vars[k] * x[j, n, k] for k in nonlatent)
                                   + beta_vars[latent_indices[1]] * x[j, n, latent_indices[1]]
                                   + epsilon[j, n, r]))
                    m.addConstr(dynr_vars[n, r, 2] <= gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in nonlatent)
                                + beta_vars[latent_indices[0]] * x[y_index[n], n, latent_indices[0]]
                                + beta_vars[latent_indices[1]] * x[y_index[n], n, latent_indices[1]]
                                + epsilon[y_index[n], n, r]
                                - (gp.quicksum(beta_vars[k] * x[j, n, k] for k in nonlatent)
                                   + beta_vars[latent_indices[0]] * x[j, n, latent_indices[0]]
                                   + beta_vars[latent_indices[1]] * x[j, n, latent_indices[1]]
                                   + epsilon[j, n, r]))
                elif len(latent_indices) == 2 and latent == 4:
                    m.addConstr(dynr_vars[n, r, 0] <= gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in nonlatent)
                                + beta_vars[latent_indices[0]] * x[y_index[n], n, latent_indices[0]]
                                + epsilon[y_index[n], n, r]
                                - (gp.quicksum(beta_vars[k] * x[j, n, k] for k in nonlatent)
                                   + beta_vars[latent_indices[0]] * x[j, n, latent_indices[0]]
                                   + epsilon[j, n, r]))
                    m.addConstr(dynr_vars[n, r, 1] <= gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in nonlatent)
                                + beta_vars[latent_indices[1]] * x[y_index[n], n, latent_indices[1]]
                                + epsilon[y_index[n], n, r]
                                - (gp.quicksum(beta_vars[k] * x[j, n, k] for k in nonlatent)
                                   + beta_vars[latent_indices[1]] * x[j, n, latent_indices[1]]
                                   + epsilon[j, n, r]))
                    m.addConstr(dynr_vars[n, r, 2] <= gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in nonlatent)
                                + beta_vars[latent_indices[0]] * x[y_index[n], n, latent_indices[0]]
                                + beta_vars[latent_indices[1]] * x[y_index[n], n, latent_indices[1]]
                                + epsilon[y_index[n], n, r]
                                - (gp.quicksum(beta_vars[k] * x[j, n, k] for k in nonlatent)
                                   + beta_vars[latent_indices[0]] * x[j, n, latent_indices[0]]
                                   + beta_vars[latent_indices[1]] * x[j, n, latent_indices[1]]
                                   + epsilon[j, n, r]))
                    m.addConstr(dynr_vars[n, r, 3] <= gp.quicksum(beta_vars[k] * x[y_index[n], n, k] for k in nonlatent)
                                + epsilon[y_index[n], n, r]
                                - (gp.quicksum(beta_vars[k] * x[j, n, k] for k in nonlatent)
                                   + epsilon[j, n, r]))

    # Objective function
    if latent == 2:
        objective = pi_vars[0] * gp.quicksum(dynr_vars[n, r, 0] for r in range(R) for n in range(N)) \
                    + (1 - pi_vars[0]) * gp.quicksum(dynr_vars[n, r, 1] for r in range(R) for n in range(N))
    elif latent == 3:
        objective = pi_vars[0] * gp.quicksum(dynr_vars[n, r, 0] for r in range(R) for n in range(N)) \
                    + pi_vars[1] * gp.quicksum(dynr_vars[n, r, 1] for r in range(R) for n in range(N)) \
                    + (1 - pi_vars[0] - pi_vars[1]) * gp.quicksum(
            dynr_vars[n, r, 2] for r in range(R) for n in range(N))
    elif latent == 4:
        objective = pi_vars[0] * gp.quicksum(dynr_vars[n, r, 0] for r in range(R) for n in range(N)) \
                    + pi_vars[1] * gp.quicksum(dynr_vars[n, r, 1] for r in range(R) for n in range(N)) \
                    + pi_vars[2] * gp.quicksum(dynr_vars[n, r, 2] for r in range(R) for n in range(N)) \
                    + (1 - pi_vars[0] - pi_vars[1] - pi_vars[2]) * gp.quicksum(dynr_vars[n, r, 3]
                                                                               for r in range(R) for n in range(N))

    m.setObjective(objective, GRB.MAXIMIZE)
    m.setParam('OutputFlag', 1)
    # m.setParam('DualReductions', 0)
    # m.setParam("TimeLimit", 3600)
    m.setParam("NonConvex", 2)
    # m.setParam("NumericFocus", 3)
    # m.setParam("Presolve", presolve)
    # m.setParam("Method", solver_method)
    # m.setParam("PreCrush", 1)
    m.optimize()

    total_time = time.time() - start_time

    print("")
    print("linMLE quad 2 Status = ", m.status)
    print("Estimated beta parameters:")
    print([beta_vars[k].x for k in range(K)])
    print("Estimated class parameters:")
    print([pi_vars[c].x for c in range(latent - 1)])
    print("Time: ", time.time() - start_time)

    bestbeta = [beta_vars[k].x for k in range(K)]
    for c in range(latent - 1):
        bestbeta.append(pi_vars[c].x)

    m.dispose()

    # omega, Unr, Uinr, beta, s_dict, z_dict, total_obj = compute_sol_from_beta(J, N, R, x, y, epsilon, bestbeta,
    #                                                                           linFix=1)
    best_loglike = 25

    return total_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_vers1(y, x, K, R, epsilon, logOfZer, time_limit, warm_start):
    start_time = time.time()
    N = len(y[0])
    J = int(len(x.keys()) / (N * K))
    logL = np.array([logOfZer] +
                    [(1 + r) * np.log(r) - r * np.log(1 + r) for r in range(1, R + 1)])
    logK = np.array([logOfZer] +
                    [np.log(r) - np.log(1 + r) for r in range(1, R + 1)])

    # initialize model
    m = gp.Model("full MILP")

    # initialize variables
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}
    Uinr_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"Uinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    omega_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.BINARY, name=f"omega_{i}_{n}_{r}")
                  for i in range(J) for n in range(N) for r in range(R)}
    d_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.BINARY, name=f"d_{i}_{n}_{r}")
              for i in range(J) for n in range(N) for r in range(R)}
    f_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.BINARY, name=f"f_{i}_{n}_{r}")
              for i in range(J) for n in range(N) for r in range(R)}
    h_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.BINARY, name=f"h_{i}_{n}_{r}")
              for i in range(J) for n in range(N) for r in range(R)}
    s_vars = {(i, n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"s_{i}_{n}")
              for i in range(J) for n in range(N)}
    z_vars = {(i, n): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"z_{i}_{n}")
              for i in range(J) for n in range(N)}

    if warm_start is not None:
        omega_s = warm_start[0]
        Unr_s = warm_start[1]
        Uinr_s = warm_start[2]
        beta_s = warm_start[3]
        s_dict_s = warm_start[4]
        z_dict_s = warm_start[5]
        eta_s = warm_start[6]

        for k in range(K):
            beta_vars[k].start = float(beta_s[k])
        for i in range(J):
            for n in range(N):
                s_vars[i, n].start = float(s_dict_s[i, n])
                z_vars[i, n].start = float(z_dict_s[i, n])
                for r in range(R):
                    omega_vars[i, n, r].start = float(omega_s[i, n, r])
                    Uinr_vars[i, n, r].start = float(Uinr_s[i, n, r])

    # creating the choice
    ##################################################################

    # choice variable simplex constraint
    for n in range(N):
        for r in range(R):
            m.addConstr(gp.quicksum(omega_vars[i, n, r] for i in range(J)) == 1)

    # define Utilities
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(
                    Uinr_vars[i, n, r] - gp.quicksum(beta_vars[k] * x[i, n, k] for k in range(K))
                    == epsilon[i, n, r])

    # define d_vars
    for n in range(N):
        for i in range(J):
            for r in range(R):
                # for j not equal to i
                for j in [j for j in range(J) if not j == i]:
                    m.addConstr(d_vars[i, n, r] <= Uinr_vars[i, n, r] - Uinr_vars[j, n, r])

    # define f_vars
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(f_vars[i, n, r] == y[i, n] * d_vars[i, n, r])

    # define h_vars
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(
                    h_vars[i, n, r] == y[i, n] * (gp.quicksum(f_vars[i, n, r] for i in range(J)) - d_vars[i, n, r]))

    # concluding choice constraints
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(omega_vars[i, n, r] <= bigM * f_vars[i, n, r] + bigM * gp.quicksum(
                    h_vars[i, n, r] for i in range(J)))

    ##################################################################

    # Rest of the constraints (log linearization)

    # define s_vars
    for n in range(N):
        for i in range(J):
            m.addConstr(s_vars[i, n] - gp.quicksum(omega_vars[i, n, r] for r in range(R)) == 0)

    # define z_vars
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(z_vars[i, n] + logK[r] * s_vars[i, n] <= logL[r])

    # Objective function
    objective = gp.quicksum(-y[i, n] * z_vars[i, n] for i in range(J) for n in range(N))
    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    m.setParam('DualReductions', 0)
    m.setParam("TimeLimit", time_limit)
    # m.setParam("NonConvex", 0)
    # m.setParam("NumericFocus", numeric_focus)
    # m.setParam("Presolve", presolve)
    # m.setParam("Method", solver_method)
    # m.setParam("PreCrush", 1)
    m.optimize()

    print("linMLE Status = ", m.status)

    if m.status == GRB.TIME_LIMIT:
        if m.SolCount >= 1:
            objective_value = sum(-y[i, n] * z_vars[i, n].X for i in range(J) for n in range(N)) \
                              + N * np.log(R)
        else:
            objective_value = 1000
    else:
        objective_value = sum(-y[i, n] * z_vars[i, n].x for i in range(J) for n in range(N)) \
                          + N * np.log(R)

    total_time = time.time() - start_time
    try:
        best_lowerbound = m.ObjBoundC + N * np.log(R)
    except:
        best_lowerbound = -1000
    if m.SolCount >= 1:
        bestbeta = [beta_vars[k].x for k in range(K)]
    else:
        bestbeta = [1000 for k in range(K)]

    best_loglike = -objective_value

    # for i in range(J):
    #     for n in range(N):
    #         for r in range(R):
    #             print(f"w_{i}{n}{r} = {omega_vars[i, n, r].x}")

    m.dispose()

    return total_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_rel2(y, x, K, R, epsilon, logOfZer, beta, time_limit, warm_start):
    start_time = time.time()
    N = len(y[0])
    J = int(len(x.keys()) / (N * K))
    logL = np.array([logOfZer] +
                    [(1 + r) * np.log(r) - r * np.log(1 + r) for r in range(1, R + 1)])
    logK = np.array([logOfZer] +
                    [np.log(r) - np.log(1 + r) for r in range(1, R + 1)])

    # initialize model
    m = gp.Model("full MILP")

    # initialize variables
    omega_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}_{r}")
                  for i in range(J) for n in range(N) for r in range(R)}
    chi_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"chi_{i}_{r}")
                for i in range(J) for n in range(N) for r in range(R)}
    s_vars = {(i, n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"s_{i}_{n}")
              for i in range(J) for n in range(N)}
    phi_vars = {(i, n, r, k): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"phi_{i}_{r}_{k}")
                for i in range(J) for n in range(N) for r in range(R) for k in range(K)}
    z_vars = {(i, n): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"z_{i}_{n}")
              for i in range(J) for n in range(N)}
    U_vars = {(n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"U_{r}")
              for n in range(N) for r in range(R)}
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}

    for n in range(N):
        # Constraints mu
        for r in range(R):
            m.addConstr(gp.quicksum(omega_vars[i, n, r] for i in range(J)) == 1)

        # Constraints alpha
        for i in range(J):
            for r in range(R):
                m.addConstr(gp.quicksum(beta_vars[k] * x[i, n, k] for k in range(K)) - U_vars[n, r]
                            <= - epsilon[i, n, r])

        # Constraints zeta
        for r in range(R):
            m.addConstr(U_vars[n, r] - gp.quicksum((phi_vars[i, n, r, k] - chi_vars[i, n, r] * beta[k]) * x[i, n, k]
                                                   for i in range(J) for k in range(K))
                        - gp.quicksum(omega_vars[i, n, r] * epsilon[i, n, r] for i in range(J)) <= 0)

        # Constraints pi
        pi_constr = dict()
        for i in range(J):
            for r in range(R):
                pi_constr[i, r] = m.addConstr(chi_vars[i, n, r] + omega_vars[i, n, r] == 1)

        # Constraints lambda_0

        # these are the constraints we fully relax in this version

        # lambda_0_constr = dict()
        # for i in range(J):
        #     for r in range(R):
        #         for k in range(K):
        #             lambda_0_constr[i, n, r, k] = m.addConstr(phi_vars[i, n, r, k] == phi[i, n, r, k])

        # Constraint beta fixed
        for k in range(K):
            m.addConstr(beta_vars[k] - gp.quicksum((phi_vars[i, n, 0, k] - chi_vars[i, n, 0] * beta[k])
                                                   for i in range(J))
                        == 0)

        # Constraints theta
        for i in range(J):
            m.addConstr(s_vars[i, n] - gp.quicksum(omega_vars[i, n, r] for r in range(R)) == 0)

        # Constraints xi
        for i in range(J):
            for r in range(R):
                m.addConstr(z_vars[i, n] + logK[r] * s_vars[i, n] <= logL[r])

    # Objective function
    objective = gp.quicksum(-y[i, n] * z_vars[i, n] for i in range(J) for n in range(N))
    m.setObjective(objective, GRB.MINIMIZE)

    # solve model
    m.setParam('OutputFlag', 1)
    # m.setParam("Presolve", SP_presolve)
    # m.setParam("Method", SP_method)
    # m.setParam("NumericFocus", SP_numericFocus)
    # m.setParam("FeasibilityTol", SP_feasTol)
    # m.setParam("OptimalityTol", SP_dualFeasTol)
    m.optimize()
    print("Rel2 Status = ", m.status)

    if m.status == GRB.TIME_LIMIT:
        if m.SolCount >= 1:
            objective_value = sum(-y[i, n] * z_vars[i, n].X for i in range(J) for n in range(N)) \
                              + N * np.log(R)
        else:
            objective_value = 1000
    else:
        objective_value = sum(-y[i, n] * z_vars[i, n].x for i in range(J) for n in range(N)) \
                          + N * np.log(R)

    total_time = time.time() - start_time
    try:
        best_lowerbound = m.ObjBoundC + N * np.log(R)
    except:
        best_lowerbound = -1000
    if m.SolCount >= 1:
        bestbeta = [beta_vars[k].x for k in range(K)]
    else:
        bestbeta = [1000 for k in range(K)]

    best_loglike = -objective_value

    for i in range(J):
        for n in range(N):
            for r in range(R):
                print(f"w_{i}{n}{r} = {omega_vars[i, n, r].x}")

    m.dispose()

    return total_time, bestbeta, best_loglike


def maximumlikelihood_linMLE_rel1(y, x, K, R, epsilon, logOfZer, time_limit, warm_start):
    start_time = time.time()
    N = len(y[0])
    J = int(len(x.keys()) / (N * K))
    logL = np.array([logOfZer] +
                    [(1 + r) * np.log(r) - r * np.log(1 + r) for r in range(1, R + 1)])
    logK = np.array([logOfZer] +
                    [np.log(r) - np.log(1 + r) for r in range(1, R + 1)])

    # initialize model
    m = gp.Model("full MILP")

    # initialize variables
    omega_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}_{r}")
                  for i in range(J) for n in range(N) for r in range(R)}
    chi_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"chi_{i}_{r}")
                for i in range(J) for n in range(N) for r in range(R)}
    s_vars = {(i, n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"s_{i}_{n}")
              for i in range(J) for n in range(N)}
    eta_vars = {(i, n, r, k): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{r}_{k}")
                for i in range(J) for n in range(N) for r in range(R) for k in range(K)}
    z_vars = {(i, n): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"z_{i}_{n}")
              for i in range(J) for n in range(N)}
    U_vars = {(n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"U_{r}")
              for n in range(N) for r in range(R)}
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}

    for n in range(N):
        # Constraints mu
        for r in range(R):
            m.addConstr(gp.quicksum(omega_vars[i, n, r] for i in range(J)) == 1)

        # Constraints alpha
        for i in range(J):
            for r in range(R):
                m.addConstr(gp.quicksum(beta_vars[k] * x[i, n, k] for k in range(K)) - U_vars[n, r]
                            <= - epsilon[i, n, r])

        # Constraints zeta
        for r in range(R):
            m.addConstr(U_vars[n, r] - gp.quicksum(eta_vars[i, n, r, k] * x[i, n, k]
                                                   for i in range(J) for k in range(K))
                        - gp.quicksum(omega_vars[i, n, r] * epsilon[i, n, r] for i in range(J)) <= 0)

        # Constraints pi
        pi_constr = dict()
        for i in range(J):
            for r in range(R):
                pi_constr[i, r] = m.addConstr(chi_vars[i, n, r] + omega_vars[i, n, r] == 1)

        # Constraints lambda_0

        # these are the constraints we fully relax in this version

        lambda_0_constr = dict()
        for i in range(J):
            for r in range(R):
                for k in range(K):
                    lambda_0_constr[i, n, r, k] = m.addConstr(
                        eta_vars[i, n, r, k] + beta_vars[k] * chi_vars[i, n, r] == beta_vars[k])

        # Constraint beta fixed
        for k in range(K):
            m.addConstr(beta_vars[k] - gp.quicksum(eta_vars[i, n, 0, k] for i in range(J)) == 0)

        # Constraints theta
        for i in range(J):
            m.addConstr(s_vars[i, n] - gp.quicksum(omega_vars[i, n, r] for r in range(R)) == 0)

        # Constraints xi
        for i in range(J):
            for r in range(R):
                m.addConstr(z_vars[i, n] + logK[r] * s_vars[i, n] <= logL[r])

    # Objective function
    objective = gp.quicksum(-y[i, n] * z_vars[i, n] for i in range(J) for n in range(N))
    m.setObjective(objective, GRB.MINIMIZE)

    # solve model
    m.setParam('OutputFlag', 1)
    m.setParam("NonConvex", 2)
    # m.setParam("Presolve", SP_presolve)
    # m.setParam("Method", SP_method)
    # m.setParam("NumericFocus", SP_numericFocus)
    # m.setParam("FeasibilityTol", SP_feasTol)
    # m.setParam("OptimalityTol", SP_dualFeasTol)
    m.optimize()
    print("Rel1 Status = ", m.status)

    if m.status == GRB.TIME_LIMIT:
        if m.SolCount >= 1:
            objective_value = sum(-y[i, n] * z_vars[i, n].X for i in range(J) for n in range(N)) \
                              + N * np.log(R)
        else:
            objective_value = 1000
    else:
        objective_value = sum(-y[i, n] * z_vars[i, n].x for i in range(J) for n in range(N)) \
                          + N * np.log(R)

    total_time = time.time() - start_time
    try:
        best_lowerbound = m.ObjBoundC + N * np.log(R)
    except:
        best_lowerbound = -1000
    if m.SolCount >= 1:
        bestbeta = [beta_vars[k].x for k in range(K)]
    else:
        bestbeta = [1000 for k in range(K)]

    best_loglike = -objective_value

    # for i in range(J):
    #     for n in range(N):
    #         for r in range(R):
    #             print(f"w_{i}{n}{r} = {omega_vars[i, n, r].x}")

    m.dispose()

    return total_time, bestbeta, best_loglike
