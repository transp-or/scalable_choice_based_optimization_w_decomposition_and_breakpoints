import numpy as np
import time

import gurobipy as gp
from gurobipy import GRB

import datetime

# Get the current date and time
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

inf = float("inf")
logOfZero = -100
bigM = 1000


def MILP_MSLE(y, x, epsilon, time_limit):
    start_time = time.time()
    N = len(y[0])
    J = x.shape[0]
    K = x.shape[2]
    R = epsilon.shape[2]
    logL = np.array([logOfZero] +
                    [(1 + r) * np.log(r) - r * np.log(1 + r) for r in range(1, R + 1)])
    logK = np.array([logOfZero] +
                    [np.log(r) - np.log(1 + r) for r in range(1, R + 1)])

    # initialize model
    m = gp.Model("full MILP")

    # initialize variables
    omega_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.BINARY, name=f"omega_{i}_{n}_{r}")
                  for i in range(J) for n in range(N) for r in range(R)}
    s_vars = {(i, n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"s_{i}_{n}")
              for i in range(J) for n in range(N)}
    z_vars = {(i, n): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"z_{i}_{n}")
              for i in range(J) for n in range(N)}
    U_vars = {(n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"U_{n}_{r}")
              for n in range(N) for r in range(R)}
    Uinr_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"Uinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    eta_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{n}_{r}")
                for i in range(J) for n in range(N) for r in range(R)}
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}

    # "nonlinear" constraints
    for n in range(N):
        for r in range(R):
            m.addConstr(U_vars[n, r] - gp.quicksum(eta_vars[i, n, r] for i in range(J)) == 0)

    # linearizing constraints
    for i in range(J):
        for n in range(N):
            for r in range(R):
                m.addConstr(eta_vars[i, n, r] >= -omega_vars[i, n, r] * bigM)
                m.addConstr(eta_vars[i, n, r] <= omega_vars[i, n, r] * bigM)
                m.addConstr(eta_vars[i, n, r] >= Uinr_vars[i, n, r] - (1 - omega_vars[i, n, r]) * bigM)
                m.addConstr(eta_vars[i, n, r] <= Uinr_vars[i, n, r] + (1 - omega_vars[i, n, r]) * bigM)

    # Constraints mu
    for n in range(N):
        for r in range(R):
            m.addConstr(gp.quicksum(omega_vars[i, n, r] for i in range(J)) == 1)

    # Constraints theta
    for n in range(N):
        for i in range(J):
            m.addConstr(s_vars[i, n] - gp.quicksum(omega_vars[i, n, r] for r in range(R)) == 0)

    # Constraints alpha
    for i in range(J):
        for n in range(N):
            for r in range(R):
                m.addConstr(Uinr_vars[i, n, r] - U_vars[n, r] <= 0)

    # Constraints xi
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(z_vars[i, n] + logK[r] * s_vars[i, n] <= logL[r])

    # Constraints kappa
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(
                    Uinr_vars[i, n, r] - gp.quicksum(beta_vars[k] * x[i, n, k] for k in range(K))
                    == epsilon[i, n, r])

    # Objective function
    objective = gp.quicksum(-y[i, n] * z_vars[i, n] for i in range(J) for n in range(N))
    m.setObjective(objective, GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)
    m.setParam("TimeLimit", time_limit)
    m.optimize()

    total_time = time.time() - start_time
    objective_value = sum(-y[i, n] * z_vars[i, n].x for i in range(J) for n in range(N)) \
                      + N * np.log(R)
    best_lowerbound = m.ObjBoundC + N * np.log(R)
    bestbeta = [beta_vars[k].x for k in range(K)]
    best_loglike = -objective_value
    m.dispose()

    return total_time, bestbeta, best_loglike, best_lowerbound


def MILP_MixedMSLE(y, x, epsilon, sigma, time_limit):
    start_time = time.time()
    N = len(y[0])
    J = x.shape[0]
    K = x.shape[2]
    R = epsilon.shape[2]

    logL = np.array([logOfZero] +
                    [(1 + r) * np.log(r) - r * np.log(1 + r) for r in range(1, R + 1)])
    logK = np.array([logOfZero] +
                    [np.log(r) - np.log(1 + r) for r in range(1, R + 1)])

    # initialize model
    m = gp.Model("fullMixedMILP")

    # initialize variables
    omega_vars = {(i, n, r): m.addVar(lb=0, ub=inf, vtype=GRB.BINARY, name=f"omega_{i}_{n}_{r}")
                  for i in range(J) for n in range(N) for r in range(R)}
    s_vars = {(i, n): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"s_{i}_{n}")
              for i in range(J) for n in range(N)}
    z_vars = {(i, n): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"z_{i}_{n}")
              for i in range(J) for n in range(N)}
    U_vars = {(n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"U_{n}_{r}")
              for n in range(N) for r in range(R)}
    Uinr_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"Uinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    eta_vars = {(i, n, r): m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{n}_{r}")
                for i in range(J) for n in range(N) for r in range(R)}
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}

    # "nonlinear" constraints
    for n in range(N):
        for r in range(R):
            m.addConstr(U_vars[n, r] - gp.quicksum(eta_vars[i, n, r] for i in range(J)) == 0)

    # linearizing constraints
    for i in range(J):
        for n in range(N):
            for r in range(R):
                m.addConstr(eta_vars[i, n, r] >= -omega_vars[i, n, r] * bigM)
                m.addConstr(eta_vars[i, n, r] <= omega_vars[i, n, r] * bigM)
                m.addConstr(eta_vars[i, n, r] >= Uinr_vars[i, n, r] - (1 - omega_vars[i, n, r]) * bigM)
                m.addConstr(eta_vars[i, n, r] <= Uinr_vars[i, n, r] + (1 - omega_vars[i, n, r]) * bigM)

    # Constraints mu
    for n in range(N):
        for r in range(R):
            m.addConstr(gp.quicksum(omega_vars[i, n, r] for i in range(J)) == 1)

    # Constraints theta
    for n in range(N):
        for i in range(J):
            m.addConstr(s_vars[i, n] - gp.quicksum(omega_vars[i, n, r] for r in range(R)) == 0)

    # Constraints alpha
    for i in range(J):
        for n in range(N):
            for r in range(R):
                m.addConstr(Uinr_vars[i, n, r] - U_vars[n, r] <= 0)

    # Constraints xi
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(z_vars[i, n] + logK[r] * s_vars[i, n] <= logL[r])

    # Constraints kappa
    for n in range(N):
        for i in range(J):
            for r in range(R):
                m.addConstr(Uinr_vars[i, n, r] -
                            (beta_vars[0] * x[i, n, 0] + beta_vars[1] * x[i, n, 1]
                             + (beta_vars[2] + beta_vars[3] * sigma[n, r]) * x[i, n, 2])
                            == epsilon[i, n, r])

    # Objective function
    objective = gp.quicksum(-y[i, n] * z_vars[i, n] for i in range(J) for n in range(N))
    m.setObjective(objective, GRB.MINIMIZE)
    # Create a log file name with the timestamp
    m.setParam('LogFile', f"MixedMILP_{N}_{R}_{J}_{timestamp}.txt")
    m.setParam('OutputFlag', 1)
    m.optimize()

    total_time = time.time() - start_time
    objective_value = sum(-y[i, n] * z_vars[i, n].x for i in range(J) for n in range(N)) \
                      + N * np.log(R)
    best_lowerbound = m.ObjBoundC + N * np.log(R)
    bestbeta = [beta_vars[k].x for k in range(K)]
    best_loglike = -objective_value
    m.dispose()

    return total_time, bestbeta, best_loglike, best_lowerbound


def MinLossMNL(y, x, K, R, epsilon, time_limit):
    start_time = time.time()
    N = len(y[0])
    J = x.shape[0]

    # initialize model
    m = gp.Model("MinLossMNL")

    # initialize variables
    dnr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dnr_{n}_{r}")
                for n in range(N) for r in range(R)}
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}

    # Objective function
    objective = gp.quicksum(dnr_vars[n, r] for n in range(N) for r in range(R))
    m.setObjective(objective, GRB.MINIMIZE)

    # linearizing constraints
    for i in range(J):
        for n in range(N):
            y_ind = int(np.where(y[:, n] == 1)[0])
            for r in range(R):
                Uinr = gp.quicksum(beta_vars[k] * x[i, n, k] for k in range(K)) + epsilon[i, n, r]
                Uynr = gp.quicksum(beta_vars[k] * x[y_ind, n, k] for k in range(K)) + epsilon[y_ind, n, r]
                m.addConstr(dnr_vars[n, r] >= Uinr - Uynr)

    m.setParam('OutputFlag', 0)
    m.setParam("TimeLimit", time_limit)
    m.optimize()

    total_time = time.time() - start_time
    objective_value = sum(dnr_vars[n, r].x for n in range(N) for r in range(R))
    bestbeta = [beta_vars[k].x for k in range(K)]
    best_loglike = -objective_value
    m.dispose()

    return total_time, bestbeta, best_loglike


def MinLossMixedMNL(y, x, epsilon, sigma, time_limit):
    start_time = time.time()
    N = len(y[0])
    J = x.shape[0]
    K = x.shape[2]
    R = epsilon.shape[2]

    # initialize model
    m = gp.Model("MinLossMixedMNL")

    # initialize variables
    dnr_vars = {(n, r): m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"dnr_{n}_{r}")
                for n in range(N) for r in range(R)}
    beta_vars = {k: m.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"beta_{k}")
                 for k in range(K)}

    # Objective function
    objective = gp.quicksum(dnr_vars[n, r] for n in range(N) for r in range(R))
    m.setObjective(objective, GRB.MINIMIZE)

    # linearizing constraints
    for i in range(J):
        for n in range(N):
            y_ind = int(np.where(y[:, n] == 1)[0])
            for r in range(R):
                # 0: ASC_RAIL
                # 1: BETA_COST
                # 2: BETA_TIME
                # 3: BETA_TIME_S
                Uinr = beta_vars[0] * x[i, n, 0] + beta_vars[1] * x[i, n, 1] + \
                       (beta_vars[2] + beta_vars[3] * sigma[n, r]) * x[i, n, 2] + epsilon[i, n, r]

                Uynr = beta_vars[0] * x[y_ind, n, 0] + beta_vars[1] * x[y_ind, n, 1] + \
                       (beta_vars[2] + beta_vars[3] * sigma[n, r]) * x[y_ind, n, 2] + epsilon[y_ind, n, r]

                m.addConstr(dnr_vars[n, r] >= Uinr - Uynr)

    m.setParam('OutputFlag', 0)
    m.setParam("TimeLimit", time_limit)
    m.optimize()

    total_time = time.time() - start_time
    objective_value = sum(dnr_vars[n, r].x for n in range(N) for r in range(R))
    bestbeta = [beta_vars[k].x for k in range(K)]
    best_loglike = -objective_value
    m.dispose()

    return total_time, bestbeta, best_loglike
