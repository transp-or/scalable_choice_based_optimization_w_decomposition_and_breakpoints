import numpy as np
from datetime import datetime
import time
import logging

from solve_full_MILP_BnB_cpp import compute_fixed_tuples_onep, compute_fixed_tuples

import gurobipy as gp
from gurobipy import GRB

console = False


def cpp_MILP(time_limit, threads, p_L, p_U, parking_data, exo_utility, endo_coef, breakpoints, onep, fixed_choices,
             J_PSP, J_PUP, start_price):
    N = parking_data['N']
    R = parking_data['R']
    J = parking_data['J_tot']
    if onep:
        start_time = time.time()

        # initialize model
        cpp = gp.Model("Continuous pricing problem")
        inf = GRB.INFINITY

        # initialize variables
        eta_vars = {n * R + r: cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{n * R + r}")
                    for n in range(N) for r in range(R)}
        omega_vars = {n * R + r: cpp.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"omega_{n * R + r}")
                      for n in range(N) for r in range(R)}
        p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in [2]}
        _, omega, _ = compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef, start_price,
                                                 onep, J_PSP, J_PUP)

        for i in [2]:
            p_vars[i].start = start_price[0]
            p_vars[i].setAttr("BranchPriority", 1000)
            for n in range(N):
                for r in range(R):
                    omega_vars[n * R + r].start = omega[n * R + r]

        # Objective function
        objective = 1 / R * gp.quicksum(
            eta_vars[n * R + r] for n in range(N) for r in range(R))
        cpp.setObjective(objective, GRB.MAXIMIZE)

        if fixed_choices:
            fixed_none = {(n, r): 3 for n in range(N) for r in range(R)}
            fixed = compute_fixed_tuples_onep(N, R, p_L[2], p_U[2], breakpoints, fixed_none)
            PUPs = list(fixed.values()).count(2)
            objective = 1 / R * (gp.quicksum(
                eta_vars[n * R + r] for n in range(N) for r in range(R)
                if fixed[n, r] == 3) + PUPs * p_vars[2])

            cpp.setObjective(objective, GRB.MAXIMIZE)

        # best choice constraint
        for n in range(N):
            for r in range(R):
                # 0: Opt-Out
                # 1: PSP
                # 2: PUP

                cpp.addConstr((1 - omega_vars[n * R + r]) * exo_utility[0, n * R + r]
                              + omega_vars[n * R + r] * exo_utility[1, n * R + r]
                              + endo_coef[n * R + r] * eta_vars[n * R + r]
                              >= exo_utility[0, n * R + r])

                cpp.addConstr((1 - omega_vars[n * R + r]) * exo_utility[0, n * R + r]
                              + omega_vars[n * R + r] * exo_utility[1, n * R + r]
                              + endo_coef[n * R + r] * eta_vars[n * R + r]
                              >= exo_utility[1, n * R + r] + endo_coef[n * R + r] * p_vars[2])

        # define eta with bigM constraints, where M = upper bound on p_2, i.e. p_U[2]
        for n in range(N):
            for r in range(R):
                # cpp.addConstr(eta_vars[n * R + r] == omega_vars[n * R + r] * p_vars[2])
                cpp.addConstr(eta_vars[i, n * R + r] <= omega_vars[n * R + r] * p_U[2])
                cpp.addConstr(eta_vars[i, n * R + r] <= p_vars[2])
                cpp.addConstr(eta_vars[i, n * R + r] >= 0)
                cpp.addConstr(eta_vars[i, n * R + r] >= p_vars[2] - (1 - omega_vars[n * R + r]) * p_U[2])
    else:
        start_time = time.time()

        # initialize model
        cpp = gp.Model("Continuous pricing problem")
        inf = GRB.INFINITY

        # initialize variables
        eta_vars = {(i, n * R + r): cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{n}_{r}")
                    for i in range(1, J) for n in range(N) for r in range(R)}
        omega_vars = {(i, n * R + r): cpp.addVar(lb=0, ub=inf, vtype=GRB.BINARY, name=f"omega_{i}_{n}_{r}")
                      for i in range(J) for n in range(N) for r in range(R)}
        p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in range(1, J)}

        # first try prices
        p = dict()
        for i in range(1, J):
            p[i] = start_price[i - 1]
        _, omega, eta = compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef, p, onep, J_PSP, J_PUP)

        for i in range(1, J):
            p_vars[i].start = start_price[i - 1]
            p_vars[i].setAttr("BranchPriority", 1000)
        for n in range(N):
            for r in range(R):
                for i in range(J):
                    omega_vars[i, n * R + r].start = omega[i, n * R + r]
                for i in range(1, J):
                    eta_vars[i, n * R + r].start = eta[i, n * R + r]

        # Objective function
        objective = 1 / R * gp.quicksum(
            eta_vars[i, n * R + r] for i in range(1, J) for n in range(N) for r in range(R))
        cpp.setObjective(objective, GRB.MAXIMIZE)

        if fixed_choices:
            fixed_none = {(i, n, r): 2 for i in range(J) for n in range(N) for r in range(R)}
            fixed = compute_fixed_tuples(N, R, p_L, p_U, exo_utility, endo_coef, J_PSP, J_PUP, fixed_none)

            PSPs = np.zeros(J_PSP)
            for i in range(J_PSP):
                PSPs[i] = np.count_nonzero(fixed[i + 1, :, :] == 1)
            PUPs = np.zeros(J_PUP)
            for i in range(J_PUP):
                PUPs[i] = np.count_nonzero(fixed[i + J_PSP + 1, :, :] == 1)

            objective = 1 / R * (gp.quicksum(
                eta_vars[i, n * R + r] for i in range(1, J) for n in range(N) for r in range(R)
                if not sum(fixed[j, n, r] for j in range(J)) == 1)
                                 + gp.quicksum(PSPs[i] * p_vars[i + 1] for i in range(J_PSP))
                                 + gp.quicksum(PUPs[i] * p_vars[i + J_PSP + 1] for i in range(J_PUP))
                                 )
            cpp.setObjective(objective, GRB.MAXIMIZE)

            for n in range(N):
                for r in range(R):
                    if sum(fixed[i, n, r] for i in range(J)) == 1:
                        for i in range(J):
                            omega_vars[i, n * R + r].lb = fixed[i, n, r]
                            omega_vars[i, n * R + r].ub = fixed[i, n, r]
                    else:
                        for i in range(J):
                            if not fixed[i, n, r] == 2:  # those alternatives that are fixed to 0 get fixed
                                omega_vars[i, n * R + r].lb = fixed[i, n, r]
                                omega_vars[i, n * R + r].ub = fixed[i, n, r]

        # one choice constraint
        for n in range(N):
            for r in range(R):
                cpp.addConstr(gp.quicksum(omega_vars[i, n * R + r] for i in range(J)) == 1)

        # best choice constraint
        for n in range(N):
            for r in range(R):
                cpp.addConstr(omega_vars[0, n * R + r] * exo_utility[0, n * R + r] +
                              gp.quicksum(omega_vars[i, n * R + r]
                                          * exo_utility[i, n * R + r]
                                          + endo_coef[i, n * R + r] * eta_vars[i, n * R + r]
                                          for i in range(1, J))
                              >= exo_utility[0, n * R + r])
                for i in range(1, J):
                    cpp.addConstr(omega_vars[0, n * R + r] * exo_utility[0, n * R + r] +
                                  gp.quicksum(omega_vars[i, n * R + r]
                                              * exo_utility[i, n * R + r]
                                              + endo_coef[i, n * R + r] * eta_vars[i, n * R + r]
                                              for i in range(1, J))
                                  >= exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_vars[i])

        # define eta with bigM constraints, where M = upper bound on p_i, i.e. p_U[i]
        for n in range(N):
            for r in range(R):
                for i in range(1, J):
                    # cpp.addConstr(eta_vars[i, n * R + r] == omega_vars[i, n * R + r] * p_vars[i])
                    cpp.addConstr(eta_vars[i, n * R + r] <= omega_vars[i, n * R + r] * p_U[i])
                    cpp.addConstr(eta_vars[i, n * R + r] <= p_vars[i])
                    cpp.addConstr(eta_vars[i, n * R + r] >= 0)
                    cpp.addConstr(eta_vars[i, n * R + r] >= p_vars[i] - (1 - omega_vars[i, n * R + r]) * p_U[i])

    # solve model
    # cpp.setParam('OutputFlag', 0)
    # cpp.setParam("Threads", 1)
    # cpp.update()
    # cpp.write("00_MILP.mps")
    # print("Wrote 00_MILP.mps")
    # exit()

    cpp.setParam('ScaleFlag', 1)
    cpp.setParam('SimplexPricing', 0)
    cpp.setParam('NormAdjust', 1)
    cpp.setParam('Heuristics', 0)
    cpp.setParam('MIPFocus', 2)
    cpp.setParam('Cuts', 0)
    cpp.setParam('AggFill', 1000)

    cpp.setParam('OutputFlag', 1)
    cpp.setParam("TimeLimit", time_limit)
    cpp.setParam("Threads", threads)

    dates = datetime.now()
    filenamee = f"newLog_{N}_{R}_{J_PSP}_{J_PUP}_{time_limit}_{2}_{threads}_{dates}.txt"
    cpp.setParam('LogFile', filenamee)

    cpp.optimize()

    if cpp.SolCount >= 1:
        objective_value = cpp.ObjVal
        if onep:
            bestprice = [0.6, p_vars[2].X]
        else:
            bestprice = [p_vars[i].X for i in range(1, J)]
    else:
        objective_value = 1000
        bestprice = [1000] * (J - 1)

    total_time = time.time() - start_time

    try:
        best_lowerbound = cpp.ObjBoundC
    except AttributeError:
        best_lowerbound = -1000

    best_obj = objective_value
    gap = cpp.MIPGap
    nodes = cpp.NodeCount

    cpp.dispose()

    logging.basicConfig(filename=filenamee, level=logging.INFO, format='%(message)s')
    if console:
        print(f"N = {N}, R = {R}, J = {J}, Total time = {total_time}, Iterations = {nodes}, "
              f"Best Price = {bestprice}, objective = {best_obj}, Gap = {gap}%")
    else:
        logging.info(f"N = {N}, R = {R}, J = {J}, Total time = {total_time}, Iterations = {nodes}, "
                     f"Best Price = {bestprice}, objective = {best_obj}, Gap = {gap}%")

    return total_time, bestprice, best_obj, best_lowerbound, gap, nodes


def cpp_QCLP(time_limit, threads, p_L, p_U, parking_data, exo_utility, endo_coef, breakpoints, onep, fixed_choices,
             J_PSP, J_PUP, start_price):
    N = parking_data['N']
    R = parking_data['R']
    J = parking_data['J_tot']
    if onep:
        start_time = time.time()

        # initialize model
        cpp = gp.Model("Continuous pricing problem")
        inf = GRB.INFINITY

        # initialize variables
        eta_vars = {n * R + r: cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{n * R + r}")
                    for n in range(N) for r in range(R)}
        omega_vars = {n * R + r: cpp.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"omega_{n * R + r}")
                      for n in range(N) for r in range(R)}
        p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in [2]}

        _, omega, eta = compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef, start_price,
                                                   onep, J_PSP, J_PUP)
        p_vars[2].start = start_price[0]
        p_vars[2].setAttr("BranchPriority", 1000)
        for n in range(N):
            for r in range(R):
                omega_vars[n * R + r].start = omega[n * R + r]
                eta_vars[n * R + r].start = eta[n * R + r]

        # Objective function
        objective = 1 / R * gp.quicksum(
            eta_vars[n * R + r] for n in range(N) for r in range(R))
        cpp.setObjective(objective, GRB.MAXIMIZE)

        if fixed_choices:
            fixed_none = {(n, r): 3 for n in range(N) for r in range(R)}
            fixed = compute_fixed_tuples_onep(N, R, p_L[2], p_U[2], breakpoints, fixed_none)
            PUPs = list(fixed.values()).count(2)
            objective = 1 / R * (gp.quicksum(
                eta_vars[n * R + r] for n in range(N) for r in range(R)
                if fixed[n, r] == 3) + PUPs * p_vars[2])

            cpp.setObjective(objective, GRB.MAXIMIZE)

        # best choice constraint
        for n in range(N):
            for r in range(R):
                # 0: Opt-Out
                # 1: PSP
                # 2: PUP

                cpp.addConstr((1 - omega_vars[n * R + r]) * exo_utility[0, n * R + r]
                              + omega_vars[n * R + r] * exo_utility[1, n * R + r]
                              + endo_coef[n * R + r] * eta_vars[n * R + r]
                              >= exo_utility[0, n * R + r])

                cpp.addConstr((1 - omega_vars[n * R + r]) * exo_utility[0, n * R + r]
                              + omega_vars[n * R + r] * exo_utility[1, n * R + r]
                              + endo_coef[n * R + r] * eta_vars[n * R + r]
                              >= exo_utility[1, n * R + r] + endo_coef[n * R + r] * p_vars[2])

        # define eta
        for n in range(N):
            for r in range(R):
                cpp.addConstr(eta_vars[n * R + r] == omega_vars[n * R + r] * p_vars[2])
    else:
        start_time = time.time()

        # initialize model
        cpp = gp.Model("Continuous pricing problem")
        inf = GRB.INFINITY

        # initialize variables
        eta_vars = {(i, n * R + r): cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{n}_{r}")
                    for i in range(1, J) for n in range(N) for r in range(R)}
        omega_vars = {(i, n * R + r): cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}_{n}_{r}")
                      for i in range(J) for n in range(N) for r in range(R)}
        p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in range(1, J)}

        # first try prices
        p = dict()
        for i in range(1, J):
            p[i] = start_price[i - 1]
        _, omega, eta = compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef, p, onep, J_PSP, J_PUP)

        for i in range(1, J):
            p_vars[i].start = start_price[i - 1]
            p_vars[i].setAttr("BranchPriority", 1000)
        for n in range(N):
            for r in range(R):
                for i in range(J):
                    omega_vars[i, n * R + r].start = omega[i, n * R + r]
                for i in range(1, J):
                    eta_vars[i, n * R + r].start = eta[i, n * R + r]

        # Objective function
        objective = 1 / R * gp.quicksum(
            eta_vars[i, n * R + r] for i in range(1, J) for n in range(N) for r in range(R))
        cpp.setObjective(objective, GRB.MAXIMIZE)

        if fixed_choices:
            fixed_none = {(i, n, r): 2 for i in range(J) for n in range(N) for r in range(R)}
            fixed = compute_fixed_tuples(N, R, p_L, p_U, exo_utility, endo_coef, J_PSP, J_PUP, fixed_none)

            PSPs = np.zeros(J_PSP)
            for i in range(J_PSP):
                PSPs[i] = np.count_nonzero(fixed[i + 1, :, :] == 1)
            PUPs = np.zeros(J_PUP)
            for i in range(J_PUP):
                PUPs[i] = np.count_nonzero(fixed[i + J_PSP + 1, :, :] == 1)

            objective = 1 / R * (gp.quicksum(
                eta_vars[i, n * R + r] for i in range(1, J) for n in range(N) for r in range(R)
                if not sum(fixed[j, n, r] for j in range(J)) == 1)
                                 + gp.quicksum(PSPs[i] * p_vars[i + 1] for i in range(J_PSP))
                                 + gp.quicksum(PUPs[i] * p_vars[i + J_PSP + 1] for i in range(J_PUP))
                                 )
            cpp.setObjective(objective, GRB.MAXIMIZE)

            for n in range(N):
                for r in range(R):
                    if sum(fixed[i, n, r] for i in range(J)) == 1:
                        for i in range(J):
                            omega_vars[i, n * R + r].lb = fixed[i, n, r]
                            omega_vars[i, n * R + r].ub = fixed[i, n, r]
                    else:
                        for i in range(J):
                            if not fixed[i, n, r] == 2:  # those alternatives that are fixed to 0 get fixed
                                omega_vars[i, n * R + r].lb = fixed[i, n, r]
                                omega_vars[i, n * R + r].ub = fixed[i, n, r]

        # one choice constraint
        for n in range(N):
            for r in range(R):
                cpp.addConstr(gp.quicksum(omega_vars[i, n * R + r] for i in range(J)) == 1)

        # best choice constraint
        for n in range(N):
            for r in range(R):
                cpp.addConstr(omega_vars[0, n * R + r] * exo_utility[0, n * R + r] +
                              gp.quicksum(omega_vars[i, n * R + r]
                                          * exo_utility[i, n * R + r]
                                          + endo_coef[i, n * R + r] * eta_vars[i, n * R + r]
                                          for i in range(1, J))
                              >= exo_utility[0, n * R + r])
                for i in range(1, J):
                    cpp.addConstr(omega_vars[0, n * R + r] * exo_utility[0, n * R + r] +
                                  gp.quicksum(omega_vars[i, n * R + r]
                                              * exo_utility[i, n * R + r]
                                              + endo_coef[i, n * R + r] * eta_vars[i, n * R + r]
                                              for i in range(1, J))
                                  >= exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_vars[i])

        # define eta
        for n in range(N):
            for r in range(R):
                for i in range(1, J):
                    cpp.addConstr(eta_vars[i, n * R + r] == omega_vars[i, n * R + r] * p_vars[i])

    # solve model
    # cpp.setParam("NonConvex", 2)
    # cpp.setParam('OutputFlag', 0)
    # cpp.setParam("Threads", 1)
    # cpp.update()
    # cpp.write("00_QCLP.mps")
    # print("Wrote 00_QCLP.mps")
    # exit()
    cpp.setParam('OutputFlag', int(R > 1))  # when used in the heuristic we suppress the log

    cpp.setParam('ScaleFlag', 0)
    cpp.setParam('SimplexPricing', 2)
    cpp.setParam('NormAdjust', 0)
    cpp.setParam('MIPFocus', 3)
    cpp.setParam('Cuts', 0)

    cpp.setParam("TimeLimit", time_limit)
    cpp.setParam("Threads", threads)
    cpp.setParam("NonConvex", 2)

    dates = datetime.now()
    if R > 1:
        filenamee = f"newLog_{N}_{R}_{J_PSP}_{J_PUP}_{time_limit}_{4}_{threads}_{dates}.txt"
        cpp.setParam('LogFile', filenamee)

    cpp.optimize()

    if cpp.SolCount >= 1:
        objective_value = cpp.ObjVal
        if onep:
            bestprice = [0.6, p_vars[2].X]
        else:
            bestprice = [p_vars[i].X for i in range(1, J)]
    else:
        objective_value = 1000
        bestprice = [1000] * (J - 1)

    total_time = time.time() - start_time

    try:
        best_lowerbound = cpp.ObjBoundC
    except AttributeError:
        best_lowerbound = -1000

    best_obj = objective_value
    gap = cpp.MIPGap
    nodes = cpp.NodeCount

    cpp.dispose()

    if R > 1:
        logging.basicConfig(filename=filenamee, level=logging.INFO, format='%(message)s')
        if console:
            print(f"N = {N}, R = {R}, J = {J}, Total time = {total_time}, Iterations = {nodes}, "
                  f"Best Price = {bestprice}, objective = {best_obj}, Gap = {gap}%")
        else:
            logging.info(f"N = {N}, R = {R}, J = {J}, Total time = {total_time}, Iterations = {nodes}, "
                         f"Best Price = {bestprice}, objective = {best_obj}, Gap = {gap}%")

    return total_time, bestprice, best_obj, best_lowerbound, gap, nodes


def cpp_QCQP(time_limit, threads, p_L, p_U, parking_data, exo_utility, endo_coef, breakpoints, onep, fixed_choices,
             J_PSP, J_PUP, start_price):
    N = parking_data['N']
    R = parking_data['R']
    J = parking_data['J_tot']
    if onep:
        start_time = time.time()

        # initialize model
        cpp = gp.Model("Continuous pricing problem")

        # initialize variables
        omega_vars = {n * R + r: cpp.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"omega_{n * R + r}")
                      for n in range(N) for r in range(R)}
        p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in [2]}

        _, omega, _ = compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef, start_price,
                                                 onep, J_PSP, J_PUP)

        for i in [2]:
            p_vars[i].start = start_price[0]
            p_vars[i].setAttr("BranchPriority", 1000)
            for n in range(N):
                for r in range(R):
                    omega_vars[n * R + r].start = omega[n * R + r]

        # Objective function
        objective = 1 / R * gp.quicksum(
            omega_vars[n * R + r] * p_vars[2] for n in range(N) for r in range(R))
        cpp.setObjective(objective, GRB.MAXIMIZE)

        if fixed_choices:
            fixed_none = {(n, r): 3 for n in range(N) for r in range(R)}
            fixed = compute_fixed_tuples_onep(N, R, p_L[2], p_U[2], breakpoints, fixed_none)
            PUPs = list(fixed.values()).count(2)
            objective = 1 / R * (gp.quicksum(
                omega_vars[n * R + r] * p_vars[2] for n in range(N) for r in range(R)
                if fixed[n, r] == 3) + PUPs * p_vars[2])

            cpp.setObjective(objective, GRB.MAXIMIZE)

        # best choice constraint
        for n in range(N):
            for r in range(R):
                # 0: Opt-Out
                # 1: PSP
                # 2: PUP

                cpp.addConstr((1 - omega_vars[n * R + r]) * exo_utility[0, n * R + r]
                              + omega_vars[n * R + r] * exo_utility[1, n * R + r]
                              + endo_coef[n * R + r] * omega_vars[n * R + r] * p_vars[2]
                              >= exo_utility[0, n * R + r])

                cpp.addConstr((1 - omega_vars[n * R + r]) * exo_utility[0, n * R + r]
                              + omega_vars[n * R + r] * exo_utility[1, n * R + r]
                              + endo_coef[n * R + r] * omega_vars[n * R + r] * p_vars[2]
                              >= exo_utility[1, n * R + r] + endo_coef[n * R + r] * p_vars[2])
    else:
        start_time = time.time()

        # initialize model
        cpp = gp.Model("Continuous pricing problem")
        inf = GRB.INFINITY

        # initialize variables
        omega_vars = {(i, n * R + r): cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}_{n}_{r}")
                      for i in range(J) for n in range(N) for r in range(R)}
        p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in range(1, J)}

        # first try prices
        p = dict()
        for i in range(1, J):
            p[i] = start_price[i - 1]
        _, omega, _ = compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef, p, onep, J_PSP, J_PUP)

        for i in range(1, J):
            p_vars[i].start = start_price[i - 1]
            p_vars[i].setAttr("BranchPriority", 1000)
        for n in range(N):
            for r in range(R):
                for i in range(J):
                    omega_vars[i, n * R + r].start = omega[i, n * R + r]

        # Objective function
        objective = 1 / R * gp.quicksum(
            omega_vars[i, n * R + r] * p_vars[i] for i in range(1, J) for n in range(N) for r in range(R))
        cpp.setObjective(objective, GRB.MAXIMIZE)

        if fixed_choices:
            fixed_none = {(i, n, r): 2 for i in range(J) for n in range(N) for r in range(R)}
            fixed = compute_fixed_tuples(N, R, p_L, p_U, exo_utility, endo_coef, J_PSP, J_PUP, fixed_none)

            PSPs = np.zeros(J_PSP)
            for i in range(J_PSP):
                PSPs[i] = np.count_nonzero(fixed[i + 1, :, :] == 1)
            PUPs = np.zeros(J_PUP)
            for i in range(J_PUP):
                PUPs[i] = np.count_nonzero(fixed[i + J_PSP + 1, :, :] == 1)

            objective = 1 / R * (gp.quicksum(
                omega_vars[i, n * R + r] * p_vars[i] for i in range(1, J) for n in range(N) for r in range(R)
                if not sum(fixed[j, n, r] for j in range(J)) == 1)
                                 + gp.quicksum(PSPs[i] * p_vars[i + 1] for i in range(J_PSP))
                                 + gp.quicksum(PUPs[i] * p_vars[i + J_PSP + 1] for i in range(J_PUP))
                                 )
            cpp.setObjective(objective, GRB.MAXIMIZE)

            for n in range(N):
                for r in range(R):
                    if sum(fixed[i, n, r] for i in range(J)) == 1:
                        for i in range(J):
                            omega_vars[i, n * R + r].lb = fixed[i, n, r]
                            omega_vars[i, n * R + r].ub = fixed[i, n, r]
                    else:
                        for i in range(J):
                            if not fixed[i, n, r] == 2:  # those alternatives that are fixed to 0 get fixed
                                omega_vars[i, n * R + r].lb = fixed[i, n, r]
                                omega_vars[i, n * R + r].ub = fixed[i, n, r]

        # one choice constraint
        for n in range(N):
            for r in range(R):
                cpp.addConstr(gp.quicksum(omega_vars[i, n * R + r] for i in range(J)) == 1)

        # best choice constraint
        for n in range(N):
            for r in range(R):
                cpp.addConstr(omega_vars[0, n * R + r] * exo_utility[0, n * R + r] +
                              gp.quicksum(omega_vars[i, n * R + r]
                                          * exo_utility[i, n * R + r]
                                          + endo_coef[i, n * R + r] * omega_vars[i, n * R + r] * p_vars[i]
                                          for i in range(1, J))
                              >= exo_utility[0, n * R + r])
                for i in range(1, J):
                    cpp.addConstr(omega_vars[0, n * R + r] * exo_utility[0, n * R + r] +
                                  gp.quicksum(omega_vars[i, n * R + r]
                                              * exo_utility[i, n * R + r]
                                              + endo_coef[i, n * R + r] * omega_vars[i, n * R + r] * p_vars[i]
                                              for i in range(1, J))
                                  >= exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_vars[i])

    # solve model
    # cpp.setParam("NonConvex", 2)
    # cpp.setParam('OutputFlag', 0)
    # cpp.setParam("Threads", 1)
    # cpp.update()
    # cpp.write("00_QCQP.mps")
    # print("Wrote 00_QCQP.mps")
    # exit()

    cpp.setParam('ScaleFlag', 0)
    cpp.setParam('SimplexPricing', 2)
    cpp.setParam('NormAdjust', 0)
    cpp.setParam('MIPFocus', 3)
    cpp.setParam('Cuts', 0)

    cpp.setParam('OutputFlag', 1)
    cpp.setParam("TimeLimit", time_limit)
    cpp.setParam("Threads", threads)
    cpp.setParam("NonConvex", 2)

    dates = datetime.now()
    filenamee = f"newLog_{N}_{R}_{J_PSP}_{J_PUP}_{time_limit}_{3}_{threads}_{dates}.txt"
    cpp.setParam('LogFile', filenamee)

    cpp.optimize()

    if cpp.SolCount >= 1:
        objective_value = cpp.ObjVal
        if onep:
            bestprice = [0.6, p_vars[2].X]
        else:
            bestprice = [p_vars[i].X for i in range(1, J)]
    else:
        objective_value = 1000
        bestprice = [1000] * (J - 1)

    total_time = time.time() - start_time

    try:
        best_lowerbound = cpp.ObjBoundC
    except AttributeError:
        best_lowerbound = -1000

    best_obj = objective_value
    gap = cpp.MIPGap
    nodes = cpp.NodeCount

    cpp.dispose()

    logging.basicConfig(filename=filenamee, level=logging.INFO, format='%(message)s')
    if console:
        print(f"N = {N}, R = {R}, J = {J}, Total time = {total_time}, Iterations = {nodes}, "
              f"Best Price = {bestprice}, objective = {best_obj}, Gap = {gap}%")
    else:
        logging.info(f"N = {N}, R = {R}, J = {J}, Total time = {total_time}, Iterations = {nodes}, "
                     f"Best Price = {bestprice}, objective = {best_obj}, Gap = {gap}%")

    return total_time, bestprice, best_obj, best_lowerbound, gap, nodes


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
        p[0] = 0
        obj = 0
        omega = np.zeros((J, N * R))
        eta = {(i, n * R + r): 0 for i in range(1, J) for n in range(N) for r in range(R)}
        for n in range(N):
            for r in range(R):
                util_list = [exo_utility[0, n * R + r]]
                for i in range(J_PSP):
                    util_list.append(exo_utility[i + 1, n * R + r] + endo_coef[i + 1, n * R + r] * p[i + 1])
                for i in range(J_PUP):
                    util_list.append(exo_utility[i + J_PSP + 1, n * R + r]
                                     + endo_coef[i + J_PSP + 1, n * R + r] * p[i + J_PSP + 1])

                # get best alternative, with tie-breaks based on price values
                max_value = max(util_list)
                candidate_indices = [i for i, val in enumerate(util_list) if abs(val - max_value) < 1e-9]
                max_util = max(candidate_indices, key=lambda idx: p[idx])

                if 1 <= max_util:
                    obj -= (1 / R) * p[max_util]
                    omega[max_util, n * R + r] = 1
                    eta[max_util, n * R + r] = p[max_util]
    return obj, omega, eta


def cpp_nonlinearNH_old(time_limit, threads, p_L, p_U, parking_data, exo_utility, endo_coef, breakpoints, onep):
    N = parking_data['N']
    R = parking_data['R']
    if onep:
        J = 3
        start_time = time.time()

        # initialize model
        cpp = gp.Model("Continuous pricing problem")

        # initialize variables
        omega_vars = {(n, r): cpp.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"omega_{n}_{r}")
                      for n in range(N) for r in range(R)}
        p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in [2]}
        _, omega, _ = compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef, [(p_L[2] + p_U[2]) / 2],
                                                 onep, J_PSP, J_PUP)

        for i in [2]:
            p_vars[i].start = (p_L[i] + p_U[i]) / 2
            p_vars[i].setAttr("BranchPriority", 1000)
            for n in range(N):
                for r in range(R):
                    omega_vars[n, r].start = omega[n * R + r]

        # Objective function
        objective = (1 / R) * gp.quicksum(
            p_vars[2] * omega_vars[n, r] for n in range(N) for r in range(R))
        cpp.setObjective(objective, GRB.MAXIMIZE)

        if isinstance(breakpoints, np.ndarray):
            fixed_none = {(n, r): 3 for n in range(N) for r in range(R)}
            fixed = compute_fixed_tuples_onep(N, R, p_L[2], p_U[2], breakpoints, fixed_none)

            for n in range(N):
                for r in range(R):
                    if not fixed[n, r] == 3:
                        omega_vars[n, r].ub = int(fixed[n, r] == 2)
                        omega_vars[n, r].Obj = 0

        # best choice constraint
        for n in range(N):
            for r in range(R):
                # 0: Opt-Out
                # 1: PSP
                # 2: PUP

                cpp.addConstr((1 - omega_vars[n, r]) * exo_utility[0, n * R + r]
                              + omega_vars[n, r] * (exo_utility[1, n * R + r] + endo_coef[n * R + r] * p_vars[2])
                              >= exo_utility[0, n * R + r])

                cpp.addConstr((1 - omega_vars[n, r]) * exo_utility[0, n * R + r]
                              + omega_vars[n, r] * (exo_utility[1, n * R + r] + endo_coef[n * R + r] * p_vars[2])
                              >= exo_utility[1, n * R + r] + endo_coef[n * R + r] * p_vars[2])
    else:
        J = parking_data['I_tot']
        start_time = time.time()

        # initialize model
        cpp = gp.Model("Continuous pricing problem")
        inf = GRB.INFINITY

        # initialize variables
        omega_vars = {(i, n * R + r): cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}_{n}_{r}")
                      for i in range(J) for n in range(N) for r in range(R)}
        p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in [1, 2]}

        # first try prices
        p = np.zeros(J)
        for i in range(J):
            p[i] = (p_L[i] + p_U[i]) / 2
        _, omega, _ = compute_cpp_from_p_parking(J, N, R, exo_utility, endo_coef, p, onep, J_PSP, J_PUP)

        for i in [1, 2]:
            p_vars[i].start = (p_L[i] + p_U[i]) / 2
            p_vars[i].setAttr("BranchPriority", 1000)
            for n in range(N):
                for r in range(R):
                    omega_vars[i, n * R + r].start = omega[i, n * R + r]

        if isinstance(breakpoints, np.ndarray):
            fixed_none = {(i, n, r): 2 for i in range(3) for n in range(N) for r in range(R)}
            fixed = compute_fixed_tuples(N, R, p_L, p_U, exo_utility, endo_coef, J_PSP, J_PUP, fixed_none)

            for n in range(N):
                for r in range(R):
                    if sum(fixed[i, n, r] for i in range(3)) == 1:
                        for i in range(J):
                            omega_vars[i, n * R + r].lb = fixed[i, n, r]
                            omega_vars[i, n * R + r].ub = fixed[i, n, r]
                    else:
                        for i in range(J):
                            if not fixed[i, n, r] == 2:  # those alternatives that are fixed to 0 get fixed
                                omega_vars[i, n * R + r].lb = fixed[i, n, r]
                                omega_vars[i, n * R + r].ub = fixed[i, n, r]

        # one choice constraint
        for n in range(N):
            for r in range(R):
                cpp.addConstr(gp.quicksum(omega_vars[i, n * R + r] for i in range(J)) == 1)

        # best choice constraint
        for n in range(N):
            for r in range(R):
                # 0: Opt-Out
                # 1: PSP
                # 2: PUP

                cpp.addConstr(omega_vars[0, n * R + r] * (exo_utility[0, n * R + r])
                              + omega_vars[1, n * R + r] * (exo_utility[1, n * R + r]
                                                            + endo_coef[1, n * R + r] * p_vars[1])
                              + omega_vars[2, n * R + r] * (exo_utility[2, n * R + r]
                                                            + endo_coef[2, n * R + r] * p_vars[2])
                              >= exo_utility[0, n * R + r])

                cpp.addConstr(omega_vars[0, n * R + r] * (exo_utility[0, n * R + r])
                              + omega_vars[1, n * R + r] * (exo_utility[1, n * R + r]
                                                            + endo_coef[1, n * R + r] * p_vars[1])
                              + omega_vars[2, n * R + r] * (exo_utility[2, n * R + r]
                                                            + endo_coef[2, n * R + r] * p_vars[2])
                              >= exo_utility[1, n * R + r] + endo_coef[1, n * R + r] * p_vars[
                                  1])

                cpp.addConstr(omega_vars[0, n * R + r] * (exo_utility[0, n * R + r])
                              + omega_vars[1, n * R + r] * (exo_utility[1, n * R + r]
                                                            + endo_coef[1, n * R + r] * p_vars[1])
                              + omega_vars[2, n * R + r] * (exo_utility[2, n * R + r]
                                                            + endo_coef[2, n * R + r] * p_vars[2])
                              >= exo_utility[2, n * R + r] + endo_coef[2, n * R + r] * p_vars[
                                  2])

        # Objective function
        objective = 1 / R * gp.quicksum(
            p_vars[i] * omega_vars[i, n * R + r] for i in [1, 2] for n in range(N) for r in range(R))
        cpp.setObjective(objective, GRB.MAXIMIZE)
    # solve model
    cpp.setParam('OutputFlag', 1)
    cpp.setParam("TimeLimit", time_limit)
    cpp.setParam("NonConvex", 2)
    cpp.setParam("Threads", threads)

    cpp.optimize()

    if cpp.status == GRB.TIME_LIMIT:
        if cpp.SolCount >= 1:
            objective_value = cpp.ObjVal
        else:
            objective_value = 1000
    else:
        objective_value = cpp.ObjVal

    total_time = time.time() - start_time
    try:
        best_lowerbound = cpp.ObjBoundC
    except:
        best_lowerbound = -1000
    if cpp.SolCount >= 1:
        if onep:
            bestprice = [0.6, p_vars[2].X]
        else:
            bestprice = [p_vars[1].X, p_vars[2].X]
    else:
        bestprice = [1000, 1000]

    best_obj = objective_value
    try:
        gap = cpp.MIPGap
    except AttributeError:
        gap = 0
    nodes = cpp.NodeCount

    cpp.dispose()

    return total_time, bestprice, best_obj, best_lowerbound, gap, nodes


def cpp_MILP_convex(time_limit, threads, p_L, p_U, parking_data, exo_utility, endo_coef):
    N = parking_data['N']
    R = parking_data['R']
    J = parking_data['J_tot']

    start_time = time.time()

    # initialize model
    cpp = gp.Model("Continuous pricing problem")
    inf = GRB.INFINITY

    e_c = 2.71828182845904
    add_to_util = 0  # make them positive for exp approx to work

    # initialize variables
    prob_vars = {(i, n): cpp.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"prob_{i}_{n}")
                 for i in range(J) for n in range(N)}
    V_vars = {(i, n): cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"V_{i}_{n}")
              for i in range(J) for n in range(N)}
    p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
              for i in range(1, J)}
    probxprice_vars = {(i, n): cpp.addVar(lb=0, ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"probxprice_{i}_{n}")
                       for i in range(1, J) for n in range(N)}
    expV_vars = {(i, n): cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"expV_{i}_{n}")
                 for i in range(J) for n in range(N)}
    expxProb_vars = {(i, j, n): cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"expxProb_{i}_{j}_{n}")
                     for i in range(1, J) for j in range(J) for n in range(N)}

    # Objective function
    objective = gp.quicksum(
        probxprice_vars[i, n] for i in range(1, J) for n in range(N))
    cpp.setObjective(objective, GRB.MAXIMIZE)

    for n in range(N):
        # define V_0n
        # cpp.addConstr(V_vars[0, n] == exo_utility[0, n * R] + add_to_util)
        # define exp(V_0n)
        # cpp.addGenConstrExpA(V_vars[0, n], expV_vars[0, n], e_c)
        # cpp.addGenConstrExp(V_vars[0, n], expV_vars[0, n])

        cpp.addConstr(expV_vars[0, n] == e_c**(exo_utility[0, n] + add_to_util))

        # define probabilites
        cpp.addConstr(gp.quicksum(prob_vars[i, n] for i in range(J)) == 1)

        for i in range(1, J):
            # define price * P_in
            cpp.addConstr(probxprice_vars[i, n] == prob_vars[i, n] * p_vars[i])

            # define probability P_in
            for j in range(J):
                cpp.addConstr(expxProb_vars[i, j, n] == prob_vars[i, n] * expV_vars[j, n])
            cpp.addConstr(expV_vars[i, n] == gp.quicksum(expxProb_vars[i, j, n] for j in range(J)))

            # define V_in
            cpp.addConstr(V_vars[i, n] == exo_utility[i, n] + endo_coef[i, n] * p_vars[i] + add_to_util)

            # define exp(V_in)
            # cpp.addGenConstrExpA(V_vars[i, n], expV_vars[i, n], e_c
            cpp.addGenConstrExp(V_vars[i, n], expV_vars[i, n])

    cpp.setParam('OutputFlag', 1)
    cpp.setParam('NonConvex', 2)
    cpp.setParam('DualReductions', 0)
    cpp.setParam("TimeLimit", time_limit)
    # cpp.setParam("Threads", threads)

    # dates = datetime.now()
    # filenamee = f"newLog_{N}_{R}_{J_PSP}_{J_PUP}_{time_limit}_{2}_{threads}_{dates}.txt"
    # cpp.setParam('LogFile', filenamee)

    cpp.optimize()

    # cpp.computeIIS()
    # cpp.write("0_convex_cpp.ilp")

    print("model status = ", cpp.status)

    for n in range(N):
        print("")
        for i in range(J):
            print(f"V_{n}_{i} = {V_vars[i, n].x}")
        for i in range(J):
            print(f"exp(V_{n}_{i}) = {expV_vars[i, n].x}")
        for i in range(J):
            print(f"Probability P_{n}({i}) = {prob_vars[i, n].x}")

    objective_value = cpp.ObjVal
    bestprice = [p_vars[i].X for i in range(1, J)]

    total_time = time.time() - start_time

    best_lowerbound = cpp.ObjBoundC
    best_obj = objective_value
    nodes = cpp.NodeCount

    cpp.dispose()

    return total_time, bestprice, best_obj, best_lowerbound, nodes
