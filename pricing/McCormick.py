import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time


class CppModel:
    def __init__(self, model, MCconstraints, eta_bounds_constraints, p_vars, omega_vars=None, eta_vars=None,
                 om_v_constr=None, et_v_constr=None, ut_v_constr=None, bp_constr=None, objConstr=None,
                 optvalid_constr=None, opt_z_vars=None):
        self.model = model
        self.eta_bounds_constraints = eta_bounds_constraints
        self.MCconstraints = MCconstraints
        self.p_vars = p_vars
        self.omega_vars = omega_vars
        self.eta_vars = eta_vars
        self.om_v_constr = om_v_constr
        self.et_v_constr = et_v_constr
        self.ut_v_constr = ut_v_constr
        self.bp_constr = bp_constr
        self.objConstr = objConstr
        self.optvalid_constr = optvalid_constr
        self.opt_z_vars = opt_z_vars


def initialize_McCormick(N, R, J, exo_utility, endo_coef, time_limit, threads, p_U, p_L, onep, gapcuts, mcm2, caps,
                         optcut, objcut, validcuts=None, start_choices=None, first_guess_obj=0, pl_VIs=False,
                         do_vis=False, addeta=0, optvalid=0, start_prices=None, start_etas=None):
    if onep:
        # initialize model
        mccormmick_cpp = gp.Model("Continuous pricing problem")
        inf = GRB.INFINITY

        # initialize variables
        eta_vars = {n * R + r: mccormmick_cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{n * R + r}")
                    for n in range(N) for r in range(R)}
        omega_vars = {n * R + r: mccormmick_cpp.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"omega_{n * R + r}")
                      for n in range(N) for r in range(R)}
        p_vars = {i: mccormmick_cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in [2]}

        # best choice constraint
        for n in range(N):
            for r in range(R):
                # 0: Opt-Out
                # 1: PSP
                # 2: PUP

                mccormmick_cpp.addConstr((1 - omega_vars[n * R + r]) * exo_utility[0, n * R + r]
                                         + omega_vars[n * R + r] * exo_utility[1, n * R + r]
                                         + endo_coef[n * R + r] * eta_vars[n * R + r]
                                         >= exo_utility[0, n * R + r])

                mccormmick_cpp.addConstr((1 - omega_vars[n * R + r]) * exo_utility[0, n * R + r]
                                         + omega_vars[n * R + r] * exo_utility[1, n * R + r]
                                         + endo_coef[n * R + r] * eta_vars[n * R + r]
                                         >= exo_utility[1, n * R + r] + endo_coef[n * R + r] * p_vars[2])

        # define eta with mcCormick
        MCconstraints = dict()
        for n in range(N):
            for r in range(R):
                MCconstraints[0, n, r, 2] = mccormmick_cpp.addConstr(
                    eta_vars[n * R + r] - p_L[2] * omega_vars[n * R + r] >= 0)
                MCconstraints[1, n, r, 2] = mccormmick_cpp.addConstr(
                    eta_vars[n * R + r] - p_U[2] * omega_vars[n * R + r] - p_vars[2] >= - p_U[2])
                MCconstraints[2, n, r, 2] = mccormmick_cpp.addConstr(
                    eta_vars[n * R + r] - p_L[2] * omega_vars[n * R + r] - p_vars[2] <= - p_L[2])
                MCconstraints[3, n, r, 2] = mccormmick_cpp.addConstr(
                    eta_vars[n * R + r] - p_U[2] * omega_vars[n * R + r] <= 0)

        # Objective function
        objective = - (1 / R) * gp.quicksum(eta_vars[n * R + r] for n in range(N) for r in range(R))
        mccormmick_cpp.setObjective(objective, GRB.MINIMIZE)

        # solve model
        mccormmick_cpp.setParam('OutputFlag', 0)
        mccormmick_cpp.setParam("TimeLimit", time_limit)
        mccormmick_cpp.setParam('Threads', threads)

        mccormmick_cpp.setParam('ScaleFlag', 1)
        mccormmick_cpp.setParam('PrePasses', 2)
        mccormmick_cpp.setParam('Presolve', 1)
        mccormmick_cpp.setParam('NormAdjust', 1)
        mccormmick_cpp.setParam('SimplexPricing', 2)
        mccormmick_cpp.setParam('AggFill', 1000)
        mccormmick_cpp.setParam('PreDepRow', 1)
        mccormmick_cpp.setParam('PreDual', 0)
        mccormmick_cpp.setParam('NumericFocus', 1)

        # mccormmick_cpp.setParam('tuneTrials', 20)
        # mccormmick_cpp.setParam('tuneTimeLimit', 7200)
        # mccormmick_cpp.tune()
        #
        # exit()

        cpp_model = CppModel(mccormmick_cpp, MCconstraints, p_vars, omega_vars, eta_vars)
    else:
        # initialize model
        mccormmick_cpp = gp.Model("Continuous pricing problem")
        inf = GRB.INFINITY

        # initialize variables
        eta_vars = {(i, n * R + r): mccormmick_cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{n}_{r}")
                    for i in range(1, J) for n in range(N) for r in range(R)}
        p_vars = {i: mccormmick_cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in range(1, J)}
        omega_vars = {
            (i, n * R + r): mccormmick_cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}_{n}_{r}")
            for i in range(J) for n in range(N) for r in range(R)}
        if optvalid == 2:
            opt_z_vars = {
                (j, n * R + r): mccormmick_cpp.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"opt_z_{j}_{n}_{r}")
                for j in range(1, J) for n in range(N) for r in range(R)}
        else:
            opt_z_vars = None

        if caps is not None:
            Uinr_vars = {(i, n * R + r): mccormmick_cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS,
                                                               name=f"Uinr_{i}_{n}_{r}")
                         for i in range(J) for n in range(N) for r in range(R)}
            Unr_vars = {(n * R + r): mccormmick_cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS,
                                                           name=f"Unr_{n}_{r}")
                        for n in range(N) for r in range(R)}
            y_vars = {(i, n * R + r): mccormmick_cpp.addVar(lb=0, ub=inf, vtype=GRB.BINARY,
                                                            name=f"y_{i}_{n}_{r}")
                      for i in range(J) for n in range(N) for r in range(R)}
            zinr_vars = {(i, n * R + r): mccormmick_cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS,
                                                               name=f"zinr_{i}_{n}_{r}")
                         for i in range(J) for n in range(N) for r in range(R)}
            bigM = 1000

            # # price bounding constraints

        # one choice constraint
        for n in range(N):
            for r in range(R):
                mccormmick_cpp.addConstr(gp.quicksum(omega_vars[i, n * R + r] for i in range(J)) == 1)

        # best choice constraint
        for n in range(N):
            for r in range(R):
                if caps is None:
                    mccormmick_cpp.addConstr(omega_vars[0, n * R + r] * exo_utility[0, n * R + r] +
                                             gp.quicksum(omega_vars[i, n * R + r]
                                                         * exo_utility[i, n * R + r]
                                                         + endo_coef[i, n * R + r] * eta_vars[i, n * R + r]
                                                         for i in range(1, J))
                                             >= exo_utility[0, n * R + r])
                    for i in range(1, J):
                        mccormmick_cpp.addConstr(omega_vars[0, n * R + r] * exo_utility[0, n * R + r] +
                                                 gp.quicksum(omega_vars[i, n * R + r]
                                                             * exo_utility[i, n * R + r]
                                                             + endo_coef[i, n * R + r] * eta_vars[i, n * R + r]
                                                             for i in range(1, J))
                                                 >= exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_vars[i])
                else:
                    mccormmick_cpp.addConstr(exo_utility[0, n * R + r] * omega_vars[0, n * R + r] +
                                             gp.quicksum(exo_utility[i, n * R + r] * omega_vars[i, n * R + r]
                                                         + endo_coef[i, n * R + r] * eta_vars[i, n * R + r] for i in
                                                         range(1, J))
                                             == Unr_vars[n * R + r])

                    mccormmick_cpp.addConstr(Uinr_vars[0, n * R + r] == exo_utility[0, n * R + r])
                    # z is smoll if y = 0
                    mccormmick_cpp.addConstr(zinr_vars[0, n * R + r] <= -100 + bigM * y_vars[0, n * R + r])
                    # z is always smoller than Uinr
                    mccormmick_cpp.addConstr(zinr_vars[0, n * R + r] <= Uinr_vars[0, n * R + r])
                    # if y == 1 then z is equal to Uinr
                    mccormmick_cpp.addConstr(
                        zinr_vars[0, n * R + r] >= Uinr_vars[0, n * R + r] - bigM * (1 - y_vars[0, n * R + r]))
                    # y_0nr == 1
                    mccormmick_cpp.addConstr(y_vars[0, n * R + r] == 1)

                    # Replace Uinr here by discounted util z
                    # cpp.addConstr(Unr_vars[n * R + r] >= Uinr_vars[0, n * R + r])
                    mccormmick_cpp.addConstr(Unr_vars[n * R + r] >= zinr_vars[0, n * R + r])

                    # omega is smaller than y
                    mccormmick_cpp.addConstr(omega_vars[0, n * R + r] <= y_vars[0, n * R + r])

                    for i in range(1, J):
                        mccormmick_cpp.addConstr(
                            Uinr_vars[i, n * R + r] == exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_vars[i])
                        # z is smoll if y = 0
                        mccormmick_cpp.addConstr(zinr_vars[i, n * R + r] <= -100 + bigM * y_vars[i, n * R + r])
                        # z is always smoller than Uinr
                        mccormmick_cpp.addConstr(zinr_vars[i, n * R + r] <= Uinr_vars[i, n * R + r])
                        # if y == 1 then z is equal to Uinr
                        mccormmick_cpp.addConstr(
                            zinr_vars[i, n * R + r] >= Uinr_vars[i, n * R + r] - bigM * (1 - y_vars[i, n * R + r]))

                        # Replace Uinr here by discounted util z
                        # cpp.addConstr(Unr_vars[n * R + r] >= Uinr_vars[i, n * R + r])
                        mccormmick_cpp.addConstr(Unr_vars[n * R + r] >= zinr_vars[i, n * R + r])

                        # omega is smaller than y
                        mccormmick_cpp.addConstr(omega_vars[i, n * R + r] <= y_vars[i, n * R + r])

                        # define y vars
                        if n >= caps[i - 1]:
                            mccormmick_cpp.addConstr(gp.quicksum(omega_vars[i, m * R + r] for m in range(n)) <=
                                                     (caps[i - 1] - 1) * y_vars[i, n * R + r] + (n - 1) * (
                                                             1 - y_vars[i, n * R + r]))
                        if n >= 1:
                            mccormmick_cpp.addConstr(gp.quicksum(omega_vars[i, m * R + r] for m in range(n)) >=
                                                     caps[i - 1] * (1 - y_vars[i, n * R + r]))
        # define eta with McCormick envelope
        MCconstraints = dict()
        eta_bounds_constraints = dict()
        om_v_constr = dict()
        et_v_constr = dict()
        ut_v_constr = dict()
        bp_constr = dict()
        optvalid_constr = dict()
        for n in range(N):
            for r in range(R):
                for i in range(1, J):
                    if not mcm2:
                        MCconstraints[0, n, r, i] = mccormmick_cpp.addConstr(
                            eta_vars[i, n * R + r] - p_L[i] * omega_vars[i, n * R + r] >= 0)
                        MCconstraints[1, n, r, i] = mccormmick_cpp.addConstr(
                            eta_vars[i, n * R + r] - p_U[i] * omega_vars[i, n * R + r] - p_vars[i] >= - p_U[i])
                        MCconstraints[2, n, r, i] = mccormmick_cpp.addConstr(
                            eta_vars[i, n * R + r] - p_L[i] * omega_vars[i, n * R + r] - p_vars[i] <= - p_L[i])
                        MCconstraints[3, n, r, i] = mccormmick_cpp.addConstr(
                            eta_vars[i, n * R + r] - p_U[i] * omega_vars[i, n * R + r] <= 0)
                    else:
                        MCconstraints[1, n, r, i] = mccormmick_cpp.addConstr(
                            eta_vars[i, n * R + r] - p_U[i] * omega_vars[i, n * R + r] - p_vars[i] >= - p_U[i])
                        MCconstraints[2, n, r, i] = mccormmick_cpp.addConstr(
                            eta_vars[i, n * R + r] - p_U[i] * omega_vars[i, n * R + r] <= 0)
                        mccormmick_cpp.addConstr(eta_vars[i, n * R + r] - p_vars[i] <= 0)  # needs no update

                    if do_vis:
                        bp_constr[i, n, r, 0] = mccormmick_cpp.addConstr(p_vars[i]
                                                                         + 0 * omega_vars[i, n * R + r]
                                                                         <= p_U[i])  # placeholder rhs
                        if addeta >= 1:
                            if addeta == 4:
                                bp_constr[i, n, r, 2] = mccormmick_cpp.addConstr(eta_vars[i, n * R + r]
                                                                                 + 0 * omega_vars[i, n * R + r]
                                                                                 <= p_U[i])  # placeholder rhs
                            elif addeta == 1:
                                bp_constr[i, n, r, 2] = mccormmick_cpp.addConstr(p_vars[i]
                                                                                 + 0 * eta_vars[i, n * R + r]
                                                                                 <= p_U[i])  # placeholder rhs
                        if pl_VIs:
                            bp_constr[i, n, r, 1] = mccormmick_cpp.addConstr(p_vars[i]
                                                                             + 0 * omega_vars[i, n * R + r]
                                                                             >= p_L[i])  # placeholder rhs
                            if addeta == 2:
                                bp_constr[i, n, r, 3] = mccormmick_cpp.addConstr(p_vars[i]
                                                                                 + 0 * eta_vars[i, n * R + r]
                                                                                 >= p_L[i])  # placeholder rhs

                    if validcuts and optvalid == 2:
                        if start_choices[i, n * R + r] == 1:
                            optvalid_constr[i, n, r] = \
                                mccormmick_cpp.addConstr(p_vars[i] +
                                                         (start_prices[i - 1] - p_L[i]) * opt_z_vars[i, n * R + r]
                                                         >= start_prices[i - 1])
                            for j in range(1, J):
                                if j != i:
                                    optvalid_constr[j, n, r] = \
                                        mccormmick_cpp.addConstr(- p_vars[j] +
                                                                 (p_U[j] - start_prices[j - 1]) * opt_z_vars[
                                                                     j, n * R + r]
                                                                 >= - start_prices[j - 1])

                            mccormmick_cpp.addConstr(omega_vars[i, n * R + r] >=
                                                     gp.quicksum(opt_z_vars[j, n * R + r]
                                                                 for j in range(1, J))
                                                     - (J - 2))
                            mccormmick_cpp.addConstr(eta_vars[i, n * R + r] >=
                                                     p_L[i] * (gp.quicksum(opt_z_vars[j, n * R + r]
                                                                           for j in range(1, J))
                                                               - (J - 2)))

        # Objective function
        objective = - (1 / R) * gp.quicksum(eta_vars[i, n * R + r] for i in range(1, J)
                                            for n in range(N) for r in range(R))

        # mccormmick_cpp.addConstr(objective <= start_obj)

        mccormmick_cpp.setObjective(objective, GRB.MINIMIZE)

        if optcut or objcut:
            objConstr = mccormmick_cpp.addConstr(objective <= first_guess_obj)
        else:
            objConstr = mccormmick_cpp.addConstr(objective <= 0)

        mccormmick_cpp.setParam('OutputFlag', 0)
        mccormmick_cpp.setParam("TimeLimit", time_limit)
        mccormmick_cpp.setParam('Threads', threads)

        mccormmick_cpp.setParam('ScaleFlag', 1)
        mccormmick_cpp.setParam('PrePasses', 2)
        mccormmick_cpp.setParam('Presolve', 1)
        mccormmick_cpp.setParam('NormAdjust', 1)
        mccormmick_cpp.setParam('SimplexPricing', 2)
        mccormmick_cpp.setParam('AggFill', 1000)
        mccormmick_cpp.setParam('PreDepRow', 1)
        mccormmick_cpp.setParam('PreDual', 0)
        mccormmick_cpp.setParam('NumericFocus', 1)

        cpp_model = CppModel(mccormmick_cpp, MCconstraints, eta_bounds_constraints, p_vars, omega_vars, eta_vars,
                             om_v_constr, et_v_constr, ut_v_constr, bp_constr, objConstr, optvalid_constr, opt_z_vars)

    return cpp_model


def cpp_QCLP_cap(N, R, J, exo_utility, endo_coef, time_limit, threads, p_L, p_U, caps):
    # initialize model
    cpp = gp.Model("Continuous pricing problem")
    inf = GRB.INFINITY

    # caps = [int(c * R) for c in caps]

    # initialize variables
    eta_vars = {(i, n * R + r): cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{n}_{r}")
                for i in range(1, J) for n in range(N) for r in range(R)}
    omega_vars = {(i, n * R + r): cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}_{n}_{r}")
                  for i in range(J) for n in range(N) for r in range(R)}
    p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
              for i in range(1, J)}
    Uinr_vars = {(i, n * R + r): cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS,
                                            name=f"Uinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    Unr_vars = {(n * R + r): cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS,
                                        name=f"Unr_{n}_{r}")
                for n in range(N) for r in range(R)}
    y_vars = {(i, n * R + r): cpp.addVar(lb=0, ub=inf, vtype=GRB.BINARY,
                                         name=f"y_{i}_{n}_{r}")
              for i in range(J) for n in range(N) for r in range(R)}
    zinr_vars = {(i, n * R + r): cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS,
                                            name=f"zinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    bigM = 1000

    # Objective function
    objective = - 1 / R * gp.quicksum(
        eta_vars[i, n * R + r] for i in range(1, J) for n in range(N) for r in range(R))
    cpp.setObjective(objective, GRB.MINIMIZE)

    # one choice constraint
    for n in range(N):
        for r in range(R):
            cpp.addConstr(gp.quicksum(omega_vars[i, n * R + r] for i in range(J)) == 1)

    # best choice constraint
    for n in range(N):
        for r in range(R):
            cpp.addConstr(exo_utility[0, n * R + r] * omega_vars[0, n * R + r] +
                          gp.quicksum(exo_utility[i, n * R + r] * omega_vars[i, n * R + r]
                                      + endo_coef[i, n * R + r] * eta_vars[i, n * R + r] for i in range(1, J))
                          == Unr_vars[n * R + r])

            cpp.addConstr(Uinr_vars[0, n * R + r] == exo_utility[0, n * R + r])
            # z is smoll if y = 0
            cpp.addConstr(zinr_vars[0, n * R + r] <= -100 + bigM * y_vars[0, n * R + r])
            # z is always smoller than Uinr
            cpp.addConstr(zinr_vars[0, n * R + r] <= Uinr_vars[0, n * R + r])
            # if y == 1 then z is equal to Uinr
            cpp.addConstr(
                zinr_vars[0, n * R + r] >= Uinr_vars[0, n * R + r] - bigM * (1 - y_vars[0, n * R + r]))
            # y_0nr == 1
            cpp.addConstr(y_vars[0, n * R + r] == 1)

            # Replace Uinr here by discounted util z
            # cpp.addConstr(Unr_vars[n * R + r] >= Uinr_vars[0, n * R + r])
            cpp.addConstr(Unr_vars[n * R + r] >= zinr_vars[0, n * R + r])

            # omega is smaller than y
            cpp.addConstr(omega_vars[0, n * R + r] <= y_vars[0, n * R + r])

            for i in range(1, J):
                cpp.addConstr(
                    Uinr_vars[i, n * R + r] == exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_vars[i])
                # z is smoll if y = 0
                cpp.addConstr(zinr_vars[i, n * R + r] <= -100 + bigM * y_vars[i, n * R + r])
                # z is always smoller than Uinr
                cpp.addConstr(zinr_vars[i, n * R + r] <= Uinr_vars[i, n * R + r])
                # if y == 1 then z is equal to Uinr
                cpp.addConstr(
                    zinr_vars[i, n * R + r] >= Uinr_vars[i, n * R + r] - bigM * (1 - y_vars[i, n * R + r]))

                # Replace Uinr here by discounted util z
                # cpp.addConstr(Unr_vars[n * R + r] >= Uinr_vars[i, n * R + r])
                cpp.addConstr(Unr_vars[n * R + r] >= zinr_vars[i, n * R + r])

                # omega is smaller than y
                cpp.addConstr(omega_vars[i, n * R + r] <= y_vars[i, n * R + r])

                # define y vars
                if n >= caps[i - 1]:
                    cpp.addConstr(gp.quicksum(omega_vars[i, m * R + r] for m in range(n)) <=
                                  (caps[i - 1] - 1) * y_vars[i, n * R + r] + (n - 1) * (
                                          1 - y_vars[i, n * R + r]))
                if n >= 1:
                    cpp.addConstr(gp.quicksum(omega_vars[i, m * R + r] for m in range(n)) >=
                                  caps[i - 1] * (1 - y_vars[i, n * R + r]))

    # for n in range(N):
    #     for r in range(R):
    #         cpp.addConstr(omega_vars[0, n * R + r] * exo_utility[0, n * R + r] +
    #                       gp.quicksum(omega_vars[i, n * R + r]
    #                                   * exo_utility[i, n * R + r]
    #                                   + endo_coef[i, n * R + r] * eta_vars[i, n * R + r]
    #                                   for i in range(1, J))
    #                       >= exo_utility[0, n * R + r])
    #         for i in range(1, J):
    #             cpp.addConstr(omega_vars[0, n * R + r] * exo_utility[0, n * R + r] +
    #                           gp.quicksum(omega_vars[i, n * R + r]
    #                                       * exo_utility[i, n * R + r]
    #                                       + endo_coef[i, n * R + r] * eta_vars[i, n * R + r]
    #                                       for i in range(1, J))
    #                           >= exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_vars[i])

    # define eta
    for n in range(N):
        for r in range(R):
            for i in range(1, J):
                cpp.addConstr(eta_vars[i, n * R + r] == omega_vars[i, n * R + r] * p_vars[i])

    # cpp.setParam('OutputFlag', int(R > 1))  # when used in the heuristic we suppress the log
    #
    # cpp.setParam('ScaleFlag', 0)
    # cpp.setParam('SimplexPricing', 2)
    # cpp.setParam('NormAdjust', 0)
    # cpp.setParam('MIPFocus', 3)
    # cpp.setParam('Cuts', 0)

    cpp.setParam('OutputFlag', 0)
    cpp.setParam("TimeLimit", time_limit)
    cpp.setParam("Threads", threads)
    cpp.setParam("NonConvex", 2)

    cpp.optimize()

    # print("Solving QCLP without capacities")
    # print("status = ", cpp.status)
    # print("obj = ", cpp.ObjVal)

    # for n in range(N):
    #     for r in range(R):
    #         if Uinr_vars[1, n * R + r].x - zinr_vars[1, n * R + r].x > 10:
    #             print(f"n,r = {n},{r}")
    #             print([Uinr_vars[i, n * R + r].x for i in range(J)])
    #             print([zinr_vars[i, n * R + r].x for i in range(J)])
    #         if any(Uinr_vars[i, n * R + r].x - zinr_vars[i, n * R + r].x > 10 for i in range(J)):
    #             print(f"{n},{r} was blocked")

    omega_values = {(i, n * R + r): omega_vars[i, n * R + r].x for i in range(J) for n in range(N) for r in range(R)}

    return omega_values, cpp.ObjVal


def solve_MILP_cap(N, R, J, exo_utility, endo_coef, time_limit, threads, fixed_price, p_l, p_u, caps, binar=False):
    # initialize model
    mccormmick_cpp = gp.Model("Continuous pricing problem")
    inf = GRB.INFINITY

    if p_l is None and p_u is None:
        p_l = {i: fixed_price[i - 1] for i in range(1, J)}
        p_u = {i: fixed_price[i - 1] for i in range(1, J)}

    # initialize variables
    eta_vars = {(i, n * R + r): mccormmick_cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{n}_{r}")
                for i in range(1, J) for n in range(N) for r in range(R)}
    if binar:
        omega_vars = {
            (i, n * R + r): mccormmick_cpp.addVar(lb=0, ub=inf, vtype=GRB.BINARY, name=f"omega_{i}_{n}_{r}")
            for i in range(J) for n in range(N) for r in range(R)}
    else:
        omega_vars = {
            (i, n * R + r): mccormmick_cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}_{n}_{r}")
            for i in range(J) for n in range(N) for r in range(R)}
    p_vars = {i: mccormmick_cpp.addVar(lb=p_l[i], ub=p_u[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
              for i in range(1, J)}
    Uinr_vars = {(i, n * R + r): mccormmick_cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS,
                                                       name=f"Uinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    Unr_vars = {(n * R + r): mccormmick_cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS,
                                                   name=f"Unr_{n}_{r}")
                for n in range(N) for r in range(R)}
    y_vars = {(i, n * R + r): mccormmick_cpp.addVar(lb=0, ub=inf, vtype=GRB.BINARY,
                                                    name=f"y_{i}_{n}_{r}")
              for i in range(J) for n in range(N) for r in range(R)}
    zinr_vars = {(i, n * R + r): mccormmick_cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS,
                                                       name=f"zinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    bigM = 1000

    # # fixing the price
    # for i in range(1, J):
    #     mccormmick_cpp.addConstr(p_vars[i] == fixed_price[i - 1])

    # p_U = {i: fixed_price[i - 1] for i in range(1, J)}

    # one choice constraint
    for n in range(N):
        for r in range(R):
            mccormmick_cpp.addConstr(gp.quicksum(omega_vars[i, n * R + r] for i in range(J)) == 1)
            for i in range(J):
                # cpp.addConstr(Unr_vars[n * R + r] <= Uinr_vars[i, n * R + r] + bigM * (1 - omega_vars[i, n * R + r]))
                mccormmick_cpp.addConstr(
                    Unr_vars[n * R + r] <= zinr_vars[i, n * R + r] + bigM * (1 - omega_vars[i, n * R + r]))

    # best choice constraint
    for n in range(N):
        for r in range(R):
            mccormmick_cpp.addConstr(Uinr_vars[0, n * R + r] == exo_utility[0, n * R + r])
            # z is smoll if y = 0
            mccormmick_cpp.addConstr(zinr_vars[0, n * R + r] <= -100 + bigM * y_vars[0, n * R + r])
            # z is always smoller than Uinr
            mccormmick_cpp.addConstr(zinr_vars[0, n * R + r] <= Uinr_vars[0, n * R + r])
            # if y == 1 then z is equal to Uinr
            mccormmick_cpp.addConstr(
                zinr_vars[0, n * R + r] >= Uinr_vars[0, n * R + r] - bigM * (1 - y_vars[0, n * R + r]))
            # y_0nr == 1
            mccormmick_cpp.addConstr(y_vars[0, n * R + r] == 1)

            # Replace Uinr here by discounted util z
            # cpp.addConstr(Unr_vars[n * R + r] >= Uinr_vars[0, n * R + r])
            mccormmick_cpp.addConstr(Unr_vars[n * R + r] >= zinr_vars[0, n * R + r])

            # omega is smaller than y
            mccormmick_cpp.addConstr(omega_vars[0, n * R + r] <= y_vars[0, n * R + r])

            for i in range(1, J):
                mccormmick_cpp.addConstr(
                    Uinr_vars[i, n * R + r] == exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_vars[i])
                # z is smoll if y = 0
                mccormmick_cpp.addConstr(zinr_vars[i, n * R + r] <= -100 + bigM * y_vars[i, n * R + r])
                # z is always smoller than Uinr
                mccormmick_cpp.addConstr(zinr_vars[i, n * R + r] <= Uinr_vars[i, n * R + r])
                # if y == 1 then z is equal to Uinr
                mccormmick_cpp.addConstr(
                    zinr_vars[i, n * R + r] >= Uinr_vars[i, n * R + r] - bigM * (1 - y_vars[i, n * R + r]))

                # Replace Uinr here by discounted util z
                # cpp.addConstr(Unr_vars[n * R + r] >= Uinr_vars[i, n * R + r])
                mccormmick_cpp.addConstr(Unr_vars[n * R + r] >= zinr_vars[i, n * R + r])

                # omega is smaller than y
                mccormmick_cpp.addConstr(omega_vars[i, n * R + r] <= y_vars[i, n * R + r])

                # define y vars
                if n >= caps[i - 1]:
                    mccormmick_cpp.addConstr(gp.quicksum(omega_vars[i, m * R + r] for m in range(n)) <=
                                             (caps[i - 1] - 1) * y_vars[i, n * R + r] + (n - 1) * (
                                                     1 - y_vars[i, n * R + r]))
                if n >= 1:
                    mccormmick_cpp.addConstr(gp.quicksum(omega_vars[i, m * R + r] for m in range(n)) >=
                                             caps[i - 1] * (1 - y_vars[i, n * R + r]))
    # define eta with McCormick envelope
    for n in range(N):
        for r in range(R):
            for i in range(1, J):
                # mccormmick_cpp.addConstr(
                #     eta_vars[i, n * R + r] - p_u[i] * omega_vars[i, n * R + r] - p_vars[i] >= - p_u[i])
                # mccormmick_cpp.addConstr(
                #     eta_vars[i, n * R + r] - p_u[i] * omega_vars[i, n * R + r] <= 0)
                # mccormmick_cpp.addConstr(eta_vars[i, n * R + r] - p_vars[i] <= 0)
                mccormmick_cpp.addConstr(
                    eta_vars[i, n * R + r] - p_l[i] * omega_vars[i, n * R + r] >= 0)
                mccormmick_cpp.addConstr(
                    eta_vars[i, n * R + r] - p_u[i] * omega_vars[i, n * R + r] - p_vars[i] >= - p_u[i])
                mccormmick_cpp.addConstr(
                    eta_vars[i, n * R + r] - p_l[i] * omega_vars[i, n * R + r] - p_vars[i] <= - p_l[i])
                mccormmick_cpp.addConstr(
                    eta_vars[i, n * R + r] - p_u[i] * omega_vars[i, n * R + r] <= 0)

    # Objective function
    objective = - (1 / R) * gp.quicksum(eta_vars[i, n * R + r] for i in range(1, J)
                                        for n in range(N) for r in range(R))

    mccormmick_cpp.setObjective(objective, GRB.MINIMIZE)

    mccormmick_cpp.setParam('OutputFlag', 0)
    mccormmick_cpp.setParam("TimeLimit", time_limit)
    mccormmick_cpp.setParam('Threads', threads)
    mccormmick_cpp.setParam('DualReductions', 0)

    mccormmick_cpp.setParam('ScaleFlag', 1)
    mccormmick_cpp.setParam('PrePasses', 2)
    mccormmick_cpp.setParam('Presolve', 1)
    mccormmick_cpp.setParam('NormAdjust', 1)
    mccormmick_cpp.setParam('SimplexPricing', 2)
    mccormmick_cpp.setParam('AggFill', 1000)
    mccormmick_cpp.setParam('PreDepRow', 1)
    mccormmick_cpp.setParam('PreDual', 0)
    mccormmick_cpp.setParam('NumericFocus', 1)

    mccormmick_cpp.optimize()

    # print("Solving MILP with capacities")
    # print("status = ", mccormmick_cpp.status)
    # print("obj = ", mccormmick_cpp.ObjVal)

    omega_values = {(i, n * R + r): omega_vars[i, n * R + r].x for i in range(J) for n in range(N) for r in range(R)}

    return omega_values, mccormmick_cpp.ObjVal


def cpp_MILP_cap(time_limit, threads, p_L, p_U, N, R, J, exo_utility, endo_coef, J_PSP, J_PUP, caps, p_value):
    start_time = time.time()

    # initialize model
    cpp = gp.Model("Continuous pricing problem")
    inf = GRB.INFINITY

    # initialize variables
    eta_vars = {(i, n * R + r): cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{n}_{r}")
                for i in range(1, J) for n in range(N) for r in range(R)}
    Uinr_vars = {(i, n * R + r): cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"Uinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}
    Unr_vars = {(n * R + r): cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"Unr_{n}_{r}")
                for n in range(N) for r in range(R)}
    omega_vars = {(i, n * R + r): cpp.addVar(lb=0, ub=inf, vtype=GRB.BINARY, name=f"omega_{i}_{n}_{r}")
                  for i in range(J) for n in range(N) for r in range(R)}
    p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
              for i in range(1, J)}

    y_vars = {(i, n * R + r): cpp.addVar(lb=0, ub=inf, vtype=GRB.BINARY, name=f"y_{i}_{n}_{r}")
              for i in range(J) for n in range(N) for r in range(R)}
    zinr_vars = {(i, n * R + r): cpp.addVar(lb=-inf, ub=inf, vtype=GRB.CONTINUOUS, name=f"zinr_{i}_{n}_{r}")
                 for i in range(J) for n in range(N) for r in range(R)}

    bigM = 1000

    for i in range(1, J):
        cpp.addConstr(p_vars[i] == p_value[i - 1])

    # Objective function
    objective = 1 / R * gp.quicksum(
        eta_vars[i, n * R + r] for i in range(1, J) for n in range(N) for r in range(R))
    cpp.setObjective(objective, GRB.MAXIMIZE)

    # define utilities and max util U_nr and z_utils and eeh y not y
    for n in range(N):
        for r in range(R):
            cpp.addConstr(Uinr_vars[0, n * R + r] == exo_utility[0, n * R + r])

            # z is smoll if y = 0
            cpp.addConstr(zinr_vars[0, n * R + r] <= -100 + bigM * y_vars[0, n * R + r])
            # z is always smoller than Uinr
            cpp.addConstr(zinr_vars[0, n * R + r] <= Uinr_vars[0, n * R + r])
            # if y == 1 then z is equal to Uinr
            cpp.addConstr(zinr_vars[0, n * R + r] >= Uinr_vars[0, n * R + r] - bigM * (1 - y_vars[0, n * R + r]))

            # y_0nr == 1
            cpp.addConstr(y_vars[0, n * R + r] == 1)

            # Replace Uinr here by discounted util z
            # cpp.addConstr(Unr_vars[n * R + r] >= Uinr_vars[0, n * R + r])
            cpp.addConstr(Unr_vars[n * R + r] >= zinr_vars[0, n * R + r])

            # omega is smaller than y
            cpp.addConstr(omega_vars[0, n * R + r] <= y_vars[0, n * R + r])

            for i in range(1, J):
                cpp.addConstr(
                    Uinr_vars[i, n * R + r] == exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_vars[i])
                # z is smoll if y = 0
                cpp.addConstr(zinr_vars[i, n * R + r] <= -100 + bigM * y_vars[i, n * R + r])
                # z is always smoller than Uinr
                cpp.addConstr(zinr_vars[i, n * R + r] <= Uinr_vars[i, n * R + r])
                # if y == 1 then z is equal to Uinr
                cpp.addConstr(zinr_vars[i, n * R + r] >= Uinr_vars[i, n * R + r] - bigM * (1 - y_vars[i, n * R + r]))

                # Replace Uinr here by discounted util z
                # cpp.addConstr(Unr_vars[n * R + r] >= Uinr_vars[i, n * R + r])
                cpp.addConstr(Unr_vars[n * R + r] >= zinr_vars[i, n * R + r])

                # omega is smaller than y
                cpp.addConstr(omega_vars[i, n * R + r] <= y_vars[i, n * R + r])

                # define y vars
                if n >= caps[i - 1]:
                    cpp.addConstr(gp.quicksum(omega_vars[i, m * R + r] for m in range(n)) <=
                                  (caps[i - 1] - 1) * y_vars[i, n * R + r] + (n - 1) * (1 - y_vars[i, n * R + r]))
                if n >= 1:
                    cpp.addConstr(gp.quicksum(omega_vars[i, m * R + r] for m in range(n)) >=
                                  caps[i - 1] * (1 - y_vars[i, n * R + r]))

    # define omega
    for n in range(N):
        for r in range(R):
            cpp.addConstr(gp.quicksum(omega_vars[i, n * R + r] for i in range(J)) == 1)
            for i in range(J):
                # cpp.addConstr(Unr_vars[n * R + r] <= Uinr_vars[i, n * R + r] + bigM * (1 - omega_vars[i, n * R + r]))
                cpp.addConstr(Unr_vars[n * R + r] <= zinr_vars[i, n * R + r] + bigM * (1 - omega_vars[i, n * R + r]))

    # define eta with bigM constraints, where M = upper bound on p_i, i.e. p_U[i]
    for n in range(N):
        for r in range(R):
            for i in range(1, J):
                # cpp.addConstr(eta_vars[i, n * R + r] == omega_vars[i, n * R + r] * p_vars[i])
                cpp.addConstr(eta_vars[i, n * R + r] <= omega_vars[i, n * R + r] * p_U[i])
                cpp.addConstr(eta_vars[i, n * R + r] <= p_vars[i])
                cpp.addConstr(eta_vars[i, n * R + r] >= 0)
                cpp.addConstr(eta_vars[i, n * R + r] >= p_vars[i] - (1 - omega_vars[i, n * R + r]) * p_U[i])

    cpp.setParam('OutputFlag', 0)
    #     cpp.setParam("TimeLimit", time_limit)
    cpp.setParam("Threads", threads)

    # Write the model to an LP file
    #     cpp.write("model_gurobipy.lp")

    cpp.optimize()

    if cpp.SolCount >= 1:
        objective_value = cpp.ObjVal
        bestprice = [p_vars[i].X for i in range(1, J)]
    else:
        objective_value = 1000
        bestprice = [1000] * (J - 1)

    for i in range(J):
        print(f"# of people choosing {i} = {sum(omega_vars[i, n * R + r].X for n in range(N) for r in range(R))}")

    #     print("")
    #     for n in range(N):
    #         for r in range(R):
    #             for i in range(D+1):
    #                 print(f"Person {n} in scen {r} has y_{i} = {y_vars[i, n * R + r].X}, om_{i} = {omega_vars[i,n * R + r].X}")

    total_time = time.time() - start_time

    try:
        best_lowerbound = cpp.ObjBoundC
    except AttributeError:
        best_lowerbound = -1000

    best_obj = objective_value
    gap = cpp.MIPGap
    nodes = cpp.NodeCount

    # cpp.dispose()

    return total_time, bestprice, best_obj


def compute_obj_value_caps(J, N, R, exo_utility, endo_coef, p, J_PSP, J_PUP, caps):
    p.append(0)  # this is for tie breaks
    obj = 0
    omega = np.zeros((J, N * R))
    occ = np.zeros(J)
    eta = {(i, n * R + r): 0 for i in range(1, J) for n in range(N) for r in range(R)}
    capso = [int(c * R) for c in caps]
    for n in range(N):
        for r in range(R):
            util_list = [exo_utility[0, n * R + r]]
            for i in range(J_PSP):
                if occ[i + 1] < capso[i]:
                    util_list.append(exo_utility[i + 1, n * R + r] + endo_coef[i + 1, n * R + r] * p[i])
                else:
                    util_list.append(-500)
            for i in range(J_PUP):
                if occ[i + J_PSP + 1] < capso[i + J_PSP]:
                    util_list.append(exo_utility[i + J_PSP + 1, n * R + r]
                                     + endo_coef[i + J_PSP + 1, n * R + r] * p[i + J_PSP])
                else:
                    util_list.append(-500)
            # get best alternative, with tie-breaks based on price values
            max_value = max(util_list)
            candidate_indices = [i for i, val in enumerate(util_list) if
                                 abs(round(val, 3) - round(max_value, 3)) == 0]
            max_util = max(candidate_indices, key=lambda idx: p[idx - 1])
            omega[max_util, n * R + r] = 1

            occ[max_util] += 1

            if 1 <= max_util:
                obj -= (1 / R) * p[max_util - 1]
                eta[max_util, n * R + r] = p[max_util - 1]
    del p[-1]
    return occ, obj, omega
