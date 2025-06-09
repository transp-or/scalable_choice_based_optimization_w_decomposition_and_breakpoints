import datetime

import numpy as np
import time
import warnings
import gurobipy as gp
from gurobipy import GRB
import logging
from joblib import Parallel, delayed
import os
import sys
import copy

console = False 


# Define a context manager to suppress output


class SuppressOutput:
    def __enter__(self):
        self.stdout_original = sys.stdout
        self.stderr_original = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.stdout_original
        sys.stderr = self.stderr_original


class Inputs:
    def __init__(self, N, R, J, exo_utility, endo_coef, p_L, p_U, threads, J_PSP, J_PUP, pres, viol):
        self.N = N
        self.J = J
        self.R = R
        self.exo_utility = exo_utility
        self.endo_coef = endo_coef
        self.p_L = p_L
        self.p_U = p_U
        self.threads = threads
        self.J_PSP = J_PSP
        self.J_PUP = J_PUP
        self.iteration = 0
        self.pres = pres
        self.viol = viol


class MCMModel:
    def __init__(self, cpp, alpha, omega_vars, eta_vars, p_vars, opt_z_vars, varphi, p_L, p_U, onep, bp_constr,
                 optvalid_constr):
        self.cpp = cpp
        self.alpha = alpha
        self.omega_vars = omega_vars
        self.eta_vars = eta_vars
        self.p_vars = p_vars
        self.opt_z_vars = opt_z_vars
        self.varphi = varphi
        self.p_L = p_L
        self.p_U = p_U
        self.onep = onep
        self.bp_constr = bp_constr
        self.optvalid_constr = optvalid_constr


class MasterProblem:
    def __init__(self, inputs, pres, lwbound, fixed, onep, outsource, mascut_constraint, best_upper_bound, threads):
        self.inputs = inputs
        self.upperbounding = 0
        self.cut_counter = 0
        self.upperbound = 0
        self.lowerbound = lwbound  # - self.inputs.N * max(inputs.p_U[1], inputs.p_U[2])
        self.start_time = time.time()
        self.onep = onep
        self.outsource = outsource
        self.pres = pres

        # if console:
        #     print(f"outsource = {outsource}")
        # else:
        #     logging.info(f"outsource = {outsource}")

        # print("initialized master attributes")
        # print("creating GRM model (master)")
        mastero = gp.Model("master")

        # skeleton = gp.Env(empty=True)
        # skeleton.setParam("OutputFlag", 0)
        # skeleton.start()
        #  mastero = gp.read(f"000_MP_skeleton.mps", env=skeleton)

        # print("created GRB model (master)")

        inf = GRB.INFINITY
        N = inputs.N
        R = inputs.R
        p_L = inputs.p_L
        p_U = inputs.p_U
        J_PSP = inputs.J_PSP
        J_PUP = inputs.J_PUP
        J = inputs.J

        # print("read master inputs")

        obj_var = mastero.addVar(lb=-inf, ub=0, name="obj")

        if onep:
            p_vars = mastero.addMVar(shape=1, lb=-inf, ub=inf, name="p")
        else:
            p_vars = mastero.addMVar(shape=J - 1, lb=-inf, ub=inf, name="p")
        # print("initialized p variables")
        obj_nr_var = mastero.addMVar(shape=N * R, lb=-inf, ub=0, name="obj_nr")
        # print("initialized obj_nr variables")

        if mascut_constraint:
            mastero.addConstr(obj_var <= best_upper_bound, name="mascut_constraint")

        mastero.setObjective(obj_var, GRB.MINIMIZE)
        # print("set objective")

        # add bounds on price
        # print("current bounds are")
        # print(f"p_l = {p_L}")
        # print(f"p_u = {p_U}")
        # p_l = {1: 0.5375000000000001, 2: 0.5, 3: 0.65, 4: 0.6605156476417611, 5: 0.65}
        # p_u = {1: 0.55, 2: 0.517773754257123, 3: 0.6625000000000001, 4: 0.671031295283522, 5: 0.6980950715066943}
        if onep:
            mastero.addConstr(p_vars[0] >= p_L[2], name="p2_low")
            mastero.addConstr(p_vars[0] <= p_U[2], name="p2_high")
        else:
            for i in range(1, J):
                mastero.addConstr(p_vars[i - 1] >= p_L[i], name=f"p{i}_low")
                mastero.addConstr(p_vars[i - 1] <= p_U[i], name=f"p{i}_high")

        # define total profit
        # print("added bounds on price")
        # print("adding bound constraint")
        mastero.addConstr(obj_var == np.ones(N * R) @ obj_nr_var)
        # print("added sum constraint")
        self.lowBoundConstr = mastero.addConstr(obj_var >= self.lowerbound, name="lowb")
        self.upBoundConstr = mastero.addConstr(obj_var <= 0, name="upb")

        if fixed is not None:
            if onep:
                for n in range(N):
                    for r in range(R):
                        if not fixed[n, r] == 3:
                            obj_nr_var[n * R + r].Obj = 0
            else:
                for n in range(N):
                    for r in range(R):
                        if sum(fixed[i, n, r] for i in range(3)) == 1:
                            obj_nr_var[n * R + r].Obj = 0

        self.vars = {'obj_nr': obj_nr_var,
                     'obj': obj_var,
                     'p': p_vars}

        mastero.setParam('OutputFlag', 0)
        mastero.setParam("Threads", self.inputs.threads)

        mastero.setParam('ScaleFlag', 1)
        mastero.setParam('SimplexPricing', 3)
        mastero.setParam('NormAdjust', 1)
        mastero.setParam('PreDual', 1)
        mastero.setParam('NumericFocus', 1)
        mastero.setParam('PreDepRow', 0)

        mastero.setParam('Presolve', 1)
        if timelimit_tricks:
            mastero.setParam('TimeLimit', R / 10)
        # mastero.setParam('SolutionLimit', 1)

        mastero.setParam('Threads', threads)

        self.model = mastero

    def add_cut(self, p, profit, varphi):
        add_only_violated = self.inputs.viol

        if add_only_violated:
            N = self.inputs.N
            R = self.inputs.R
            J = self.inputs.J
            epsilon_viol = 1e-9

            violated_count = 0
            violations = []

            # Adding cuts in bulk with addConstrs and LinExpr

            # # Step 1: Identify violated cuts
            # violated_cuts = []
            # for n in range(N):
            #     for r in range(R):
            #         # Initialize a LinExpr for the left-hand side of the cut
            #         cut_lhs = gp.LinExpr()
            #         cut_lhs += self.vars["obj_nr"][n * R + r]
            #         for i in range(J - 1):
            #             cut_lhs -= varphi[n * R + r, i] * self.vars["p"][i]
            #
            #         # Calculate the right-hand side (similar to before)
            #         cut_rhs = profit[n * R + r] - np.sum(
            #             varphi[n * R + r, i] * p[i] for i in range(J - 1)) - epsilon_viol
            #
            #         # Check if the cut is violated
            #         if cut_lhs.getValue() <= cut_rhs:
            #             violated_cuts.append((n, r))
            #
            # # Step 2: Build Linear Expressions for each violated cut
            # cut_expressions = {}
            # for index, (n, r) in enumerate(violated_cuts):
            #     violated_count += 1
            #     add_to_cut_counter = 1
            #     # Initialize a LinExpr for the left-hand side of the cut
            #     cut_lhs = gp.LinExpr()
            #     cut_lhs += self.vars["obj_nr"][n * R + r]
            #     for i in range(J - 1):
            #         cut_lhs -= varphi[n * R + r, i] * self.vars["p"][i]
            #
            #     # Calculate the right-hand side (similar to before)
            #     cut_rhs = profit[n * R + r] - np.sum(varphi[n * R + r, i] * p[i] for i in range(J - 1))
            #
            #     # Add to cut_expressions
            #     cut_expressions[index] = [cut_lhs, cut_rhs]
            #
            # # Step 3: Add all violated cuts to the model at once
            # if violated_count >= 2:
            #     self.model.addConstrs(cut_expressions[expr][0] >= cut_expressions[expr][1]
            #                           for expr in cut_expressions.keys())
            # elif violated_count == 1:
            #     self.model.addConstr(cut_expressions[0][0] >= cut_expressions[0][
            #         1])  # adding a single constraint with addConstrs yields an error
            # else:
            #     pass

            # Adding cuts in bulk with addConstrs
            #
            # # Step 1: Identify violated cuts
            # violated_cuts = []
            # for n in range(N):
            #     for r in range(R):
            #         cut_lhs = self.vars["obj_nr"][n * R + r] \
            #                   - gp.quicksum(varphi[n * R + r, i] * self.vars["p"][i] for i in range(J - 1))
            #         cut_rhs = profit[n * R + r] - np.sum(
            #             varphi[n * R + r, i] * p[i] for i in range(J - 1)) - epsilon_viol
            #         if cut_lhs.getValue() <= cut_rhs:
            #             violated_cuts.append((n, r))
            #
            # # Step 2: Build Linear Expressions for each violated cut
            # cut_expressions = {}
            # for index, (n, r) in enumerate(violated_cuts):
            #     violated_count += 1
            #     add_to_cut_counter = 1
            #     cut_lhs = self.vars["obj_nr"][n * R + r] \
            #               - gp.quicksum(varphi[n * R + r, i] * self.vars["p"][i] for i in range(J - 1))
            #     cut_rhs = profit[n * R + r] - np.sum(varphi[n * R + r, i] * p[i] for i in range(J - 1))
            #     cut_expressions[index] = [cut_lhs, cut_rhs]
            #
            # # Step 3: Add all violated cuts to the model at once
            # if violated_count >= 2:
            #     self.model.addConstrs(cut_expressions[expr][0] >= cut_expressions[expr][1]
            #                           for expr in cut_expressions.keys())
            # else:
            #     self.model.addConstr(cut_expressions[0][0] >= cut_expressions[0][
            #         1])  # adding a single constraint with addConstrs yields an error

            # Adding cuts in one by one with addConstr

            all_vars = self.model.getVars()
            price_vars = all_vars[1:J]
            # price_values = self.model.getAttr("X", price_vars)
            obj_nr_vars = all_vars[-(N * R):]
            obj_nr_values = np.array(self.model.getAttr("X", obj_nr_vars))
            profitMinusEps = np.array(profit - epsilon_viol)

            cut_violated = obj_nr_values <= profitMinusEps

            # iterate through n, r to check if the cut we would be adding is currently violated or not
            for n in range(N):
                for r in range(R):
                    # add the cut for (n, r) only if it is violated in the current MP solution
                    if cut_violated[n * R + r]:
                        # print(f"violation found in tupels: [({n}, {r})]")
                        violated_count += 1
                        violations.append(profit[n * R + r] - obj_nr_values[n * R + r])
                        self.model.addConstr(obj_nr_vars[n * R + r]
                                             - gp.quicksum(varphi[n * R + r, i] * price_vars[i] for i in range(J - 1))
                                             >= profit[n * R + r]
                                             - np.sum(varphi[n * R + r, i] * p[i] for i in range(J - 1))
                                             )
            # if len(violations) >= 1:
            #     print("average violation = ", np.sum(violations) / len(violations))
            # print("number of violated cuts = ", violated_count)
        else:
            self.model.addConstr(self.vars["obj_nr"] - varphi @ self.vars["p"] >= profit - varphi @ p)
            violated_count = self.inputs.N * self.inputs.R

        self.cut_counter += violated_count

    def solve(self):
        J = self.inputs.J
        if self.outsource and ((self.inputs.iteration <= 1) or (self.pres != 2)):
            self.model.update()
            dates = str(datetime.datetime.now())
            self.model.write(f"MP_{dates}.lp")
            outsource = gp.Env(empty=True)
            outsource.setParam("OutputFlag", 0)
            outsource.start()
            outsource_model = gp.read(f"MP_{dates}.lp", env=outsource)
            outsource_model.optimize()
            objval = outsource_model.ObjVal
            p_vars_0 = outsource_model.getVarByName("p[0]")
            if not self.onep:
                p = np.array([outsource_model.getVarByName(f"p[{i}]").x for i in range(J - 1)])
            else:
                p = np.array([p_vars_0.x])
            obj_master = objval
            outsource_model.dispose()
        else:
            self.model.optimize()
            if self.model.status == 2 or self.model.status == 9:
                if self.onep:
                    p = np.array([self.vars["p"][0].x])
                else:
                    p = np.array([self.vars["p"][i].x for i in range(J - 1)])
                obj_master = self.vars["obj"].x
            else:
                # print(f"Master infeasible, status {self.model.status}")
                return None, False
                print(self.inputs.p_L)
                print(self.inputs.p_U)
                exit()
                # if console:
                #     print(f"MP infeasible, try outsourcing")
                # else:
                #     logging.info(f"MP infeasible, try outsourcing")

                self.model.update()
                dates = str(datetime.datetime.now())
                self.model.write(f"MP_{dates}.lp")
                outsource = gp.Env(empty=True)
                outsource.setParam("OutputFlag", 0)
                outsource.start()
                outsource_model = gp.read(f"MP_{dates}.lp", env=outsource)
                outsource_model.optimize()
                objval = outsource_model.ObjVal
                p_vars_0 = outsource_model.getVarByName("p[0]")
                if not self.onep:
                    p = np.array([outsource_model.getVarByName(f"p[{i}]").x for i in range(J - 1)])
                else:
                    p = np.array([p_vars_0.x])
                obj_master = objval
                outsource_model.dispose()

        if obj_master > self.lowerbound:
            self.lowerbound = obj_master

        return p, True

    def add_upperbound(self, new_upperbound):
        if self.upperbounding:
            self.upBoundConstr.rhs = new_upperbound
        self.upperbound = new_upperbound


# Define a function to solve the Gurobi model for a specific n and r
def solve_gurobi_model(J, R, p_L, p_U, threads, eta, p, n, r, exo_utility, endo_coef, fixed,
                       omega_dict, eta_dict, obb, vvarphi):
    fix = (sum(fixed[i, n, r] for i in range(3)) == 1)
    if fix:
        # create redundant cut for tuples that can be ignored
        if eta:
            for i in range(J):
                omega_dict[i, n * R + r] = fixed[i, n, r]
            for i in range(1, J):
                eta_dict[i, n * R + r] = omega_dict[i, n * R + r] * p[i - 1]

        o = - (1 / R) * sum(fixed[i, n, r] * p[i - 1] for i in range(1, J))
        v = [fixed[i, n, r] * (- 1 / R) for i in range(1, J)]
    else:
        # initialize model
        cpp = gp.Model("Continuous pricing problem")
        inf = GRB.INFINITY

        # initialize variables
        eta_vars = {i: cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{i}")
                    for i in range(1, J)}
        omega_vars = {i: cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}")
                      for i in range(J)}
        p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in range(1, J)}

        # for i in range(J):
        #     if fixed[i, n, r] == 0:
        #         omega_vars[i].ub = 0

        # one choice constraint
        cpp.addConstr(gp.quicksum(omega_vars[i] for i in range(J)) == 1)

        alpha = dict()

        alpha[0] = cpp.addConstr(omega_vars[0] * exo_utility[0, n * R + r]
                                 + gp.quicksum(omega_vars[j] * exo_utility[j, n * R + r]
                                               + eta_vars[j] * endo_coef[j, n * R + r] for j in range(1, J))
                                 >= exo_utility[0, n * R + r], name=f"alpha_0")

        for i in range(1, J):
            alpha[i] = cpp.addConstr(omega_vars[0] * exo_utility[0, n * R + r]
                                     + gp.quicksum(omega_vars[j] * exo_utility[j, n * R + r]
                                                   + eta_vars[j] * endo_coef[j, n * R + r] for j in range(1, J))
                                     - endo_coef[i, n * R + r] * p_vars[i]
                                     >= exo_utility[i, n * R + r], name=f"alpha_{i}")

        # mcCormick
        for i in range(1, J):
            cpp.addConstr(eta_vars[i] - p_L[i] * omega_vars[i] >= 0)
            cpp.addConstr(eta_vars[i] - p_U[i] * omega_vars[i] - p_vars[i] >= - p_U[i])
            cpp.addConstr(eta_vars[i] - p_L[i] * omega_vars[i] - p_vars[i] <= - p_L[i])
            cpp.addConstr(eta_vars[i] - p_U[i] * omega_vars[i] <= 0)

        varphi = dict()
        for i in range(1, J):
            varphi[i] = cpp.addConstr(p_vars[i] == p[i - 1])

        # Objective function
        cpp.setObjective(- (1 / R) * gp.quicksum(eta_vars[i] for i in range(1, J)), GRB.MINIMIZE)

        cpp.setParam('OutputFlag', 0)
        cpp.setParam("Threads", threads)

        cpp.setParam('ScaleFlag', 1)
        cpp.setParam('PrePasses', 2)
        cpp.setParam('Presolve', 1)
        cpp.setParam('NormAdjust', 1)
        cpp.setParam('SimplexPricing', 2)
        cpp.setParam('AggFill', 1000)
        cpp.setParam('PreDepRow', 1)
        cpp.setParam('PreDual', 0)
        cpp.setParam('NumericFocus', 1)

        cpp.setParam('FeasibilityTol', 1e-8)

        # cpp.Params.LogFile = f'gurobi_log_{n}_{r}.txt'

        cpp.optimize()
        o = cpp.ObjVal
        v = [varphi[i].Pi for i in range(1, J)]

        if eta:
            for i in range(J):
                omega_dict[i, n * R + r] = omega_vars[i].x
            for i in range(1, J):
                eta_dict[i, n * R + r] = eta_vars[i].x
        cpp.dispose()

    obb[n * R + r] = o
    for i in range(J - 1):
        vvarphi[n * R + r, i] = v[i]
    return [o] + [v[i] for i in range(J - 1)]


def cpp_nonlinear_McCM_sep_update(N, R, J, mcm_model, exo_utility, endo_coef, p, eta, fixed, onep,
                                  pl_VIs, validcuts, do_vis, addeta, optvalid, start_prices, start_choices, start_etas,
                                  first_dual_bb):
    if eta:
        omega_dict = np.zeros((J, N * R))
        eta_dict = np.zeros((J, N * R))
    else:
        omega_dict = None
        eta_dict = None
    obb = np.zeros(shape=(N * R))
    if onep:
        vvarphi = np.zeros(shape=(N * R, 1))
    else:
        vvarphi = np.zeros(shape=(N * R, J - 1))

    mcm = mcm_model.cpp
    alpha = mcm_model.alpha
    omega_vars = mcm_model.omega_vars
    eta_vars = mcm_model.eta_vars
    p_vars = mcm_model.p_vars
    if optvalid == 2:
        opt_z_vars = mcm_model.opt_z_vars
    varphi = mcm_model.varphi
    bp_constr = mcm_model.bp_constr
    optvalid_constr = mcm_model.optvalid_constr

    p_L = copy.copy(mcm_model.p_L)
    p_U = copy.copy(mcm_model.p_U)

    if onep:
        p = p[0]
        varphi[2].rhs = p
    else:
        for i in range(1, J):
            varphi[i].rhs = p[i - 1]

    # mcm.update()

    if not onep:
        if any(p[i - 1] < p_L[i] or p[i - 1] > p_U[i] for i in range(1, J)):
            for i in range(1, J):
                if p[i - 1] < p_L[i]:
                    p[i - 1] = p_L[i]
                elif p[i - 1] > p_U[i]:
                    p[i - 1] = p_U[i]
                varphi[i].rhs = p[i - 1]
    else:
        if p < p_L[2] or p > p_U[2]:
            if p < p_L[2]:
                p = p_L[2]
            elif p > p_U[2]:
                p = p_U[2]
            varphi[2].rhs = p

    # lets try to isolate the loops for onep and multiple prices
    if onep:
        p2 = p
        if fixed is not None:
            for n in range(N):
                for r in range(R):
                    fix = (not fixed[n, r] == 3)
                    if fix:
                        o = - (1 / R) * int(fixed[n, r] == 2) * p2
                        v2 = int(fixed[n, r] == 2) * (- 1 / R)
                    else:
                        mcm.chgCoeff(alpha[0], omega_vars[2],
                                     exo_utility[1, n * R + r] - exo_utility[0, n * R + r])
                        mcm.chgCoeff(alpha[0], eta_vars[2], endo_coef[n * R + r])

                        mcm.chgCoeff(alpha[1], omega_vars[2],
                                     exo_utility[1, n * R + r] - exo_utility[0, n * R + r])
                        mcm.chgCoeff(alpha[1], eta_vars[2], endo_coef[n * R + r])
                        mcm.chgCoeff(alpha[1], p_vars[2], -endo_coef[n * R + r])
                        alpha[1].rhs = exo_utility[1, n * R + r] - exo_utility[0, n * R + r]

                        mcm.optimize()
                        o = mcm.ObjVal
                        v2 = varphi[2].Pi

                    obb[n * R + r] = o
                    vvarphi[n * R + r, 0] = v2
        else:
            for n in range(N):
                for r in range(R):
                    mcm.chgCoeff(alpha[0], omega_vars[2],
                                 exo_utility[1, n * R + r] - exo_utility[0, n * R + r])
                    mcm.chgCoeff(alpha[0], eta_vars[2], endo_coef[n * R + r])

                    mcm.chgCoeff(alpha[1], omega_vars[2],
                                 exo_utility[1, n * R + r] - exo_utility[0, n * R + r])
                    mcm.chgCoeff(alpha[1], eta_vars[2], endo_coef[n * R + r])
                    mcm.chgCoeff(alpha[1], p_vars[2], -endo_coef[n * R + r])
                    alpha[1].rhs = exo_utility[1, n * R + r] - exo_utility[0, n * R + r]

                    mcm.optimize()
                    o = mcm.ObjVal
                    v2 = varphi[2].Pi

                    obb[n * R + r] = o
                    vvarphi[n * R + r, 0] = v2
    else:
        if fixed is not None:
            # # Create a list of arguments to pass to each process
            # threads = 1
            #
            # with SuppressOutput():
            #     # Parallelize the partial function with joblib
            #     results = Parallel(n_jobs=-1)(
            #         delayed(solve_gurobi_model)(J, R, p_L, p_U, threads, eta, p, n, r, exo_utility, endo_coef, fixed,
            #                omega_dict, eta_dict, obb, vvarphi) for n in range(N) for r in range(R))
            #
            # obb = [sublist[0] for sublist in results]
            # for i in range(J - 1):
            #     vvarphi[:, i] = [sublist[i+1] for sublist in results]

            if first_dual_bb:
                smallest_breakbounds_l = {i: 100 for i in range(1, J)}
                highest_breakbounds_u = {i: -100 for i in range(1, J)}

            if not pl_VIs:
                smallest_breakbounds_l = {i: -100 for i in range(1, J)}  # will thus never be > p_l

            for n in range(N):
                for r in range(R):
                    fix = (sum(fixed[i, n, r] for i in range(3)) == 1)
                    if fix:
                        # print("fixed")
                        # create redundant cut for tuples that can be ignored
                        if eta:
                            for i in range(J):
                                omega_dict[i, n * R + r] = fixed[i, n, r]
                            for i in range(1, J):
                                eta_dict[i, n * R + r] = omega_dict[i, n * R + r] * p[i - 1]

                        o = - (1 / R) * sum(fixed[i, n, r] * p[i - 1] for i in range(1, J))
                        v = [fixed[i, n, r] * (- 1 / R) for i in range(1, J)]
                    else:
                        # print(f"({n}, {r}) not fixed?")
                        # for u in range(1, J):
                        #     print(fixed[u, n, r])
                        # print(p)
                        # if p[0] == 0.6418557465371142 and p[1] == 0.6170906789057812 and p[2] == 0.7910535745571039 and p[3] == 0.7892573935667294:
                        # # if p == [0.6418557465371142, 0.6170906789057812, 0.7910535745571039, 0.7892573935667294]:
                        #     print(f"{n}, {r} is not fixed")

                        for i in range(J):
                            if fixed[i, n, r] == 0:
                                omega_vars[i].ub = 0
                            # else:
                            #     omega_vars[i].ub = 1

                        for j in range(J):
                            mcm.chgCoeff(alpha[0], omega_vars[j], exo_utility[j, n * R + r])
                        for j in range(1, J):
                            mcm.chgCoeff(alpha[0], eta_vars[j], endo_coef[j, n * R + r])
                        alpha[0].rhs = exo_utility[0, n * R + r]

                        for i in range(1, J):
                            for j in range(J):
                                mcm.chgCoeff(alpha[i], omega_vars[j], exo_utility[j, n * R + r])
                            for j in range(1, J):
                                mcm.chgCoeff(alpha[i], eta_vars[j], endo_coef[j, n * R + r])
                            mcm.chgCoeff(alpha[i], p_vars[i], -endo_coef[i, n * R + r])
                            alpha[i].rhs = exo_utility[i, n * R + r]

                        # here we update the new things. Objcut/optcut are fine. Its just about...
                        # bp_constr (if do_vis)
                        # ==> with this comes the breakbound computation (if firtstdual)
                        # opt_valid_constr (if opt_valid)

                        # so do_vis, opt_valid, thats it?
                        # the do_vis include addeta of course

                        for opt_choice in range(1, J):
                            low_utils = [exo_utility[0, n * R + r]]
                            if pl_VIs:
                                high_utils = [exo_utility[0, n * R + r]]
                            for i in range(1, J):
                                low_utils.append(exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_U[i])
                                if pl_VIs:
                                    high_utils.append(exo_utility[i, n * R + r] + endo_coef[i, n * R + r] * p_L[i])
                            low_util_max = max(low_utils)
                            if pl_VIs:
                                high_util_max = max(high_utils)
                            bp_high = (low_util_max - exo_utility[opt_choice, n * R + r]) / endo_coef[
                                opt_choice, n * R + r]
                            if pl_VIs:
                                bp_low = (high_util_max - exo_utility[opt_choice, n * R + r]) / endo_coef[
                                    opt_choice, n * R + r]
                            if do_vis:
                                bp_constr[opt_choice, 0].rhs = p_U[opt_choice]
                                mcm.chgCoeff(bp_constr[opt_choice, 0],
                                             omega_vars[opt_choice],
                                             (p_U[opt_choice] - bp_high))
                                if pl_VIs:
                                    bp_constr[opt_choice, 1].rhs = bp_low
                                    mcm.chgCoeff(bp_constr[opt_choice, 1],
                                                 omega_vars[opt_choice],
                                                 (bp_low - p_L[opt_choice]))
                                if addeta >= 1:
                                    bp_constr[opt_choice, 2].rhs = p_U[opt_choice]
                                    if addeta == 1:
                                        mcm.chgCoeff(bp_constr[opt_choice, 2],
                                                     eta_vars[opt_choice],
                                                     (p_U[opt_choice] - bp_high) / bp_high)
                                    elif addeta == 4:
                                        mcm.chgCoeff(bp_constr[opt_choice, 2],
                                                     omega_vars[opt_choice],
                                                     (p_U[opt_choice] - bp_high))
                                    if pl_VIs and addeta == 2:
                                        bp_constr[opt_choice, 3].rhs = bp_low
                                        mcm.chgCoeff(bp_constr[opt_choice, 3],
                                                     eta_vars[opt_choice],
                                                     (bp_low - p_L[opt_choice]) / p_U[opt_choice])
                                        # (bp_low - p_L[opt_choice]) / bp_low)

                            if first_dual_bb:
                                if bp_high > highest_breakbounds_u[opt_choice]:
                                    highest_breakbounds_u[opt_choice] = bp_high
                                if pl_VIs:
                                    if bp_low < smallest_breakbounds_l[opt_choice]:
                                        smallest_breakbounds_l[opt_choice] = bp_low

                            if validcuts and optvalid == 2:
                                # print(f"optvalid_constr = {optvalid_constr.keys()}")
                                # both the MAIN and all SIDE constraints have to be updated
                                # J times!
                                # they are all the same direction so it should be fine
                                if start_choices[opt_choice, n * R + r] == 1:
                                    # make the MAIN constraint valid for this i
                                    mcm.chgCoeff(optvalid_constr[opt_choice],
                                                 p_vars[opt_choice],
                                                 1)
                                    mcm.chgCoeff(optvalid_constr[opt_choice],
                                                 opt_z_vars[opt_choice],
                                                 (start_prices[opt_choice - 1] - p_L[opt_choice]))
                                    optvalid_constr[opt_choice].rhs = start_prices[opt_choice - 1]
                                    # update the 0, J constraint for opt_choice
                                    mcm.chgCoeff(optvalid_constr[0],
                                                 omega_vars[opt_choice],
                                                 1)
                                    mcm.chgCoeff(optvalid_constr[J],
                                                 eta_vars[opt_choice],
                                                 1)
                                    # optvars is multiplied by start eta for all i in 1,J in constr J
                                    # in constr 0 its always -1 so no need to update
                                    mcm.chgCoeff(optvalid_constr[J],
                                                 opt_z_vars[opt_choice],
                                                 -p_L[opt_choice])
                                    optvalid_constr[J].rhs = -p_L[opt_choice] * (J - 2)
                                    for j in range(1, J):
                                        if j != opt_choice:
                                            # make the MAIN constraint invalid for this j
                                            mcm.chgCoeff(optvalid_constr[opt_choice],
                                                         p_vars[j],
                                                         0)
                                            mcm.chgCoeff(optvalid_constr[opt_choice],
                                                         opt_z_vars[j],
                                                         0)
                                            # make the j-th SIDE constraint invalid for opt choice
                                            mcm.chgCoeff(optvalid_constr[j],
                                                         p_vars[opt_choice],
                                                         0)
                                            mcm.chgCoeff(optvalid_constr[j],
                                                         opt_z_vars[opt_choice],
                                                         0)
                                            # make the j-th SIDE constraint valid for this j
                                            mcm.chgCoeff(optvalid_constr[j],
                                                         p_vars[j],
                                                         -1)
                                            mcm.chgCoeff(optvalid_constr[j],
                                                         opt_z_vars[j],
                                                         (p_U[j] - start_prices[j - 1]))
                                            optvalid_constr[j].rhs = - start_prices[j - 1]
                                            # update the 0, J constraint for this j
                                            mcm.chgCoeff(optvalid_constr[0],
                                                         omega_vars[j],
                                                         0)
                                            mcm.chgCoeff(optvalid_constr[J],
                                                         eta_vars[j],
                                                         0)
                                            mcm.chgCoeff(optvalid_constr[J],
                                                         opt_z_vars[j],
                                                         -p_L[opt_choice])
                                else:
                                    for j in range(1, J):
                                        optvalid_constr[j].rhs = -100
                        mcm.optimize()
                        try:
                            o = mcm.ObjVal
                            v = [varphi[i].Pi for i in range(1, J)]
                        except AttributeError:
                            resolvo = False
                            for i in range(1, J):
                                if p[i - 1] > p_U[i]:
                                    resolvo = True
                                    # print(f"p[{i}] = {p[i-1]} > p_U[{i}] = {p_U[i]}]")
                                    p[i - 1] = p_U[i]
                                    # print("correct it and resolve: ")
                                    try:
                                        o = mcm.ObjVal
                                        v = [varphi[i].Pi for i in range(1, J)]
                                        print("resolving bounds helped")
                                    except AttributeError:
                                        print("Damn, still not")
                                        print(f"Subproblem ({n}, {r}) infeasible, status {mcm.status}")
                                        print("current bounds")
                                        print(f"p_l = {p_L}")
                                        print(f"p_u = {p_U}")
                                        print(f"current fixed = {p}")
                                        exit()
                                elif p[i - 1] < p_L[i]:
                                    resolvo = True
                                    # print(f"p[{i}] = {p[i-1]} < p_L[{i}] = {p_L[i]}]")
                                    p[i - 1] = p_L[i]
                                    # print("correct it and resolve: ")
                                    try:
                                        o = mcm.ObjVal
                                        v = [varphi[i].Pi for i in range(1, J)]
                                        print("resolving bounds helped")
                                    except AttributeError:
                                        print("Damn, still not")
                                        exit()
                            if not resolvo:
                                # print("yeah not sure what the issue is. I wrote the thing as a 0000debug.lp")
                                mcm.write("0000debug.lp")
                                outsource = gp.Env(empty=True)
                                outsource.setParam("OutputFlag", 0)
                                # outsource.setParam("DualReductions", 0)
                                outsource.start()
                                outsource_model = gp.read(f"0000debug.lp", env=outsource)
                                outsource_model.optimize()

                                if outsource_model.status == 2:
                                    all_constr = outsource_model.getConstrs()
                                    p_constr = all_constr[- (J - 1):]
                                    o = outsource_model.ObjVal
                                    v = [pc.Pi for pc in p_constr]
                                    outsource_model.dispose()
                                    # print("outsourcing helped")
                                else:
                                    test = outsource_model
                                    print(f"tupel ({n}, {r}) cant solve")
                                    print("Status = ", test.status)
                                    print("run computeIIS")
                                    test.computeIIS()
                                    test.write("0001debug.lp")

                                    for constr in test.getConstrs():
                                        if constr.IISConstr == 1:
                                            print("constr = ", constr)

                                    for i in range(J):
                                        if start_choices[i, n*R+r] == 1:
                                            print(f"startchoice was {i}")
                                    exit()
                        # neutralize the effect of ome
                        for i in range(J):
                            if fixed[i, n, r] == 0:
                                omega_vars[i].ub = 1

                        if eta:
                            for i in range(J):
                                omega_dict[i, n * R + r] = omega_vars[i].x
                            for i in range(1, J):
                                eta_dict[i, n * R + r] = eta_vars[i].x

                    obb[n * R + r] = o
                    for i in range(J - 1):
                        vvarphi[n * R + r, i] = v[i]
        else:
            for n in range(N):
                for r in range(R):
                    mcm.chgCoeff(alpha[0], omega_vars[0], exo_utility[0, n * R + r])
                    mcm.chgCoeff(alpha[0], omega_vars[1], exo_utility[1, n * R + r])
                    mcm.chgCoeff(alpha[0], omega_vars[2], exo_utility[2, n * R + r])
                    mcm.chgCoeff(alpha[0], eta_vars[1], endo_coef[1, n * R + r])
                    mcm.chgCoeff(alpha[0], eta_vars[2], endo_coef[2, n * R + r])
                    alpha[0].rhs = exo_utility[0, n * R + r]

                    for i in range(1, J):
                        for j in range(J):
                            mcm.chgCoeff(alpha[i], omega_vars[j], exo_utility[j, n * R + r])
                        for j in range(1, J):
                            mcm.chgCoeff(alpha[i], eta_vars[j], endo_coef[j, n * R + r])
                        mcm.chgCoeff(alpha[i], p_vars[i], -endo_coef[i, n * R + r])
                        alpha[i].rhs = exo_utility[i, n * R + r]

                    mcm.optimize()

                    o = mcm.ObjVal
                    v = [varphi[i].Pi for i in range(1, J)]

                    if eta:
                        for i in range(J):
                            omega_dict[i, n * R + r] = omega_vars[i].x
                        for i in range(1, J):
                            eta_dict[i, n * R + r] = eta_vars[i].x

                    obb[n * R + r] = o
                    for i in range(J - 1):
                        vvarphi[n * R + r, i] = v[i]

    if not first_dual_bb:
        return obb, vvarphi, omega_dict, eta_dict
    else:
        return obb, vvarphi, omega_dict, eta_dict, smallest_breakbounds_l, highest_breakbounds_u


def initialize_MCM_sep_one_nr(R, exo_utility, endo_coef, p_L, p_U, p, threads, onep, J, subcut, best_upper_bound,
                              validcuts, mcm2, pl_VIs, do_vis, addeta, optvalid, start_prices):
    if onep:
        # initialize model
        cpp = gp.Model("Continuous pricing problem")
        inf = GRB.INFINITY

        # initialize variables
        eta_vars = {i: cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{i}")
                    for i in [2]}
        omega_vars = {i: cpp.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"omega_{i}")
                      for i in [2]}
        p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in [2]}

        alpha = dict()
        alpha[0] = cpp.addConstr(omega_vars[2] * (exo_utility[1, 0] - exo_utility[0, 0])
                                 + eta_vars[2] * endo_coef[0]
                                 >= 0)
        alpha[1] = cpp.addConstr(omega_vars[2] * (exo_utility[1, 0] - exo_utility[0, 0])
                                 + eta_vars[2] * endo_coef[0]
                                 - endo_coef[0] * p_vars[2]
                                 >= exo_utility[1, 0] - exo_utility[0, 0])
        alpha[2] = None

        # mcCormick
        cpp.addConstr(eta_vars[2] - p_L[2] * omega_vars[2] >= 0)
        cpp.addConstr(eta_vars[2] - p_U[2] * omega_vars[2] - p_vars[2] >= - p_U[2])
        cpp.addConstr(eta_vars[2] - p_L[2] * omega_vars[2] - p_vars[2] <= - p_L[2])
        cpp.addConstr(eta_vars[2] - p_U[2] * omega_vars[2] <= 0)

        varphi = dict()
        varphi[1] = None
        varphi[2] = cpp.addConstr(p_vars[2] == p[0])

        # Objective function
        cpp.setObjective(- (1 / R) * eta_vars[2], GRB.MINIMIZE)
    else:
        # initialize model
        cpp = gp.Model("Continuous pricing problem")
        # skeleton = gp.Env(empty=True)
        #  skeleton.setParam("OutputFlag", 0)
        # skeleton.start()
        # cpp = gp.read(f"000_SP_skeleton.mps", env=skeleton)

        inf = GRB.INFINITY

        # initialize variables
        eta_vars = {i: cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"eta_{i}")
                    for i in range(1, J)}
        omega_vars = {i: cpp.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name=f"omega_{i}")
                      for i in range(J)}
        p_vars = {i: cpp.addVar(lb=p_L[i], ub=p_U[i], vtype=GRB.CONTINUOUS, name=f"p_{i}")
                  for i in range(1, J)}
        if validcuts and optvalid == 2:
            opt_z_vars = {j: cpp.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"opt_z_{j}")
                          for j in range(1, J)}
        else:
            opt_z_vars = None

        # one choice constraint
        cpp.addConstr(gp.quicksum(omega_vars[i] for i in range(J)) == 1)

        alpha = dict()

        alpha[0] = cpp.addConstr(omega_vars[0] * exo_utility[0, 0]
                                 + gp.quicksum(omega_vars[j] * exo_utility[j, 0]
                                               + eta_vars[j] * endo_coef[j, 0] for j in range(1, J))
                                 >= exo_utility[0, 0], name=f"alpha_0")

        for i in range(1, J):
            alpha[i] = cpp.addConstr(omega_vars[0] * exo_utility[0, 0]
                                     + gp.quicksum(omega_vars[j] * exo_utility[j, 0]
                                                   + eta_vars[j] * endo_coef[j, 0] for j in range(1, J))
                                     - endo_coef[i, 0] * p_vars[i]
                                     >= exo_utility[i, 0], name=f"alpha_{i}")

        # mcCormick
        if not mcm2:
            for i in range(1, J):
                cpp.addConstr(eta_vars[i] - p_L[i] * omega_vars[i] >= 0)
                cpp.addConstr(eta_vars[i] - p_U[i] * omega_vars[i] - p_vars[i] >= - p_U[i])
                cpp.addConstr(eta_vars[i] - p_L[i] * omega_vars[i] - p_vars[i] <= - p_L[i])
                cpp.addConstr(eta_vars[i] - p_U[i] * omega_vars[i] <= 0)
        else:
            for i in range(1, J):
                cpp.addConstr(eta_vars[i] - p_U[i] * omega_vars[i] - p_vars[i] >= - p_U[i])
                cpp.addConstr(eta_vars[i] - p_U[i] * omega_vars[i] <= 0)
                cpp.addConstr(eta_vars[i] - p_vars[i] <= 0)

        bp_constr = dict()
        optvalid_constr = dict()

        if do_vis:  # need to give the right values here wesh... this has to do with the
            # breakpoint constraint computations... maybe it should happen outside?
            # maybe. but I mean it could also just happen here, since were already
            # looping over N, R anyway
            for i in range(1, J):
                bp_constr[i, 0] = cpp.addConstr(p_vars[i] + 0 * omega_vars[i] <= p_U[i])
                if addeta >= 1:
                    if addeta == 1:
                        bp_constr[i, 2] = cpp.addConstr(p_vars[i] + 0 * eta_vars[i] <= p_U[i])
                    elif addeta == 4:
                        bp_constr[i, 2] = cpp.addConstr(eta_vars[i] + 0 * omega_vars[i] <= p_U[i])
                if pl_VIs:
                    bp_constr[i, 1] = cpp.addConstr(p_vars[i] + 0 * omega_vars[i] >= p_L[i])  # placeholder rhs
                    if addeta >= 2:
                        bp_constr[i, 3] = cpp.addConstr(p_vars[i] + 0 * eta_vars[i] >= p_L[i])  # placeholder rhs

        # if validcuts and optvalid == 1:
        #     i = 0  # will be updated to be the opt choice for each n,r
        #     optvalid_constr[i] = cpp.addConstr(p_vars[i] - gp.quicksum(p_vars[j] for j in
        #                                                                [j for j in range(1, J) if j != i])
        #                                        + (start_prices[i - 1] - p_L[i] + sum(
        #         p_U[j] - start_prices[j - 1] for j in [j for j in range(1, J) if j != i]))
        #                                        * omega_vars[i]
        #                                        >= start_prices[i - 1]
        #                                        - sum(start_prices[j - 1] for j in
        #                                              [j for j in range(1, J) if j != i]))
        if validcuts and optvalid == 2:
            i = 1  # will be updated to be the opt choice for each n,r
            optvalid_constr[i] = \
                cpp.addConstr(p_vars[i] + (start_prices[i - 1] - p_L[i]) * opt_z_vars[i]
                              >= -100, name=f"diese_{i}?")  # RHS to be updated
            for j in range(1, J):
                if j != i:
                    optvalid_constr[j] = \
                        cpp.addConstr(- p_vars[j] +
                                      (p_U[j] - start_prices[j - 1]) * opt_z_vars[
                                          j]
                                      >= -100, name=f"diese_{j}?")  # RHS to be updated

            optvalid_constr[0] = cpp.addConstr(omega_vars[i] - gp.quicksum(opt_z_vars[j]
                                                                           for j in range(1, J)) >= - (J - 2))
            optvalid_constr[J] = cpp.addConstr(eta_vars[i] - 0 * gp.quicksum(opt_z_vars[j]
                                                                             for j in range(1, J)) >= - 0 * (J - 2))

        varphi = dict()
        for i in range(1, J):
            varphi[i] = cpp.addConstr(p_vars[i] == p[i - 1])

        # Objective function
        cpp.setObjective(- (1 / R) * gp.quicksum(eta_vars[i] for i in range(1, J)), GRB.MINIMIZE)

        # if subcut:
        # cpp.addConstr(- (1 / R) * gp.quicksum(eta_vars[i] for i in range(1, J)) <= best_upper_bound)

    # solve model
    # cpp.setParam('OutputFlag', 0)
    # cpp.setParam("Threads", 1)
    # cpp.update()
    # cpp.write("00_Sub.mps")
    # #print("Wrote 00_Sub.mps")

    cpp.setParam('OutputFlag', 0)
    cpp.setParam("Threads", threads)

    cpp.setParam('ScaleFlag', 1)
    cpp.setParam('PrePasses', 2)
    cpp.setParam('Presolve', 1)
    cpp.setParam('NormAdjust', 1)
    cpp.setParam('SimplexPricing', 2)
    cpp.setParam('AggFill', 1000)
    cpp.setParam('PreDepRow', 1)
    cpp.setParam('PreDual', 0)
    cpp.setParam('NumericFocus', 1)

    cpp.setParam('FeasibilityTol', 1e-8)

    return cpp, alpha, omega_vars, eta_vars, p_vars, opt_z_vars, varphi, bp_constr, optvalid_constr


def get_dual_vars(N, R, mcm_model, inputs, p, eta, fixed, onep, pl_VIs, validcuts, do_vis, addeta, optvalid,
                  start_prices,
                  start_choices, start_etas, first_dual_bb=False):
    exo_utility = inputs.exo_utility
    endo_coef = inputs.endo_coef
    if not first_dual_bb:
        obj, varphi, omega_dict, eta_dict = cpp_nonlinear_McCM_sep_update(N, R, inputs.J, mcm_model, exo_utility,
                                                                          endo_coef,
                                                                          p, eta, fixed, onep,
                                                                          pl_VIs, validcuts, do_vis, addeta, optvalid,
                                                                          start_prices, start_choices, start_etas,
                                                                          first_dual_bb)
    else:
        obj, varphi, omega_dict, eta_dict, smallest_breakbounds_l, highest_breakbounds_u = cpp_nonlinear_McCM_sep_update(
            N, R, inputs.J, mcm_model, exo_utility,
            endo_coef,
            p, eta, fixed, onep,
            pl_VIs, validcuts, do_vis, addeta, optvalid,
            start_prices,
            start_choices, start_etas, first_dual_bb)
    if not first_dual_bb:
        return obj, varphi, omega_dict, eta_dict
    else:
        return obj, varphi, omega_dict, eta_dict, smallest_breakbounds_l, highest_breakbounds_u


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


def bnb_benders(N, R, exo_utility, endo_coef, p_l, p_u, eta, lwbound, pres, threads, fixed, onep, outsource,
                fixed_choices, J_PSP, J_PUP, filenamee, timetricks, viol, pl_VIs, subcut, mascut, best_upper_bound,
                validcuts, mcm2, breakbounds, do_vis, addeta, optvalid, start_prices, start_choices, start_etas,
                outsidebreakbounds, optBDstart):

    # # Print all input values
    # parameters = locals()
    # for param, value in parameters.items():
    #     print(f"{param} = {value}")

    # so, optcut and objcut area controlled.. I mean and so are the others, at this point. So just look what
    # each of them does and translate it to here
    global timelimit_tricks

    if mascut:
        mascut_constraint = True
        mascut_after_solve = True
    else:
        mascut_constraint = False
        mascut_after_solve = False

    # infosable = False
    # for i in range(1, J_PSP + J_PUP + 1):
    #     if p_l[i] > p_u[i]:
    #         infosable = True
    # if infosable:
    #     # print("Ashley ,look at me")
    #     # print(p_l)
    #     # print(p_u)
    #     # exit()
    #     return None, None, None, None, None, None, None, None, False

    p_L = copy.copy(p_l)
    p_U = copy.copy(p_u)

    if not console:
        logging.basicConfig(filename=filenamee, level=logging.INFO, format='%(message)s')

    timelimit_tricks = timetricks
    # timelimit_tricks = False
    termination = False

    J = J_PSP + J_PUP + 1

    lwbound = - N * R * max(p_u[i] for i in range(1, J))

    verbose = False

    if outsidebreakbounds and breakbounds:
        # do a loop here that only computes breakbounds but does not change any constraints
        smallest_breakbounds_l, highest_breakbounds_u = compute_breakbounds(N, R, J, p_L, p_U, exo_utility, endo_coef,
                                                                            pl_VIs)

        improvo = False
        for i in range(1, J):
            if smallest_breakbounds_l[i] > p_L[i]:
                print(f"changed p_l[{i}] from {p_L[i]} to {smallest_breakbounds_l[i]}")
                p_L[i] = smallest_breakbounds_l[i]
                improvo = True
            if highest_breakbounds_u[i] < p_U[i]:
                p_U[i] = highest_breakbounds_u[i]
                improvo = True
        if improvo:
            if any(p_L[i] > p_U[i] for i in range(1, J)):
                # print("Ashley,look at me (in the BD)")
                # print(p_L)
                # print(p_U)
                # exit()
                return None, None, None, None, None, None, None, None, False

    # define first guess for optimal price
    # could be replaced by a heuristic
    if not onep:
        p = np.array([(p_L[i] + p_U[i]) / 2 for i in range(1, J)])
    else:
        p = [(p_L[2] + p_U[2]) / 2]

    # print("original start_prices = ", p)

    if optBDstart:
        # for i in range(1, J):
        #     if p_L[i] <= start_prices[i - 1] <= p_U[i]:
        #         p[i - 1] = start_prices[i - 1]
        pass
        # print("as we have optBDstart we get")
        # print("new start_prices = ", p)

    # print("initialize price = ", p)

    inputs = Inputs(N, R, J, exo_utility, endo_coef, p_L, p_U, threads, J_PSP, J_PUP, pres, viol)
    # print("initiaized inputs")
    master = MasterProblem(inputs, pres, lwbound, fixed, onep, outsource, mascut_constraint, best_upper_bound, threads)
    # print(f"initiaize MP with UB = {master.upperbound}, LB = {master.lowerbound}")
    # logging.info("initiaized MP")

    # initialize subproblem

    if not outsidebreakbounds and breakbounds:
        # this means the parameter that returns bb in the first dual call should be true
        first_dual_bb = True
    else:
        first_dual_bb = False

    cpp, alpha, omega_vars, eta_vars, p_vars, opt_z_vars, varphi, bp_constr, optvalid_constr = \
        initialize_MCM_sep_one_nr(R, exo_utility, endo_coef, p_L, p_U, p, threads, onep, J, subcut, best_upper_bound,
                                  validcuts, mcm2, pl_VIs, do_vis, addeta, optvalid, start_prices)

    # print("initialized SP")
    # logging.info("initiaized SP")

    if mascut_after_solve:
        mascutbuffer = 0
    else:
        mascutbuffer = None

    reporto = False

    p0, feaso = master.solve()

    if not feaso:
        return None, None, None, None, None, None, None, None, False

    if mascut_after_solve:
        # print(0)
        # print("master.model.ObjVal", master.model.ObjVal)
        # print("best_upper_bound", best_upper_bound)
        if master.model.ObjVal > best_upper_bound + mascutbuffer:
            return None, None, None, None, None, None, None, None, False

    # print("solving 0st MP gives p_new = ", p0)
    # print("and objective value = ", master.model.objVal)
    # print("and lets take a look at the solution values of the P_nr variables:")
    # all_vars = master.model.getVars()
    # obj_nr_vars = all_vars[-(N * R):]
    # obj_nr_values = np.array(master.model.getAttr("X", obj_nr_vars))
    # print(obj_nr_values)

    mcm_model = MCMModel(cpp, alpha, omega_vars, eta_vars, p_vars, opt_z_vars, varphi, p_L, p_U, onep, bp_constr,
                         optvalid_constr)

    if not first_dual_bb:
        profit, varphi, omega_dict, eta_dict = get_dual_vars(N, R, mcm_model, inputs, p, eta, fixed, onep,
                                                             pl_VIs, validcuts, do_vis, addeta, optvalid, start_prices,
                                                             start_choices, start_etas)
    else:
        profit, varphi, omega_dict, eta_dict, smallest_breakbounds_l, highest_breakbounds_u = get_dual_vars(N, R,
                                                                                                            mcm_model,
                                                                                                            inputs, p,
                                                                                                            eta, fixed,
                                                                                                            onep,
                                                                                                            pl_VIs,
                                                                                                            validcuts,
                                                                                                            do_vis,
                                                                                                            addeta,
                                                                                                            optvalid,
                                                                                                            start_prices,
                                                                                                            start_choices,
                                                                                                            start_etas,
                                                                                                            first_dual_bb)
        improvo = False
        for i in range(1, J):
            if smallest_breakbounds_l[i] > p_L[i]:
                print(f"changed p_l[{i}] from {p_L[i]} to {smallest_breakbounds_l[i]}")
                p_L[i] = smallest_breakbounds_l[i]
                improvo = True
            if highest_breakbounds_u[i] < p_U[i]:
                p_U[i] = highest_breakbounds_u[i]
                improvo = True
        if improvo:
            if any(p_L[i] > p_U[i] for i in range(1, J)):
                # print("Ashley,look at me (in the BD)")
                # print(p_L)
                # print(p_U)
                # exit()
                return None, None, None, None, None, None, None, None, False
        if improvo:
            # update bounds in the MP (maybe better if we port them as variables eeh)
            for i in range(1, J):
                master.model.getConstrByName(f"p{i}_low").rhs = p_L[i]
                master.model.getConstrByName(f"p{i}_high").rhs = p_U[i]

    # print("compute dual values and P_nr^c")
    # # print("These are the dual values: ")
    # # print(varphi)
    # print("These are the P_nr^c")
    # print(profit)

    # print("Solved first SP")
    # logging.info("Solved first SP")
    best_p = p
    best_profit = np.sum(profit)
    best_omega = omega_dict
    best_eta = eta_dict

    # print("Current best price = ", best_p)
    # print("Current best sum(P_nr^c) = ", best_profit)

    profsum = np.sum(profit)
    if profsum < master.upperbound:
        master.add_upperbound(profsum)

    prev_cut_number = master.cut_counter
    master.add_cut(p, profit, varphi)
    after_cut_number = master.cut_counter

    if not after_cut_number > prev_cut_number:
        termination = True

    # logging.info("Added first cut")
    # logging.info("Solving first MP")
    p, feaso = master.solve()

    if not feaso:
        return None, None, None, None, None, None, None, None, False

    if mascut_after_solve:
        # print(0.5)
        # print("master.model.ObjVal", master.model.ObjVal)
        # print("best_upper_bound", best_upper_bound)
        if master.model.ObjVal > best_upper_bound + mascutbuffer:
            return None, None, None, None, None, None, None, None, False
            # reporto = True
    for i in range(1, J):
        if p[i - 1] > p_U[i]:
            p[i - 1] = p_U[i]
        if p[i - 1] < p_L[i]:
            p[i - 1] = p_L[i]

    # print("solving first MP gives p_new = ", p)
    # print("and objective value = ", master.model.objVal)
    # # print("and lets take a look at the solution values of the P_nr variables:")
    # all_vars = master.model.getVars()
    # obj_nr_vars = all_vars[-(N * R):]
    # obj_nr_values = np.array(master.model.getAttr("X", obj_nr_vars))
    # print(obj_nr_values)

    # print("solved first MP")
    # logging.info("Solved first MP")

    inputs.iteration = 0
    total_lowerbound = master.lowerbound
    total_upperbound = master.upperbound

    if not total_upperbound == 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            relative_gap = abs((total_upperbound - total_lowerbound) / total_upperbound) * 100
            if verbose:
                if console:
                    print(f"UB = {total_upperbound}")
                    print(f"LB = {total_lowerbound}")
                    print(f"rel gap = {relative_gap}")
                    print("")
                    print(f"current price = {p}")
                else:
                    logging.info(f"UB = {total_upperbound}")
                    logging.info(f"LB = {total_lowerbound}")
                    logging.info(f"rel gap = {relative_gap}")
                    logging.info("")
                    logging.info(f"current price = {p}")
    else:
        relative_gap = 0

    requ_gap = 0.01
    iteration_count = 1
    stuck_counter = 0
    while relative_gap > requ_gap:
        # while not termination:
        profit, varphi, omega_dict, eta_dict = get_dual_vars(N, R, mcm_model, inputs, p, eta, fixed, onep,
                                                             pl_VIs, validcuts, do_vis, addeta, optvalid, start_prices,
                                                             start_choices, start_etas)
        iteration_count += 1
        if np.sum(profit) < best_profit:
            best_p = p
            # print("improved the bounds to")
            # print(p_L)
            # print(p_U)
            # print("best profit:")
            # for n in range(N):
            #     for r in range(R):
            #         print(f"profit[{n},{r}] = ", - R * profit[n * R + r])

            if eta:
                best_omega = omega_dict
                best_eta = eta_dict
            #     print("best p = ", best_p)
            #     for n in range(N):
            #         for r in range(R):
            #             for i in range(1, J):
            #                 if fixed[i, n, r] == 0:
            #                     print(f"profit[{i},{n},{r}] = ", eta_dict[i, n * R + r])
            #                     print(f"omega[{i},{n},{r}] = ", omega_dict[i, n * R + r], "(fixed=0)")
            #                 else:
            #                     print(f"profit[{i},{n},{r}] = ", eta_dict[i, n * R + r])
            #                     print(f"omega[{i},{n},{r}] = ", omega_dict[i, n * R + r])
            #
            # print("total = ", np.sum(profit))

        profsum = np.sum(profit)
        if profsum < master.upperbound:
            master.add_upperbound(profsum)
            # logging.info(f"Added new upperbound to MP")

        prev_cut_number = master.cut_counter
        master.add_cut(p, profit, varphi)
        after_cut_number = master.cut_counter

        if not after_cut_number > prev_cut_number:
            termination = True

        # logging.info(f"Added new cut MP")

        # if iteration == 3:
        #     master.model.setParam('OutputFlag', 0)
        #     master.model.setParam("Threads", 1)
        #     master.model.update()
        #     master.model.write("00_Master.mps")
        #     #print("Wrote 00_Master.mps")
        #     exit()
        # logging.info(f"start updating MP")
        # ste = time.time()
        # master.model.update()
        # # logging.info(f"done updating MP, this took {time.time()-ste}s")
        # # logging.info(f"start solving MP")
        # ste = time.time()
        p, feaso = master.solve()

        if not feaso:
            return None, None, None, None, None, None, None, None, False

        if mascut_after_solve:
            # print(iteration_count)
            # print("master.model.ObjVal", master.model.ObjVal)
            # print("best_upper_bound", best_upper_bound)
            if master.model.ObjVal > best_upper_bound + mascutbuffer:
                return None, None, None, None, None, None, None, None, False
                # reporto = True

        for i in range(1, J):
            if p[i - 1] > p_U[i]:
                p[i - 1] = p_U[i]
            if p[i - 1] < p_L[i]:
                p[i - 1] = p_L[i]

        # logging.info(f"done solving MP, this took {time.time()-ste}s")

        # print(f"solving MP in iteration {iteration_count} gives p_new = ", p)
        # print("and objective value = ", master.model.objVal)
        # print("and lets take a look at the solution values of the P_nr variables:")
        # all_vars = master.model.getVars()
        # obj_nr_vars = all_vars[-(N * R):]
        # obj_nr_values = np.array(master.model.getAttr("X", obj_nr_vars))
        # print(obj_nr_values)

        inputs.iteration += 1

        total_lowerbound = master.lowerbound
        total_upperbound = master.upperbound

        if not total_upperbound == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_rel_gap = abs((total_upperbound - total_lowerbound) / total_upperbound) * 100

                if timelimit_tricks:
                    if new_rel_gap < requ_gap:
                        master.model.setParam("TimeLimit", GRB.INFINITY)
                        master.model.setParam("OptimalityTol", 1e-2)
                        # ste = time.time()
                        p, feaso = master.solve()
                        if not feaso:
                            return None, None, None, None, None, None, None, None, False
                        # if console:
                        #     print(f"done resolving MP, this took {time.time() - ste}s")
                        # else:
                        #     logging.info(f"done resolving MP, this took {time.time()-ste}s")

                        total_lowerbound = master.lowerbound
                        total_upperbound = master.upperbound

                        new_rel_gap = abs((total_upperbound - total_lowerbound) / total_upperbound) * 100

                if relative_gap == new_rel_gap:
                    stuck_counter += 1
                if stuck_counter >= 5:
                    requ_gap = 1e-3
                relative_gap = new_rel_gap
                if verbose:
                    # print("UB = ", total_upperbound)
                    # print("LB = ", total_lowerbound)
                    # print("rel gap = ", relative_gap)
                    # print("current price = ", p)
                    if console:
                        print(f"UB = {total_upperbound}")
                        print(f"LB = {total_lowerbound}")
                        print(f"rel gap = {relative_gap}")
                        print("")
                        print(f"current price = {p}")
                    else:
                        logging.info(f"UB = {total_upperbound}")
                        logging.info(f"LB = {total_lowerbound}")
                        logging.info(f"rel gap = {relative_gap}")
                        logging.info("")
                        logging.info(f"current price = {p}")
        else:
            relative_gap = 0

    best_profit = master.upperbound
    total_lowerbound = master.lowerbound
    if not best_profit == 0:
        gap = round(((best_profit - total_lowerbound) / best_profit) * 100, 2)
    else:
        gap = round((best_profit - total_lowerbound) * 100, 2)

    prrrofit = best_profit
    best_p = list(best_p)
    # print("Total number of cuts added = ", master.cut_counter, "in", iteration_count, "iterations")
    # exit()

    # new_p_l = copy.copy(p_L)
    # new_p_u = copy.copy(p_U)


    return inputs.iteration, best_p, best_omega, best_eta, prrrofit, gap, p_L, p_U, True
