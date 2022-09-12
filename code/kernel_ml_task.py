from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, laplacian_kernel
from sklearn.utils import shuffle
import pandas as pd
import gurobipy as grb

def dist_rob_ksvm(param, data):
    """ kernelized distributionally robust SVM """
    kernel_train = data['K']
    y_train = data['y'].flatten()

    all_epsilon = list(param['epsilon'])
    all_epsilon.sort(reverse=True)
    all_kappa = list(param['kappa'])
    all_kappa.sort(reverse=True)
    if float('Inf') in all_kappa:
        all_kappa.remove(float('Inf'))
    if 0 in all_epsilon:
        all_epsilon.remove(0)

    row = kernel_train.shape[0]
    optimal = {}

    # Step 0: create model
    model = grb.Model('Ker_DRSVM')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_alpha = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_alpha[i] = model.addVar(
            vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    chg_cons = {}
    for i in range(row):
        model.addConstr(
            1 - y_train[i] * grb.quicksum(var_alpha[k] * kernel_train[k, i]
                                          for k in range(row)) <= var_s[i])
        chg_cons[i] = model.addConstr(
            1 + y_train[i] * grb.quicksum(var_alpha[k] * kernel_train[k, i]
                                          for k in range(row)) -
            all_kappa[0] * var_lambda <= var_s[i])
    model.addQConstr(
        grb.quicksum(var_alpha[k1] * kernel_train[k1, k2] * var_alpha[k2]
                     for k1 in range(row)
                     for k2 in range(row)) <= var_lambda * var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))
    for index, kappa in enumerate(all_kappa):
        # Change model for different kappa
        if index > 0:
            for i in range(row):
                model.chgCoeff(chg_cons[i], var_lambda, -kappa)
        for epsilon in all_epsilon:
            obj = var_lambda * epsilon + 1 / row * sum_var_s
            model.setObjective(obj, grb.GRB.MINIMIZE)

            # Step 5: solve the problem
            model.optimize()

            # Step 6: store results
            alpha_opt = np.array([var_alpha[i].x for i in range(row)])
            tmp = {
                (kappa, epsilon): {
                    'alpha': alpha_opt,
                    'objective': model.ObjVal,
                    'diagnosis': model.status
                }
            }
            optimal.update(tmp)

    return optimal

def regularized_ksvm(param, data):
    """ kernelized robust/regularized SVM """
    kernel_train = data['K']
    y_train = data['y'].flatten()

    all_epsilon = list(param['epsilon'])
    all_epsilon.sort(reverse=True)
    if 0 in all_epsilon:
        all_epsilon.remove(0)

    row = kernel_train.shape[0]
    optimal = {}

    # Step 0: create model
    model = grb.Model('Ker_DRSVM')
    model.setParam('OutputFlag', False)

    # Step 1: define decision variables
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
    var_s = {}
    var_alpha = {}
    for i in range(row):
        var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_alpha[i] = model.addVar(
            vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)

    # Step 2: integerate variables
    model.update()

    # Step 3: define constraints
    for i in range(row):
        model.addConstr(
            1 - y_train[i] * grb.quicksum(var_alpha[k] * kernel_train[k, i]
                                          for k in range(row)) <= var_s[i])
    model.addQConstr(
        grb.quicksum(var_alpha[k1] * kernel_train[k1, k2] * var_alpha[k2]
                     for k1 in range(row)
                     for k2 in range(row)) <= var_lambda * var_lambda)

    # Step 4: define objective value
    sum_var_s = grb.quicksum(var_s[i] for i in range(row))
    for epsilon in all_epsilon:
        obj = var_lambda * epsilon + 1 / row * sum_var_s
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        alpha_opt = np.array([var_alpha[i].x for i in range(row)])
        tmp = {
            (float('Inf'), epsilon): {
                'alpha': alpha_opt,
                'objective': model.ObjVal,
                'diagnosis': model.status
            }
        }
        optimal.update(tmp)

    return optimal

def ksvm(param, data):
    """ kernelized SVM """
    certif = np.linalg.eigvalsh(data['K'])[0]
    if certif < 0:
        data['K'] = data['K'] - 2 * certif * np.eye(data['K'].shape[0])
    optimal = {}
    if len(param['kappa']) > 1 or float('inf') not in param['kappa']:
        optimal.update(dist_rob_ksvm(param, data))
    if float('Inf') in param['kappa']:
        optimal.update(regularized_ksvm(param, data))

    return optimal

def test_performance(kernel_train, y_train, kernel_test, y_test, param):
    """ Re-train the model with all data and return the performance
    on the test dataset """
    training_data = {'K': kernel_train, 'y': y_train}
    optimal = ksvm(param, training_data)
    alpha_opt = optimal[(param['kappa'][0], param['epsilon'][0])]['alpha']
    y_pred = np.sign(kernel_test.dot(alpha_opt))
    return accuracy_score(y_test, y_pred)