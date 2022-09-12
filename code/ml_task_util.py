"""
same preprocessing from the codes without kernelization
https://github.com/sorooshafiee/Regularization-via-Transportation/blob/master/parallel_process.py
in the paper
https://arxiv.org/pdf/1710.10016.pdf

"""
from collections import defaultdict
from distutils.dist import Distribution
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, laplacian_kernel
from sklearn.utils import shuffle


import pandas as pd

from ml_task import *

def uci_classification_kernel(*args):
    """
    Use Gaussian Kernel as default
    """


def uci_classification(*args):
    # Setting parameters
    all_param = {
        'epsilon': [1e-4, 5e-4, 1e-3, 5e-2, 1e-2, 5e-2, 1e-1],
        'kappa': [0.1, 0.2, 0.3, 0.4, 0.5, 1, float('inf')],
        'd': [],
        'C': []
    }
    pnorms = [1, 2, float('Inf')]

    # Initialize output
    DRSVM_AUC = {}
    RSVM_AUC = {}

    # Load input data
    nargin = len(args)
    if nargin == 2:
        x_data = args[0]
        y_data = args[1]
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.25)
    elif nargin == 4:
        x_train = args[0]
        y_train = args[1]
        x_train, y_train = shuffle(x_train, y_train)
        x_test = args[2]
        y_test = args[3]

    # Fit classical svm model, hinge loss minimization
    stand_scaler = StandardScaler()
    x_train_nrm = stand_scaler.fit_transform(x_train)
    x_test_nrm = stand_scaler.transform(x_test)
    training_data = {'x': x_train_nrm, 'y': y_train}
    dro_model = Logit_Loss(x_train_nrm, y_train)
    
    print('Without Robustness:')
    dro_model.sklearn_in_built()
    
    print('AUC under standard scikit-learn is', dro_model.predict_auc(x_test_nrm, y_test))

    dro_model.standard_solver()
    print('AUC under CVXPY is', dro_model.predict_auc(x_test_nrm, y_test))
    
    dist_model = Distribution_Learner(x_train_nrm, y_train)
    X_new, y_new = dist_model.condition_wgan_X_y()
    dro_model.standard_solver(X = X_new, y = y_new)
    print('AUC under CVXPY with WGAN data is', dro_model.predict_auc(x_test_nrm, y_test), 0)
    for eps in [0.1, 0.2, 0.5, 1, 2, 5]:
        dro_model.standard_solver(X = X_new, y = y_new, robust_param = eps)
        print('AUC under CVXPY with WGAN data is', dro_model.predict_auc(x_test_nrm, y_test), eps)

    # Parameter selection and then test the model performance
    # skf = StratifiedKFold(n_splits=5)
    # for pnorm in pnorms:
    #     all_param['pnorm'] = pnorm
    #     total_score = defaultdict(list)
    #     # K-fold cross validation
    #     for train_index, val_index in skf.split(x_train, y_train):
    #         x_train_k, x_val_k = x_train[train_index], x_train[val_index]
    #         y_train_k, y_val_k = y_train[train_index], y_train[val_index]
    #         x_train_k = stand_scaler.fit_transform(x_train_k)
    #         x_val_k = stand_scaler.transform(x_val_k)
    #         #data_k = {'x': x_train_k, 'y': y_train_k}
    #         dro_model = Logit_Loss(x_train_k, y_train_k)
    #         # optimal = dro_model.standard_solver()
    #         # for key, value in optimal.items():
    #         w_opt = dro_model.standard_solver()
    #         y_scores = 1 / (1 + np.exp(-x_val_k.dot(w_opt)))
    #         total_score.append(roc_auc_score(y_val_k, y_scores))
        # Select the best model
    #     tot_score = pd.DataFrame(total_score)
    #     ave_score = tot_score.mean()
    #     best_kappa, best_epsilon = ave_score.idxmax()
    #     best_reg = ave_score[float('inf')].idxmax()

    #     param = {
    #         'epsilon': [best_epsilon],
    #         'kappa': [best_kappa],
    #         'pnorm': pnorm,
    #         'C': [],
    #         'd': []
    #     }
    #     optimal = dro_model.svm(param, training_data)
    #     w_opt = optimal[(best_kappa, best_epsilon)]['w']
    #     y_scores = 1 / (1 + np.exp(-x_test_nrm.dot(w_opt)))
    #     DRSVM_AUC[pnorm] = roc_auc_score(y_test, y_scores)

    #     param = {
    #         'epsilon': [best_reg],
    #         'kappa': [float('inf')],
    #         'pnorm': pnorm,
    #         'C': [],
    #         'd': []
    #     }
    #     optimal = dro_model.svm(param, training_data)
    #     w_opt = optimal[(float('inf'), best_reg)]['w']
    #     y_scores = 1 / (1 + np.exp(-x_test_nrm.dot(w_opt)))
    #     RSVM_AUC[pnorm] = roc_auc_score(y_test, y_scores)

    # return (DRSVM_AUC, RSVM_AUC, SVM_AUC)