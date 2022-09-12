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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, laplacian_kernel
from sklearn.utils import shuffle


import pandas as pd

from ml_task import Distribution_Learner
import kernel_ml_task
PARAM = {
    'epsilon': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    'kappa': [0.05, 0.1, 0.25, 0.5, float('inf')],
    'deg': [2, 3, 4, 5],
    'gamma_rbf': [1/100, 1/64, 1/36, 1/25],
    'gamma_lap': [1/100, 1/64, 1/36, 1/25]
}
kernel_functions = ['polynomial', 'rbf', 'laplacian']


def uci_classification_kernel(*args):
    """
    Use Gaussian Kernel as default
    """
    gamma = 1 / 100
    sel_degree = 3

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
    x_train_kernel = polynomial_kernel(x_train_nrm, degree = sel_degree, gamma = gamma)
    x_test_kernel = polynomial_kernel(x_test_nrm, x_train_nrm, degree = sel_degree, gamma = gamma)
    #data_k = {'x': x_train_kernel, 'y': y_train}
    for eps in PARAM['epsilon']:
        sel_param = {
        'epsilon': [eps], 'kappa': [0.5]
        }
        print(kernel_ml_task.test_performance(x_train_kernel, y_train, x_test_kernel, y_test, sel_param))

    #wgan to refit first
    dist_model = Distribution_Learner(x_train_nrm, y_train)
    X_new, y_new = dist_model.condition_wgan_X_y()
    print(X_new.shape)
    x_train_new = stand_scaler.transform(X_new)
    x_train_kernel_new = polynomial_kernel(x_train_new, degree = sel_degree, gamma = gamma)
    x_test_kernel_new = polynomial_kernel(x_test, x_train_new, degree = sel_degree, gamma = gamma)
    print(x_train_kernel_new.shape)
    for eps in PARAM['epsilon']:
        sel_param = {
        'epsilon': [eps], 'kappa': [0.5]
        }
        print(kernel_ml_task.test_performance(x_train_kernel_new, y_new, x_test_kernel_new, y_test, sel_param))