import pandas as pd
import numpy as np
import math
import gurobipy as grb
import cvxpy as cp
import wgan
import os

from scipy.special import beta
from add_param import *
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import wgan
import os
from sklearn.metrics import roc_auc_score
import torch
from scipy.optimize import brent
from scipy.spatial.distance import pdist, squareform
from torch import optim
from torch.autograd import Variable
from torch.nn import Module, Parameter
from torch.nn.functional import binary_cross_entropy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Loss:
    def __init__(self, X, y):
        self.X = X
        self.y = y.flatten()
        self.feature_dim = X.shape[0]
        self.test_size = X.shape[1]
        
#still not involved in nn
#use SGD to solver binary-classification-entropy problem
## can handle more complex nn cases
class Logit_SGD_Loss(Module, Loss):
    def __init__(self):
        pass
    def forward(self):
        pass
    def project(self):
    #project into the regularization space
        pass

# in ml with x and y
class Distribution_Learner:
    def __init__(self, X, y, is_regression = False):
        self.data_X = X
        self.data_y = y.reshape(-1, 1)
        self.resample_times = 2
        self.sample_size = self.data_X.shape[0]
        self.is_regression = is_regression
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        #in case of labelling
        self.X_columns = list(range(0, self.data_X.shape[1]))
        self.y_column = [self.data_X.shape[1]]
    def joint_wgan_X_y(self):
        """
        joint -> representation, directly learn P_{X, Y}
        """
        data = pd.DataFrame(np.hstack((self.data_X, self.data_y)),  columns =self.X_columns + self.y_column)

        #directly learn P_{X, Y} through gan
        if self.is_regression == False: 
            data_wrapper = wgan.DataWrapper(df = data, continuous_vars = self.X_columns, categorical_vars = self.y_column, context_vars = [])
        else:
            data_wrapper = wgan.DataWrapper(df = data, continuous_vars = self.X_columns + self.y_column, categorical_vars = [], context_vars = [])
        spec = wgan.Specifications(data_wrapper, batch_size = min(64, self.sample_size), max_epochs = 2000, critic_d_hidden = [32, 32],
        generator_d_hidden = [32, 32],
        critic_lr = 1e-3, 
        generator_lr = 1e-3)
        
        generator = wgan.Generator(spec)
        critic = wgan.Critic(spec)
        tr_X, context = data_wrapper.preprocess(data)
        wgan.train(generator, critic, tr_X, context, spec)
        df_gen = data_wrapper.apply_generator(generator, data.sample(self.resample_times * self.sample_size, replace = True))
        df_new_gen = np.array(df_gen)
        return df_new_gen[:, self.X_columns], df_new_gen[:, self.y_column].reshape(-1, 1)
    def condition_wgan_X_y(self):
        """
        conditional -> representation, learn P_X and P_{Y | X}
        """
        data = pd.DataFrame(np.hstack((self.data_X, self.data_y)),  columns =self.X_columns + self.y_column)
        #data wrapper for X
        data_wrappers = [wgan.DataWrapper(df = data, continuous_vars = self.X_columns, categorical_vars = [], context_vars = [])]
        #data wrapper for Y | X
        if self.is_regression == False: 
            data_wrappers.append(wgan.DataWrapper(df = data, continuous_vars = [], categorical_vars = self.y_column, context_vars = self.X_columns))
        else:
            data_wrappers.append(wgan.DataWrapper(df = data, continuous_vars = self.y_column, categorical_vars = [], context_vars = self.X_columns))
        specs = [wgan.Specifications(dw, batch_size = min(64, self.sample_size), max_epochs = 1000, critic_d_hidden = [24, 24],
        generator_d_hidden = [24, 24],
        critic_lr = 1e-3, 
        generator_lr = 1e-3) for dw in data_wrappers]
        generators = [wgan.Generator(spec) for spec in specs]
        critics = [wgan.Critic(spec) for spec in specs]
    
        #train X 
        tr_X, context = data_wrappers[0].preprocess(data)
        wgan.train(generators[0], critics[0], tr_X, context, specs[0])
    
        #train Y | X
        tr_y, context = data_wrappers[1].preprocess(data)
        wgan.train(generators[1], critics[1], tr_y, context, specs[1])

        df_gen_X = data_wrappers[0].apply_generator(generators[0], data.sample(self.resample_times * self.sample_size, replace = True))
        df_gen_Y_X = data_wrappers[1].apply_generator(generators[1], df_gen_X)
        df_gen_Y_X = np.array(df_gen_Y_X)
        return df_gen_Y_X[:, self.X_columns], df_gen_Y_X[:, self.y_column].reshape(-1, 1)
#use the standard cvx solver / in-built module in scikit-learn to solve binary-classification-entropy problem, output the best param
class Logit_Loss(Loss):
    def __init__(self, X, y):
        Loss.__init__(self, X, y)
        self.robust = False
    def reload(self, X, y):
        self.X = X
        self.y = y.flatten()
        self.feature_dim = X.shape[0]
        self.test_size = X.shape[1]
    def standard_solver(self, X = None, y = None, option = 'CVXPY', robust_param = False):
        #can input new data
        assert((X is not None and y is not None) or (X is None and y is None))
        if X is not None:
            self.reload(X, y)

        assert (robust_param is False or type(robust_param) == float or type(robust_param) == int)
    #here we mainly focus on the chi2-divergence
        self.robust = robust_param        
        #y should be in {-1, 1}
        ## consider the obj to be log(1 + exp(-y theta^T x))
        theta = cp.Variable(self.test_size)
        if self.robust == False:
            loss = cp.sum(cp.logistic(cp.multiply(-self.y, self.X@theta)))
            
        else:
            #use the dual form to calculate chi2 loss
            eta = cp.Variable()
            loss = math.sqrt(1 + 2 * self.robust) / math.sqrt(self.test_size) * cp.norm(cp.pos(cp.logistic(cp.multiply(-self.y, self.X@theta)) - eta), 2) + eta

        problem = cp.Problem(cp.Minimize(loss))
        problem.solve(solver = cp.MOSEK)
        #print(theta.value)
        self.w_opt = theta.value

    def sklearn_in_built(self):
        #only work under nonrobust cases
        assert (self.robust == False)
        print(self.X.shape)
        lr_model = LogisticRegression(penalty = 'none', fit_intercept = False)
        lr_model.fit(self.X, self.y)
        #print(lr_model.coef_)
        self.w_opt = lr_model.coef_.reshape(-1, 1)

    def standard_solver_svm(self):
        """ classical SVM by minimizing hinge loss function solely """
        x_train = self.X
        y_train = self.y

        row, col = x_train.shape
    
        # Step 0: create model
        model = grb.Model('classical_hinge_loss_minimization')
        model.setParam('OutputFlag', False)

        # Step 1: define decision variables
        var_s = {}
        var_w = {}
        for i in range(row):
            var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
        for j in range(col):
            var_w[j] = model.addVar(
                vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)

        # Step 2: integerate variables
        model.update()

        # Step 3: define constraints
        for i in range(row):
            model.addConstr(
                1 - y_train[i] * grb.quicksum(var_w[j] * x_train[i, j] for j in range(col)) <= var_s[i])

        # Step 4: define objective value
        sum_var_s = grb.quicksum(var_s[i] for i in range(row))
        obj = 1 / row * sum_var_s
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        w_opt = np.array([var_w[i].x for i in range(col)])

        self.w_opt = w_opt
    def predict_auc(self, x_test_nrm, y_test):
        #output the auc of each model
        y_scores = 1 / (1 + np.exp(-x_test_nrm.dot(self.w_opt)))
        auc = roc_auc_score(y_test, y_scores)
        return auc




"""
param fit with the parametric model
train the ERM / DRO with different parameters

"""

if __name__ == '__main__':
    #dataloader

    #train the model w/o param fitting for GAN
    ## 

    

    #test metrics

    pass