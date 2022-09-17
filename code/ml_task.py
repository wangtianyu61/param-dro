from re import L
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
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
import wgan
import os
from sklearn.metrics import roc_auc_score, r2_score
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
        self.X = np.array(X)
        self.y = np.array(y).flatten()
        self.test_size = self.X.shape[0]
        self.feature_dim = self.X.shape[1]
    def reload(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y).flatten()
        self.test_size = self.X.shape[0]
        self.feature_dim = self.X.shape[1]
        
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
        if type(X)!= pd.DataFrame:
            data_X = pd.DataFrame(X)
            data_y = pd.Series(y)
            self.df_origin = pd.concat([data_X, data_y], axis = 1)
            self.df_origin.columns = list(range(0, data_X.shape[1] + 1))
            
            self.sample_size = data_X.shape[0]
        
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            #in case of labelling
            self.X_columns = list(range(0, data_X.shape[1]))
            self.y_column = [data_X.shape[1]]
        else:
            self.df_origin = pd.concat([X, y], axis = 1)
            self.X_columns = list(X.columns)
            self.y_column = [y.name]
            self.sample_size = len(self.df_origin)
        self.is_regression = is_regression
        self.resample_times = 5
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
        spec = wgan.Specifications(data_wrapper, batch_size = 512, max_epochs = 1000, critic_d_hidden = [32, 32],
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
    def conditional_wgan_X_y(self):
        """
        conditional -> representation, learn P_X and P_{Y | X}
        """
        data = self.df_origin
        #data wrapper for X
        data_wrappers = [wgan.DataWrapper(df = data, continuous_vars = self.X_columns, categorical_vars = [], context_vars = [])]
        #data wrapper for Y | X
        if self.is_regression == False: 
            data_wrappers.append(wgan.DataWrapper(df = data, continuous_vars = [], categorical_vars = self.y_column, context_vars = self.X_columns))
        else:
            data_wrappers.append(wgan.DataWrapper(df = data, continuous_vars = self.y_column, categorical_vars = [], context_vars = self.X_columns))
        specs = [wgan.Specifications(dw, batch_size = min(512, self.sample_size), max_epochs = 1000, critic_d_hidden = [128, 128, 128],
        generator_d_hidden = [128, 128, 128],
        critic_lr = 1e-3, 
        generator_lr = 1e-3, print_every = 500) for dw in data_wrappers]
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
        #df_gen_Y_X = np.array(df_gen_Y_X)
        return df_gen_Y_X[self.X_columns], df_gen_Y_X[self.y_column]

class Distribution_Learner_LDW:
    def __init__(self, X, y, model_name):
        self.df_origin = pd.concat([X, y], axis = 1)
        self.X_columns = X.columns
        self.y_column = ['re78']
        batch_size_selector = {'nswre74': 128, 'psid': 512, 'cps': 4096}
        assert (model_name in ['nswre74', 'psid', 'cps'])
        self.chosen_batch_size = batch_size_selector[model_name]
        
    def conditional_wgan_X_y(self):
        continuous_vars_0 = ["age", "education", "re74", "re75"]
        continuous_lower_bounds_0 = {"re74": 0, "re75": 0}
        categorical_vars_0 = ["black", "hispanic", "married", "nodegree"]
        context_vars_0 = ["t"]
        
        # Y | X, t
        continuous_vars_1 = ["re78"]
        continuous_lower_bounds_1 = {"re78": 0}
        categorical_vars_1 = []
        context_vars_1 = ["t", "age", "education", "re74", "re75", "black", "hispanic", "married", "nodegree"]
        
        # Initialize objects
        data_wrappers = [wgan.DataWrapper(self.df_origin, continuous_vars_0, categorical_vars_0, 
                                        context_vars_0, continuous_lower_bounds_0),
                        wgan.DataWrapper(self.df_origin, continuous_vars_1, categorical_vars_1, 
                                        context_vars_1, continuous_lower_bounds_1)]
        specs = [wgan.Specifications(dw, batch_size = self.chosen_batch_size, max_epochs=1000, critic_d_hidden = [128, 128, 128],
        generator_d_hidden = [128, 128, 128], critic_lr=1e-3, generator_lr=1e-3,
                                    print_every=500) for dw in data_wrappers]
        generators = [wgan.Generator(spec) for spec in specs]
        critics = [wgan.Critic(spec) for spec in specs]
        
        #train X | t
        x, context = data_wrappers[0].preprocess(self.df_origin)
        wgan.train(generators[0], critics[0], x, context, specs[0])
        
        # train Y | X, t
        y, context = data_wrappers[1].preprocess(self.df_origin)
        wgan.train(generators[1], critics[1], y, context, specs[1])
        
        # simulate data with conditional WGANs
        df_generated = data_wrappers[0].apply_generator(generators[0], self.df_origin.sample(int(1e4), replace=True))
        df_generated = data_wrappers[1].apply_generator(generators[1], df_generated)
        return df_generated[self.X_columns], df_generated[self.y_column]

#regression tasks
class Sq_Loss(Loss):
    def __init__(self, X, y):
        Loss.__init__(self, X, y)
    def standard_solver(self, X = None, y = None, option = 'CVXPY', robust_param = False, reg = False):
        assert((X is not None and y is not None) or (X is None and y is None))
        if X is not None:
            self.reload(X, y)

        assert (robust_param is False or type(robust_param) == float or type(robust_param) == int)
        #chi2-divergence
        self.robust = robust_param
        #consider the obj to be 1/n||y - THETA X||^2
        theta = cp.Variable(self.feature_dim)
        if self.robust == False or self.robust == 0:
            #sq loss
            #loss = cp.sum_squares(self.X @ theta - self.y) / self.test_size
            #abs loss
            loss = cp.sum(cp.abs(self.X @ theta - self.y)) / self.test_size
        else:
            #not use them, chi2 does not perform well in this case
            eta = cp.Variable()
            loss = math.sqrt(1 + 2 * self.robust) / math.sqrt(self.test_size) * cp.norm(cp.pos(cp.power(self.X@theta - self.y, 2) - eta), 2) + eta
        cons = []

        #ridge or lasso?
        if reg != False:
            if reg[0] == 'Ridge':
                loss += reg[1] * (cp.norm(theta, 2)) / self.test_size
            elif reg[0] == 'Lasso':
                loss += 2 * reg[1] * cp.norm(theta, 1)
            elif reg[0] == 'W1-2':
                #t = cp.Variable()
                #cons = [t >= 0, cp.sum(cp.power(theta, 2)) + 1 <= t ** 2]
                loss += reg[1] * cp.norm(cp.hstack([theta, 1]), 2) 
            elif reg[0] == 'W1-inf':
                loss += reg[1] * cp.norm(cp.hstack([theta, 1]), 1) 

        problem = cp.Problem(cp.Minimize(loss), cons)
        problem.solve(solver = cp.MOSEK)
        #print(theta.value)
        self.w_opt = theta.value
        
    
    def sklearn_in_built(self, reg = False):
        if reg == False:
            reg = LinearRegression(fit_intercept = False)
        elif reg[0] == 'Ridge':
            reg = Ridge(alpha = reg[1], fit_intercept = False)
        elif reg[0] == 'Lasso':
            reg = Lasso(alpha = reg[1], fit_intercept = False)

        reg.fit(self.X, self.y)
        self.w_opt = reg.coef_.reshape(-1, 1)       
    
    def predict(self, x_test_nrm, y_test):
        x_test_nrm = np.array(x_test_nrm)
        y_pred = np.array(x_test_nrm @ self.w_opt).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)
        print('Empirical Excess Risk is:', np.mean(np.abs(y_test - y_pred)))
        return r2_score(y_pred, y_test)
    

#use the standard cvx solver / in-built module in scikit-learn to solve binary-classification-entropy problem, output the best param
class Logit_Loss(Loss):
    def __init__(self, X, y):
        Loss.__init__(self, X, y)
        self.robust = False

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
        theta = cp.Variable(self.feature_dim)
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