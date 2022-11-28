import numpy as np
import pandas as pd

import math
import random
from gurobipy import *
from full_model import *
from add_param import *
from scipy import stats


feature_dim = 20
sample_size = 100
test_size = 2000
#window_size = 120
#loop_num_reparam = 50

#distribution shift
contaminate_rate = 0.2
shift_prop = 0.8


outer_test_num = 200
resample_times = 10






np.random.seed(42)
feature_name = ['Mkt-RF', 'SMB', 'HML']
# return_mean = np.random.uniform(-2, 2, port_num)    
# A = np.random.uniform(-1, 1, (port_num, port_num))
# B = np.dot(A, A.transpose())
# return_cov = (B + B.T)/4


dom_order = 1
class DGP:
    def __init__(self, sample_size, feature_dim, test_size, dist_type):
        self.sample_size = sample_size
        self.test_size = test_size
        self.port_num = feature_dim
        self.distribution_type = dist_type
        if dist_type == 'normal':
            self.return_mean = np.random.uniform(-2, 2, feature_dim)    
            A = np.random.uniform(-1, 1, (feature_dim, feature_dim))
            B = np.dot(A, A.transpose())
            self.return_cov = (B + B.T)/4
        elif dist_type == 'beta' or dist_type == 'WGAN':
            self.alpha = np.random.uniform(alpha_lb, alpha_ub, feature_dim)
            self.dist_bd = dist_bound
            self.beta = np.ones(feature_dim)*2
            
    def noise_generator(self, size):
        #uniform noise
        return np.random.uniform(-2, 2, (size, self.port_num))
        #return 0

    def DGP_train_normal(self):
        noise1 = self.noise_generator(self.sample_size)
        #noise = np.random.exponential(noise_rate, (sample_size, port_num)) - noise_rate
        return np.random.multivariate_normal(self.return_mean, self.return_cov, self.sample_size) + noise1
        #np.random.uniform(-2, 2, (sample_size, port_num))
        #return np.random.multivariate_normal(return_mean, return_cov, sample_size) + noise

    def DGP_test_normal(self):
        #noise1 = 2
        noise1 = self.noise_generator(self.test_size)
        return np.random.multivariate_normal(self.return_mean, self.return_cov, self.test_size) + noise1
    #np.random.uniform(-2, 2, (test_size, port_num))
    def DGP_train_beta(self):
        data = np.ones((self.port_num, self.sample_size))
        for i in range(self.port_num):
            data[i] = 2*self.dist_bd[i] * np.random.beta(self.alpha[i], self.beta[i], size = self.sample_size) - self.dist_bd[i]
        # ##model 0: no noise
        # return data.T
        #model 1: convolution
        return data.T + self.noise_generator(self.sample_size)
        ##model 2: contaminate
        # data = data.T
        # for i in range(self.sample_size):
        #     if np.random.random() < contaminate_rate:
        #         for j in range(self.port_num):
        #             data[i][j] = np.random.uniform(-self.dist_bd[j], self.dist_bd[j])
        # return data
    def DGP_test_beta(self):
        #model 3: distribution shift
        #shift_prop2 = np.random.uniform(-shift_prop, shift_prop)
        # for i in range(self.port_num):
        #     shift_prop2 = np.random.uniform(0, shift_prop)
        #     flag = -1
        #     if self.alpha[i] < (alpha_lb + alpha_ub) / 2:
        #         flag = 1
        #     # if self.alpha[i] < (alpha_lb + alpha_ub)/2:
        #     #     self.alpha[i] = self.alpha[i] + (alpha_ub - self.alpha[i])/shift_prop
        #     # else:
        #     #     self.alpha[i] = self.alpha[i] - (self.alpha[i] - alpha_lb)/shift_prop
        #     self.alpha[i] = self.alpha[i] + flag * shift_prop2 * min(self.alpha[i] - alpha_lb, alpha_ub - self.alpha[i])
            
            
        
        data = np.ones((self.port_num, self.test_size))
        for i in range(self.port_num):
            data[i] = 2*self.dist_bd[i] * np.random.beta(self.alpha[i], self.beta[i], size = self.test_size) - self.dist_bd[i]
        # ##base model: no noise
        # return data.T
        #model 1: convolution
        return data.T + self.noise_generator(self.test_size)
    
        ##model 2: contaminate
        # data = data.T
        # for i in range(self.test_size):
        #     if np.random.random() < contaminate_rate:
        #         for j in range(self.port_num):
        #             data[i][j] = np.random.uniform(-self.dist_bd[j], self.dist_bd[j])
        # return data
        
"""
The underlying distribution is Beta + Uniform, but we fit with normal and empirical (nonparam) 
"""
def ambiguity_level(sample_size, param = False):
    if param == False:
        return math.sqrt(20 / sample_size)
    else:
        return math.sqrt(1 / sample_size) + 0.02

np.random.seed(42)
if __name__ == '__main__':
    CV = 0
    if CV != 0:
        CV = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
    
    SAMPLE_RANGE = [800]    
    DRO_nonparam = np.zeros((len(SAMPLE_RANGE), outer_test_num))
    ERM_nonparam = np.zeros((len(SAMPLE_RANGE), outer_test_num))
    DRO_normal = np.zeros((len(SAMPLE_RANGE), outer_test_num))
    ERM_normal = np.zeros((len(SAMPLE_RANGE), outer_test_num))
    
    dom_order = 2
    for j in range(outer_test_num):
        print(j)
        data = DGP(sample_size, feature_dim, test_size, dist_type)
        PORT_DRO_ERM = data_opt_portfolio()
        
    
        
        for i, sample_size in enumerate(SAMPLE_RANGE):
            # DGP for each round
            data.sample_size = sample_size
            ## fit with beta 
            train_data = data.DGP_train_beta()
            new_data = data.DGP_test_beta()
            PORT_DRO_ERM.simulate_test_base(train_data, new_data, 'beta')
            
            ERM_nonparam[i][j] = PORT_DRO_ERM.simulate_test_method('Down_Risk_ERM', 0, dom_order)
            PORT_DRO_ERM.reparam = False
        
            PORT_DRO_ERM.ambiguity_size = ambiguity_level(sample_size)
            DRO_nonparam[i][j] = PORT_DRO_ERM.simulate_test_method('Down_Risk_DRO', CV, dom_order)
            
            PORT_DRO_ERM.reparam = resample_times
            PORT_DRO_ERM.ambiguity_size = ambiguity_level(sample_size, True)
            ERM_normal[i][j] = PORT_DRO_ERM.simulate_test_method('Down_Risk_ERM', 0, dom_order)
            DRO_normal[i][j] = PORT_DRO_ERM.simulate_test_method('Down_Risk_DRO', CV, dom_order)
    
