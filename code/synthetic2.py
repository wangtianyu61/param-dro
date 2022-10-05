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
shift_prop = 0.2


outer_test_num = 20
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
        #return np.random.uniform(-2, 2, (size, self.port_num))
        return 0

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

if __name__ == '__main__':
    
    for dom_order in [1, 2, 4]:
        for sample_size in [25, 50, 100, 200]:
            CV = 0
            if CV == 0:
                #ambiguity_set_choice = [0.01]
                #misspecified
                ambiguity_set_choice = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
                #ambiguity_set_choice = [0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
            else:
                ambiguity_set_choice = [1]
                CV = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            PORT_DRO_ERM = data_opt_portfolio()
            DRO_nonparam = np.zeros((len(ambiguity_set_choice), outer_test_num))
            ERM_nonparam = np.zeros(outer_test_num)
            DRO_normal = np.zeros((len(ambiguity_set_choice), outer_test_num))
            ERM_normal = np.zeros(outer_test_num)  
            DRO_beta = np.zeros((len(ambiguity_set_choice), outer_test_num))
            ERM_beta = np.zeros(outer_test_num)
            for i in range(outer_test_num):
                print(i,sample_size)
                data = DGP(sample_size, feature_dim, test_size, dist_type)
                if dist_type == 'beta':
                    train_data = data.DGP_train_beta()
                    new_data = data.DGP_test_beta()
                
                    PORT_DRO_ERM.simulate_test_base(train_data, new_data, dist_type)
                    PORT_DRO_ERM.true_alpha = data.alpha
                    PORT_DRO_ERM.true_beta = data.beta
                    
                #-----------------------------checkpoint------------------------#
                elif dist_type == 'normal':
                    train_data = data.DGP_train_normal()
                    new_data = data.DGP_test_normal()
                    PORT_DRO_ERM.simulate_test_base(train_data, new_data, dist_type)
                elif dist_type == 'WGAN':
                    train_data = data.DGP_train_beta()
                    new_data = data.DGP_test_beta()
                    PORT_DRO_ERM.simulate_test_base(train_data, new_data, dist_type)
                    
                PORT_DRO_ERM.reparam = False
                ERM_nonparam[i] = PORT_DRO_ERM.simulate_test_method('Down_Risk_ERM', 0, dom_order)
            
                # PORT_DRO_ERM.reparam = resample_times
                # ERM_param[i] = PORT_DRO_ERM.simulate_test_method('Down_Risk_ERM', 0, dom_order)
                
            
                
                if CV == 0:          
                    for j, param in enumerate(ambiguity_set_choice):
                        PORT_DRO_ERM.ambiguity_size = param
                        PORT_DRO_ERM.reparam = False
                        #print(PORT_DRO_ERM.history_return.shape)
                        DRO_nonparam[j][i] = PORT_DRO_ERM.simulate_test_method('Down_Risk_DRO', CV, dom_order)
                        
                    # PORT_DRO_ERM.reparam = resample_times
                    # PORT_DRO_ERM.param_est()
                    # for j, param in enumerate(ambiguity_set_choice):
                    #     #print(PORT_DRO_ERM.ambiguity_size)
                    #     PORT_DRO_ERM.ambiguity_size = param
                    #     PORT_DRO_ERM.reparam = False
                    #     DRO_param[j][i] = PORT_DRO_ERM.simulate_test_method('Down_Risk_DRO', CV, dom_order)    
                    give_type = 'beta'
                    PORT_DRO_ERM.dist_type = give_type
                    PORT_DRO_ERM.simulate_test_base(train_data, new_data, give_type)
                    PORT_DRO_ERM.reparam = resample_times
                    ERM_beta[i] = PORT_DRO_ERM.simulate_test_method('Down_Risk_ERM', 0, dom_order)
                    
                    PORT_DRO_ERM.param_est()
                    for j, param in enumerate(ambiguity_set_choice):
                        #print(PORT_DRO_ERM.ambiguity_size)
                        PORT_DRO_ERM.ambiguity_size = param
                        PORT_DRO_ERM.reparam = False
                        DRO_beta[j][i] = PORT_DRO_ERM.simulate_test_method('Down_Risk_DRO', CV, dom_order)   
            
                    give_type = 'normal'
                    PORT_DRO_ERM.dist_type = give_type
                    PORT_DRO_ERM.simulate_test_base(train_data, new_data, give_type)
                    PORT_DRO_ERM.reparam = resample_times
                    ERM_normal[i] = PORT_DRO_ERM.simulate_test_method('Down_Risk_ERM', 0, dom_order)
                    
                    PORT_DRO_ERM.param_est()
                    for j, param in enumerate(ambiguity_set_choice):
                        #print(PORT_DRO_ERM.ambiguity_size)
                        PORT_DRO_ERM.ambiguity_size = param
                        PORT_DRO_ERM.reparam = False
                        DRO_normal[j][i] = PORT_DRO_ERM.simulate_test_method('Down_Risk_DRO', CV, dom_order)   
                
            
            
            
            a1 = [np.mean(ERM_nonparam)]
            a1.extend([np.mean(DRO_nonparam[j]) for j in range(len(ambiguity_set_choice))])
            
            b1 = [np.mean(ERM_beta)]
            b1.extend([np.mean(DRO_beta[j]) for j in range(len(ambiguity_set_choice))])
            
            c1 = [np.mean(ERM_normal)]
            c1.extend([np.mean(DRO_normal[j]) for j in range(len(ambiguity_set_choice))])
            
            #stat significance
            a2 = [1]
            a2.extend([stats.ttest_rel(DRO_nonparam[j], ERM_nonparam)[1] for j in range(len(ambiguity_set_choice))])
            
            b2 = [1]
            b2.extend([stats.ttest_rel(DRO_beta[j], ERM_beta)[1] for j in range(len(ambiguity_set_choice))])
            
            b3 = [stats.ttest_rel(ERM_beta, ERM_nonparam)[1]]
            b3.extend([stats.ttest_rel(DRO_beta[j], DRO_nonparam[j])[1] for j in range(len(ambiguity_set_choice))])
            
            c2 = [1]
            c2.extend([stats.ttest_rel(DRO_normal[j], ERM_normal)[1] for j in range(len(ambiguity_set_choice))])
                
            c3 = [stats.ttest_rel(ERM_normal, ERM_nonparam)[1]]
            c3.extend([stats.ttest_rel(DRO_normal[j], DRO_nonparam[j])[1] for j in range(len(ambiguity_set_choice))])
            
            df = pd.DataFrame([a1, a2, b1, b2, b3, c1, c2, c3])    
            
            suffix = str(sample_size) + '_' + str(PORT_DRO_ERM.lower_weight) 
            #str(PORT_DRO_ERM.lower_weight) + '_' + str(PORT_DRO_ERM.target_return) + '_' + str(shift_prop)
            #print(a1, b1, c1)
            
            df.to_csv('../result/1003-rerun/temp1_' + str(dom_order) + '_' + str(suffix) + '.csv', index = None)
    
    
    
    
    
    
    