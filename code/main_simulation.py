import pandas as pd
import numpy as np
import math
from gurobipy import *
from full_model import *
from full_simulation import *


feature_dim = 50
loop_num = 50
sample_size = 50

dom_order = 2
noise_mean = 5

def DGP(sample_size, noise_mean):
    sample_mean = np.zeros(feature_dim)
    A = np.random.uniform(-1, 1, (feature_dim, feature_dim))
    B = np.dot(A, A.transpose())
    return_cov = (B + B.T)/4
    sample_cov = return_cov
    true = np.random.multivariate_normal(sample_mean, sample_cov, sample_size)
    
    #add noise or not: Exp, Cauchy guarantee to be zero mean is ok
    noise_rate = noise_mean
    noise = np.random.exponential(noise_rate, (sample_size, feature_dim)) - noise_rate
    return true + noise
    


if __name__ == '__main__':
    for sample_size in [50, 100, 150, 200]:
        for noise_mean in [2, 5]:
            ambiguity_set_choice = [0.1, 0.2, 0.5, 1, 2.5, 5, 10]
            simu_model = simulation_test(sample_size, feature_dim)
            
            res_DRO_nonparam_chi2 = np.zeros((len(ambiguity_set_choice), loop_num))
            res_ERM_nonparam = np.zeros(loop_num)
            res_DRO_param_chi2 = np.zeros((len(ambiguity_set_choice), loop_num))
            
            res_DRO_nonparam_W1 = np.zeros((len(ambiguity_set_choice), loop_num))
            res_DRO_param_W1 = np.zeros((len(ambiguity_set_choice), loop_num))
            
            res_ERM_param = np.zeros(loop_num)    
            for i in range(loop_num):
                print(i)
                train_data = DGP(sample_size, noise_mean)
                simu_model.simulate_test_base(train_data)
                simu_model.reparam = False
        
                res_ERM_nonparam[i] = simu_model.ERM(dom_order)
                simu_model.reparam = 10
        
                res_ERM_param[i] = simu_model.ERM(dom_order)
                
                for j, param in enumerate(ambiguity_set_choice):
                    simu_model.ambiguity_size = param
                    simu_model.reparam = False
                    res_DRO_nonparam_chi2[j][i] = simu_model.DRO_chi2(dom_order)
                    res_DRO_nonparam_W1[j][i] = simu_model.DRO_W1(dom_order)
                    
                    simu_model.reparam = 10
                    res_DRO_param_chi2[j][i] = simu_model.DRO_chi2(dom_order)
                    res_DRO_param_W1[j][i] = simu_model.DRO_W1(dom_order)
                    
                    
                    #print(simu_model.sample.shape)
            a1 = [np.mean(res_ERM_nonparam), np.mean(res_ERM_param)]
            a1.extend([np.mean(res_DRO_nonparam_chi2[j]) for j in range(len(ambiguity_set_choice))])
            a2 = [0,0]
            a2.extend([np.mean(res_DRO_param_chi2[j]) for j in range(len(ambiguity_set_choice))])
            a3 = [0,0]
            a3.extend([np.mean(res_DRO_nonparam_W1[j]) for j in range(len(ambiguity_set_choice))])
            a4 = [0,0]
            a4.extend([np.mean(res_DRO_param_W1[j]) for j in range(len(ambiguity_set_choice))])
            
            
            b1 = [np.std(res_ERM_nonparam), np.std(res_ERM_param)]
            b1.extend([np.std(res_DRO_nonparam_chi2[j]) for j in range(len(ambiguity_set_choice))])
            b2 = [0,0]
            b2.extend([np.std(res_DRO_param_chi2[j]) for j in range(len(ambiguity_set_choice))])
            b3 = [0,0]
            b3.extend([np.std(res_DRO_nonparam_W1[j]) for j in range(len(ambiguity_set_choice))])
            b4 = [0,0]
            b4.extend([np.std(res_DRO_param_W1[j]) for j in range(len(ambiguity_set_choice))])
            
            
            df = pd.DataFrame([a1, b1, a2, b2, a3, b3, a4, b4])
            suffix = str(feature_dim) + '_' + str(sample_size) + '_' + str(noise_mean) + '_' + str(dom_order) + '_' + str(simu_model.bd)
            df.to_csv('../result/simulation-quadratic/temp2_' + str(suffix) + '.csv', index = None)
            
