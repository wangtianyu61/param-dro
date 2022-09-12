import pandas as pd
import numpy as np
import math
import random
from gurobipy import *
from full_model import *
from add_param import *
from scipy import stats

feature_dim = 20
sample_size = 200
test_size = 2000
window_size = 60
#loop_num_reparam = 50

#distribution shift
contaminate_rate = 0.2
shift_prop = 5


outer_test_num = 50
resample_times = 10


dom_order = 2
np.random.seed(42)
feature_name = ['Mkt-RF', 'SMB', 'HML']

if __name__ == '__main__':
    #ambiguity_set_choice = [0.02, 0.05, 0.1, 0.2, 0.5, 1]
    port_name = '30_Industry'
    print(port_name)
    CV = 1
    if CV == 0:
        #ambiguity_set_choice = [0.01]
        #ambiguity_set_choice = [0.1, 0.2, 0.5, 1, 2.5, 5, 10]
        ambiguity_set_choice = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]
    else:
        ambiguity_set_choice = [1]
        CV = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]
    PORT_DRO_ERM = data_opt_portfolio(window_size)
    df_data = pd.read_csv('../data/portfolio/3_factor_' + str(port_name) + '.csv')[0:70]
    
    DRO_noparam_obj = np.zeros(len(ambiguity_set_choice) + 1)
    DRO_beta_obj = np.zeros(len(ambiguity_set_choice) + 1)
    DRO_normal_obj = np.zeros(len(ambiguity_set_choice) + 1)
    
    rv_name = df_data.columns[4:]
    dist_type = 'WGAN'
    PORT_DRO_ERM.roll_window_test_base(df_data, feature_name, rv_name, dist_type)
    
    #no reparameterize
    PORT_DRO_ERM.reparam = False
    DRO_noparam_obj[0] = PORT_DRO_ERM.roll_window_test_method('Down_Risk_ERM', 0, dom_order)
    for j, param in enumerate(ambiguity_set_choice):
        PORT_DRO_ERM.ambiguity_size = param
        DRO_noparam_obj[j + 1] = PORT_DRO_ERM.roll_window_test_method('Down_Risk_DRO', CV, dom_order)
    # #print(np.mean(return1)/np.std(return1))
    
    #reparameterize
    PORT_DRO_ERM.reparam = resample_times


    # ERM_GAN_obj = PORT_DRO_ERM.roll_window_test_method('Down_Risk_ERM', 0, dom_order)
    # DRO_GAN_obj = PORT_DRO_ERM.roll_window_test_method('Down_Risk_DRO', CV, dom_order)
    
    PORT_DRO_ERM.dist_type = 'beta'
    DRO_beta_obj[0] = PORT_DRO_ERM.roll_window_test_method('Down_Risk_ERM', 0, dom_order)
    for j, param in enumerate(ambiguity_set_choice):
        PORT_DRO_ERM.ambiguity_size = param
        DRO_beta_obj[j + 1] = PORT_DRO_ERM.roll_window_test_method('Down_Risk_DRO', CV, dom_order)

    PORT_DRO_ERM.dist_type = 'normal'
    DRO_normal_obj[0] = PORT_DRO_ERM.roll_window_test_method('Down_Risk_ERM', 0, dom_order)
    for j, param in enumerate(ambiguity_set_choice):
        PORT_DRO_ERM.ambiguity_size = param
        DRO_normal_obj[j + 1] = PORT_DRO_ERM.roll_window_test_method('Down_Risk_DRO', CV, dom_order)

    df = pd.DataFrame([list(DRO_noparam_obj), list(DRO_beta_obj), list(DRO_normal_obj)])
    df.to_csv('../result/portfolio/' + port_name + '_cv.csv', index = None)
    
    
    # print('nonparam', ERM_noparam_obj, DRO_noparam_obj)
    # print('GAN', ERM_GAN_obj, DRO_GAN_obj)
    # print('beta', ERM_beta_obj, DRO_beta_obj)
    # print('normal', ERM_normal_obj, DRO_normal_obj)