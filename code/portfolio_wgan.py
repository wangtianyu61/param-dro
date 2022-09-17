import pandas as pd
import numpy as np
import math
import random
from gurobipy import *
from full_model import *
from add_param import *
from scipy import stats

from portfolio_util import *

feature_dim = 20
sample_size = 200
test_size = 2000
window_size = 120
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
    port_name = '6_FF'
    ambiguity_size = 0.5

    PORT_WGAN = data_opt_portfolio(window_size)
    df_data = pd.read_csv('../data/portfolio/3_factor_' + str(port_name) + '.csv')[122:]
        
    rv_name = df_data.columns[4:]
    dist_type = 'WGAN'
    PORT_WGAN.roll_window_test_base(df_data, feature_name, rv_name, dist_type)    

    wgan_train_num = 5
    ERM_return = np.zeros(PORT_WGAN.test_number)
    DRO_return = np.zeros(PORT_WGAN.test_number)
    PORT_WGAN.dom_order = 2

    #for each time period
    PORT_WGAN.reparam = False
    for i in range(PORT_WGAN.test_number):
        if i%10 == 0:
            print(i)
        history_return_base = PORT_WGAN.all_return[i:(i + window_size)]
        test_return = PORT_WGAN.all_return[window_size + i]
        #pre-train to obtain the best WGAN
        wgan_data = []
        perform_metric = np.zeros(wgan_train_num)
        for j in range(wgan_train_num):
            PORT_WGAN.history_return = history_return_base
    
            PORT_WGAN.WGAN_train(resample_times)
            wgan_data.append(PORT_WGAN.history_return)
            PORT_WGAN.Downside_Risk_ERM()
            PORT_WGAN.port_return = np.dot(history_return_base, PORT_WGAN.weight)
            perform_metric[j] = PORT_WGAN.evaluate_downside_risk()
        print(perform_metric)
        PORT_WGAN.history_return = wgan_data[np.argmin(perform_metric)]

    
        PORT_WGAN.Downside_Risk_ERM()
        ERM_return[i] = np.dot(PORT_WGAN.weight, test_return)
        PORT_WGAN.Downside_Risk_DRO_chi2_div()
        DRO_return[i] = np.dot(PORT_WGAN.weight, test_return)
    
    PORT_WGAN.port_return = ERM_return
    return_to_csv(PORT_DRO_ERM.port_return, port_name, 'WGAN-ERM')
    print('Excess Risk for WGAN-ERM is ', PORT_WGAN.evaluate_downside_risk())
    PORT_WGAN.port_return = DRO_return
    return_to_csv(PORT_DRO_ERM.port_return, port_name, 'WGAN-DRO')
    print('Excess Risk for WGAN-DRO is ', PORT_WGAN.evaluate_downside_risk())    
    





