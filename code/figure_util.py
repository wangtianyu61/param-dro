import pandas as pd
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import *
from matplotlib import rcParams



# config = {
#     "font.family":'serif',
#     #"font.size":7.5,
#     "mathtext.fontset":'stix',
#     "font.serif":['SimHei'],    
# }

# rcParams.update(config)

def main_simulation():
    """
    x-axis: the size of the ambiguity ball
    y-axis: cost
    each line represent method, corresponding csv
    Column 3+: DRO
    Row / 2 == 0: std, Row / 2 == 1: mean
    Line 1: ERM_nonparam ERM_param DRO_nonparam_chi2
    Line 3: DRO_param_chi2
    Line 5: DRO_nonparam_W1
    Line 7: DRO_param_W1
    """
    DS_DIR = '../result/simulation-quadratic/temp2_50_'
    suffix = '_2_2_2'
    eps = [0, 0.1, 0.2, 0.5, 1, 2.5, 5, 10]
    method_name = ['Nonparam-Chi2', 'Param-Chi2', 'Nonparam-W1', 'Param-W1']
    marker_choice = ['*', 'o', '*', 'o']
    for sample_size in [50, 100, 150, 200]:
        sns.set_style('darkgrid')
        plt.figure(figsize = (10, 6), dpi = 100)
        filename = DS_DIR + str(sample_size) + suffix
        df = pd.read_csv(filename + '.csv')
        res_mean = dict.fromkeys(method_name)
        res_std = dict.fromkeys(method_name)
        for i, method in enumerate(method_name):  
            res_mean[method] = [df.iloc[0, int(i%2)]]
            res_std[method] = [df.iloc[1, int(i%2)]]
            res_mean[method].extend(list(df.iloc[2*i][2:]))
            res_std[method].extend(list(df.iloc[2*i + 1][2:]))
            y_err = list(1.96 * np.array(res_std[method]))
            plt.plot(eps, res_mean[method], label = method, marker = marker_choice[i], linewidth = 1)
            #plt.errorbar(x = eps, y = res_mean[method], yerr = y_err, capsize = 2, label =  method)
        #pyplot.xticks(eps)
        plt.legend(loc = 'upper right')
        plt.xlabel("Size of Ambiguity Set (eps)", size = 14)
        plt.ylabel("2-norm of true decision", size = 14)
        plt.yscale('log')
        plt.savefig('../figures/simu1_' + str(sample_size) + '.pdf')
        #plt.show()




if __name__ == '__main__':
    main_simulation()


