import pandas as pd
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import *
from matplotlib import rcParams
from scipy import stats


# config = {
#     "font.family":'serif',
#     #"font.size":7.5,
#     "mathtext.fontset":'stix',
#     "font.serif":['SimHei'],    
# }

# rcParams.update(config)

def main_simulation(bound, noise_rate):
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
    bd = bound
    noise = noise_rate
    suffix = '_' + str(noise) + '_2' + '_' + str(bd)
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
        plt.legend(loc = 'lower left')
        plt.xlabel("Size of Ambiguity Set (eps)", size = 14)
        plt.ylabel("Exact Generalization Error", size = 14)
        plt.yscale('log')
        plt.savefig('../figures/simu1_' + str(sample_size) + '_' + str(noise) + '_'+ str(bd) +'.pdf')
        #plt.show()

def synthetic2_mis():
    """
    Row 0: Nonparam
    Row 2: Beta
    Row 5: Normal
    """

    DS_DIR = '../result/1003-rerun/temp4_'
    eps = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    dom_range = [1, 2, 4]
    low_weight = [-2, -10]
    model_name = {'Empirical': 0, 'Beta': 2, 'Normal': 5}

    sample_range = [100, 200]
    for sample_size in sample_range:
        for dom_order in dom_range:
            sns.set_style('darkgrid')
            plt.figure(figsize = (10, 6), dpi = 100)
            marker_choice = ['*', 'o', '*', 'o', '*', 'o']
        
            for j, lw in enumerate(low_weight):
                filename = DS_DIR + str(dom_order) + '_' + str(sample_size) + '_' + str(lw) 
                df = pd.read_csv(filename + '.csv')
                
                for i, method in enumerate(list(model_name.keys())):
                    plt.plot(eps, list(df.iloc[model_name[method]]), label = method + r', $\tau$ = ' + str(-1 * lw), marker = marker_choice[int(2 * i + j)], linewidth = 1)
            
            
            plt.legend(loc = 'upper right')
            plt.xlabel(r'Size of Ambiguity Set ($\varepsilon$)', size = 14)
            plt.ylabel(r'Z($\hat{x}$)', size = 14)
            plt.yscale('log')
            plt.savefig('../figures/simu2_temp4_' + str(sample_size) + '_' + str(dom_order) + '.pdf')

def synthetic2_dist_shift():
    """
    Row 0: Nonparam
    Row 2: Beta
    Row 5: Normal
    """

    
    eps = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    #eps = [0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    dom_range = [1, 2, 4]
    shift_C = [0.2, 0.8]
    model_name = {'Empirical': 0, 'Beta': 2, 'Normal': 5}

    sample_range = [25, 50, 100, 200]
    for sample_size in sample_range:
        for dom_order in dom_range:
            sns.set_style('darkgrid')
            plt.figure(figsize = (10, 6), dpi = 100)
            marker_choice = ['*', 'o', '*', 'o', '*', 'o']
            # if sample_size in [100, 200] and dom_order in [1, 2]:
            #     DS_DIR = '../result/1003-rerun/temp8_'
            # else:
            #     DS_DIR = '../result/1003-rerun/temp7_'
            DS_DIR = '../result/1003-rerun/temp7_'
            for j, shift in enumerate(shift_C):
                filename = DS_DIR + str(dom_order) + '_' + str(sample_size) + '_-2_' + str(shift) 
                df = pd.read_csv(filename + '.csv')
                
                for i, method in enumerate(list(model_name.keys())):
                    res_org = list(df.iloc[model_name[method]])
                    #res = [res_org[0]] + res_org[(len(res_org) - 6): (len(res_org) - 1)]
                    plt.plot(eps, res_org, label = method + r', $C$ = ' + str(shift), marker = marker_choice[int(2 * i + j)], linewidth = 1)
            
            
            plt.legend(loc = 'upper right')
            plt.xlabel(r'Size of Ambiguity Set ($\varepsilon$)', size = 14)
            plt.ylabel(r'Z($\hat{x}$)', size = 14)
            plt.yscale('log')
            plt.savefig('../figures/simu3_temp7_' + str(sample_size) + '_' + str(dom_order) + '.pdf')

def synthetic2_param():
    #main_simulation()
    """
    Row 0: Nonparam
    Row 2: Beta
    Row 5: Normal
    """

    DS_DIR = '../result/1003-rerun/temp0_'
    eps = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    dom_range = [1, 2, 4]
    low_weight = [-2, -10]
    model_name = {'Empirical': 0, 'Beta': 2, 'Normal': 5}

    sample_range = [25, 50, 100, 200]
    for sample_size in sample_range:
        for dom_order in dom_range:
            sns.set_style('darkgrid')
            plt.figure(figsize = (10, 6), dpi = 100)
            marker_choice = ['*', 'o', '*', 'o', '*', 'o']
        
            for j, lw in enumerate(low_weight):
                filename = DS_DIR + str(dom_order) + '_' + str(sample_size) + '_' + str(lw) 
                df = pd.read_csv(filename + '.csv')
                
                for i, method in enumerate(list(model_name.keys())):
                    plt.plot(eps, list(df.iloc[model_name[method]]), label = method + r', $\tau$ = ' + str(-1 * lw), marker = marker_choice[int(2 * i + j)], linewidth = 1)
            
            
            plt.legend(loc = 'upper right')
            plt.xlabel(r'Size of Ambiguity Set ($\varepsilon$)', size = 14)
            plt.ylabel(r'Z($\hat{x}$)', size = 14)
            plt.yscale('log')
            plt.savefig('../figures/simu2_temp0_' + str(sample_size) + '_' + str(dom_order) + '.pdf') 
            
def ldw_table():
    DS_DIR = '../result/LDW/'
    model = 'age'
    dtype = 'cps'
    sample_range = {'age': [200, 500, 1000], 'temp': [200, 500, 1000, 2000]}
    output_dta = [([0] * 5) for i in range(len(sample_range[model]))]
    #output_dta = np.array([np.arange(len(sample_range[model])), np.arange(5)], dtype = str)
    for i, sample_size in enumerate(sample_range[model]):

        filename = DS_DIR + model + '_' + dtype + '_2_' + str(sample_size)
        df = pd.read_csv(filename + '.csv')
        index_list = ['0', '2', '4', '6']
        res_mean = [np.mean(df[index])*100 for index in index_list]
        res_std = [np.std(df[index])*100 for index in index_list]
        k = 0
        for a, b in zip(res_mean, res_std):
            output_dta[i][k] = str(round(res_mean[k], 2)) + ' ( ' + str(round(res_std[k], 2)) + ' )'
            k += 1
            
            
            # compare NP-DRO and P-DRO
        output_dta[i][k] = stats.ttest_rel(list(df['2']), list(df['6']))[1]
    df = pd.DataFrame(output_dta, index = None, columns = ['NP-ERM', 'NP-DRO', 'P-ERM', 'P-DRO', 'pvalue'])
    df.to_csv(DS_DIR + model + '_' + dtype + '.csv', index = None)

def synthetic2_concept(dist_name):
    """
    Row 0: Nonparam
    Row 2: Beta
    Row 5: Normal
    temp1/4/7_2_(sample size)_-2.csv
    """
    dist = dist_name
    dist_model = {'True': 1, 'Misspecified': 4, 'Shift': 7}
    DS_DIR = '../result/1003-rerun/temp'
    model_name = {'Empirical': 0, 'Beta': 2, 'Normal': 5}
    model_name_err = {'Empirical': 8, 'Beta': 9, 'Normal': 10}
    sample_range = [25, 50, 100, 200]
    model_perform = {}
    model_perform_err = {}
    for name in model_name.keys():
        model_perform[name + '-ERM'] = list(np.zeros(len(sample_range)))
        model_perform[name + '-DRO'] = list(np.zeros(len(sample_range)))
        model_perform_err[name + '-ERM'] = list(np.zeros(len(sample_range)))
        model_perform_err[name + '-DRO'] = list(np.zeros(len(sample_range)))    
    for i, sample_size in enumerate(sample_range):
        filename = DS_DIR + str(dist_model[dist]) + '_2_' + str(sample_size) + '_-2'
        if dist == 'Shift':
            filename += '_0.8'
        df = pd.read_csv(filename + '.csv')
        
        for name in model_name.keys():
            model_perform[name + '-ERM'][i] = df.iloc[model_name[name]][0]
            model_perform[name + '-DRO'][i] = min(df.iloc[model_name[name]][1:])
            model_perform_err[name + '-ERM'][i] = df.iloc[model_name_err[name]][0]
            model_perform_err[name + '-DRO'][i] = min(df.iloc[model_name_err[name]][1:])
    
    #print(model_perform)
    colors=['#4daf4a', '#ff7f00', '#984ea3', '#FFD43B', '#a65628', '#f781bf', '#e41a1c', '#377eb8']
    sns.set_style('darkgrid')
    plt.figure(figsize = (10, 6), dpi = 100)
    marker_choice = ['*', 'o', '*', 'o', '*', 'o']
    for j, model_name_setup in enumerate(list(model_perform.keys())):
        plt.plot(sample_range, model_perform[model_name_setup], label = model_name_setup, marker = marker_choice[j], color = colors[j], linewidth = 1)
        #plt.errorbar(x = sample_range, y = model_perform[model_name_setup], yerr = model_perform_err[model_name_setup], color = colors[j], capsize = 2)
    
    
    plt.legend(loc = 'upper right')
    plt.xlabel('sample size', size = 14)
    plt.ylabel(r'Z($\hat{x}$)', size = 14)
    #plt.yscale('log')
    plt.savefig('../figures/test/simu0_tempX_' + str(dist) + '.pdf') 
def unfinished():
    """
    Row 0: Nonparam
    Row 2: Beta
    Row 5: Normal
    temp0/4/7_2_(sample size)_-2.csv
    """
    DS_DIR = '../result/LDW/'
    model = 'temp'
    dtype = 'psid'
    sample_range = {'age': [200, 500, 1000, 1500], 'temp': [200, 500, 1000, 1500]}
    output_dta = np.zeros((4, len(sample_range[model])))
    output_dta_std = np.zeros((4, len(sample_range[model])))
    method_name_list = ['NP-ERM (Empirical)', 'NP-DRO (Empirical)', 'P-ERM (Mixture Gaussian)', 'P-DRO (Mixture Gaussian)']
    #output_dta = [([0] * 5) for i in range(len(sample_range[model]))]
    #output_dta = np.array([np.arange(len(sample_range[model])), np.arange(5)], dtype = str)
    for i, sample_size in enumerate(sample_range[model]):

        filename = DS_DIR + model + '_' + dtype + '_2_' + str(sample_size)
        df = pd.read_csv(filename + '.csv')
        index_list = ['0', '2', '4', '6']
        res_mean = [np.mean(df[index])*100 for index in index_list]
        res_std = [np.std(df[index])*100 / math.sqrt(len(df)) for index in index_list]
        
        for k in range(len(res_mean)):
            output_dta[k][i] = res_mean[k]
            output_dta_std[k][i] = res_std[k]
    sns.set_style('darkgrid')
    plt.figure(figsize = (10, 6), dpi = 100)
    marker_choice = ['*', 'o', '*', 'o', '*', 'o']
    colors=['#4daf4a', '#ff7f00', '#984ea3', '#FFD43B', '#a65628', '#f781bf', '#e41a1c', '#377eb8']
    for j, model_name_setup in enumerate(method_name_list):
        res_mean = output_dta[j]
        res_std = output_dta_std[j]
        for k in range(len(res_mean)):
            if res_mean[k] > 20:
                break
        plt.plot(sample_range[model][k:], res_mean[k:], label = model_name_setup, marker = marker_choice[j], color = colors[j], linewidth = 1)
        plt.errorbar(x = sample_range[model][k:] , y = res_mean[k:], yerr = res_std[k:], color = colors[j], capsize = 2)
    plt.legend(loc = 'lower right')
    plt.xlabel('sample size', size = 14)
    plt.ylabel(r'$R^2 (\%)$', size = 14)
    #plt.yscale('log')
    plt.savefig('../figures/LDW2_' + str(model) + '.pdf') 
            
            
            # compare NP-DRO and P-DRO
    #     output_dta[i][k] = stats.ttest_rel(list(df['2']), list(df['6']))[1]
    # df = pd.DataFrame(output_dta, index = None, columns = ['NP-ERM', 'NP-DRO', 'P-ERM', 'P-DRO', 'pvalue'])
    # df.to_csv(DS_DIR + model + '_' + dtype + '.csv', index = None)
if __name__ == '__main__':
    synthetic2_concept('True')
    #main_simulation(bound = 10, noise_rate = 5)
    
    

