import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DS_DIR = '../result/portfolio/'
def return_to_csv(return_array, name_tag, model_tag, model_suffix = None):
    df = pd.Series(return_array)
    df.to_csv(DS_DIR + name_tag + '_cv_' + model_tag + model_suffix + '.csv', index = None)

def return_to_plot():
    #draw the plot    
    DS_DIR = '../result/portfolio'    
    width = 0.12  # the width of the bars
    lw_weight = -2
    
    labels = ['10-In','6-FF', '30-In', '25-FF']
    true_label = ['10_Industry','6_FF', '30_Industry', '25_FF']
    x = np.arange(len(labels))  # the label locations
    colors=['#4daf4a', '#ff7f00', '#984ea3', '#FFD43B', '#a65628', '#f781bf', '#e41a1c', '#377eb8']
    model_names = ['NP-ERM (Empirical)', 'NP-DRO (Empirical)', 
                   'P-ERM (beta)', 'P-DRO (beta)', 'P-ERM (normal)', 'P-DRO (normal)']
    true_model_names = ['ERM_noparam', 'DRO_noparam', 'ERM_beta', 'DRO_beta',
                        'ERM_normal', 'DRO_normal']
    fig, ax = plt.subplots(figsize=(10, 6))
    # for m_name in model_names:
    #     for
    #     mean_dict[] = 
    mean_dict = []
    rect = {}
    for (i,m) in enumerate(model_names):
        rect[m] = ax.bar(x + (-1.1+i)*width, np.array(mean_dict[m]), width, 
                         label = model_names[i], color=colors[i])
                         # yerr = 0, 
                         # error_kw=dict(lw=0.5, capsize=1, capthick=-1))    
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'Empirical $\hat{h}$', fontsize=15)
    ax.set_xlabel('Dataset Name', fontsize=15)
    # ax.set_title('Warfarin, CVaR metric at varying thresholds')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.legend(loc='upper left', fontsize=14)
    

    plt.savefig('../figures/portfolio_' + str(lw_weight), bbox_inches="tight")
    pass

if __name__ == '__main__':
    DS_DIR = '../result/portfolio/'      
    true_label = ['10_Industry','6_FF', '30_Industry', '25_FF']
    true_model_names = ['ERM_noparam', 'DRO_noparam', 'ERM_beta', 'DRO_beta',
                        'ERM_normal', 'DRO_normal']
    def res_gen(lw_weight):
        output_data = np.zeros((len(true_label), len(true_model_names) + 5))
        for j, dset in enumerate(true_label):
            res_dset = [[] for i in range(len(true_model_names))]
            for i, model in enumerate(true_model_names):
                df = pd.read_csv(DS_DIR + dset + '_cv_' + model + '_' + str(lw_weight) + '_60.csv')
                temp = list(df['0'])
                res_dset[i] = [max(5-el, 0)**2 for el in temp]
            for i in range(len(true_model_names)):                    
                output_data[j][i] = np.mean(res_dset[i])
            #compare the three DRO with their own ERM
            for i in range(3):
                output_data[j][len(true_model_names) + i] = stats.ttest_rel(res_dset[2*i + 1], res_dset[2*i])[1]
            #compare beta-dro, normal-dro with empirical-dro:
            for i in range(2):
                output_data[j][len(true_model_names) + 3 + i] = stats.ttest_rel(res_dset[2*i + 3], res_dset[1])[1]
        return output_data
    a1 = res_gen(-2)
    a2 = res_gen(-10)
    output_dta = [([0] * 11) for i in range(len(true_label))]
    for i in range(len(true_label)):
        for j in range(len(true_model_names)):
            output_dta[i][j] = str(round(a1[i][j],2)) + ' (' + str(round(a2[i][j]/a1[i][j], 2)) + ')'
        for j in range(len(true_model_names), len(true_model_names) + 5):
            output_dta[i][j] = a1[i][j]
            
    df_out = pd.DataFrame(output_dta)
    df_out.to_csv(DS_DIR + 'all_portfolio_summary.csv')
    
