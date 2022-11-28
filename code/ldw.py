from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score, r2_score
import wgan
import pandas as pd
import numpy as np
import os
from ml_task import Sq_Loss, Distribution_Learner_LDW
import warnings

from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import rbf_kernel



from itertools import product
"""
read txt file into csv
"""
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
DS_DIR = '../data/LDW/'
def preprocessing(name_tag):
    label = ['t', 'age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75', 're78']

    data = pd.read_csv(DS_DIR + name_tag + '.txt', sep = ' ', header = None)
    #label = ['treatment', 'age', 'education', 'Black', 'Hispanic', 'married', 'nodegree', 'RE75', 'RE78', 'outcome']
    true_label = list(2*(np.array(list(range(10))) + 1))
    data = data[true_label]
    data.columns = label
    data['re74'] = data['re74']/1000
    data['re75'] = data['re75']/1000
    data['re78'] = data['re78']/1000    
    data.to_csv(DS_DIR + name_tag + '.csv', index = None)

def ldw_reg(name_tag):
    pass

def ldw_normal(x_train, y_train):
    data_train = pd.concat([x_train, y_train], axis = 1)
    
    items = [0, 1]
    category_var = ["black", "hispanic", "married", "nodegree"]
    category_num = len(category_var)
    dict_category = {}
    threshold = 10
    resample_times = 5
    for c in list(product(items, repeat = category_num)):
        data = data_train
        for j in range(category_num):
            data = data[data[category_var[j]] == c[j]]
            data_columns = data.columns
        if data.shape[0] <= threshold:
            newdf = pd.DataFrame(np.repeat(data.values, resample_times, axis = 0))
    
        else:
            data = np.array(data)
            sample_mean = np.mean(data, axis = 0)
            sample_cov = np.cov(data.T)
            newdata = np.random.multivariate_normal(mean = sample_mean, 
                                                    cov = sample_cov,
                                                    size = resample_times * data.shape[0])
            
            newdf = pd.DataFrame(newdata)
        newdf.columns = data_columns
        dict_category[c] = newdf
        #postprocessing
        data_train = pd.concat([data_train, newdf], axis = 0)
    data_train['re74'] = data_train['re74'].apply(lambda x: 0 if x < 0 else x)
    data_train['re75'] = data_train['re75'].apply(lambda x: 0 if x < 0 else x)
    data_train['re78'] = data_train['re78'].apply(lambda x: 0 if x < 0 else x)
    data_train['age'] = data_train['age'].apply(lambda x: 25 if x < 25 else x)
    data_train['age'] = data_train['age'].apply(lambda x: 60 if x > 60 else x)
    x_train_new = data_train[feature]
    y_train_new = data_train[label]
    return x_train_new, y_train_new


     
poly_size = 2
Dist_Shift = 'No'
Eval_Loss = 'l2'
eps_list = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
#eps_list = [1]

regr_setup = 'W2-2'
solver_setup = 'GUROBI'
loop_num = 50
all_test_num = 1500

#problem parameter
feature = ['t', 'age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75']
label = 're78'
names_control = ['nswre74', 'psid', 'cps']
name = names_control[1]
normal_train_num = 2


#pick the best eps for dro model as domain knowledge with a separate dataset, however, we do not augment them as in the real downside training tasks.
def dist_shift_eps_selection(data_train, data_test, PARAM = False):

    eps_list = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]

    poly = PolynomialFeatures(poly_size)
    x_train = data_train[feature]
    x_test = data_test[feature]
    y_train = data_train[label]
    y_test = data_test[label]

    if PARAM == True:
        x_train, y_train = ldw_normal(x_train, y_train)

    X_train_aug = poly.fit_transform(x_train)
    X_test_aug = poly.fit_transform(x_test)
    
    model_perform = np.zeros(len(eps_list))
    
    dro_model = Sq_Loss(np.array(X_train_aug), np.array(y_train), Eval_Loss)
    
    for k, eps in enumerate(eps_list):
        dro_model.standard_solver(reg = [regr_setup, eps], option = solver_setup)
        __, model_perform[k] = dro_model.predict(X_test_aug, y_test)
    best_eps = eps_list[np.argmin(model_perform)]
    print(PARAM, best_eps)
    return best_eps


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category = DataConversionWarning)
    #param_cons = ['Ridge', 1]
    seeds_all = {'Yes': [0], 'No': [0, 1, 2, 9, 10, 11, 14, 21, 27, 30], 'Educ': [0], 'age': [0], 'No-Yes':[0]}
    seeds_choice = seeds_all[Dist_Shift]

    np_erm = np.zeros((loop_num, 2))
    np_dro = np.zeros((loop_num, 2))
    p_erm = np.zeros((loop_num, 2))
    p_dro = np.zeros((loop_num, 2))
    for i in list(range(loop_num)):
        print(i)
        np.random.seed(i)

        #cases of no distribution shift
        if Dist_Shift in ['No', 'No-Yes']:
            data_c = pd.read_csv(DS_DIR + name + '_control.csv', sep = ',')
            #print(data_c.describe())
            #made it as the intercept term
            data_c['t'] = 1
            
            x_data = data_c[feature]
            y_data = data_c[label]
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = all_test_num)
 
            #x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5)
            
            if Dist_Shift == 'No-Yes':
                x_train = x_data
                y_train = y_data
                data_psid = pd.read_csv(DS_DIR + names_control[2] + '_control.csv', sep = ',')
                data_psid['t'] = 1
                x_test = data_psid[feature]
                y_test = data_psid[label]
                
                
              
                
        #situation of distribution shift
        else:
            data_train = pd.read_csv(DS_DIR + name + '_control.csv', sep = ',')
            data_test = pd.read_csv(DS_DIR + name + '_controL_new.csv', sep = ',')
            data_train['t'] = 1
            data_test['t'] = 1
            data_all = pd.concat([data_train, data_test], axis = 0)
            if Dist_Shift == 'Educ':
                data_train = data_all[data_all['education'] >= 10]
                data_test = data_all[data_all['education'] < 10]
            elif Dist_Shift == 'age':
                data_train = data_all[data_all['age'] >= 25]
                data_test = data_all[data_all['age'] < 25]
            all_data_train, ___ = train_test_split(data_train, train_size = int(all_test_num * 1.2))
            
            data_train, data_ct = train_test_split(all_data_train, train_size = 0.8)
            data_test, data_ct_test = train_test_split(data_test, train_size = 0.8)
            best_eps = dist_shift_eps_selection(data_ct, data_ct_test)
            eps_list = [best_eps]
 
            x_train = data_train[feature]
            x_test = data_test[feature]
            y_train = data_train[label]
            y_test = data_test[label]

            #x_train_new, y_train_new = ldw_normal(x_train, y_train)

        dro_model = Sq_Loss(np.array(x_train), np.array(y_train), Eval_Loss)
        # print('Without augmenting features:')
        
        # for eps in eps_list:
            
        #     dro_model.standard_solver(robust_param = 0, reg = ['W1-2', eps])
        #     print(dro_model.predict(np.array(x_train), np.array(y_train)), dro_model.predict(np.array(x_test), np.array(y_test)), eps)
        # print('=================================')
       
        
        poly = PolynomialFeatures(poly_size)
        X_train_aug = poly.fit_transform(x_train)
        X_test_aug = poly.fit_transform(x_test)
        #X_train_kernel = rbf_kernel(x_train)
        
        
        #X_test_kernel = rbf_kernel(x_test)
        ## Nonparam-ERM
        dro_model.standard_solver(reg = [regr_setup, 0], X = X_train_aug, y = y_train, option = solver_setup)
        np_erm[i][0], np_erm[i][1] = dro_model.predict(np.array(X_test_aug), np.array(y_test))
        
        ## Nonparam-DRO
        if Dist_Shift in ['age', 'Educ']:
            best_eps = dist_shift_eps_selection(data_ct, data_ct_test, True)    
            eps_list = [best_eps]
        ### Validation Part
        model_perform = np.zeros(len(eps_list))
        X_tr_aug, X_val_aug, y_tr, y_val = train_test_split(X_train_aug, y_train, test_size = 0.2)
        for k, eps in enumerate(eps_list):
            
            # dro_model.standard_solver(reg = ['W2-2', eps], X = X_train_kernel, y = y_train, option = 'GUROBI')
            # print('kernel: ', dro_model.predict(x_test, y_test, x_train, 'kernel'), eps)
            
            dro_model.standard_solver(reg = [regr_setup, eps], X = X_tr_aug, y = y_tr, option = solver_setup)
            __, model_perform[k] = dro_model.predict(np.array(X_val_aug), np.array(y_val))
        
        best_eps = eps_list[np.argmin(model_perform)]
        dro_model.standard_solver(reg = [regr_setup, best_eps], X = X_train_aug, y = y_train, option = solver_setup)
        np_dro[i][0], np_dro[i][1] = dro_model.predict(np.array(X_test_aug), np.array(y_test))

    #     ####################WGAN##################################
    #     wgan_train_num = 5
    #     wgan_data = []
    #     perform_metric = np.zeros(wgan_train_num)
    #     validate_metric = np.zeros(wgan_train_num)
    #     dist_model = Distribution_Learner_LDW(x_train, y_train, name) 
    
    #     for j in range(wgan_train_num):
    #         X_train_new, y_train_new = dist_model.conditional_wgan_X_y()
    #         wgan_data.append({'X': X_train_new, 'y': y_train_new})
            
            
    #         reg_svr = SVR(C = 1)
    #         reg_svr.fit(X_train_new, y_train_new)
    #         y_pred_new = reg_svr.predict(x_train)
    #         perform_metric[j] = np.mean(np.abs(y_pred_new - y_train))
    #         print('SVR Result: ', perform_metric[j])
    #         #dro_model.standard_solver(robust_param = 0, reg = False, X = X_train_new, y = y_train_new)
    #         #perform_metric[j] = dro_model.predict(x_train, y_train)
    #         #validate_metric[j] = dro_model.predict(x_test, y_test)
    #     print(perform_metric)
    #     #SELECT THE BEST in sample model from them
    #     X_train_new = wgan_data[np.argmin(perform_metric)]['X']
    #     y_train_new = wgan_data[np.argmin(perform_metric)]['y']
    #     print('Without augmenting features:')
        # for eps in eps_list:
            
        #     dro_model.standard_solver(robust_param = 0, reg = ['W1-2', eps], X = x_train_new, y = y_train_new)
        #     print(dro_model.predict(np.array(x_test), np.array(y_test)), eps)
        # print("===============================")
        
        # Param-ERM
        new_data = []
        new_data_tr = []
        
        X_tr, X_val, y_tr, y_val = train_test_split(x_train, y_train, test_size = 0.2)
        X_tr_aug = poly.fit_transform(X_tr)
        
        inner_perform_metric = np.zeros(normal_train_num)
        inner_perform_metric2 = np.zeros(normal_train_num)
        ### select the best by in-sample performance
        for j in range(normal_train_num):
            x_train_new, y_train_new = ldw_normal(x_train, y_train) 
            X_train_aug2 = poly.fit_transform(x_train_new)
            dro_model.standard_solver(reg = [regr_setup, 0], X = X_train_aug2, y = y_train_new, option = solver_setup)
            new_data.append({'X': X_train_aug2, 'y': y_train_new})
            #mark: TODO (Out-of-sample Training)
            __, inner_perform_metric[j] = dro_model.predict(np.array(X_test_aug), np.array(y_test))
            
            X_tr_new, y_tr_new = ldw_normal(X_tr, y_tr)
            X_tr_aug2 = poly.fit_transform(X_tr_new)
            dro_model.standard_solver(reg = [regr_setup, 0], X = X_tr_aug2, y = y_tr_new, option = solver_setup)
            new_data.append({'X': X_tr_aug2, 'y': y_tr_new})
            #mark: TODO
            __, inner_perform_metric2[j] = dro_model.predict(np.array(X_val_aug), np.array(y_val))
            
        X_train_aug2 = new_data[np.argmin(inner_perform_metric)]['X']
        y_train_new = new_data[np.argmin(inner_perform_metric)]['y']
        
        X_tr_aug2 = new_data[np.argmin(inner_perform_metric2)]['X']
        y_tr_new = new_data[np.argmin(inner_perform_metric2)]['y']
        
        ### retrain
        dro_model.standard_solver(reg = [regr_setup, 0], X = X_train_aug2, y = y_train_new, option = solver_setup)

        p_erm[i][0], p_erm[i][1] = dro_model.predict(np.array(X_test_aug), np.array(y_test))
        
        
        ## Param-DRO
        model_perform = np.zeros(len(eps_list))
        for k, eps in enumerate(eps_list):
            # reg = SVR(C = eps + 0.001)
            # reg.fit(x_train_new, y_train_new)
            # y_pred = reg.predict(x_test)
            # print('kernel: ', r2_score(y_test, y_pred))
            dro_model.standard_solver(reg = [regr_setup, eps], X = X_tr_aug2, y = y_tr_new, option = solver_setup)
            __, model_perform[k] = dro_model.predict(np.array(X_val_aug), np.array(y_val))
        best_eps = eps_list[np.argmin(model_perform)]
        dro_model.standard_solver(reg = [regr_setup, best_eps], X = X_train_aug2, y = y_train_new, option = solver_setup)
        p_dro[i][0], p_dro[i][1] = dro_model.predict(np.array(X_test_aug), np.array(y_test))


    print(np_erm, np_dro, p_erm, p_dro)
    res = np.hstack((np_erm, np_dro, p_erm, p_dro))
    df = pd.DataFrame(res)
    if Dist_Shift == 'age':
        df.to_csv('../result/LDW/age_' + name + '_' + str(poly_size) + '_' + str(all_test_num) + '.csv')
    else:
        df.to_csv('../result/LDW/temp_' + name + '_' + str(poly_size) + '_' + str(all_test_num) + '.csv')
























