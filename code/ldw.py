from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
import wgan
import pandas as pd
import numpy as np
import os
from ml_task import Sq_Loss, Distribution_Learner_LDW

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

poly_size = 1
if __name__ == '__main__':
    param_cons = ['Ridge', 1]
    seeds_choice = [0, 1, 2, 9, 10, 11, 14, 21, 27, 30]
    
    for i in seeds_choice:
        print(i)
        np.random.seed(i)
        feature = ['t', 'age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75']
        label = 're78'
        names_control = ['nswre74', 'psid', 'cps']
        name = names_control[1]
        data_c = pd.read_csv(DS_DIR + name + '_control.csv', sep = ',')
        #print(data_c.describe())
        #made it as the intercept term
        data_c['t'] = 1

        x_data = data_c[feature]
        y_data = data_c[label]
        poly = PolynomialFeatures(2)
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2)
        dro_model = Sq_Loss(np.array(x_train), np.array(y_train))
        print('Without augmenting features:')
        for eps in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
            dro_model.standard_solver(robust_param = 0, reg = ['W1-2', eps])
            print(dro_model.predict(np.array(x_train), np.array(y_train)), dro_model.predict(np.array(x_test), np.array(y_test)), eps)
        print('=================================')
        print('With augmenting features:')
        X_train_aug = poly.fit_transform(x_train)
        x_test_aug = poly.fit_transform(x_test)
        for eps in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
            dro_model.standard_solver(robust_param = 0, reg = ['W1-2', eps], X = X_train_aug, y = y_train)
            print(dro_model.predict(np.array(x_test_aug), np.array(y_test)), eps)


        ######################WGAN##################################
        wgan_train_num = 10
        wgan_data = []
        perform_metric = np.zeros(wgan_train_num)
        validate_metric = np.zeros(wgan_train_num)
        dist_model = Distribution_Learner_LDW(x_train, y_train, name) 
        for j in range(wgan_train_num):
            X_train_new, y_train_new = dist_model.conditional_wgan_X_y()
            wgan_data.append({'X': X_train_new, 'y': y_train_new})
            dro_model.standard_solver(robust_param = 0, reg = False, X = X_train_new, y = y_train_new)
            perform_metric[j] = dro_model.predict(x_train, y_train)
            validate_metric[j] = dro_model.predict(x_test, y_test)
        print(perform_metric, validate_metric)
        #SELECT THE BEST in sample model from them
        X_train_new = wgan_data[np.argmin(perform_metric)]['X']
        y_train_new = wgan_data[np.argmin(perform_metric)]['y']
        print('Without augmenting features:')
        for eps in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
            dro_model.standard_solver(robust_param = 0, reg = ['W1-2', eps], X = X_train_new, y = y_train_new)
            print(dro_model.predict(np.array(x_test), np.array(y_test)), eps)
        print("===============================")
        print('With augmenting features:')
        X_train_aug = poly.fit_transform(X_train_new)
        x_test_aug = poly.fit_transform(x_test)
        for eps in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
            dro_model.standard_solver(robust_param = 0, reg = ['W1-2', eps], X = X_train_aug, y = y_train_new)
            print(dro_model.predict(np.array(x_test_aug), np.array(y_test)), eps)
        #dro_model.standard_solver(reg = param_cons, X = X_train_new, y = y_train_new)
        #print(dro_model.predict(np.array(x_test), np.array(y_test)))


        print('================')




