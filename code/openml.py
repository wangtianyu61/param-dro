from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import os, wgan
from ml_task_util import Sq_Loss, Distribution_Learner

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DS_DIR = '../data/openml/'

target = {"houses": "median_house_value",
          "house_16H": "price",
          "ailerons": "goal",
          "sulfur": "y1",
          "elevators": "Goal"}

#house_16H and sulfur seems really bad for linear regression.

def arff2csv():
    for filename in target.keys():
        data, meta = arff.loadarff(DS_DIR + filename + '.arff')
        df = pd.DataFrame(data)
        df.to_csv(DS_DIR + filename + '.csv', index = None)  

poly_size = 1
if __name__ == '__main__':
    filenames = ['houses', 'ailerons', 'elevators']
    
    #filenames = list(target.keys())
    for file_name in filenames[1:2]:        
        print(file_name)
        df = pd.read_csv(DS_DIR + file_name + '.csv')
        y_label = target[file_name]
        all_label = set(df.columns)
        all_label.remove(y_label)
        x_data = df[list(all_label)]
        y_data = df[y_label]
        #preprocessing
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2)

        # poly = PolynomialFeatures(1)
        # x_train_new = poly.fit_transform(x_train)
        # x_test_new = poly.fit_transform(x_test)
        dro_model = Sq_Loss(x_train, y_train)
        for eps in [0, 0.01, 0.05, 0.1, 0.5, 1]:
            dro_model.standard_solver(reg = ['W1-2', eps])
            print(eps, dro_model.predict(x_train, y_train), dro_model.predict(x_test, y_test))
        
        wgan_train_num = 5
        wgan_data = []
        perform_metric = np.zeros(wgan_train_num)
        validate_metric = np.zeros(wgan_train_num)
        dist_model = Distribution_Learner(X = x_train, y = y_train, is_regression = True)
        for j in range(wgan_train_num):
            X_train_new, y_train_new = dist_model.conditional_wgan_X_y()
            wgan_data.append({'X': X_train_new, 'y': y_train_new})
            dro_model.standard_solver(reg = False, X = X_train_new, y = y_train_new)
            perform_metric[j] = dro_model.predict(x_train, y_train)
            validate_metric[j] = dro_model.predict(x_test, y_test)
            print(perform_metric[j], validate_metric[j])
        #SELECT THE BEST in sample model from them
        X_train_new = wgan_data[np.argmax(perform_metric)]['X']
        y_train_new = wgan_data[np.argmax(perform_metric)]['y']
        for eps in [0, 0.01, 0.05, 0.1, 0.5, 1]:
            dro_model.standard_solver(robust_param = 0, reg = ['Ridge', eps], X = X_train_new, y = y_train_new)
            print(dro_model.predict(np.array(x_test), np.array(y_test)), eps)
            
        
            
        

        