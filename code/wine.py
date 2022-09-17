import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import os, wgan
from ml_task_util import Sq_Loss, Distribution_Learner
import io
from sklearn.ensemble import RandomForestRegressor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DS_DIR = '../data/wine/'

def load_wine(ws_path):
    with io.open(ws_path, "r", encoding="utf-8") as f:
        pack = []
        nlines = 0
        y_list = []
        for line in f:
            if nlines > 0:
                tokens = line.strip().split(";")
                features = np.array(tokens[0:10]).astype(float)
                labels = int(tokens[11])
                y_list.append(labels)
                pack.append(features)
            nlines += 1
        return pack, y_list


if __name__ == '__main__':
    wine_data_red = DS_DIR + "winequality-red.csv"
    wine_data_white = DS_DIR + "winequality-white.csv"
    red_data, red_eval = load_wine(wine_data_red)
    white_data, white_eval = load_wine(wine_data_white)
    dro_model = Sq_Loss(red_data, red_eval)
    #regr = RandomForestRegressor(max_depth = 5, random_state = 0)
    for eps in [0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]:
        dro_model.standard_solver(robust_param = eps, reg = ['Ridge', 1])
        print(eps, dro_model.predict(red_data, red_eval), dro_model.predict(white_data, white_eval))
    
    #WGAN
    wgan_train_num = 5
    wgan_data = []
    perform_metric = np.zeros(wgan_train_num)
    validate_metric = np.zeros(wgan_train_num)
    dist_model = Distribution_Learner(X = red_data, y = red_eval, is_regression = True)
    for j in range(wgan_train_num):
        X_train_new, y_train_new = dist_model.conditional_wgan_X_y()
        wgan_data.append({'X': X_train_new, 'y': y_train_new})
        dro_model.standard_solver(reg = False, X = X_train_new, y = y_train_new)
        perform_metric[j] = dro_model.predict(red_data, red_eval)
        validate_metric[j] = dro_model.predict(white_data, white_eval)
        print(perform_metric[j], validate_metric[j])
        
    X_train_new = wgan_data[np.argmax(perform_metric)]['X']
    y_train_new = wgan_data[np.argmax(perform_metric)]['y']
    for eps in [0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000]:
        dro_model.standard_solver(robust_param = eps, reg = ['Ridge', 0], X = X_train_new, y = y_train_new)
        print(dro_model.predict(np.array(white_data), np.array(white_eval)), eps)