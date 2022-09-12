"""
ref from 
https://github.com/sorooshafiee/Regularization-via-Transportation/blob/master/table2_code.ipynb
"""
from sklearn.datasets import load_svmlight_file, load_svmlight_files
import glob
from ml_task_util import *
from kernel_ml_task_util import *
import pandas as pd
import numpy as np
import warnings

from sklearn.exceptions import ConvergenceWarning
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category = FutureWarning)
    warnings.filterwarnings("ignore", category = ConvergenceWarning)
    
    DIR_DATA = '../data/UCI/'
    
    FILE_NAMES = glob.glob(DIR_DATA + "*.txt")
    FILE_NAMES = [fname for fname in FILE_NAMES if '_test.txt' not in fname]
    
    for fname in FILE_NAMES[0:4]:
        print(fname[len(DIR_DATA):-4])
        try:
            X_train, y_train, X_test, y_test = load_svmlight_files(
                (fname, fname[:-4] + '_test.txt'))
            X_train = X_train.todense()
            X_test = X_test.todense()
            labels = np.unique(y_train)
            y_train[y_train == labels[0]] = -1
            y_train[y_train == labels[1]] = 1
            y_test[y_test == labels[0]] = -1
            y_test[y_test == labels[1]] = 1
            is_test = True
        except FileNotFoundError:
            data = load_svmlight_file(fname)
            X_data = data[0]
            y_data = data[1]
            X_data = X_data.todense()
            labels = np.unique(y_data)
            y_data[y_data == labels[0]] = -1
            y_data[y_data == labels[1]] = 1
            is_test = False
        results = []
        if is_test:
            uci_classification_kernel(X_train, y_train, X_test, y_test)
        else:
            uci_classification_kernel(X_data, y_data)