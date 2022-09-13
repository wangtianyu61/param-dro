import pandas as pd
import matplotlib.pyplot as plt

DS_DIR = '../result/portfolio/'
def return_to_csv(return_array, name_tag, model_tag):
    df = pd.Series(return_array)
    df.to_csv(DS_DIR + name_tag + '_cv_' + model_tag + '.csv', index = None)

def return_to_plot():
#draw the plot
    pass