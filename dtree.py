# Simple Decision Tree Algorithm
# How to determine optimum threshold
# 1 - Split data into regions with given thresholds
# 2 - Calculate average in each region
# 3 - Calculate error in each region
# 4 - Repeat the steps above for a series of threshold to find optimum threshold

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DecisionTreeRegressor():

    def __init__(self, data:pd.DataFrame, threshold_list:list) -> None:
        assert isinstance(data, pd.DataFrame), f"data accepts DataFrame, got {type(data)}"
        assert isinstance(threshold_list, (list, tuple)), f"threshold_list accepts list or tuple, got {type(threshold_list)}"
        self.data = data
        self.threshold_list = threshold_list
        self.optimum_threshold = None
        
    def split_data(self, data:pd.DataFrame, threshold_x:int) -> pd.DataFrame:
        """Split data into two regions"""
        y_left = data[data['X'] < threshold_x]
        y_right = data[data['X'] >= threshold_x]
        return y_left, y_right
        
    def calc_avg(self, data:pd.DataFrame) -> float:
        """Calculate average -> predict data"""
        return np.mean(data)
    
    def calc_rmse(self, data_pred:float, data_act:np.ndarray) -> float:
        """Calculate RMSE"""
        err = data_act - data_pred
        rmse = (np.mean(err**2))**0.5
        return rmse
    
    @property
    def optimum_threshold(self):
        return self._optimum_threshold
    
    @optimum_threshold.setter
    def optimum_threshold(self, data):
        """Find optimum threshold"""
        opt_thr = 'cek'
        self._optimum_threshold = opt_thr
    

def main():
    data = pd.read_csv('play_tennis.csv')
    thre_list = [1, 2, 3]
    thre_list2 = 1
    dt = DecisionTreeRegressor(data, thre_list)
    print(dt.optimum_threshold)

if __name__ == '__main__':
    main()