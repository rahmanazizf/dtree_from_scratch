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

    def __init__(self, data:pd.DataFrame, threshold_list:list, output_vars:list, input_vars:list=[]) -> None:
        assert isinstance(data, pd.DataFrame), f"data accepts DataFrame, got {type(data)}"
        assert isinstance(threshold_list, (list, tuple)), f"threshold_list accepts list or tuple, got {type(threshold_list)}"
        assert isinstance(input_vars, list), f"input_vars accepts list, got {type(input_vars)}"
        assert isinstance(output_vars, list), f"output_vars accepts list, got {type(output_vars)}"
        self.data = data
        self.threshold_list = threshold_list
        self._input_vars = input_vars
        self._output_vars = output_vars
        self.optimum_threshold = False

    @property
    def input_vars(self):
        return self._input_vars
    
    @input_vars.setter
    def input_vars(self, value):
        if not value:
            self._input_vars = self.data.columns[~self.data.columns.isin(self._output_vars)]
        else:
            self._input_vars = value

    # @property
    # def output_vars(self):
    #     return self._output_vars
    
    # @output_vars.setter
    # def output_vars(self, value):
    #     self._output_vars = value
        
    #TODO : output_vars pake getter dan setter aja kali yak
    def split_data(self, data:pd.DataFrame, threshold_x:int) -> pd.DataFrame:
        """Split data into two regions"""
        data_left = data[data[self.input_vars] < threshold_x]
        data_right = data[data[self.input_vars] >= threshold_x]
        return data_left, data_right
        
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
    def optimum_threshold(self, value):
        """Find optimum threshold"""
        rmsel_list = []
        rmser_list = []
        rmsetot_list = []

        for thre in self.threshold_list:
            data_left, data_right = self.split_data(self.data, thre)

            if len(data_left)==0 or len(data_right)==0:
                pass

            # hitung nilai prediksi
            pred_l = self.calc_avg(data_left[self._output_vars])
            pred_r = self.calc_avg(data_right[self._output_vars])

            rmse_l = self.calc_rmse(pred_l, data_left[self._output_vars])
            rmse_r = self.calc_rmse(pred_r, data_right[self._output_vars])

            rmsel_list.append(rmse_l)
            rmser_list.append(rmse_r)
            rmsetot_list.append(rmse_l+rmse_r)

        rmsel_list = np.array(rmsel_list)
        rmser_list = np.array(rmser_list)
        rmsetot_list = np.array(rmsetot_list)
        self._optimum_threshold = self.threshold_list[np.argmin(rmsetot_list)]

    
    # TODO: tambahkan fungsi untuk menampilkan kurva RMSE vs threshold

def main():
    data = pd.DataFrame({"X": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                     "y": np.array([1, 1.2, 1.4, 1.1, 1.0, 5.5, 6.1, 6.7, 6.4, 6, 5.9, 3, 3.2, 3.1])})
    # buat list untuk variasi threshold
    thre_list = [*range(2, 15, 1)]
    output_vars = ['y']
    dt = DecisionTreeRegressor(data, thre_list, output_vars)
    print(dt.optimum_threshold)
    # hasil yang seharusnya adalah 6

if __name__ == '__main__':
    main()