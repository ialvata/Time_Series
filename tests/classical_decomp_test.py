from preprocessing.preprocess_base import Preprocess
from preprocessing.preprocess_input_base import PreprocessInput
from preprocessing.data_spliting.set_splitting import Split
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path

path_to_data=Path(
    "/home/ivo/Programming_Personal_Projects/Time_Series/datasets/csv/AirPassengers.csv"
)

input = PreprocessInput(path_to_data)

class AirPassengersPreprocessing(Preprocess):

    def __init__(self,input:PreprocessInput):
        Preprocess.__init__(self,input=input)
        self.cleaned = False

    def clean_dataframe(self):
        if not self.cleaned:
            self.dataframe.drop(["Unnamed: 0", "time"],axis=1, inplace=True)
            time=pd.date_range("1949-01","1961-01",freq="M")
            self.dataframe.index=time
            self.cleaned = True
        else:
            print("Dataframe has already been cleaned")

air_passengers = AirPassengersPreprocessing(input=input)
air_passengers.clean_dataframe()
air_passengers_cleaned = air_passengers.dataframe

################ Dataset splitting into Train and Test set ######################
rol_fold = Split(air_passengers_cleaned)


################            Classical Decomposition           ##################
# import warnings
# warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose

model_results = seasonal_decompose(rol_fold.train_df, period=12)
model_results.plot()