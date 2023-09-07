from sklearn.metrics import (
    median_absolute_error as mae,
    mean_absolute_percentage_error as mape,
    mean_squared_error as mse,
)
from math import sqrt
from preprocessing.preprocess_input_base import PreprocessInput
from preprocessing.preprocess_base import Preprocess
from preprocessing.roll_windows.roll_window_base import ClassicalWindow
from preprocessing.feature_engineering.feature_engineering import (
    FeatureEngineering, SeasonLength
)
import pandas as pd
from pathlib import Path
from models.random_forest import RandForestModel

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

###############       Dataset splitting into Train and Test set           #####################
rol_fold = ClassicalWindow(air_passengers_cleaned,train_prop = 0.8)
train_test_generator = rol_fold.create_folds()
train_set,test_set = next(train_test_generator)

###############               Feature Engineering                   #####################
feat_eng = FeatureEngineering(train_set)
feat_eng.add_fourier_features([SeasonLength.DAY_OF_YEAR,SeasonLength.MONTH_OF_YEAR])

##############                    Model Instatiation                   #####################
rf_model = RandForestModel(optimization_metric = mse)
rf_model.find_best(train_set.features, train_set.labels)
rf_model.forecast(test_set.features)


print("Olá")