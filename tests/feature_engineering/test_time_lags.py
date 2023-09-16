
from preprocessing.preprocess_base import Preprocess
from preprocessing.preprocess_input_base import PreprocessInput
from preprocessing.feature_engineering.feature_engineering import (
    FeatureEngineering
)
from preprocessing.data_loaders.classical_loader import ClassicalLoader
from preprocessing.feature_engineering.feature_engineering import (
    FeatureEngineering
)
from preprocessing.feature_engineering.fourier import FourierFeature,SeasonLength
from preprocessing.feature_engineering.time_lags import LagFeature

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

###############       Dataset splitting into Train and Test set           #####################
rol_fold = ClassicalLoader(air_passengers_cleaned, labels_names=["value"],train_prop = 0.8)
train_test_generator = rol_fold.create_folds()
train_set, test_set = next(train_test_generator)
###############               Feature Engineering (TrainSet)              #####################
# do we want to change in place in the train_set and test_set? This would be implicit...
# setting the dataset on which to create the features should be allowed to be done at a later
# stage, such as inside the CrossValidation class.
feat_eng = FeatureEngineering(labels_names=["value"])
feat_eng.add_to_pipeline(
    features = [
        FourierFeature(
            seasonal_lengths = [SeasonLength.DAY_OF_YEAR,SeasonLength.MONTH_OF_YEAR]
        ),
        # Feature below will result in error since 'sin_term_4_MONTH_OF_YEAR' is not part of 
        # test_set original columns.
        # LagFeature(columns=["value",'sin_term_4_MONTH_OF_YEAR'], lags=[3,4])
        LagFeature(columns=["value"], lags=[3,4])
    ]
)
feat_eng.set_datasets(train_set=train_set,test_set=test_set)
feat_eng.create_features(destiny_set="test")
feat_eng.test_set.dataframe 
# has 1 row, which was the size of the original test_set
# has 23 columns: value + 20 trig columns + lag 3 + lag 4
print("OLA")