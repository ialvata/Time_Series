from preprocessing.preprocess_input_base import PreprocessInput
from preprocessing.preprocess_base import Preprocess
from preprocessing.data_loaders.classical_loader import ClassicalLoader
from preprocessing.feature_engineering.feature_engineering import (
    FeatureEngineering
)
from preprocessing.feature_engineering.fourier import FourierFeature,SeasonLength
import pandas as pd
from pathlib import Path
from models.lightgbm import LGBMModel
from evaluation.metrics import MSE,MAE,MAPE,RMSE

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
train_set,test_set = next(train_test_generator)

###############               Feature Engineering (TrainSet)              #####################
# do we want to change in place in the train_set and test_set? This would be implicit...
feat_eng = FeatureEngineering(dataframe=train_set, labels_names=["value"])
feat_eng.add_to_pipeline(
    features = [
        FourierFeature(
            seasonal_lengths = [SeasonLength.DAY_OF_YEAR,SeasonLength.MONTH_OF_YEAR]
        )
    ]
)
feat_eng.create_features(destiny_set = "feat_eng")
##############                    Model Instatiation                   #####################
rf_model = LGBMModel(optimization_metric = MSE)
rf_model.find_best(feat_eng.features, feat_eng.labels)

##############                    Model Forecast                   #####################
rf_model.forecast(feat_eng.features, use_best_model = True)



print("Olá")