from preprocessing.preprocess_input_base import PreprocessInput
from preprocessing.preprocess_base import Preprocess
from preprocessing.data_loaders.classical_loader import ClassicalLoader
from preprocessing.feature_engineering.feature_engineering import (
    FeatureEngineering
)
from preprocessing.feature_engineering.fourier import FourierFeature,SeasonLength
from preprocessing.feature_engineering.rolling_features import RollingFeature
import pandas as pd
from pathlib import Path
from models.random_forest import RandForestModel
from evaluation.cross_validation import CrossValidation, TuningOption
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
        RollingFeature(columns=["value"], rolling_periods=[3,4])
    ]
)

#################                   Model Instantiation                 #######################
rf_model_1 = RandForestModel(name = "rf_model_1", 
                             optimization_metric = MSE, hyperparameters={"max_depth": 10})
rf_model_2 = RandForestModel(name = "rf_model_2",
                             optimization_metric = MSE, 
                             hyperparameters={"max_features": "log2"})

#################                    Cross-Validation                  ########################
cross_val = CrossValidation(
    models = [rf_model_1,rf_model_2],
    metrics = [MSE, RMSE, MAPE, MAE],
    data_generator = train_test_generator,
    feat_eng = feat_eng, # in the future it will take a list of several different pipelines.
)

cross_val.evaluate(tuning_option = TuningOption.FIT_PARAM_ONLY)
cross_val.show_results() # should return


print("Olá")