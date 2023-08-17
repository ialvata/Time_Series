from preprocessing.preprocess_base import Preprocess
from preprocessing.preprocess_input_base import PreprocessInput
from preprocessing.data_spliting.set_splitting import Split
from preprocessing.feature_engineering.feature_engineering import FeatureEngineering, TrigSeason
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

feat_eng = FeatureEngineering(air_passengers_cleaned)
feat_eng.add_trig_season(TrigSeason.YEAR)
# feat_eng.dataframe.plot.scatter("YEAR_sin_season","YEAR_cos_season").set_aspect("equal")

# Usually data splitting is done before feature engineering, but in this case it's a simple way
# to test with multiple column dataframe.
data_split = Split(feat_eng.dataframe, label_columns=["value"])
data_split.test_labels
data_split.test_labels
data_split.val_labels
assert (
    len(data_split.train_df) + len(data_split.test_df)+len(data_split.val_df) 
    == 
    feat_eng.dataframe.shape[0]
)
print("OLA")