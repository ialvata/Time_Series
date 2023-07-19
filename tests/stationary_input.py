from preprocessing.stationary import StationaryInput
from preprocessing.stationary import Stationary
from preprocessing.preprocess_base import Preprocess
from preprocessing.preprocess_input_base import PreprocessInput
from preprocessing.transformations import Difference

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
TRAIN_PROP = 0.9
test_rows = int((1-TRAIN_PROP) * air_passengers_cleaned.shape[0])
test_set = air_passengers_cleaned.iloc[-test_rows:]
train_set = air_passengers_cleaned.iloc[:-test_rows]

train_input = StationaryInput(train_set)
stat_pipeline = Stationary(train_input)
diff_transform = Difference(periods=3,stationary=stat_pipeline)
diff_df = diff_transform.apply(["value"])
undiff_df = diff_transform.invert(diff_df)
assert (undiff_df == train_set).all().values

print("OLá")