from preprocessing.stationary import StationaryInput
from preprocessing.stationary import Stationary
from preprocessing.preprocess_base import Preprocess
from preprocessing.preprocess_input_base import PreprocessInput
from preprocessing.transformations import Difference,BoxCox

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
TRAIN_PROP = 0.9
test_rows = int((1-TRAIN_PROP) * air_passengers_cleaned.shape[0])
test_set = air_passengers_cleaned.iloc[-test_rows:]
train_set = air_passengers_cleaned.iloc[:-test_rows]

################      Stationarizing Data                #########################
# class AirPassengersStationary(Stationary):
#     def stationarize(self):
#         self.input.dataframe
#         diff_transform = Difference(periods=1,stationary=self)


train_input = StationaryInput(train_set)
stat_pipeline = Stationary(train_input)
diff_transform = Difference(periods=3,stationary=stat_pipeline)
diff_transform.apply(["value"])
diff_df = stat_pipeline.tranformed_data
if diff_df is not None:
    diff_transform.invert()
    undiff_df = stat_pipeline.tranformed_data
    if undiff_df is not None:
        print("Doing assert for Difference")
        assert (undiff_df == train_set).all().values

stat_pipeline_2 = Stationary(train_input)
box_cox_transform = BoxCox(stat_pipeline_2)
box_cox_transform.apply(["value"])
box_cox_df = stat_pipeline_2.tranformed_data
if box_cox_df is not None:
    box_cox_transform.invert()
    invert_box_cox_df = stat_pipeline_2.tranformed_data
    if invert_box_cox_df is not None:
        print("Doing assert for BoxCox")
        assert (invert_box_cox_df - train_input.dataframe < 10**(-12) ).all().values

print("OLá")