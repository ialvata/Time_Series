from preprocessing.preprocess_base import Preprocess
from preprocessing.preprocess_input_base import PreprocessInput
from pathlib import Path
from pandas import DatetimeIndex,to_datetime
from preprocessing.set_splitting import Split



#####################      Downloading first time, only       #################################

# from zipfile import ZipFile
# import os
# import keras

# uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
# zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
# zip_file = ZipFile(zip_path)
# zip_file.extractall()
# csv_path = "jena_climate_2009_2016.csv"

path_to_data=Path(
    "/home/ivo/Programming_Personal_Projects/Time_Series/datasets/csv/"
)/"temperature_jena_climate_2009_2016.csv"

input = PreprocessInput(path_to_data)


class AirTemperature(Preprocess):
    def __init__(self,input:PreprocessInput):
        Preprocess.__init__(self,input=input)
        self.cleaned = False

    def clean_dataframe(self):
        if not self.cleaned:
            self.dataframe.index = DatetimeIndex(
                # use of to_datetime makes the conversion must faster
                to_datetime(
                    self.dataframe["Date Time"],
                    format="%d.%m.%Y %H:%M:%S"
                ))
            self.dataframe.drop(["Date Time"],axis=1, inplace=True)
            self.cleaned = True
        else:
            print("Dataframe has already been cleaned")

air_temp = AirTemperature(input)
air_temp.clean_dataframe()
air_temp.plot_heatmap()

data_split = Split(air_temp.dataframe, label_columns=["T (degC)"])
data_split.test_labels
data_split.test_labels
data_split.val_labels
assert (
    len(data_split.train_df) + len(data_split.test_df)+len(data_split.val_df) 
    == 
    air_temp.dataframe.shape[0]
)
print("ola")