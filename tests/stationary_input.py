from preprocessing.stationary import StationaryInput
from preprocessing.stationary import Stationary
from preprocessing.preprocess_base import Preprocess
import pandas as pd

path_to_data="/home/ivo/Programming_Personal_Projects/Time_Series/datasets/csv/AirPassengers.csv"

input = StationaryInput(path_to_data)

class AirPassengersPreprocessing(Preprocess, Stationary):

    def __init__(self,input:StationaryInput):
        Preprocess.__init__(self,input=input)
        Stationary.__init__(self,input=input)
        self.cleaned = False

    def clean_dataframe(self):
        if not self.cleaned:
            self.dataframe.drop(["Unnamed: 0"],axis=1, inplace=True)
            time=pd.date_range("1949-01","1961-01",freq="M")
            self.dataframe.index=time
            self.cleaned = True
        else:
            print("Dataframe has already been cleaned")

air_passengers = AirPassengersPreprocessing(input=input)
air_passengers.clean_dataframe()
print("OLá")