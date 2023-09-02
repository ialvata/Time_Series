from preprocessing.stationarity.stationary import StationaryInput
from preprocessing.stationarity.stationary import Stationary
from preprocessing.preprocess_base import Preprocess
from preprocessing.preprocess_input_base import PreprocessInput
from preprocessing.stationarity.transformations import Difference,BoxCox
import matplotlib.pyplot as plt
from models.sarimax import SARIMAXModel,SARIMAXOrder, NonSeasonalOrder, SeasonalOrder
from preprocessing.roll_windows.roll_window_base import ClassicalWindow

import pandas as pd
from pathlib import Path

# import warnings
# warnings.filterwarnings('ignore')

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
rol_fold = ClassicalWindow(air_passengers_cleaned,train_prop = 0.8)
train_test_generator = rol_fold.create_folds()
train_set,test_set = next(train_test_generator)

###############      Stationarizing Data                #########################
class AirPassengersStationary(Stationary):
    _type = "AirPassenger"
    def plot_data(self, column:str, save:bool = False):
        fig, ax = plt.subplots()
        ax.plot(self.tranformed_data.index,self.tranformed_data[column], 
                "g-.", label="Transformed Data")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Passengers")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig(f"{self._type}_plot.png")

    def stationarize(self, plot_flag:bool = False):
        box_cox_transform = BoxCox(stationary=self)
        box_cox_transform.transform(["value"])
        # the transform method will automatically add the transformation 
        # to the stationary transformation pipeline
        diff_transform = Difference(periods=1,stationary=self)
        diff_transform.transform(["value"])
        diff_transform_seasonal = Difference(periods=12,stationary=self)
        diff_transform_seasonal.transform(["value"])
        if plot_flag:
            self.plot_data("value")


train_input = StationaryInput(train_set)
# train_set
#             value
# 1949-01-31    112
# 1949-02-28    118

stat_pipeline = AirPassengersStationary(train_input)
stat_pipeline.stationarize()
# stat_pipeline.tranformed_data
#                value
# 1950-02-28  0.048688
# 1950-03-31  0.000861

sarimax = SARIMAXModel()
# sarimax.find_best(endogenous_data=stat_pipeline.tranformed_data)
# print(sarimax.best_order) 
sarimax.fit_custom_model(
    endogenous_data=stat_pipeline.tranformed_data,
    simple_differencing=True,
    order=SARIMAXOrder(NonSeasonalOrder(p=0, d=0, q=1), SeasonalOrder(P=0, D=0, Q=1, s=12))
)
# SARIMAXOrder(NonSeasonalOrder(p=0, d=0, q=1), SeasonalOrder(P=0, D=0, Q=1, s=12)
# 1958-08-31   -0.028699
# print(sarimax.forecast())
forecast_series = pd.DataFrame(sarimax.forecast(), columns=["value"])
train_plus_forecast_df = pd.concat([stat_pipeline.tranformed_data,forecast_series])

destationarized_df = stat_pipeline.destationarize(train_plus_forecast_df)
if destationarized_df is not None:
    assert (destationarized_df.iloc[:-1].apply(round).astype("int") == train_set).all().values
else:
    Exception("destationarized_df is None!")
# the transformations create very small rounding errors,, which we need to account for.
print("OLá")