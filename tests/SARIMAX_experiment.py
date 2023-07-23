from preprocessing.stationary import StationaryInput
from preprocessing.stationary import Stationary
from preprocessing.preprocess_base import Preprocess
from preprocessing.preprocess_input_base import PreprocessInput
from preprocessing.transformations import Difference,BoxCox
from models.sarimax import SARIMAXModel
from preprocessing.management import RollingFold
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

################ Dataset splitting into Train and Test set ######################
rol_fold = RollingFold(air_passengers_cleaned,train_prop = 0.8)
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
        box_cox_transform = BoxCox(self)
        box_cox_transform.apply(["value"])
        diff_transform = Difference(periods=1,stationary=self)
        diff_transform.apply(["value"])
        diff_transform_seasonal = Difference(periods=12,stationary=self)
        diff_transform_seasonal.apply(["value"])
        if plot_flag:
            self.plot_data("value")


train_input = StationaryInput(train_set)
stat_pipeline = AirPassengersStationary(train_input)
stat_pipeline.stationarize()
assert stat_pipeline.is_trend_stationarity(["value"], print_output= True)

sarimax = SARIMAXModel()
sarimax.find_best(endogenous_data=stat_pipeline.tranformed_data)
print(sarimax.best_order) 
# SARIMAXOrder(NonSeasonalOrder(p=0, d=0, q=1), SeasonalOrder(P=0, D=0, Q=1, s=12))
# 1958-08-31   -0.028699
print(sarimax.forecast(use_best_model=True))
residuals_df = sarimax.check_residuals(use_best_model=True, plot_diagnostics=True)
############# Sequential rolling forecasts #########################
forecasts = []
rol_fold_stationarized = RollingFold(stat_pipeline.tranformed_data)
stationarized_generator = rol_fold_stationarized.create_folds()
for train,test in stationarized_generator:
    sarimax.fit_custom_model(endogenous_data = train, order = sarimax.best_order)
    forecast = sarimax.forecast()
    forecasts.append(forecast.iloc[0])

#################         Invert transformation on the SARIMAX forescasts        ##############







# box_cox_transform = BoxCox(stat_pipeline)
# box_cox_transform.apply(["value"])
# diff_transform = Difference(periods=1,stationary=stat_pipeline)
# diff_transform.apply(["value"])
# stat_pipeline.plot_data("value")
# diff_transform.invert()
# box_cox_transform.invert()
# stat_pipeline.plot_data("value")
# diff_df = stat_pipeline.tranformed_data
# if diff_df is not None:
#     diff_transform.invert()
#     undiff_df = stat_pipeline.tranformed_data
#     if undiff_df is not None:
#         print("Doing assert for Difference")
#         assert (undiff_df == train_set).all().values

# stat_pipeline_2 = Stationary(train_input)
# box_cox_transform = BoxCox(stat_pipeline_2)
# box_cox_transform.apply(["value"])
# box_cox_df = stat_pipeline_2.tranformed_data
# if box_cox_df is not None:
#     box_cox_transform.invert()
#     invert_box_cox_df = stat_pipeline_2.tranformed_data
#     if invert_box_cox_df is not None:
#         print("Doing assert for BoxCox")
#         assert (invert_box_cox_df - train_input.dataframe < 10**(-12) ).all().values

print("OLá")