import os
wd=os.getcwd()
print(wd)
from preprocessing.stationary import StationaryInput

path_to_data="/home/ivo/Programming_Personal_Projects/Time-Series/datasets/csv/AirPassengers.csv"

StationaryInput(path_to_data)


print("OLá")