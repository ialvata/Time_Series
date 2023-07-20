from tqdm import tqdm
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from collections import namedtuple
from typing import NamedTuple
from itertools import product

SeasonalOrder = namedtuple(typename="SeasonalOrder", 
                       field_names=["P","D", "Q","s"])
NonSeasonalOrder = namedtuple(typename="NonSeasonalOrder", 
                       field_names=["p","d", "q"])
class SARIMAXOrder:
    def __init__(self, non_seasonal:NonSeasonalOrder,seasonal:SeasonalOrder) -> None:
        self.non_seasonal = non_seasonal
        self.seasonal = seasonal

class SARIMAXModel:
    def __init__(self, 
                 p_non_seasonal_max: int = 4, q_non_seasonal_max: int = 4,
                 diff_non_seasonal: int = 0,
                 p_seasonal_max:int = 4, q_seasonal_max:int = 4,
                 diff_seasonal: int = 0,
                 frequency_cycle:int = 12) -> None:
        # p
        p_non_season_range = range(0,p_non_seasonal_max)
        # q
        q_non_season_range = range(0,q_non_seasonal_max)
        # P
        p_season_range = range(0,p_seasonal_max)
        # Q
        q_season_range = range(0,q_seasonal_max)
        # d
        self.diff_non_seasonal = diff_non_seasonal
        # D
        self.diff_seasonal = diff_seasonal
        # s
        self.frequency_cycle = frequency_cycle

        cartesian_prod = set(
            product(p_non_season_range,[self.diff_non_seasonal],q_non_season_range,
                    p_season_range,[self.diff_seasonal],q_season_range,[self.frequency_cycle])
        )
        order_list = [
            SARIMAXOrder(
                NonSeasonalOrder(tuple_[0],tuple_[1], tuple_[2]),
                SeasonalOrder(tuple_[3],tuple_[4], tuple_[5], tuple_[6])
            )
            for tuple_ in list(cartesian_prod)
        ]


    def build_model(self, 
                    endogenous_data:pd.DataFrame | pd.Series,
                    exogenous_data: pd.DataFrame | pd.Series,
                    order_list:list[SARIMAXOrder], d:int):
        results = []

        for order in tqdm(order_list):
            model = SARIMAX(endog=endogenous_data, exog=exogenous_data, 
                            order= order.non_seasonal,
                            seasonal_order=order.seasonal)
            fitted_model = model.fit(disp=False)
            aic = fitted_model.aic
            results.append([order, aic])
