from tqdm import tqdm
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from collections import namedtuple
from typing import NamedTuple
from itertools import product
from  math import inf as INFINITY
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox as ljung_box

SeasonalOrder = namedtuple(typename="SeasonalOrder", 
                       field_names=["P","D", "Q","s"])
NonSeasonalOrder = namedtuple(typename="NonSeasonalOrder", 
                       field_names=["p","d", "q"])
class SARIMAXOrder:
    def __init__(self, non_seasonal:NonSeasonalOrder,seasonal:SeasonalOrder) -> None:
        self.non_seasonal = non_seasonal
        self.seasonal = seasonal
    def __repr__(self) -> str:
        return f"SARIMAXOrder({self.non_seasonal}, {self.seasonal})"
    
class SARIMAXModel:
    def __init__(self, 
                 p_non_seasonal_max: int = 3, q_non_seasonal_max: int = 3,
                 diff_non_seasonal: int = 0,
                 p_seasonal_max:int = 3, q_seasonal_max:int = 3,
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
        self.order_list = [
            SARIMAXOrder(
                NonSeasonalOrder(tuple_[0],tuple_[1], tuple_[2]),
                SeasonalOrder(tuple_[3],tuple_[4], tuple_[5], tuple_[6])
            )
            for tuple_ in list(cartesian_prod)
        ]
        self.best_order = None
        self.best_model = None
        self.custom_model = None

    def find_best(
            self, 
            endogenous_data:pd.DataFrame | pd.Series,
            exogenous_data: pd.DataFrame | pd.Series | None = None,
            order_list:list[SARIMAXOrder] | None = None,
            simple_differencing:bool = False,
            **kwargs
        ):
        """
        Parameters
        ----------

        `simple_differencing`
        
        If simple_differencing = True is used, then the endog and exog data are differenced 
        prior to putting the model in state-space form. This has the same effect as if the 
        user differenced the data prior to constructing the model, which has implications for 
        using the results:
        * Forecasts and predictions will be about the differenced data, not about the 
            original data. (while if simple_differencing = False is used, then forecasts and 
            predictions will be about the original data).

        * If the original data has an Int64Index, a new RangeIndex will be created for the 
            differenced data that starts from one, and forecasts and predictions will use this 
            new index.

        """
        
        best_aic = INFINITY
        if order_list is None:
            order_list = self.order_list
        for order in tqdm(order_list):
            model = SARIMAX(endog=endogenous_data, exog=exogenous_data, 
                            order= order.non_seasonal,
                            seasonal_order=order.seasonal,
                            simple_differencing = simple_differencing,
                            **kwargs)
            fitted_model = model.fit(disp=False)
            aic = fitted_model.aic
            if aic < best_aic:
                best_aic = aic
                self.best_model = fitted_model
                self.best_order = order


    def forecast(self,steps=1, signal_only=False, use_best_model:bool = False, **kwargs):
        """
        Out-of-sample forecasts
        ----------------------- 
        (The fitted model also has a predict method for interpolation
        /in-sample forecasts)

        This was taken from the forecast docstring of the `MLEModel` class.

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default is 1.
        signal_only : bool, optional
            Whether to compute forecasts of only the "signal" component of
            the observation equation. Default is False. For example, the
            observation equation of a time-invariant model is
            :math:`y_t = d + Z \alpha_t + \varepsilon_t`, and the "signal"
            component is then :math:`Z \alpha_t`. If this argument is set to
            True, then forecasts of the "signal" :math:`Z \alpha_t` will be
            returned. Otherwise, the default is for forecasts of :math:`y_t`
            to be returned.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : array_like
            Out-of-sample forecasts (Numpy array or Pandas Series or DataFrame,
            depending on input and dimensions).
            Dimensions are `(steps x k_endog)`.

        See Also
        --------
        predict
            In-sample predictions and out-of-sample forecasts.
        get_forecast
            Out-of-sample forecasts and results **including** confidence intervals.
        get_prediction
            In-sample predictions / out-of-sample forecasts and results
            including confidence intervals.
        """
        if use_best_model:
            if self.best_model is not None:
                return self.best_model.forecast(steps=steps, signal_only=signal_only, **kwargs)
            else:
                print("Please first find a best model! Only then run the forecast method.")
        else:
            if self.custom_model is not None:
                pass
            else:
                print("Please first fit a custom model! Only then run the forecast method.")
    
    def test_residuals(self, lags:int | np.ndarray = [10], 
                       use_best_model:bool = False,
                       plot_diagnostics:bool = False, 
                       **kwargs) -> pd.DataFrame:
        """
        Based on the statsmodels Ljung-Box test of autocorrelation in residuals.

        Parameters
        ----------
        lags : {int, array_like}, default None
            If lags is an integer then this is taken to be the largest lag
            that is included, the test result is reported for all smaller lag
            length. If lags is a list or array, then all lags are included up to
            the largest lag in the list, however only the tests for the lags in
            the list are reported. If lags is None, then the default maxlag is
            min(10, nobs // 5). The default number of lags changes if period
            is set.
        boxpierce : bool, default False
            If true, then additional to the results of the Ljung-Box test also the
            Box-Pierce test results are returned.
        """

        if use_best_model:
            if plot_diagnostics:
                self.best_model.plot_diagnostics(figsize(10,8))
            residuals = self.best_model.resid
        else:
            if plot_diagnostics:
                self.custom_model.plot_diagnostics(figsize(10,8))
            residuals = self.custom_model.resid
        return ljung_box(residuals, lags = lags, **kwargs)
