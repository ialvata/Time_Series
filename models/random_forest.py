
import pandas as pd
import numpy as np
from typing import Callable
from statsmodels.stats.diagnostic import acorr_ljungbox as ljung_box
from sklearn.ensemble import RandomForestRegressor
import optuna


class RandForestModel:
    def __init__(self, optimization_metric: Callable) -> None:
        """
        This class will use by default the RandomForestRegressor by Scikit-Learn.

        To use the best_model, one must first use find_best method.
        
        Parameters
        ----------
        `optimization_metric:Callable`
            This should be a function that returns a float. This function will be used for 
            hyperparameter optimization, in the `find_best` method.
            Note: This function will be MINIMIZED during optimization. 
        """
        self.model = RandomForestRegressor
        self.optimization_metric = optimization_metric
        self.best_model = None
        self.custom_model = None
        self.residuals_test_df = None

    def find_best(
            self, 
            X_train:pd.DataFrame | pd.Series,
            y_train: pd.DataFrame | pd.Series,
            show_progress_bar: bool = True,
            **kwargs
        ):
        """
        Parameters
        ----------
        """
        # defining an objective function to be used in the Optuna optimization study
        def objective(trial: optuna.Trial):     
            params = {
                'n_jobs': -1,
                'criterion':"squared_error", # trial.suggest_categorical(
                    # 'criterion', 
                    # ["absolute_error","squared_error", "friedman_mse", "poisson"]
                # ),
                "n_estimators":500,
                "max_depth":trial.suggest_int('max_depth', 1, 10, 1),
                # "min_weight_fraction_leaf":trial.suggest_float(
                #     "min_weight_fraction_leaf",0.0,0.5, step = 0.01
                # ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 10,
                ),
                # Minimum number of samples required at each leaf node
                "min_samples_leaf" : trial.suggest_int(
                    "min_samples_leaf",1,10
                ),
                # "max_features":trial.suggest_float("max_features",0.001,1.0),
            }
            rf_regr = RandomForestRegressor(**params, random_state=0)
            n_obs = 10
            X_split = X_train[:-n_obs]
            y_split = y_train[:-n_obs]
            X_val = X_train[-n_obs:]
            y_val = y_train[-n_obs:]
            rf_regr.fit(X_split,np.ravel(y_split))
            y_pred = rf_regr.predict(X_val)
            return self.optimization_metric(y_val,y_pred)
        # creating optuna study to find best parameters
        study = optuna.create_study(direction="minimize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective, n_trials=200, show_progress_bar=show_progress_bar, n_jobs=-1
        )
        # defining best model and fitting it.
        self.best_model = RandomForestRegressor(**(study.best_trial.params),random_state=0)
        self.best_model.fit(X_train,np.ravel(y_train))

    def fit_custom_model(self, 
            dataframe:pd.DataFrame | pd.Series,
            exogenous_data: pd.DataFrame | pd.Series | None = None,
            **kwargs
        ):

        self.custom_model= self.model.fit()


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
                return self.custom_model.forecast(
                    steps=steps, signal_only=signal_only, **kwargs
                )
            else:
                print("Please first fit a custom model! Only then run the forecast method.")
    
    def check_residuals(self, lags:int | np.ndarray = [10], 
                       use_best_model:bool = False,
                       plot_diagnostics:bool = False, 
                       **kwargs) -> bool:
        """
        Based on the statsmodels Ljung-Box test of autocorrelation 
        (or dependency) in residuals.
        This method also allows for plotting some basic graphs for simple dignostic about
        the residuals.

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

        Returns
        -------
        bool value
            True, if H0 is not rejected, i.e. residuals are uncorrelated, which is consistent
            with independent residuals. 
        """

        if use_best_model:
            if plot_diagnostics:
                self.best_model.plot_diagnostics(figsize=(10,8))
            residuals = self.best_model.resid
        else:
            if plot_diagnostics:
                self.custom_model.plot_diagnostics(figsize=(10,8))
            residuals = self.custom_model.resid
        self.residuals_test_df = ljung_box(residuals, lags = lags, **kwargs)
        return self.residuals_test_df["lb_pvalue"].iloc[0]>0.05
