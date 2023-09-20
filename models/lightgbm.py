import pandas as pd
import numpy as np
from evaluation.metrics import MetricBase
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import optuna
import matplotlib.figure
import lightgbm as lgb
from lightgbm import LGBMRegressor

class LGBMModel:
    def __init__(self, name: str = "LGBMRegressor",
                 optimization_metric: MetricBase | None = None, 
                 hyperparameters:dict = {}) -> None:
        """
        This class will use by default the LGBMRegressor by LightGBM Python package.

        To use the best_model, one must first use find_best method.
        
        Parameters
        ----------
        `optimization_metric:Callable`
            This should be a function that returns a float. This function will be used for 
            hyperparameter optimization, in the `find_best` method.
            Note: This function will be MINIMIZED during optimization.
        `hyperparameters: dict | None = None`
            The entries of this dictionary should be compatible with the input for the
            LGBMRegressor model by LightGBM Python package.
            Check https://lightgbm.readthedocs.io/en/latest/Parameters.html for further
            information on the available parameters.
        """
        self.model = LGBMRegressor
        self.optimization_metric = optimization_metric
        self._best_model = None
        self._custom_model = None
        self.residuals_test_df = None
        self.hyperparameters = hyperparameters
        self.best_hyperparameters = {}
        self.name = name
    
    @property
    def best_model(self):
        if self._best_model is not None:
            return self._best_model
        else:
            raise Exception(
                "Please first set a best model!"
                "Using the find_best method, it will set one for you."
            )

    @best_model.setter
    def best_model(self, model:LGBMRegressor):
        self._best_model = model
    
    @property
    def custom_model(self):
        if self._custom_model is not None:
            return self._custom_model
        else:
            raise Exception(
                "Please first set a custom model!"
            )

    @custom_model.setter
    def custom_model(self, model:LGBMRegressor):
        self._custom_model = model

    def find_best(
            self, 
            X_train:pd.DataFrame | pd.Series,
            y_train: pd.DataFrame | pd.Series,
            X_val:pd.DataFrame | pd.Series | None = None,
            y_val: pd.DataFrame | pd.Series | None = None,
            num_obs_val: int = 10, # This needs to be changed to a proportion
            show_progress_bar: bool = True,
            **kwargs
        )-> None:
        """
        Parameters
        ----------
        """
        if X_val is None:
            X_split = X_train[:-num_obs_val]
            X_val = X_train[-num_obs_val:]
        else:
            X_split = X_train
        if y_val is None:
            y_split = y_train[:-num_obs_val]
            y_val = y_train[-num_obs_val:]
        else:
            y_split = y_train
        if self.optimization_metric is None:
            raise Exception("No optimization metric for LGBMModel! Please set one.")
        # defining an objective function to be used in the Optuna optimization study

        
        def objective(trial: optuna.Trial) -> float:     
            params = {
                'objective': 'regression', # we're in a regression problem 
                "num_threads": 8,
                "verbosity": -1,
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                'num_leaves': trial.suggest_int('num_leaves', 2, 502,10),
                'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10,3),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100,5)
            }
            # Add a callback for pruning = early stopping.
            pruning_callback = optuna.integration.LightGBMPruningCallback(
                trial, "multi_logloss"
            )
            dtrain = lgb.Dataset(X_split, label=y_split)
            lgbmc = lgb.train(params, train_set = dtrain) #verbose_eval=False)
            y_pred = lgbmc.predict(X_val) # returns the class probabilities
            result = self.optimization_metric.compute(y_val,y_pred) #type:ignore
            if isinstance(result, float):
                return result
            else:
                raise Exception("Computing the chosen metric returns a non-floating number")
        # creating optuna study to find best parameters
        study = optuna.create_study(direction="minimize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective, n_trials=200, show_progress_bar=show_progress_bar, n_jobs=-1
        )
        # defining best model and fitting it.
        self.best_model = LGBMRegressor(**(study.best_trial.params),random_state=0)
        self.best_model.fit(X_train,np.ravel(y_train))
        self.best_hyperparameters = study.best_trial.params

    def fit_custom_model(self, 
            X_train:pd.DataFrame | pd.Series,
            y_train:pd.DataFrame | pd.Series,
            hyperparameters: dict = {}
        )-> None:
        """
        Parameters
        ----------
        `kwargs`
            These are keyword arguments compatible with `LGBMRegressor` model 
            by LightGBM Python package, including hyperparameters.
        """
        if hyperparameters is None:
            # if self.hyperparameters is None:
            #     raise Exception("Hyperparameters have not been defined!")
            hyperparameters = self.hyperparameters
        self.custom_model = self.model(**hyperparameters, random_state=0)
        if y_train.shape[1] == 1:
            self.custom_model.fit(X_train,np.ravel(y_train))
        else:
            self.custom_model.fit(X_train,y_train)


    def forecast(self,
                 X_test:pd.DataFrame | pd.Series, 
                 use_best_model:bool = False, **kwargs) -> np.ndarray:
        """
        Forecasts
        ----------------------- 
        In-sample/Out-of-sample forecasts

        Parameters
        ----------
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : array_like
            Aray of forecasts (Numpy array or Pandas Series or DataFrame,
            depending on input).
        """
        if use_best_model:
            return self.best_model.predict(X_test)
        else:
            return self.custom_model.predict(X_test)

    def show_feat_importances(self,
                              columns: list[str],
                              use_best_model:bool = False) -> matplotlib.figure.Figure:
        """
        Parameters
        ----------
        `columns: list[str]`
            This list should contain the columns names which were used during fitting. 
        """
        if use_best_model:
            regr = self.best_model
        else:
            regr = self.custom_model
        importances = regr.feature_importances_
        std = np.std([tree.feature_importances_ for tree in regr.estimators_], axis=0)
        forest_importances = pd.Series(importances, index=columns)

        fig, ax = plt.subplots()
        params = {'legend.fontsize': 'x-large',
                'figure.figsize': (20, 7),
                'axes.labelsize': 'x-large',
                'axes.titlesize':'xx-large',
                'xtick.labelsize':'x-large',
                'ytick.labelsize':'x-large',
                }

        plt.rcParams.update(params)
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        return fig
    
    def check_residuals(self, lags: list[int] | np.ndarray = [10], 
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
                pass
            #     self.best_model.plot_diagnostics(figsize=(10,8))
            # residuals = self.best_model.resid
            raise Exception("Needs Implementation!")
        else:
            if plot_diagnostics:
                pass
            #     self.custom_model.plot_diagnostics(figsize=(10,8))
            # residuals = self.custom_model.resid
            raise Exception("Needs Implementation!")
        # self.residuals_test_df = ljung_box(residuals, lags = lags, **kwargs)
        # return self.residuals_test_df["lb_pvalue"].iloc[0]>0.05
