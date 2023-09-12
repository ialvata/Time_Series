from models.model_base import ModelBase
from typing import Generator
from enum import Enum
from preprocessing.feature_engineering.feature_engineering import FeatureEngineering
from evaluation.metrics import MetricBase
import pandas as pd
import numpy as np

class TuningOption(Enum):
    """
    Represents the different possibilities when doing cross validation:

        `NO_FITTING`:
            In this case, we've already fitted a custom model, and defined the hyperparameters.
            We only need to forecast the custom model with the test set.

        `FIT_PARAM_ONLY`:
            We fit the a custom model - with already defined hyperparameters - on the 
            training set and and forecast it on the test set.
        
        `FIT_HYPER_PARAM`:
            We use set the hyperparameters to different values, fit the custom model on the
            train set, and evaluate the hyperparameters on the validation set. Then we forecast
            on the test set.
    """
    FIT_HYPER_PARAM = 1
    FIT_PARAM_ONLY = 2
    NO_FITTING = 3

class TrialResults:
    def __init__(self, 
        forecast:float|pd.DataFrame|pd.Series|np.ndarray, hyperparameters:dict | None = None
    ) -> None:
        self.hyperparameters = hyperparameters
        self.forecast = forecast

class CrossValidation:
    def __init__(self, models:list[ModelBase],metrics: list[MetricBase],
                 data_generator:Generator, feat_eng: FeatureEngineering
                 ) -> None:
        self.models = models
        self.metrics = metrics
        self.data_generator = data_generator
        self.feat_eng = feat_eng
        self.forecasts = None
        self.history = []
        self.observations = []

    def evaluate(self, tuning_option: TuningOption):
        for model in self.models:
            for train_set,test_set  in self.data_generator:
                #####################              Feature Engineering             ####################
                self.feat_eng.set_datasets(train_set = train_set, test_set = test_set)
                # outside of cross validation
                # in the main code self.feat_eng.add_pipeline()
                self.feat_eng.create_features(destiny_set = "train")
                self.feat_eng.create_features(destiny_set = "test")
                ####################               Evaluation type                #####################
                match tuning_option:
                    case TuningOption.FIT_HYPER_PARAM:
                        model.find_best(
                            X_train=self.feat_eng.train_set.features,
                            y_train=self.feat_eng.train_set.labels,
                        )
                        
                        forecast = model.forecast(
                            X_test=self.feat_eng.test_set.features,use_best_model= True
                        )
                        self.history.append(
                            TrialResults(
                                hyperparameters=model.best_hyperparameters,
                                forecast=forecast                                
                            )
                        )
                        
                    case TuningOption.FIT_PARAM_ONLY:
                        model.fit_custom_model(
                            X_train=self.feat_eng.train_set.features,
                            y_train=self.feat_eng.train_set.labels
                        )
                        forecast = model.forecast(
                            X_test=self.feat_eng.test_set.features, use_best_model= False
                        )
                        self.history.append(
                            TrialResults(forecast=forecast)
                        )
                    case TuningOption.NO_FITTING:
                        forecast = model.forecast(
                            X_test=self.feat_eng.test_set.features, use_best_model= False
                        )
                        self.history.append(
                            TrialResults(forecast=forecast)
                        )
        self.observations = self.feat_eng.test_set.labels

    def show_results(self) -> pd.DataFrame:
        forecasts = np.ndarray([trial_res.forecast for trial_res in self.history])
        metrics = {
            f"{metric.name}": metric.compute(
                predictions = forecasts, observations = np.array(self.observations)
            )
            for metric in self.metrics
        }
        table_res = pd.DataFrame(metrics)
        table_res.style.format(
            '{:.4f}'
        ).highlight_min(color='green')
        return table_res
