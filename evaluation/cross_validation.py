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

class TrialHistory:
    def __init__(self,
        trial:int,
        model_name:str,
        forecast:np.ndarray, 
        observation: np.ndarray,
        hyperparameters:dict | None = None,
    ) -> None:
        self.hyperparameters = hyperparameters
        self.forecast = forecast
        self.observation = observation
        self.trial = trial
        self.model_name = model_name
    def __repr__(self) -> str:
        return f"TrialHistory(trial={self.trial}, model={self.model_name})"

class CrossValidation:
    def __init__(self, models:list[ModelBase],metrics: list[MetricBase],
                 data_generator:Generator, feat_eng: FeatureEngineering
                 ) -> None:
        self.models = models
        self.metrics = metrics
        self.data_generator = data_generator
        self.feat_eng = feat_eng
        self.forecasts = None
        self.observations = []
        self.histories = {f"{model.name}":[] for model in models}

    def evaluate(self, tuning_option: TuningOption):
        for trial,(train_set,test_set)  in enumerate(self.data_generator):
            print(len(train_set))
            ###################              Feature Engineering              #################
            self.feat_eng.set_datasets(train_set = train_set, test_set = test_set)
            # outside of cross validation
            # in the main code self.feat_eng.add_pipeline()
            self.feat_eng.create_features(destiny_set = "train")
            self.feat_eng.create_features(destiny_set = "test")
            ##################               Evaluation type              #####################
            for model in self.models:
                history = self.histories[f"{model.name}"]
                match tuning_option:
                    case TuningOption.FIT_HYPER_PARAM:
                        model.find_best(
                            X_train=self.feat_eng.train_set.features,
                            y_train=self.feat_eng.train_set.labels,
                        )
                        
                        forecast = model.forecast(
                            X_test=self.feat_eng.test_set.features,use_best_model= True
                        )
                        history.append(
                            TrialHistory(
                                trial=trial,
                                model_name= model.name,
                                hyperparameters=model.best_hyperparameters,
                                forecast=np.array(forecast),
                                observation= np.array(self.feat_eng.test_set.labels)                            
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
                        history.append(
                            TrialHistory(
                                trial=trial, model_name= model.name,
                                forecast=np.array(forecast),
                                observation= np.array(self.feat_eng.test_set.labels)
                            )
                        )
                    case TuningOption.NO_FITTING:
                        forecast = model.forecast(
                            X_test=self.feat_eng.test_set.features, use_best_model= False
                        )
                        history.append(
                            TrialHistory(
                                trial = trial, model_name= model.name,
                                forecast = np.array(forecast),
                                observation = np.array(self.feat_eng.test_set.labels)
                            )
                        )
                self.histories[f"{model.name}"]= history


    def show_results(self, model_names:str | list[str] |None = None) -> pd.DataFrame:
        """
        TODO:
            `model_names:str|list[str]|None = None` 
                We should implement for the case of list[str]-
        """
        if model_names is None:
            model_names = [model.name for model in self.models]
        if isinstance(model_names,str):
            model_names = [model_names]
        if model_names is None:
            raise Exception("List of models associated to this CrossValidation is empty!")
        
        forecasts = {
            f"{model_name}": np.ravel(
                [trial.forecast for trial in self.histories[model_name]]
            ) for model_name in model_names
        }
        observations = {
            f"{model_name}": np.ravel(
                [trial.observation for trial in self.histories[model_name]]
            ) for model_name in model_names
        }
        metrics = {f"{metric.name}": [
            metric.compute(
                predictions = forecasts[model_name], observations = observations[model_name]
            ) for model_name in model_names]for metric in self.metrics
        }

        table_res = pd.DataFrame(metrics, index=model_names)
        table_res.style.format(
                '{:.4f}'
        ).highlight_min(color='green')
        return table_res
