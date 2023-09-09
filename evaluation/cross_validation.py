from models.model_base import ModelBase
from typing import Callable,Generator
from enum import Enum
from preprocessing.feature_engineering.feature_engineering import FeatureEngineering

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


class CrossValidation:
    def __init__(self, models:list[ModelBase],metrics: list[Callable],
                 data_generator:Generator, feat_eng = FeatureEngineering
                 ) -> None:
        self.models = models
        self.metrics = metrics
        self.data_generator = data_generator
        self.feat_eng = feat_eng
        self.forecasts = None


    def _evaluate_fit_param_only(self):
        """
        Method for the case when we use the whole training set to fit a model with 
        hyperparameters already predefined
        """
        ...

    def _evaluate_no_fitting(self):...

    def _evaluate_fit_hyper_param(self):...

    def evaluate(self, tuning_option: TuningOption):
        for model in self.models:
            for train_set,test_set  in self.data_generator:
                #####################              Feature Engineering             ####################
                self.feat_eng.set_datasets(train = train_set, test = test_set)
                self.feat_eng.create_features(train_set)
                self.feat_eng.create_features(test_set)
                ####################               Evaluation type                #####################
                match tuning_option:
                    case TuningOption.FIT_HYPER_PARAM:
                        ...
                        model.forecast(X_test=self.feat_eng.features,use_best_model= True)
                    case TuningOption.FIT_PARAM_ONLY:
                        ...
                        model.forecast(X_test=self.feat_eng.features, use_best_model= False)
                    case TuningOption.NO_FITTING:
                        ...
                        model.forecast(X_test=self.feat_eng.features, use_best_model= False)
                
                        
                        

    def show_results(self):...
