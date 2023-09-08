from models.model_base import ModelBase
from typing import Callable,Generator
from enum import Enum
from preprocessing.feature_engineering.feature_engineering import FeatureEngineering

class TuningOption(Enum):
    HYPER_TUNING = 1
    ONLY_TRAINING = 2
    NO_FITTING = 3


class CrossValidation:
    def __init__(self, models:list[ModelBase],metrics: list[Callable],
                 data_generator:Generator, feat_eng_train_test = list[FeatureEngineering]
                 ) -> None:
        self.models = models
        self.metrics = metrics
        self.data_generator = data_generator
        self.feat_eng_train_test = feat_eng_train_test


    def _evaluate_only_training(self):...

    def _evaluate_no_fitting(self):...

    def _evaluate_hyper_tuning(self):...

    def evaluate(self):...

    def show_results(self):...
