from typing import Protocol
import pandas as pd

class FeatureBase(Protocol):
    def create_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:...