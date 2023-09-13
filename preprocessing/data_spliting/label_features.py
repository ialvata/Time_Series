import pandas as pd


class LabelFeatSet:
    def __init__(self, 
                 dataframe:pd.DataFrame,
                 labels_names:list[str], 
                 feature_columns:list[str] | None = None
    ) -> None:
        self.dataframe = dataframe
        self.labels_names = sorted(labels_names)
        self.feature_columns = feature_columns
    
    @property
    def labels(self)-> pd.DataFrame | pd.Series:
        return self.dataframe[self.labels_names]
    
    @property
    def features(self)-> pd.DataFrame | pd.Series:
        # if we do feature engineering, we may want to update self.dataframe after
        # TrainSet initialization. 
        self.feature_columns = self.dataframe.columns.drop(self.labels_names)
        # drop preserves list order.
        return self.dataframe[self.feature_columns]