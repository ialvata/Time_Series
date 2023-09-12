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
        # I'm not sure... Provisional pattern.
        columns_set = set(self.dataframe.columns)
        self.feature_columns = sorted(
            list(columns_set.difference(self.labels_names))
        )
        return self.dataframe[self.labels_names]