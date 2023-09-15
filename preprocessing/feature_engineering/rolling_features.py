import numpy as np
import pandas as pd

class RollingFeature:
    _type = "RollingFeature"
    def __init__(self,columns:list[str],
                rolling_periods:int|list[int]) -> None:
            """
            This function will create rolling time features, i.e. features containing averages of
            past values of other features.
            Example
            -------
            Let's assume the columns = ["Cl"]
            If the period is 3, then the current value 
                RollFeat_t = Avg(Cl_{t-1}, Cl_{t-2}, Cl_{t-3})

            Parameters
            ----------
            `rolling_periods:int|list[int]`
                \tIf `rolling_periods` is `int`, then we internally create a list from 1 to \
                `rolling_periods`+1, and use each element as a different period to create a \
                feature.

                \tIf `rolling_periods` is a `list`, then we just use the periods explicitely 
                in the list.

            `columns:list[str]`
                List of columns names for which we want to build the time lags.

            Note
            ----
            This function will transform a matrix of dim -> n_samples x n_columns
            into a matrix of dim -> (n_samples-lags) x (n_columns*lags).
            """
            self.rolling_periods= rolling_periods
            self.columns = columns
            
            if isinstance(self.rolling_periods,int):
                self.periods = range(1,self.rolling_periods+1)
            elif isinstance(self.rolling_periods,list):
                self.periods = self.rolling_periods
            else:
                raise Exception ("rolling_periods of incorrect data type!")
            # the number of rows we'll need to fetch for the train set is the maximum
            # of the periods
            self.num_rows_train = max(self.periods)
            
    def __repr__(self) -> str:
         return (
              f"RollingFeature(rolling_periods = {self.rolling_periods},"
              f"columns = {self.columns})"
         )
    
    def create_features(self, dataframe: pd.DataFrame, 
                        auxiliary_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Parameters
        ----------
            `auxiliary_df: pd.DataFrame | None = None`
                This auxiliary dataframe will be the train_set of the FeatureEngineering class.
                It will be used to fetch the last rows of the train_set to avoid NaNs in the 
                resulting test_set, while keeping the size of the test_set.
        """
        if auxiliary_df is not None:
           dataframe = pd.concat(
               [auxiliary_df.iloc[
                   -self.num_rows_train:,
                   # we want to retrieve only the feature columns for which we want to create
                   # the rolling features.
                   [list(dataframe.columns).index(col) for col in self.columns]
                ], dataframe],
                axis=0)
        appended_lags = []
        for period in self.periods:
            rolling_df = dataframe[self.columns]
            rolling_df.columns=[x+"_roll_avg_"+str(period) for x in self.columns]
            # the shift is needed, because if we use just the rolling, we get
            #       RollFeat_t = Avg(Cl_{t}, Cl_{t-1}, Cl_{t-2})
            # but we want:
            #       RollFeat_t = Avg(Cl_{t-1}, Cl_{t-2}, Cl_{t-3})
            # this we we don't have data leakage from using Cl_{t}
            rolling_df = rolling_df.shift(1).rolling(window = period).mean()
            appended_lags.append(rolling_df)
        full_list_df = [dataframe] + appended_lags
        return pd.concat(full_list_df, axis=1).dropna()