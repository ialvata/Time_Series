import numpy as np
import pandas as pd
from enum import IntEnum


class EWMAFeature:
    _type = "EWMAFeature"
    def __init__(self,
                 columns:list[str],
                 alphas:list[float] | None = None,
                 spans:list[int] | None = None,
    ) -> None:
            """
            This function will create Exponentially Weighted Moving Average features. While the
            moving average considers a rolling window, the EWMA considers every past 
            observation as part of a weighted average, determined by \\alpha: 
                EWMA_t  = \\alpha y_t + (1-\\alpha)EWMA_{t-1}

            Parameters
            ----------
            `columns: list[str] | None = None`
                List of columns names for which we want to build the Fourier terms
                If `None`, then we automatically create the EWMA terms for all the DataFrame.

            `alphas: list[0|1, ..., }] | None = None`
                List of alphas terms for each new EWMA feature.
            `spans: list[int] | None = None`
                List of spans terms for each new EWMA feature.
                Span is approximately the number of periods after which the weights approach 
                faster zero.

            Note
            ----
            Either alphas or spans must not be None.
            """
            self.columns = columns
            if alphas is not None and (max(alphas)>1 or min(alphas)<0):
                 raise Exception("alpha values should be in -> ]0,1] !")
            self.alphas = alphas
            
            self.spans = spans
            if self.alphas is None and self.spans is None:
                raise Exception("Both alphas and spans are None! Please define at least one.")
            
    def __repr__(self) -> str:
         return (
              f"{self._type}(columns = {self.columns},"
              f"alphas = {self.alphas}, spans = {self.spans})"
         )
    
    def create_features(self, dataframe: pd.DataFrame,
                        auxiliary_df: pd.DataFrame | None = None) -> pd.DataFrame:
            """
            Parameters
            ----------
                `auxiliary_df: pd.DataFrame | None = None`
                    This auxiliary dataframe will be the train_set of the FeatureEngineering 
                    class. It will be used to fetch the last row of the train_set to avoid 
                    resetting the previous time-step of EWMA, which would lead to an 
                    inconsistent pipeline.
            """
            original_start_rows = 0
            if auxiliary_df is not None:
                dataframe = pd.concat(
                    [auxiliary_df.iloc[
                        -2:,
                        # we want to retrieve only the feature columns for which we want to create
                        # the rolling features.
                        [list(dataframe.columns).index(col) for col in self.columns]
                    ], dataframe],axis = 0)
                original_start_rows = 2
            appended_features = []
            if self.alphas is not None:
                 keyword_values = self.alphas
            elif self.spans is not None:
                 keyword_values = self.spans
            else:
                 raise Exception(
                      "Both alphas and spans are None! Please define at least one."
                )
            for value in keyword_values:
                rolling_df = dataframe[self.columns]
                rolling_df.columns=[x+"_ewma_"+str(value) for x in self.columns]
                if self.alphas is not None:
                     rolling_df = rolling_df.shift(1).ewm(alpha = value).mean()
                if self.spans is not None:
                     rolling_df = rolling_df.shift(1).ewm(span = value).mean() 
                appended_features.append(rolling_df)
            full_list_df = [dataframe] + appended_features
            return pd.concat(full_list_df, axis=1).iloc[original_start_rows:]
        