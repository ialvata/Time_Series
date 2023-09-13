import numpy as np
import pandas as pd
from enum import IntEnum

class SeasonLength(IntEnum):
    """
    Length of the seasonality cycle.
    For more information, read the docstring in `_calculate_fourier_terms` method, below.
    """
    DAY_OF_YEAR = 366 # DateTimeIndex.day_of_year
    WEEK_OF_YEAR = 52 # DateTimeIndex.isocalendar().week
    MONTH_OF_YEAR = 12 # DateTimeIndex.month
    DAY_OF_WEEK = 7 # DateTimeIndex.day_of_week
    WEEK_OF_MONTH = 5
    DAY_OF_MONTH = 31 # DateTimeIndex.day


class FourierFeature:
    _type = "FourierFeature"
    def __init__(self,seasonal_lengths:list[SeasonLength],
                 columns:list[str] | None = None,
                 num_terms:list[int] | None = None
    ) -> None:
            """
            This function will create Fourier features to represent explicitely time as a 
            regressor.

            Parameters
            ----------
            `seasonal_lengths:list[SeasonLength]`

            `columns:list[str]| None = None`
                List of columns names for which we want to build the Fourier terms
                If `None`, then we automatically create the Fourier terms from the DataFrame index,
                when it's a DateTimeIndex.

            `terms:list[int] | None = None`
                List of numbers of Fourier terms for each element of seasonal length.

            Note
            ----
            seasonal_lengths, n_terms, and columns should match on the element order.

            TODO:
                Implement case for SeasonLength.WEEK_OF_MONTH.
            """
            self.seasonal_lengths = seasonal_lengths
            self.columns = columns
            if num_terms is None:
                # Unless otherwise stated, we'll use 5 as the number of Fourier Terms
                # for all new fourier features.
                self.num_terms = [5]*len(seasonal_lengths)
            else:
                self.num_terms = num_terms
    def __repr__(self) -> str:
         return (
              f"FourierFeature(seasonal_lengths = {self.seasonal_lengths},"
              f"columns = {self.columns}, num_terms = {self.num_terms})"
         )
    def _calculate_fourier_terms(self,
        seasonal_cycle: np.ndarray, cycle_len: int, n_fourier_terms: int
    ):
        """
        Calculates Fourier Terms which will be used as features.

        Parameters
        ----------
        `seasonal_cycle`:
            For a day-of-week cycle, we expect an iterable: 
                [1, 2,..., 7, 1, 2,...]
                Here `cycle_len` is 7.
            For a day-of-month cycle, we expect: 
                [1, 2, ..., 31, 1,...]
                Here `cycle_len` is 31. 
                Note that not all months will have the same length.
            For a day-of-year cycle, we expect: 
                [1,...,366,1,...]
                Here `cycle_len` is 366.
                Note that not all years will have the same length.
            For month-of-year cycle, we expect:
                [1, 2,..., 12, 1,...]
                Here `cycle_len` is 12.
        """
        sin_array = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
        cos_array = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
        for n in range(1, n_fourier_terms + 1):
            # Fourier(x) =  sum_{i=1}^{N} A_n * cos(2*pi / P * n * x) + B_n sin(2*pi/P*n*x)
            # seasonal_cycle array gives us the x values.
            sin_array[:, n - 1] = np.sin((2 * np.pi * seasonal_cycle * n) / cycle_len)
            cos_array[:, n - 1] = np.cos((2 * np.pi * seasonal_cycle * n) / cycle_len)
        return np.hstack([sin_array, cos_array])

    def create_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(dataframe.index,pd.DatetimeIndex):
                    raise Exception(
                        "Please convert (internal) DataFrame Index to a DatetimeIndex!"
                    )
            fourier_columns = []
            fourier_names = []
            if self.columns is None:
                for season_len,num_term in zip(self.seasonal_lengths,self.num_terms):
                    sin_columns = [
                        f"sin_term_{num}_{season_len.name}" for num in range(1,num_term+1)
                    ]
                    cos_columns = [
                        f"cos_term_{num}_{season_len.name}" for num in range(1,num_term+1)
                    ]
                    match season_len:
                        case SeasonLength.DAY_OF_YEAR:
                            fourier_array = self._calculate_fourier_terms(
                                seasonal_cycle=dataframe.index.day_of_year.values,
                                cycle_len=SeasonLength.DAY_OF_YEAR,
                                n_fourier_terms=num_term
                            )
                        case SeasonLength.WEEK_OF_YEAR: 
                            fourier_array = self._calculate_fourier_terms(
                                seasonal_cycle = np.array(
                                    dataframe.index.isocalendar().week.values
                                    ),
                                cycle_len=SeasonLength.WEEK_OF_YEAR,
                                n_fourier_terms=num_term
                            )
                        case SeasonLength.MONTH_OF_YEAR: 
                            fourier_array = self._calculate_fourier_terms(
                                seasonal_cycle = np.array(
                                    dataframe.index.month.values
                                    ),
                                cycle_len=SeasonLength.MONTH_OF_YEAR,
                                n_fourier_terms=num_term
                            )
                        case SeasonLength.DAY_OF_MONTH: 
                            fourier_array = self._calculate_fourier_terms(
                                seasonal_cycle = np.array(
                                    dataframe.index.day.values
                                    ),
                                cycle_len=SeasonLength.DAY_OF_MONTH,
                                n_fourier_terms=num_term
                            )
                        # TODO WEEK_OF_MONTH
                        case SeasonLength.WEEK_OF_MONTH:
                            raise Exception(
                                "The option WEEK_OF_MONTH has not been implemented yet!"
                                "Please choose another option."
                            )
                        case _:
                            raise Exception(
                                "Invalid SeasonLength option! Please choose a valid one."
                            )
                    fourier_columns.append(np.hstack([fourier_array]))
                    fourier_names.append(np.hstack([sin_columns,cos_columns]))                
            else:
                for column,season_len,num_term in zip(
                     self.columns,self.seasonal_lengths,self.num_terms
                ):
                    sin_columns = [
                        f"sin_term_{num}_{season_len.name}" for num in range(1,num_term+1)
                    ]
                    cos_columns = [
                        f"cos_term_{num}_{season_len.name}" for num in range(1,num_term+1)
                    ]
                    fourier_array = self._calculate_fourier_terms(
                                seasonal_cycle=np.array(dataframe[column]),
                                cycle_len=season_len,
                                n_fourier_terms=num_term
                            )
                    fourier_columns.append(np.hstack([fourier_array]))
                    fourier_names.append(np.hstack([sin_columns,cos_columns]))
            fourier_df = pd.DataFrame(np.hstack(fourier_columns),
                            columns = np.hstack(fourier_names),
                            index=dataframe.index)
            return pd.concat([dataframe,fourier_df],axis=1)
    