from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from enum import Enum

class OutlierOption(Enum):
    """
    Class for determining the plot data to be shown in plot_outliers method.
    """
    SD = 1
    IQR = 2


class OutlierAnalysis:
    def __init__(self, dataframe:pd.DataFrame) -> None:
        """
        Before using some of this methods, ponder the need to deseasonalize the data to avoid
        identifying a seasonal peak as an outlier.
        """
        self.dataframe = dataframe
        self.summary_df = pd.DataFrame(columns=["# of Outliers", "% of Outliers"])
        # outlier_sd_df will be a dataframe with boolean masks with the locations of the outliers.
        self.outlier_sd_df = None
        # outlier_iqr_df will be a similar dataframe for IQR method
        self.outlier_iqr_df = None
        # this will be a tuple with the dataframe's shape used in outlier detection
        self.shape = None

    def detect_outlier_sd(self,
                          dataframe:pd.DataFrame | None = None,
                          columns: list[str] | None = None,
                          sd_multiple:int = 2) -> None:
        """
        We consider the distance to the mean, using the standard deviation (SD), to identify 
        the outliers.
        The theory behind this assumes the data follows a Normal distribution.
        When it follows another distribution, we should be careful using this criteria for
        identifying outliers.
        The mean and standard deviation are influenced by extreme values (possibly outliers).
        """
        if dataframe is None:
            dataframe = self.dataframe
        if columns is None:
            columns = list(dataframe.columns)
        self.shape = dataframe.shape
        # creating dataframe with boolean masks with positions of outliers
        self.outlier_sd_df = pd.DataFrame(columns=columns)
        for column in columns:
            mean = dataframe[column].mean()
            std = dataframe[column].std()
            higher_bound = mean + sd_multiple * std
            lower_bound = mean - sd_multiple * std
            self.outlier_sd_df[column] = (
                (dataframe[column] > higher_bound) | (dataframe[column] < lower_bound)
            )
            self.summary_df.loc[
                f"{sd_multiple} SD - {column}", "# of Outliers"
            ] = self.outlier_sd_df[column].sum()
            self.summary_df.loc[
                f"{sd_multiple} SD - {column}", "% of Outliers"
            ] = self.outlier_sd_df[column].sum()/len(dataframe)*100

    def detect_outlier_iqr(self,
                           dataframe:pd.DataFrame | None = None,
                           columns: list[str] | None = None,
                           iqr_multiple: float = 1.5) -> None:
        """
        This method, since it uses quantiles, it's more robust to extreme values.
        The choice of 1.5 as the Inter-quartile range (`iqr_multiple`) comes from assuming
        a Normal distribution.
        """
        if dataframe is None:
            dataframe = self.dataframe
        if columns is None:
            columns = list(dataframe.columns)
        self.shape = dataframe.shape
        # creating dataframe with boolean masks with positions of outliers
        self.outlier_iqr_df = pd.DataFrame(columns=columns)
        for column in columns:
            q1, q2, q3 = (
                np.quantile(dataframe[column], 0.25), 
                np.quantile(dataframe[column], 0.5), 
                np.quantile(dataframe[column], 0.75)
            )
            iqr = q3 - q1
            higher_bound = q3 + iqr_multiple * iqr
            lower_bound = q1 - iqr_multiple * iqr
            self.outlier_iqr_df[column] = (
                (dataframe[column] > higher_bound) | (dataframe[column] < lower_bound)
            )
            self.summary_df.loc[
                f"{iqr_multiple} IQR - {column}", "# of Outliers"
            ] = self.outlier_iqr_df[column].sum()
            self.summary_df.loc[
                f"{iqr_multiple} IQR - {column}", "% of Outliers"
            ] = self.outlier_iqr_df[column].sum()/len(dataframe)*100

    def plot_outliers(self, 
                      option:OutlierOption = OutlierOption.IQR, 
                      dataframe:pd.DataFrame | None = None,
                      columns: list[str] | None = None, 
                      multiple: float | None = None, 
                      **kwargs) -> None:
        """
        (Currently plotting only for IQR method is available)
        Make a box plot from DataFrame columns.
        The box extends from the Q1 to Q3 quartile values of the data,
        with a line at the median (Q2). The whiskers extend from the edges
        of box to show the range of the data. By default, they extend no more than
        `1.5 * IQR (IQR = Q3 - Q1)` from the edges of the box, ending at the farthest
        data point within that interval. Outliers are plotted as separate dots.

        Parameters
        ----------
        **kwargs
            All other plotting keyword arguments to be passed to
            :func:`matplotlib.pyplot.boxplot`.


        
        """
        if dataframe is None:
                    dataframe = self.dataframe
        if columns is None:
            columns = list(dataframe.columns)
        if option == OutlierOption.IQR:
            dataframe.boxplot(column=columns, rot=45, whis = multiple,**kwargs)

    def show_summary(self):
        print("\n###########################################################################")
        self.summary_df.style.format({"% of Outliers": "{:.2f}%"})
        print(self.summary_df)
        print(f"Dataframe shape = {self.shape} ")

# def detect_outlier_isolation_forest(ts, outlier_fraction, **kwargs):
#     """
#     In this definition, time series anomalies are detected using an Isolation Forest algorithm.
#     Arguments:
#         df: Pandas dataframe
#         column_name: string. Name of the column that we want to detect anomalies in
#         outliers_fraction: float. Percentage of outliers allowed in the sequence.
#     Outputs:
#         df: Pandas dataframe with column for detected Isolation Forest anomalies (True/False)
#     """
#     # Scale the column that we want to flag for anomalies
#     min_max_scaler = StandardScaler()
#     scaled_time_series = min_max_scaler.fit_transform(ts.reshape(-1, 1))
#     # train isolation forest
#     kwargs["contamination"] = outlier_fraction
#     kwargs["random_state"] = 42
#     model = IsolationForest(**kwargs)
#     pred = model.fit_predict(scaled_time_series)
#     # -1 is outlier and 1 is outlier. Changing it to 0 and 1
#     pred = 1 - np.clip(pred, a_min=0, a_max=None)
#     return pred.astype(bool)