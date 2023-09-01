from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.preprocessing import StandardScaler

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
        # When doing a joint isolation forest analysis, we want to see the columns being
        # considered in the summary
        self.isolation_forest_columns = None

    def sd_detection(self,
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

    def iqr_detection(self,
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

    def isolation_forest_detection(self,
                                   dataframe:pd.DataFrame | None = None,
                                   columns: list[str] | None = None,
                                   outlier_fraction: float = 0.01,
                                   separate_analysis:bool = True, 
                                    **kwargs):
        """
        Outliers are identified using the Isolation Forest algorithm from Sci-kit Learn.
        
        Parameters
        ----------
            `dataframe`
                Pandas dataframe
            `columns`: list[str] | None = None. 
                Name of the column that we want to detect anomalies in
            `outliers_fraction`: float = 0.01. 
                Percentage of outliers allowed in the sequence.
            `separate_analysis`: bool = True.
                The Isolation Forest algorithm is able to identify outliers, considering each
                column separately, or considering all columns in `columns` together.
                It's similar to considering just a marginal distribution, or a joint 
                distribution.
            `kwargs`
                keyword arguments to be passed to `sklearn.ensemble.IsolationForest`.
        
        Returns
        -------
            df: Pandas dataframe with column for detected Isolation Forest anomalies (True/False)
        """
        if dataframe is None:
            dataframe = self.dataframe
        if columns is None:
            columns = list(dataframe.columns)
        self.shape = dataframe.shape
        # creating dataframe with boolean masks with positions of outliers
        self.outlier_iso_forest_df = pd.DataFrame(columns=columns)
        # Scale the column that we want to flag for anomalies
        scaler = StandardScaler()
        scaled_time_series = scaler.fit_transform(dataframe)
        column_indices = {
            name: i for i, name in enumerate(dataframe.columns)
        }
        # train isolation forest
        kwargs["contamination"] = outlier_fraction
        kwargs["random_state"] = 42
        model = IsolationForest(**kwargs)
        if separate_analysis:
            for column in columns:
                # Predict returns ndarray of shape (n_samples,). For each observation, 
                # tells whether or not (+1 or -1) it should be considered as an inlier 
                # according to the fitted model.
                predictions = model.fit_predict(
                    scaled_time_series[
                        :,[column_indices[column]]
                    ]
                )
                # -1 is outlier and 1 is outlier. Changing it to 0 and 1
                predictions = 1 - np.clip(predictions, a_min=0, a_max=None)
                self.outlier_iso_forest_df[column] = predictions.astype(bool)
                self.summary_df.loc[
                    f"Isolation Forest - {column}", "# of Outliers"
                ] = self.outlier_iso_forest_df[column].sum()
                self.summary_df.loc[
                    f"Isolation Forest - {column}", "% of Outliers"
                ] = self.outlier_iso_forest_df[column].sum()/len(dataframe)*100
        else:
            predictions = model.fit_predict(
                    scaled_time_series[
                        # collecting only the columns we desire to consider in the algorithm
                        :,[column_indices[column] for column in columns]
                    ]
                )
            # -1 is outlier and 1 is inlier. Changing it to 1 and 0, respectively
            predictions = 1 - np.clip(predictions, a_min=0, a_max=None)
            # now, predictions are is_outlier array instead of is_inlier array.
            self.outlier_iso_forest_df = predictions.astype(bool)
            self.summary_df.loc[
                "Isolation Forest - Jointly", "# of Outliers"
            ] = self.outlier_iso_forest_df.sum()
            self.summary_df.loc[
                "Isolation Forest - Jointly", "% of Outliers"
            ] = self.outlier_iso_forest_df.sum()/len(dataframe)*100
            self.isolation_forest_columns = columns


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
        if self.isolation_forest_columns is not None:
            print(
                "Isolation Forest Columns considered in Joint analysis"
                f" -> {self.isolation_forest_columns}"
            )