from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np


class OutlierAnalysis:
    def __init__(self, dataframe:pd.DataFrame) -> None:
        # outlier_sd_df will be a dataframe with boolean masks with the locations of the outliers.
        self.outlier_sd_df = None
        self.dataframe = dataframe
        self.summary_df = pd.DataFrame(columns=["# of Outliers", "% of Outliers"])

    def detect_outlier_sd(self,
                          dataframe:pd.DataFrame | None = None,
                          columns: list[str] | None = None,
                          sd_multiple:int = 2) -> None:
        if dataframe is None:
            dataframe = self.dataframe
        if columns is None:
            columns = list(dataframe.columns)
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
                f"{sd_multiple}IQR - {column}", "# of Outliers"
            ] = self.outlier_sd_df[column].sum()
            self.summary_df.loc[
                f"{sd_multiple}IQR - {column}", "% of Outliers"
            ] = self.outlier_sd_df[column].sum()/len(dataframe)*100

    def detect_outlier_iqr(self,
                           columns: list[str],
                           dataframe:pd.DataFrame | None = None,
                           iqr_multiple: float = 1.5) -> None:
        if dataframe is None:
            dataframe = self.dataframe
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

    def show_summary(self):
        self.summary_df.style.format({"% of Outliers": "{:.2f}%"})
        return self.summary_df

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