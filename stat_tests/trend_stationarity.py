from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd

# taken from https://www.statsmodels.org/devel/examples/notebooks/generated/stationarity_detrending_adf_kpss.html

class ADFResults:
    def __init__(self, adf_res: dict) -> None:
        self.test_statistic = adf_res[0]
        self.p_value = adf_res[1]
        self.num_lags_used = adf_res[2]
        self.num_observations_used = adf_res[3]
    def __repr__(self) -> str:
        return f"ADFResults({round(self.p_value,5)})"

def adf_test(timeseries:pd.DataFrame|pd.Series, print_output:bool = False)-> ADFResults:
    """
    Augmented Dickey-Fuller unit root test
        H0: Presence of a unit root 
        H1: No unit root (trend stationarity, if remaining modelling assumptions are correct)
    """
    dftest = adfuller(timeseries.values.flatten().tolist(), autolag="AIC")
    if print_output:
        print("Results of Dickey-Fuller Test:")
        dfoutput = pd.Series(
            dftest[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "#Lags Used",
                "Number of Observations Used",
            ],
        )
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)" % key] = value
        print(dfoutput)
    return ADFResults(dftest)
    
def kpss_test(timeseries, print_output:bool = False):
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    if print_output:
        print("Results of KPSS Test:")
        kpss_output = pd.Series(
            kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
        )
        for key, value in kpsstest[3].items():
            kpss_output["Critical Value (%s)" % key] = value
        print(kpss_output)
    return kpsstest