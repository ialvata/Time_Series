from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd

# taken from https://www.statsmodels.org/devel/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
def adf_test(timeseries:pd.DataFrame, print_flag:bool = False):
    """
    Augmented Dickey-Fuller unit root test
        H0: Presence of a unit root 
        H1: No unit root (trend stationarity)
    """
    dftest = adfuller(timeseries, autolag="AIC")
    if print_flag:
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
    return dftest
    
def kpss_test(timeseries, print_flag:bool = False):
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    if print_flag:
        print("Results of KPSS Test:")
        kpss_output = pd.Series(
            kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
        )
        for key, value in kpsstest[3].items():
            kpss_output["Critical Value (%s)" % key] = value
        print(kpss_output)
    return kpsstest