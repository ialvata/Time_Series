
[GitHub.io Personal Page](https://ialvata.github.io/)
### Badge Overview

| **Open Source** | [![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)]()
|---|---|


# Time_Series

![A nice picture of a Time Series.](https://github.com/ialvata/ialvata.github.io/blob/main/static/images/time_series.jpg "A nice picture of a Time Series.")


This is a repo containing a simple library for Time Series data problems, which tries to aggregate all steps of the analysis, from data preprocessing, to label/target prediction performance assessment, and graphical presentation. 

## Main Features
- Object oriented API, with a very low learning fixed cost.
- Composable elements, which can be assembled to create a Time Series data pipeline.
- Feature Engineering, like explicit time embeddings (EWMA, Fourier, etc).
- Label transformations for stationarity, such as differencing and Box-Cox transform.
- Outlier detection algorithms (Isolation Forest Detection, Inter-Quartile Range, etc).
- Statistical tests implemented for checking stationarity.
- Both classical and ML time series models wrappers for seamless integration.
- Label/target prediction performance evaluation through Cross Validation.


## TODO
- Multiple time steps forecasting for random forests based models.
- Add folder with example notebooks.
- Add Forecast Bias and Normalized Deviation metrics.
- [Add TFT model from PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html)
- N-HiTS  integration (using PyTorch Forecasting).
- Add automated tests that prove algos are working.
- Add Informer from HuggingFace package. (I don't have enough memory to put the model into GPU memory. Google Colab?)



