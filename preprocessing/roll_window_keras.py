import datetime

import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
import matplotlib.pyplot as plt
from collections.abc import Generator

from preprocessing.roll_window_base import RollWindow
from preprocessing.data_spliting.set_splitting import Split
from keras.preprocessing import timeseries_dataset_from_array

class KerasWindow(RollWindow):
    def __init__(self, 
                 data_split: Split,
                 input_length: int,
                 label_length: int,
                 n_step_forecast:int = 1,
                 sequence_stride: int = 1,
                 label_columns: list[str] | None = None,
                 shuffle:bool = False, 
                 batch_size: int = 128
                 ):
        """
        Parameters:
        -----------
        n_step_forecast: int
            Integer variable that defines the number of lookahead steps we will forecast.
            If = 1, then we're doing a single-step forecast.
            If > 1, then we're doing a multi-step forecast.

        Observations:
        -------------
        This will serve as x argument to the `fit` method of the Keras Model API.
        The fit method, in the x argument, will take a `tf.data.Dataset` like object.
        In this case, it should be a **tuple** of either (inputs, targets) or 
        (inputs, targets, sample_weights).
            See `https://keras.io/api/models/model_training_apis/`
        
        There are two main ways to construct the Window:
            - Either we use the Split object, which has the train/val/test dataframes, and from it 
            we construct a `tf.data.Dataset`, and then we filter for inputs and labels; 
            - Or we use the Split object, filter for inputs and labels, and only then do we
            construct `tf.data.Dataset` for each inputs df and labels df.
        
        The second method may be too ineficient memory wise, while the first may be too slow, for 
        a great amount of data.

        TODO: Create data using the first method. The first is already implemented.
        """
        self.data_split = data_split
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {
            name: i for i, name in enumerate(self.data_split.train_df.columns)
        }
        self.input_length = input_length
        self.label_length = label_length
        self.sequence_length = input_length + n_step_forecast
        self.sequence_stride = sequence_stride
        self.shuffle = shuffle
        self.batch_size = batch_size

    def filtered_keras_dataset(self, dataframe:pd.DataFrame) -> tf.data.Dataset:
        """
        Example
        -------
        Consider indices [0, 1, ... 98]. With sequence_length=10, sampling_rate=2, 
        sequence_stride=3, batch_size = 2, shuffle=False, the dataset will yield batches of 
        sequences composed of the following indices:
            First batch:
                Inputs:
                    First sequence:  [0  2  4  6  8 10 12 14 16 18]
                    Second sequence: [3  5  7  9 11 13 15 17 19 21]
                Label/targets:
                    [0 3]
        Label 0 is assessed by model with inputs([0  2  4  6  8 10 12 14 16 18])

            Second batch:
                Inputs
                    Third sequence:  [6  8 10 12 14 16 18 20 22 24]
                    Fourth sequence: [9 11 13 15 17 19 21 23 25 27]
                Label/targets:
                    [6 9]
            ...
            Last batch:
                ...
                Last sequence:   [78 80 82 84 86 88 90 92 94 96]

        In this case the last 2 data points are discarded since no full sequence can be 
        generated to include them (the next sequence would have started at index 81, and 
        thus its last step would have gone over 98).

        Hence, `tf.data.Dataset` is a generator of tensors of 
        dim(batch_size, sequence_length, num_features), where sequence_length will be the 
        window size (input length plus label length).
        """

        ds:tf.data.Dataset = timeseries_dataset_from_array(
            data = dataframe.to_numpy(), # must be a numpy array
            targets = None, # if we used this, we would be limited to single-step forecasts
            sequence_length = self.sequence_length,
            sequence_stride = self.sequence_stride,
            shuffle = self.shuffle,
            batch_size = self.batch_size
        )
        return ds.map(self.filter,
            # num_parallel_calls=tf.data.AUTOTUNE
        )
        
    
    def filter(self, tensor:tf.Tensor):
        """
        Each tensor is of shape dim(batch_size, sequence_length, num_features), where 
        sequence_length will be the window size (input length plus label length).
        """
        inputs = tensor[:,:self.input_length,:]
        label_start_index = self.sequence_length - self.label_length
        labels = tensor[:,label_start_index:,:]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:,:,self.column_indices[name]]
                    for name in self.label_columns
                ], 
                axis=2 # we're stacking on the 3rd axis (axis=2), the features/columns axis.
            )
        return inputs, labels
    
    @property
    def train_data(self):
         return self.filtered_keras_dataset(self.data_split.train_df)
    
    @property
    def val_data(self):
         if self.data_split.val_df is not None:
            return self.filtered_keras_dataset(self.data_split.val_df)
         else:
             raise Exception("This Split object has no Validation Data!")
    
    @property
    def test_data(self):
         return self.filtered_keras_dataset(self.data_split.test_df)


if __name__=="__main__":
    import numpy as np
    data = np.array([range(100),range(100,200),range(200,300)]).T#np.array([range(100),range(100)]).T
    # data -> array(
    #     [
    #         [  0, 100, 200],
    #         [  1, 101, 201],
    #             ...
    #         [ 98, 198, 298],
    #         [ 99, 199, 299]
    #     ]
    # )
    window = 10
    stride = 1
    ds = timeseries_dataset_from_array(
                data = data,
                targets = None,
                sequence_length = window,
                sequence_stride = stride,
                shuffle = False,
                batch_size = 2
            )
    input_length = 6
    label_columns = ["col_2","col_3"]
    column_indices = {
        "col_1": 0,
        "col_2": 1,
        "col_3": 2,
    }
    def filter(tensor:tf.Tensor):
        inputs = tensor[:,:input_length,:]
        labels = tensor[:,input_length:,:]
        if label_columns is not None:
            labels = tf.stack(
                [
                    labels[:,:,column_indices[name]]
                    for name in label_columns
                ],
                axis = 2
            )
        return inputs, labels
    ds_filter = ds.map(filter)
    [batch for batch in ds_filter]
    # label tensors respect shape when stacked by the correct axis.
    # [(<tf.Tensor: shape=(2, 6, 3), dtype=int64, numpy=
    #     array([[[  0, 100, 200],
    #             [  1, 101, 201],
    #             [  2, 102, 202],
    #             [  3, 103, 203],
    #             [  4, 104, 204],
    #             [  5, 105, 205]],
        
    #             [[  1, 101, 201],
    #             [  2, 102, 202],
    #             [  3, 103, 203],
    #             [  4, 104, 204],
    #             [  5, 105, 205],
    #             [  6, 106, 206]]])>,
    #     <tf.Tensor: shape=(2, 4, 2), dtype=int64, numpy=
    #     array([[[106, 206],
    #             [107, 207],
    #             [108, 208],
    #             [109, 209]],
        
    #             [[107, 207],
    #             [108, 208],
    #             [109, 209],
    #             [110, 210]]])>),
    #             ...