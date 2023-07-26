import datetime

import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
import matplotlib.pyplot as plt
from collections.abc import Generator

from preprocessing.roll_window_base import RollWindow
from preprocessing.set_splitting import Split
from keras.preprocessing import timeseries_dataset_from_array

class KerasWindow(RollWindow):
    def __init__(self, 
                 data_split: Split,
                 sequence_length: int,
                 sequence_stride: int,
                 label_columns: list[str] | None = None,
                 shuffle:bool = False, batch_size: int = 128
                 ):
        """
        See
        """

        self.data_split = data_split
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {
            name: i for i, name in enumerate(self.data_split.train_df.columns)
        }
        # these two parameters should be created in the data_split instead.
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.shuffle = shuffle
        self.batch_size = batch_size

    @property
    def keras_train_set(self) -> tf.data.Dataset:
        """
        Example
        -------
        Consider indices [0, 1, ... 98]. With sequence_length=10, sampling_rate=2, 
        sequence_stride=3, batch_size = 2, shuffle=False, the dataset will yield batches of sequences composed 
        of the following indices:
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
        """
        return timeseries_dataset_from_array(
            data = self.data_split.train_input,
            targets = self.data_split.train_labels,
            sequence_length = self.sequence_length,
            sequence_stride = self.sequence_stride,
            shuffle = self.shuffle,
            batch_size = self.batch_size
        )

    
