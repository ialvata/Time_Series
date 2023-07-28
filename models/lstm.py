import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell
from preprocessing.roll_window_keras import KerasWindow
from pandas import DatetimeIndex

class LSTMKeras:
    def __init__(self) -> None:
        """
        Even though the LSTM is an RNN model, the recursiveness exists only in each component
        of the input vector.
        Let's consider a dataframe as our input array.
        Each input vector is made up of 1 row, and multiple features/columns.
        The recursiveness is in the features, not the rows. If we want to add a temporal
        dependency in the features, like that of an AR model, we need to add time lags to 
        the columns. 
        """
        self.model = Sequential(
            [
                LSTM(32, return_sequences=True),
                Dense(units=1)
            ]
        )
        self.history: keras.callbacks.History | None = None
        self.model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])
        self.window: KerasWindow | None = None

    def fit(self, window:KerasWindow, max_epochs: int, patience: int = 0):
        early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')
        self.window = window
        self.history = self.model.fit(
            window.train_data,#window.train,
            epochs=max_epochs,
            validation_data=window.val_data,#window.val,
            callbacks=[early_stopping],
            use_multiprocessing=True,
            workers=4)
        
    def plot_predictions(self, column_to_plot: str, max_subplots:int = 3):
        if self.window is not None:
            # next(iter(self.window.val_data))
            inputs, labels = next(iter(self.window.val_data))
            plt.figure(figsize=(12, 8))
            column_to_plot_index = self.window.column_indices[column_to_plot]
            max_n = min(max_subplots, len(inputs))
            for n in range(max_n):
                plt.subplot(3, 1, n+1)
                plt.ylabel(f'{column_to_plot} [scaled]')
                plt.plot(np.arange(self.window.input_length), 
                         inputs[n, :, column_to_plot_index],
                         label='Inputs', marker='.', zorder=-10)
                if self.window.label_columns:
                    label_col_index = self.window.label_columns_indices.get(column_to_plot, None)
                else:
                    label_col_index = column_to_plot_index
                if label_col_index is None:
                    continue
                plt.scatter(np.arange(self.window.label_length), 
                            labels[n, :, label_col_index],
                            edgecolors='k', marker='s', label='Labels', c='green', s=64)
                predictions = self.model(inputs)
                plt.scatter(np.arange(self.window.label_length), 
                            predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='red', s=64)
                if n == 0:
                    plt.legend()
            time_index: DatetimeIndex = self.window.data_split.train_df.index
            time_unit = time_index.freq
            if time_unit is not None:
                plt.xlabel(f'Timesteps (unit={time_unit})')
            else:
                plt.xlabel('Timesteps')
        else:
            raise Exception("This model has n")

    def plot_loss(self):
        if self.history is not None:
            loss = self.history.history["loss"]
            val_loss = self.history.history["val_loss"]
            epochs = range(1, len(loss)+1)
            plt.figure()
            plt.plot(epochs, loss, "b", label="Training loss")
            plt.plot(epochs, val_loss, "r", label="Validation loss")
            plt.title("Loss Evolution Accross Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        else:
            raise Exception("Unfitted model has no History object!")