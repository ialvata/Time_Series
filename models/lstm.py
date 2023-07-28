import matplotlib.pyplot as plt

import keras
from keras import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell


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
        
    def fit(self, train_data, val_data, max_epochs: int, patience: int = 0):
        early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')
        self.history = self.model.fit(
            train_data,#window.train,
            epochs=max_epochs,
            validation_data=val_data,#window.val,
            callbacks=[early_stopping],
            use_multiprocessing=True,
            workers=4)
        
    def visualize_loss(self, title):
        if self.history is not None:
            loss = self.history.history["loss"]
            val_loss = self.history.history["val_loss"]
            epochs = range(1, len(loss)+1)
            plt.figure()
            plt.plot(epochs, loss, "b", label="Training loss")
            plt.plot(epochs, val_loss, "r", label="Validation loss")
            plt.title(title)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        else:
            raise Exception("Unfitted model has no History object!")