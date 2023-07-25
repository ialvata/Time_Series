import pandas as pd

class Split:
    def __init__(self, dataframe: pd.DataFrame, 
                 input_columns: list[str] | None = None,
                 label_columns: list[str] | None = None,
                 train_val_test_prop:tuple[float,float,float] = (0.8,0.1,0.1)
                 ) -> None:
        
        if train_val_test_prop[0] == 0 or train_val_test_prop[2] == 0:
            raise Exception("Train or Test proportions cannot be zero!")
        if sum(train_val_test_prop)!=1:
            raise Exception("Proportions must sum to 1.")
        self.train_length = int(train_val_test_prop[0]*dataframe.shape[0])
        self.train_df = dataframe.iloc[:self.train_length]
        if train_val_test_prop[1]!= 0:
            self.test_length = int(train_val_test_prop[2]*dataframe.shape[0])
            self.val_length = dataframe.shape[0]-self.train_length-self.test_length
            self.val_df =  dataframe.iloc[
                self.train_length : self.train_length + self.val_length
            ]
            self.test_df = dataframe.iloc[
                self.train_length + self.val_length:
            ]
        else:
            self.val_df = None
            self.test_length = dataframe.shape[0]-self.train_length
            self.test_df = dataframe.iloc[
                self.train_length:
            ]
        self._train_labels = None
        self._train_input = None
        self._test_labels = None
        self._test_input = None
        self._val_labels = None
        self._val_input = None
        self.label_columns = label_columns
        self.input_columns = input_columns

    ########################            label properties           ########################
    @property
    def train_labels(self):
        if self.label_columns is None:
            raise Exception("This split has no label_columns")
        if self._train_labels is None:
            self._train_labels = self.train_df[self.label_columns]
        return self._train_labels
    @property
    def test_labels(self):
        if self.label_columns is None:
            raise Exception("This split has no label_columns")
        if self._test_labels is None:
            self._test_labels = self.test_df[self.label_columns]
        return self._test_labels
    @property
    def val_labels(self):
        if self.label_columns is None:
            raise Exception("This split has no label_columns")
        if self.val_df is None:
            raise Exception("This split has no validation dataframe")
        if self._val_labels is None:
            self._val_labels = self.val_df[self.label_columns]
        return self._val_labels

    ########################            input properties           ########################
    @property
    def train_input(self):
        if self.label_columns is None:
            raise Exception("This split has no label_columns")
        if self.input_columns is None:
            set_labels = set(self.label_columns)
            set_all_columns = set(self.train_df.columns)
            self.input_columns = list(set_all_columns-set_labels)
        if self._train_input is None:
            self._train_input = self.train_df[self.input_columns]
        return self._train_input
    @property
    def test_input(self):
        if self.label_columns is None:
            raise Exception("This split has no label_columns")
        if self.input_columns is None:
            set_labels = set(self.label_columns)
            set_all_columns = set(self.test_df.columns)
            self.input_columns = list(set_all_columns-set_labels)
        if self._test_input is None:
            self._test_input = self.test_df[self.input_columns]
        return self._test_input
    @property
    def val_input(self):
        if self.label_columns is None:
            raise Exception("This split has no label_columns")
        if self.val_df is None:
            raise Exception("This split has no validation dataframe")
        if self.input_columns is None:
            set_labels = set(self.label_columns)
            set_all_columns = set(self.val_df.columns)
            self.input_columns = list(set_all_columns-set_labels)
        if self._val_labels is None:
            self._val_labels = self.val_df[self.label_columns]
        return self._val_labels