import os
import pandas as pd
import traceback

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def train_dataset(self):
        df_train = pd.read_csv(self.config["path"]["traindata"])
        return df_train

    def test_dataset(self):
        df_test = pd.read_csv(self.config["path"]["testdata"])
        return df_test
