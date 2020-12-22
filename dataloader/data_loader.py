import pandas as pd


class DataLoader:
    def __init__(self, path):
        self.base_data_path = path.base_data_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.base_data_path, encoding='utf-8')

    def get_data(self):
        return self.df
