import pandas as pd


class DataLoader:
    def __init__(self, path):
        self.base_data_path = path.base_data_path
        self.df = self.load_data(self.base_data_path)

    @staticmethod
    def load_data(path):
        return pd.read_csv(path, encoding='utf-8')

    def get_data(self):
        return self.df

    # Split data here