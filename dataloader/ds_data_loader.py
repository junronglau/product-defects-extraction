import pandas as pd


class DataLoader:
    def __init__(self, path):
        self.df = self.load_data(path.base_data_path)
        self.test_df = self.load_data(path.summary_label_path)

    @staticmethod
    def load_data(path, headers=True):
            return pd.read_csv(path, encoding='utf-8')

    def get_data(self):
        return self.df, self.test_df


