from utils.preprocess import replace_null


class BasePreprocessor:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.train_labels = train_df['has_defect']
        self.test_labels = test_df['has_defect']
        self.test_protocol = test_df['protocol']
        self.train_text = train_df['cleaned_text']
        self.test_text = test_df['cleaned_text']

    def preprocess_data(self):
        self.train_text = replace_null(self.train_text, '')
        self.test_text = replace_null(self.test_text, '')

