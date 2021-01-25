from preprocess.base.defects_classifier.base_preprocessor import BasePreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer


class SvmPreprocessor(BasePreprocessor):
    def __init__(self, train_df, test_df):
        super(SvmPreprocessor, self).__init__(train_df, test_df)
        self.vectorizer = TfidfVectorizer(analyzer='word',
                                          token_pattern=r'\w{1,}',
                                          ngram_range=(1, 2))
        self.train_features = None
        self.test_features = None
        self.prepare_data()

    def prepare_data(self):
        self.preprocess_data()
        self.vectorizer.fit(self.train_text)  # Only fit on train data
        self.train_features = self.vectorizer.transform(self.train_text)
        self.test_features = self.vectorizer.transform(self.test_text)

    def get_train_data(self):
        return {"features": self.train_features,
                "labels": self.train_labels}

    def get_test_data(self):
        return {"features": self.test_features,
                "labels": self.test_labels,
                "protocol": self.test_protocol}
