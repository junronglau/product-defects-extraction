from sklearn.feature_extraction.text import TfidfVectorizer


class SvmPreprocessor:
    def __init__(self, train_df, test_df):
        self.train_features, self.test_features = self.prepare_data(train_df, test_df)
        self.train_labels = train_df['has_defect']
        self.test_labels = test_df['has_defect']
        self.vectorizer = TfidfVectorizer(analyzer='word',
                                          token_pattern=r'\w{1,}',
                                          ngram_range=(1, 3),
                                          max_features=5000)

    def prepare_data(self, train_df, test_df):
        self.vectorizer.fit(train_df['cleaned_text'])  # Only fit on train data
        train_features = self.vectorizer.transform(train_df['cleaned_text'])
        test_features = self.vectorizer.transform(test_df['cleaned_text'])
        return train_features, test_features

    def get_data(self):
        return (self.train_features, self.train_labels), (self.test_features, self.test_labels)
