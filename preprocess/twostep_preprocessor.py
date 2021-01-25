from preprocess.base.defects_classifier.base_preprocessor import BasePreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class TwoStepPreprocessor(BasePreprocessor):
    def __init__(self, train_df, test_df):
        super(TwoStepPreprocessor, self).__init__(train_df, test_df)
        self.vectorizer = TfidfVectorizer(analyzer='word',
                                          token_pattern=r'\w{1,}',
                                          ngram_range=(1, 3),
                                          max_features=5000)
        self.prepare_data()
        self.train_features = self.vectorizer.transform(self.train_features)
        self.test_features = self.vectorizer.transform(self.test_features)
        self.reliable_neg = self.generate_reliable_neg(0.1)

    def prepare_data(self):
        self.preprocess_data()
        self.generate_reliable_neg()
        self.vectorizer.fit(self.train_features)  # Only fit on train data

    def generate_reliable_neg(self, threshold):
        """
        Generates reliable negatives to be used for the 2-step learning process by filtering out 5-star positively
        worded reviews and applying a SPY technique
        :threshold: % of the top negative examples to extract
        """
        reliable_neg = self.train_features[(self.train_labels == 0) & (self.train_features['cleaned_ratings'] == 1)]
        analyser = SentimentIntensityAnalyzer()
        senti_scores = [analyser.polarity_scores(sentence)['pos'] for sentence in reliable_neg['cleaned_text']]
        reliable_neg['senti_scores'] = senti_scores
        reliable_neg = reliable_neg.sort_values(by=['senti_scores'], ascending=False)
        reliable_neg = reliable_neg.iloc[:threshold*len(reliable_neg), :]  # get top n% of the rows
        return reliable_neg

    def get_train_data(self):
        return {"features": self.train_features,
                "labels": self.train_labels}

    def get_test_data(self):
        return {"features": self.test_features,
                "labels": self.test_labels,
                "protocol": self.test_protocol}
