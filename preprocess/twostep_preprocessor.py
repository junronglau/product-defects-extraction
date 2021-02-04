from preprocess.base.defects_classifier.base_preprocessor import BasePreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from random import shuffle
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
pd.options.mode.chained_assignment = None


class TwoStepPreprocessor(BasePreprocessor):
    def __init__(self, train_df, test_df):
        super(TwoStepPreprocessor, self).__init__(train_df, test_df)
        self.vectorizer = TfidfVectorizer(analyzer='word',
                                          token_pattern=r'\w{1,}',
                                          ngram_range=(1, 2),
                                          max_features=10000)
        self.train_features = None
        self.test_features = None
        self.train_labelled_data = None
        self.train_unlabelled_data = None
        self.train_processed_labels = None
        self.remaining_pos = None
        self.continue_training = True

    def prepare_data(self):
        self.preprocess_data()
        self.vectorizer.fit(self.train_text)  # Only fit on train data
        self.train_features = pd.DataFrame(self.vectorizer.transform(self.train_text).toarray())
        self.test_features = pd.DataFrame(self.vectorizer.transform(self.test_text).toarray())
        self.generate_split_data()

    def generate_pseudo_neg(self, threshold):
        """
        Generates pseudo negatives to be used for the 2-step learning process by filtering out 5-star positive-worded
        reviews
        :threshold: % of the top negative examples to extract
        """
        train_df = self.train_df.copy()
        pseudo_neg = train_df[(self.train_labels != 1) & (train_df['cleaned_ratings'] == 1)]
        analyser = SentimentIntensityAnalyzer()
        senti_scores = [analyser.polarity_scores(sentence)['pos'] for sentence in pseudo_neg['cleaned_text'].fillna("")]
        pseudo_neg.loc[:, 'senti_scores'] = senti_scores
        pseudo_neg = pseudo_neg.sort_values(by=['senti_scores'], ascending=False)
        pseudo_neg = pseudo_neg.iloc[:int(threshold * len(pseudo_neg)), :]  # get top n% of the rows
        print("psuedo neg", pseudo_neg.shape[0])
        return self.train_features.loc[list(pseudo_neg.index)].copy()

    def generate_reliable_neg(self, threshold, last_iter):
        """
        Train a basic model on positives and pseudo-negative data, then create a test dataset with extracted "spy"
        positives and unlabelled data. Evaluate on extract out data with threshold higher than the spy dataset.
        :param threshold: % of positive labels to infiltrate as spy
        :return: reliable negative dataset to be part of labelled data
        """
        n_samples = int(self.train_features[self.train_labels == 1].shape[0] * threshold)
        spy_data = self.train_features[self.train_labels == 1].sample(n_samples)
        spy_ind = spy_data.index
        pos_data = self.train_features[self.train_labels == 1]
        pos_data = pos_data.loc[~pos_data.index.isin(spy_ind)]

        if self.train_labelled_data is not None:
            reliable_neg = pd.DataFrame.sparse.from_spmatrix(self.train_labelled_data)
            reliable_neg = reliable_neg[(np.array(self.train_processed_labels) == -1)]
        else:
            reliable_neg = self.generate_pseudo_neg(0.05)

        new_reliable_neg = reliable_neg.sample(min(len(pos_data), len(reliable_neg)))
        X_train = pd.concat([pos_data, new_reliable_neg])
        y_train = [1 for _ in range(len(pos_data))] + [0 for _ in range(len(new_reliable_neg))]

        gnb = LogisticRegression()
        gnb.fit(np.array(X_train), y_train)
        y_pred = gnb.predict_proba(spy_data)
        min_prob_threshold = np.min(y_pred[:, 1])
        unlabelled = self.train_features[self.train_labels != 1] if self.train_unlabelled_data is None else \
            pd.DataFrame.sparse.from_spmatrix(self.train_unlabelled_data)
        if unlabelled.shape[0] == 0:
            self.continue_training = False
            return reliable_neg
        print("Q dataset (should decrease)", unlabelled.shape[0])
        probs = gnb.predict_proba(unlabelled)
        new_reliable_neg = unlabelled[probs[:, 1] < min_prob_threshold]
        if new_reliable_neg.shape[0] == 0 or last_iter:
            self.continue_training = False
            self.remaining_pos = unlabelled[probs[:, 1] >= min_prob_threshold]
        final_reliable_neg = pd.concat([reliable_neg, new_reliable_neg])
        return final_reliable_neg

    def generate_split_data(self, last_iter=False):
        reliable_neg = self.generate_reliable_neg(0.2, last_iter)
        reliable_pos = self.train_features[self.train_labels == 1]
        if self.remaining_pos is not None:
            reliable_pos = pd.concat([reliable_pos, self.remaining_pos])

        train_labelled_data = pd.concat([reliable_neg, reliable_pos])
        train_labels = [-1 for _ in range(len(reliable_neg))] + [1 for _ in range(len(reliable_pos))]

        if self.remaining_pos is not None:
            ros = RandomOverSampler(random_state=0)
            train_labelled_data, train_labels = ros.fit_resample(
                csr_matrix(train_labelled_data.values, dtype=np.float64), pd.Series(train_labels))
            train_labelled_data = pd.DataFrame.sparse.from_spmatrix(train_labelled_data)

        train_labelled_data, train_labels = self._shuffle_data(train_labelled_data.reset_index(drop=True), np.array(train_labels))
        train_labelled_idx = list(train_labelled_data.index)
        train_unlabelled_data = self.train_features.loc[~self.train_features.index.isin(train_labelled_idx)]
        print("{pos} positive labels and {neg} negative labels".format(pos=len(reliable_pos), neg=len(reliable_neg)))
        print("{} unlabelled data".format(len(train_unlabelled_data)))
        self.train_labelled_data = csr_matrix(train_labelled_data.fillna("").values, dtype=np.float64)
        self.train_unlabelled_data = csr_matrix(train_unlabelled_data.fillna("").values, dtype=np.float64)
        self.train_processed_labels = train_labels

    @staticmethod
    def _shuffle_data(features, labels):
        ind_list = [i for i in range(len(labels))]
        shuffle(ind_list)
        features = features.iloc[ind_list]
        labels = labels[ind_list]
        return features, labels

    def get_train_data(self):
        return {"labelled_features": self.train_labelled_data,
                "unlabelled_features": self.train_unlabelled_data,
                "labels": self.train_processed_labels}

    def get_test_data(self):
        return {"features": self.test_features,
                "labels": self.test_labels,
                "protocol": self.test_protocol}
