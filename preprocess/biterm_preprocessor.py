from preprocess.base.labels_generator.base_preprocessor import BasePreprocessor
from sklearn.feature_extraction.text import CountVectorizer

from biterm.utility import vec_to_biterms
import numpy as np


class BitermPreprocessor(BasePreprocessor):
    def __init__(self, df, config):
        super(BitermPreprocessor, self).__init__(df, config)
        self.dictionary = None
        self.corpus = None
        self.data = None
        self.vectorizer = CountVectorizer()

    def prepare_data(self):
        self.preprocess_data()
        self.format_data()

    def format_data(self):
        self.data = self.vectorizer.fit_transform(self.df['cleaned_text']).toarray()
        self.dictionary = np.array(self.vectorizer.get_feature_names())
        self.corpus = vec_to_biterms(self.data)

    def get_corpus(self):
        return self.corpus

    def get_dictionary(self):
        return self.dictionary

    def get_data(self):
        return self.data

