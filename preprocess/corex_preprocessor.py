from preprocess.base.labels_generator.base_preprocessor import BasePreprocessor
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import json
import numpy as np


class CorexPreprocessor(BasePreprocessor):
    def __init__(self, df, config):
        super(CorexPreprocessor, self).__init__(df, config)
        self.dictionary = None
        self.data = None
        self.vocab = None
        self.vectorizer = None
        self.seed_topics_path = config.labels_generator.paths.seeded_topics_path
        self.seed_topics = None
        self.ngram = config.labels_generator.preprocess.ngrams

    def prepare_data(self):
        self.preprocess_data()
        self.format_data()

    def format_data(self):
        if self.ngram == "unigram":
            self.vocab = list(set(word_tokenize(" ".join(self.df['cleaned_text']))))
            self.vectorizer = CountVectorizer(vocabulary=self.vocab)
            self.dictionary = self.vectorizer.vocabulary_
        elif self.ngram == "bigram":
            self.vectorizer = TfidfVectorizer(
                max_df=.5,
                min_df=10,
                max_features=None,
                norm=None,
                ngram_range=(1, 2),
                binary=True,
                use_idf=False,
                sublinear_tf=False
            )
        self.data = self.vectorizer.fit_transform(self.df['cleaned_text'])
        self.seed_topics = self.load_seeded_topics()
        self.vocab = self.vectorizer.get_feature_names()

    def load_seeded_topics(self):
        with open(self.seed_topics_path) as f:
            seed_topics_lst = json.load(f)
        seed_topics_dct = [[word for word in st] for t_id, (t_name, st) in enumerate(seed_topics_lst.items())]
        return seed_topics_dct

    def get_data(self):
        return self.data

    def get_corpus(self):
        return np.array(self.df['cleaned_text'].str.split())

    def get_raw_corpus(self):
        return self.df