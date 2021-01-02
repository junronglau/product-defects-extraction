from preprocess.base.labels_generator.base_preprocessor import BasePreprocessor
from sklearn.feature_extraction.text import CountVectorizer

import json
from nltk.tokenize import word_tokenize
import numpy as np


class GldaPreprocessor(BasePreprocessor):
    def __init__(self, df, config):
        super(GldaPreprocessor, self).__init__(df, config)
        self.dictionary = None
        self.data = None
        self.vocab = None
        self.vectorizer = None
        self.seed_topics_path = config.labels_generator.paths.seeded_topics_path
        self.seed_topics = None

    def prepare_data(self):
        self.preprocess_data()
        self.format_data()

    def format_data(self):
        self.vocab = list(set(word_tokenize(" ".join(self.df['cleaned_text']))))
        self.vectorizer = CountVectorizer(vocabulary=self.vocab)
        self.data = self.vectorizer.fit_transform(self.df['cleaned_text']).toarray()
        self.dictionary = self.vectorizer.vocabulary_
        self.seed_topics = self.load_seeded_topics()

    def load_seeded_topics(self):
        with open(self.seed_topics_path) as f:
            seed_topics_lst = json.load(f)
        seed_topics_dct = {}
        for t_id, (t_name, st) in enumerate(seed_topics_lst.items()):
            for word in st:
                seed_topics_dct[self.dictionary[word]] = t_id
        return seed_topics_dct

    def get_data(self):
        return self.data

    def get_corpus(self):
        return np.array(self.df['cleaned_text'].str.split())

