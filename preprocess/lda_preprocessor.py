from preprocess.base.labels_generator.base_preprocessor import BasePreprocessor

from utils.preprocess import sent_to_words
from utils.preprocess import generate_bigrams
from utils.preprocess import generate_dictionary


class LDAPreprocessor(BasePreprocessor):
    def __init__(self, df, config):
        super(LDAPreprocessor, self).__init__(df, config)
        self.dictionary = None
        self.corpus = None

    def prepare_data(self):
        self.preprocess_data()
        self.format_data()

    def format_data(self):
        split_words = list(sent_to_words(self.df['cleaned_text']))
        bigrams = generate_bigrams(split_words)
        # Create Dictionary
        self.dictionary = generate_dictionary(bigrams)
        # Term Document Frequency
        self.corpus = [self.dictionary.doc2bow(text) for text in bigrams]

    def get_data(self):
        return self.corpus

    def get_dictionary(self):
        return self.dictionary
