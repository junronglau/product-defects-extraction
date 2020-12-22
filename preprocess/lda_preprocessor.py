from utils.preprocess import filter_reviews_rating
from utils.preprocess import sent_to_words
from utils.preprocess import generate_bigrams
from utils.preprocess import generate_dictionary


class LDAPreprocessor:
    def __init__(self, config, reviews_df):
        self.config = config
        self.reviews_df = reviews_df
        self.dictionary = None
        self.corpus = None

    def preprocess_data(self):
        self.reviews_df = filter_reviews_rating(self.reviews_df, 0.4)
        split_words = list(sent_to_words(self.reviews_df['cleaned_text']))
        bigrams = generate_bigrams(split_words)

        # Create Dictionary
        self.dictionary = generate_dictionary(bigrams)

        # Term Document Frequency
        self.corpus = [self.dictionary.doc2bow(text) for text in bigrams]

    def get_data(self):
        return self.corpus

    def get_dictionary(self):
        return self.dictionary
