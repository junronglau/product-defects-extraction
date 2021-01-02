from utils.preprocess import replace_null
from utils.preprocess import remove_short_words
from utils.preprocess import remove_digits
from utils.preprocess import filter_pos_tags
from utils.preprocess import filter_reviews_rating
from utils.preprocess import remove_stop_words


class BasePreprocessor:
    def __init__(self, df, config):
        self.df = df
        self.config = config

    def preprocess_data(self):
        self.df = filter_reviews_rating(self.df, 0.2)
        self.df['cleaned_text'] = remove_digits(self.df['cleaned_text'])
        self.df['cleaned_text'] = remove_short_words(self.df['cleaned_text'], 3)
        self.df['cleaned_text'] = self.df['cleaned_text'].astype(str).apply(filter_pos_tags, tag_lst=['N', 'V', 'J'])
        self.df['cleaned_text'] = remove_stop_words(self.df['cleaned_text'], self.generate_stop_word_list())
        self.df['cleaned_text'] = replace_null(self.df['cleaned_text'], '')

    def generate_stop_word_list(self):
        with open(self.config.labels_generator.paths.stop_words_path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        return content
