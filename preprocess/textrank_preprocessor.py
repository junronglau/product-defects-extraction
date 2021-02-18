import pandas as pd
pd.options.mode.chained_assignment = None
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer


class TwoStepPreprocessor():
    def __init__(self, train_df):
        self.train_df = train_df
        self.limit_docs = 5
        self.limit_words = 10
        self.train_features = []

    def prepare_data(self):
        # Remove short summaries with less than N words
        # Filter by most voted for each group
        # Load document vectors

        self.remove_short_reviews()
        self.retrieve_top_docs()
        for topic in self.train_df['defect_topic'].unique():
            text = self.train_df['cleaned_text'].loc[self.train_df['defect_topic'] == topic]
            parser = PlaintextParser.from_string(text,Tokenizer("english"))
            self.train_features.append([topic, parser])

    def remove_short_reviews(self):
        self.train_df = self.train_df[self.train_df['cleaned_text'].str.split().str.len().lt(self.limit_words)]

    def retrieve_top_docs(self):
        self.train_df = self.train_df.sort_values(by='cleaned_ratings')
        self.train_df = self.train_df.groupby(['defect_topic']).head(self.limit_docs)

    def get_train_data(self):
        return {"features": self.train_features}