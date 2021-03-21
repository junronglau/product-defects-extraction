import pandas as pd

pd.options.mode.chained_assignment = None
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer


class TextRankPreprocessor():
    def __init__(self, train_df, n_docs):
        self.train_df = train_df
        self.limit_docs = n_docs
        self.min_words = 5
        self.max_words = 10
        self.train_features = []

    def prepare_data(self):
        # Remove short summaries with less than N words
        # Filter by most voted for each group
        # Load document vectors

        self.remove_short_reviews()
        self.lowercase_reviews()
        self.retrieve_top_docs()
        self.tokenize_reviews

        testdf = pd.DataFrame()
        for topic in self.train_df['defect_topic'].unique():
            # Because we need to have legible sentences, we use raw comments instead of cleaned ones
            text = '\n'.join(self.train_df['comment'].loc[self.train_df['defect_topic'] == topic])
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            self.train_features.append([topic, parser])

            testdf = pd.concat([testdf, self.train_df['comment'].loc[self.train_df['defect_topic'] == topic]],
                               sort=False)
        testdf.to_csv("./data/uc3/defects_summarizer/summary_data.csv")

    def tokenize_reviews(self):
        self.train_df['comment'] = self.train_df['comment'].str.replace(".", "\n").str.replace(",", "\n")

    def remove_short_reviews(self):
        self.train_df = self.train_df[self.train_df['comment']
            .str.split().str.len()
            .gt(self.min_words)]
        self.train_df = self.train_df[self.train_df['comment']
            .str.split().str.len()
            .lt(self.max_words)]

    def lowercase_reviews(self):
        self.train_df['comment'] = self.train_df['comment'].str.lower()

    def retrieve_top_docs(self):
        self.train_df = self.train_df.sort_values(by='cleaned_ratings')
        self.train_df = self.train_df.groupby(['defect_topic']).head(self.limit_docs)

    def get_train_data(self):
        return {"features": self.train_features}


