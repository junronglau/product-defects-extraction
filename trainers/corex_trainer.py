import numpy as np
import pandas as pd
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim


class CorexTrainer:
    def __init__(self, model, data):
        self.model = model.model
        self.data = data
        self.vocab = model.vocab
        self.seed_topics = model.seed_topics

    def train(self):
        self.model.fit(
            self.data,
            words=self.vocab,
            anchors=self.seed_topics,
            anchor_strength=2
        )

    def evaluate(self, data, corpus):
        score_lst = metric_coherence_gensim(measure='c_v',
                                            top_n=10,
                                            topic_word_distrib=self._get_topics(),
                                            dtm=data.toarray(),
                                            vocab=np.array(self.vocab),
                                            texts=corpus)
        avg_score = np.mean(score_lst)
        return score_lst, avg_score

    def generate_topics(self):
        topic_word = self.model.get_topics(n_words=10)
        for i, topic_dist in enumerate(topic_word):
            topic_words = [ngram[0] for ngram in topic_dist if ngram[1] > 0]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    def _get_topics(self):
        topic_lst = []
        topic_word = self.model.get_topics(n_words=-1, print_words=False)
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.zeros(shape=(len(self.vocab)))
            topic_words[[ngram[0] for ngram in topic_dist]] = 1
            topic_lst.append(topic_words)
        return np.array(topic_lst)

    def get_top_documents(self, topics, df, conf=0.9):
        """
        Retrieves a set of documents evaluated by the trained topic model
        :param topics: indexes of topics to extract (0-based index)
        :param df: original dataframe to extract
        :param conf: percentage of documents to extract
        :return: Extracted dataframe from the topic model evaluation
        """
        top_docs = self.model.get_top_docs(n_docs=-1, sort_by="log_prob")  # log_prob output range: [-inf, 0]
        top_docs = [top_docs[topic] for topic in topics]
        top_docs_df = pd.DataFrame()
        for topic_n, topic_docs in zip(topics, top_docs):
            docs, probs = zip(*topic_docs)
            docs, probs = np.array(docs), np.array(probs)
            limit = np.quantile(probs, conf)
            top_docs_df = pd.concat([top_docs_df, df.iloc[list(docs[probs > limit])]], ignore_index=True, sort=False)
        top_docs_df.drop_duplicates(subset=['comment'], inplace=True)
        return top_docs_df

