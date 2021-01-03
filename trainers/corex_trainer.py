import numpy as np
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
