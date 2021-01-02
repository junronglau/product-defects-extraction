import numpy as np
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim


class GldaTrainer:
    def __init__(self, model, data):
        self.model = model.model
        self.data = data
        self.vocab = model.vocab
        self.seed_topics = model.seed_topics

    def train(self):
        self.model.fit(self.data, seed_topics=self.seed_topics, seed_confidence=0.75)

    def evaluate(self, data, corpus):
        score_lst = metric_coherence_gensim(measure='c_v',
                                            top_n=10,
                                            topic_word_distrib=np.array(self.model.topic_word_),
                                            dtm=np.array(data),
                                            vocab=np.array(self.vocab),
                                            texts=corpus)
        avg_score = np.mean(score_lst)
        return score_lst, avg_score

    def generate_topics(self):
        n_top_words = 10
        topic_word = self.model.topic_word_
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(self.vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))
