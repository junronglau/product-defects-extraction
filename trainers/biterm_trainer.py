from biterm.utility import topic_summuary
from tqdm import tqdm


class BitermTrainer:
    def __init__(self, model, corpus):
        self.model = model.model
        self.dictionary = model.dictionary
        self.corpus = corpus
        self.iterations = model.iterations
        self.topics = None

    def train(self):
        for i in tqdm(range(0, len(self.corpus), 100)):
            biterms_chunk = self.corpus[i:i + 100]
            self.model.fit(biterms_chunk, iterations=self.iterations)
        self.topics = self.model.transform(self.corpus)

    def generate_topics(self, data):
        topic_summuary(self.model.phi_wz.T, data, self.dictionary, 10)
