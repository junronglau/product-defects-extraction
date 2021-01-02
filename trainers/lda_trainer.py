from gensim.models import CoherenceModel


class LDATrainer:
    def __init__(self, model):
        self.model = model.model
        self.data = model.data
        self.dictionary = model.dictionary
        self.coherence_model = CoherenceModel(model=self.model,
                                              corpus=self.data,
                                              texts=self.get_texts(),
                                              dictionary=self.dictionary,
                                              coherence='u_mass')

    def evaluate(self):
        coherence_lda = self.coherence_model.get_coherence()
        print('Coherence Score: ', coherence_lda)
        return coherence_lda

    def generate_topics(self):
        for topic in self.model.print_topics(num_topics=10, num_words=10):
            print(topic)

    def get_texts(self):
        return [[self.dictionary[word_id] for word_id, freq in doc] for doc in self.data]