import gensim


class LDAModel:
    def __init__(self, config, corpus, dictionary):
        self.model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                     id2word=dictionary,
                                                     num_topics=config.labels_generator.model.num_topics,
                                                     random_state=config.labels_generator.model.random_state,
                                                     update_every=1,
                                                     chunksize=100,
                                                     passes=config.labels_generator.model.iterations,
                                                     alpha='auto',
                                                     per_word_topics=True)
        self.data = corpus
        self.dictionary = dictionary
        self.model_path = config.labels_generator.paths.save_model_path

    def save(self):
        self.model.save(self.model_path)
