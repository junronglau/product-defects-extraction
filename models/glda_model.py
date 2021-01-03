from lda import guidedlda as glda
import pickle


class GldaModel:
    def __init__(self, config, preprocessor):
        self.model_path = config.labels_generator.paths.save_model_path
        self.model = glda.GuidedLDA(n_topics=config.labels_generator.model.num_topics,
                                    n_iter=config.labels_generator.model.iterations,
                                    random_state=config.labels_generator.model.random_state,
                                    refresh=20,
                                    alpha=0.01,
                                    eta=0.01)

        self.vocab = preprocessor.vocab
        self.seed_topics = preprocessor.seed_topics

    def save(self):
        with open(self.model_path, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)
