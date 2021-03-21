from corextopic import corextopic as ct
import pickle


class CorexModel:
    def __init__(self, config, preprocessor, load=False, seed=True):
        self.model_path = config.labels_generator.paths.save_model_path
        if load:
            self.model = pickle.load(open(self.model_path, "rb"))
        else:
            self.model = ct.Corex(n_hidden=config.labels_generator.model.num_topics,
                                  seed=config.labels_generator.model.random_state)

        self.vocab = preprocessor.vocab
        self.seed_topics = None
        if seed:
            self.seed_topics = preprocessor.seed_topics

    def save(self):
        with open(self.model_path, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)
