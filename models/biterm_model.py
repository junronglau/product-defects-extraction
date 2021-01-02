from biterm.btm import oBTM
import pickle


class BitermModel:
    def __init__(self, config, dictionary):
        self.model_path = config.labels_generator.paths.save_model_path
        self.dictionary = dictionary
        self.model = oBTM(num_topics=config.labels_generator.model.num_topics, V=self.dictionary)

    def save(self):
        with open(self.model_path, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)
