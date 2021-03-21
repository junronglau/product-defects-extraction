from sklearn import svm
import pickle


class SvmModel:
    def __init__(self, config, load=False):
        self.model_path = config.defects_classifier.paths.save_model_path
        if load:
            self.model = pickle.load(open(self.model_path, "rb"))
        else:
            self.model = svm.SVC(C=0.1, kernel='linear', probability=True)


    def save(self):
        with open(self.model_path, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)
