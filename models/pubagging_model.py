from pulearn import BaggingPuClassifier
from sklearn import svm
import pickle

class PuBaggingModel:
    def __init__(self, config, load=False):
        self.model_path = config.defects_classifier.paths.save_model_path
        if load:
            self.model = pickle.load(open(self.model_path, "rb"))
        else:
            svc = svm.SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
            self.model = BaggingPuClassifier(base_estimator=svc, n_estimators=5)

    def save(self):
        with open(self.model_path, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)
