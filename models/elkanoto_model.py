from pulearn import ElkanotoPuClassifier
from sklearn import svm
import pickle

class ElkanotoModel:
    def __init__(self, config, load=False):
        self.model_path = config.defects_classifier.paths.save_model_path
        if load:
            self.model = pickle.load(open(self.model_path, "rb"))
        else:
            svc = svm.SVC(C=10, kernel='linear', probability=True)
            self.model = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)

    def save(self):
        with open(self.model_path, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)
