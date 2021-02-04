from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


class SvmTrainer:
    def __init__(self, model, preprocessor):
        self.model = model.model
        self.preprocessor = preprocessor

    def train(self, mode="normal"):
        if mode == "2step":
            iter = 0
            last_iter = False
            while self.preprocessor.continue_training:
                if iter == 3:
                    last_iter = True
                    self.preprocessor.continue_training = False
                self.preprocessor.generate_split_data(last_iter)
                iter += 1
        data = self.preprocessor.get_train_data()
        self.model.fit(data['labelled_features'], data['labels'])

    def evaluate_all(self, **data):
        test_features = data['features']
        test_labels = data['labels']
        predictions = self.model.predict(test_features)
        acc = accuracy_score(predictions, test_labels)
        precision, recall, _, _ = precision_recall_fscore_support(predictions.astype(int), test_labels.astype(int), average='binary')
        return {"accuracy": acc, "precision": precision, "recall": recall}

    def evaluate_protocol(self, protocol_dct, **data):
        test_features = data['features']
        test_labels = data['labels']
        test_protocol = data['protocol']
        eval_dct = {}
        for proto_name, proto_code in protocol_dct.items():
            proto_features = test_features[test_protocol == proto_code]
            proto_labels = test_labels[test_protocol == proto_code]
            predictions = self.model.predict(proto_features)
            acc = accuracy_score(predictions, proto_labels)

            precision, recall, _, _ = precision_recall_fscore_support(predictions, proto_labels, average='binary')
            eval_dct[proto_name] = {"accuracy": acc, "precision": precision, "recall": recall}
        return eval_dct
