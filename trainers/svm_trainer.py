from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


class SvmTrainer:
    def __init__(self, model, **data):
        self.model = model.model
        self.train_features = data['features']
        self.train_labels = data['labels']

    def train(self):
        self.model.fit(self.train_features, self.train_labels)

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
