from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot


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

        print("predicting")
        predictions_prob = self.model.predict_proba(test_features)[:, 1]
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(test_labels.astype(int), predictions_prob.astype(int))
        # # plot the roc curve for the model
        # pyplot.plot(fpr, tpr)
        # # axis labels
        # pyplot.xlabel('False Positive Rate')
        # pyplot.ylabel('True Positive Rate')
        # pyplot.legend()
        import sklearn.metrics as metrics
        roc_auc = metrics.auc(fpr, tpr)
        import matplotlib.pyplot as plt
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        #
        # # show the plot
        # pyplot.show()
        import numpy as np
        print(fpr)
        print(tpr)
        print(thresholds)

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

    def generate_predictions(self, **data):
        test_features = data['features']
        predictions = self.model.predict(test_features)
        return predictions
