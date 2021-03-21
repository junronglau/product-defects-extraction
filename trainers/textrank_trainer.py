from rouge_score import rouge_scorer


class TextRankTrainer:
    def __init__(self, model, preprocessor):
        self.model = model.model
        self.preprocessor = preprocessor
        self.num_sentences = 3
        self.scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    def train_and_evaluate(self, test_data):
        data = self.preprocessor.get_train_data()
        avg_prec = avg_recall = avg_f1 = 0
        n = len(data['features'])
        for topic, parser in data['features']:
            summary = self.model(parser.document, self.num_sentences)
            text_summary = ""
            for sentence in summary:
                text_summary += str(sentence)
            scores = self.scorer.score(text_summary, test_data[test_data['topic'] == topic]['text'].iloc[-1])
            print(f"==={topic}===\n{text_summary}\nscores:{scores}")
            avg_prec += scores['rouge1'].precision
            avg_recall += scores['rouge1'].recall
            avg_f1 += scores['rouge1'].fmeasure

        return avg_prec/n, avg_recall/n, avg_f1/n
