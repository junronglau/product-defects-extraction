class TextRankTrainer:
    def __init__(self, model, preprocessor):
        self.model = model.model
        self.preprocessor = preprocessor
        self.num_sentences = 2

    def train(self):
        for topic, parser in self.preprocessor:
            summary = self.model(parser.document, self.num_sentences)
            text_summary = ""
            for sentence in summary:
                text_summary+=str(sentence)
                print("==={topic}===\n{summary}".format(topic=topic,summary=text_summary))
