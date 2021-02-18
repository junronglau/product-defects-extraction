from sumy.summarizers.text_rank import TextRankSummarizer
import pickle


class TextRankModel:
    def __init__(self, config, load=False):
        self.model_path = config.defects_summarizer.paths.save_model_path
        if load:
            self.model = pickle.load(open(self.model_path, "rb"))
        else:
            self.model = TextRankSummarizer()

    def save(self):
        with open(self.model_path, 'wb') as output:
            pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)






