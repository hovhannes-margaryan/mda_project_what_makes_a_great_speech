from transformers import pipeline
from preprocessors import Sentensize


class Speech:
    def __init__(self, content: str):
        self.content = content

    def get_emotion_scores(self, model: str = "j-hartmann/emotion-english-distilroberta-base"):
        """

        :param model: model name from Huggingface
        :return:
        """
        classifier = pipeline("text-classification", model=model, return_all_scores=True)
        return classifier(self.content, truncation=True)

    def get_emotion_scores_by_sentence(self, model: str = "j-hartmann/emotion-english-distilroberta-base"):
        """

        :param model: model name from Huggingface
        :return:
        """
        classifier = pipeline("text-classification", model=model, return_all_scores=True)
        emotion_scores = {}

        for i, sentence in enumerate(Sentensize()(self.content)):
            emotion_scores[i] = classifier(str(sentence), truncation=True)

        return emotion_scores