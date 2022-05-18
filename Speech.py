from transformers import pipeline
from preprocessors import Sentensize, RemoveStopwords
import spacy


class Speech:
    def __init__(self, content: str):
        self.content = content

    def get_emotion_scores(self, model: str = "j-hartmann/emotion-english-distilroberta-base") -> dict:
        """

        :param model: model name from Huggingface
        :return:
        """
        classifier = pipeline("text-classification", model=model, return_all_scores=True)
        return classifier(self.content, truncation=True)

    def get_emotion_scores_by_sentence(self, model: str = "j-hartmann/emotion-english-distilroberta-base") -> dict:
        """

        :param model: model name from Huggingface
        :return:
        """
        classifier = pipeline("text-classification", model=model, return_all_scores=True)
        emotion_scores = {}

        for i, sentence in enumerate(Sentensize()(self.content)):
            emotion_scores[i] = classifier(str(sentence), truncation=True)

        return emotion_scores

    def get_summary(self, model: str = "facebook/bart-large-cnn", max_length: int = 256, inplace: bool = False) -> str:
        """

        :param model: model name from Huggingface
        :param max_length: max length of the summary
        :param inplace: if true self.content is replaced by summary
        :return:
        """
        summarizer = pipeline("summarization", model=model)
        summary = summarizer(self.content, truncation=True, max_length=max_length)[0]["summary_text"]
        if inplace:
            self.content = summary
            return self.content
        else:
            return summary

    def get_entities(self, model: str = "en_core_web_md", labels: bool = False):
        nlp = spacy.load(model)
        doc = nlp(RemoveStopwords()(self.content))

        if labels:
            entities = [(ent.text, ent.label_) for ent in doc.ents]
        else:
            entities = [ent.text for ent in doc.ents]

        return entities
