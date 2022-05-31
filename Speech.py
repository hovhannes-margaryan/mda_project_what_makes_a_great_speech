from transformers import pipeline
from preprocessors import Sentensize, RemoveStopwords, Lower
import spacy
import numpy as np
import re


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

    def get_entities(self, model: str = "en_core_web_md", labels: bool = False) -> list:
        """

        :param model:  model name from spacy
        :param labels: if true also returns the labels of entities
        :return:
        """
        nlp = spacy.load(model)
        doc = nlp(RemoveStopwords()(Lower()(self.content)))

        if labels:
            entities = [(entity[0].title(), entity[1]) for entity in
                        list(dict.fromkeys([(ent.text, ent.label_) for ent in doc.ents]))]
        else:
            entities = [entity.title() for entity in np.unique([ent.text for ent in doc.ents]).tolist()]

        return entities

    def extract_speaker(self) -> str:
        """
        assumes content contains the speaker name and follows the pattern of pdf of speeches
        :return:
        """
        return self.content.split("\n")[0].strip()

    def extract_title(self) -> str:
        """
        assumes content contains the title and follows the pattern of pdf of speeches
        :return:
        """
        return self.content.split("\n")[1].strip()

    def extract_year(self) -> str:
        """
        assumes content contains the year and follows the pattern of pdf of speeches
        :return:
        """
        return re.search("[0-9][0-9][0-9][0-9]", self.content.split("\n")[2]).group(0)

    def get_length(self, model: str = "en_core_web_md") -> int:
        """
        :param model: model name from spacy
        :return:
        """

        nlp = spacy.load(model)
        tokens = [token for token in nlp(self.content)]

        return len(tokens)

    def get_proportion_in_speech(self, list_of_words: list, model: str = "en_core_web_md", absolute=False) -> float:
        """
        :param list_of_words: list of words to check if speech contains one
        :param model: model name from spacy to be used for tokenization of speech content
        :param absolute: if true retains an absolute count else proportion
        :return:
        """
        count = 0
        nlp = spacy.load(model)
        tokens_text = [token.text for token in nlp(self.content)]
        list_of_words = [word.strip().lower() for word in list_of_words]

        for word in list_of_words:
            if word in tokens_text:
                count += 1

        if absolute:
            return count
        else:
            return count / len(tokens_text)

    def get_mean_sentence_length(self):
        sentence_lengths = [len(sen) for sen in Sentensize()(self.content)]
        return np.mean(np.array(sentence_lengths))