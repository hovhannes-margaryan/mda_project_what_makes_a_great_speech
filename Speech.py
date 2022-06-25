from transformers import pipeline
from preprocessors import Sentensize, RemoveStopwords, Lower
import spacy
import numpy as np
import re
from textblob import TextBlob
from collections import Counter
from preprocessors import *
import textdescriptives as td


class Speech:
    def __init__(self, content: str):
        """
        :param content: raw speech without preprocessing
        """
        self.content = content

    def get_content(self, preprocessors: list = None):
        """
        :param preprocessors: list of preprocessors needed for a specific model
        :return:
        """
        if preprocessors is None:
            preprocessors = [RemoveExtras(names=True)]
        preprocessed = self.apply_preprocessors(preprocessors)
        return preprocessed

    def get_emotion_scores(self, model: str = "j-hartmann/emotion-english-distilroberta-base",
                           preprocessors: list = None, **kwargs) -> dict:
        """
        set return_all_scores=True for the default model
        :param preprocessors: list of preprocessors needed for a specific model
        :param model: model name from Huggingface
        :return: dict of emotion scores
        """
        if preprocessors is None:
            preprocessors = [RemoveExtras(names=True)]
        preprocessed = self.apply_preprocessors(preprocessors)
        pipe = pipeline("text-classification", model=model, **kwargs)
        return pipe(preprocessed, truncation=True)

    def get_emotion_scores_by_sentence(self, model: str = "j-hartmann/emotion-english-distilroberta-base",
                                       preprocessors: list = None, **kwargs) -> dict:
        """
        set return_all_scores=True for the default model
        :param preprocessors: list of preprocessors needed for a specific model
        :param model: model name from Huggingface
        :return: dict of emotion scores for each sentence in speech
        """
        if preprocessors is None:
            preprocessors = [RemoveExtras(names=True)]
        preprocessed = self.apply_preprocessors(preprocessors)

        pipe = pipeline("text-classification", model=model, **kwargs)
        emotion_scores = {}

        for i, sentence in enumerate(Sentensize()(preprocessed)):
            emotion_scores[i] = pipe(str(sentence), truncation=True)

        return emotion_scores

    def get_summary(self, model: str = "facebook/bart-large-cnn", preprocessors=None, max_length: int = 256) -> str:
        """
        :param preprocessors: list of preprocessors needed for a specific model
        :param model: model name from Huggingface
        :param max_length: max length of the summary
        :return: summary of content
        """
        if preprocessors is None:
            preprocessors = [RemoveExtras(names=True)]
        preprocessed = self.apply_preprocessors(preprocessors)
        pipe = pipeline("summarization", model=model)
        summary = pipe(preprocessed, truncation=True, max_length=max_length)[0]["summary_text"]

        return summary

    def get_entities(self, model: str = "en_core_web_md", preprocessors: list = None, labels: bool = False) -> list:
        """
        :param preprocessors: list of preprocessors needed for a specific model
        :param model:  model name from spacy
        :param labels: if true also returns the labels of entities
        :return: a list of entities
        """
        if preprocessors is None:
            preprocessors = [RemoveExtras(names=True), Lower()]
        preprocessed = self.apply_preprocessors(preprocessors)
        nlp = spacy.load(model)
        doc = nlp(preprocessed)
        if labels:
            entities = [(entity[0].title(), entity[1]) for entity in
                        list(dict.fromkeys([(ent.text, ent.label_) for ent in doc.ents]))]
        else:
            entities = [entity.title() for entity in np.unique([ent.text for ent in doc.ents]).tolist()]

        return entities

    def get_speaker(self) -> str:
        """
        assumes content contains the speaker name and follows the pattern of pdf of speeches
        :return:
        """
        preprocessed = self.apply_preprocessors([RemoveExtras(names=False)])
        return preprocessed.split("\n")[0].strip()

    def get_title(self) -> str:
        """
        assumes content contains the title and follows the pattern of pdf of speeches
        :return:
        """
        preprocessed = self.apply_preprocessors([RemoveExtras(names=False)])
        return preprocessed.split("\n")[1].strip()

    def get_year(self) -> str:
        """
        assumes content contains the year and follows the pattern of pdf of speeches
        :return:
        """
        preprocessed = self.apply_preprocessors([RemoveExtras(names=False)])
        return re.search("[0-9][0-9][0-9][0-9]", preprocessed.split("\n")[2]).group(0)

    def get_length(self, model: str = "en_core_web_md") -> int:
        """
        :param model: model name from spacy
        :return: the number of tokens in speech preprocessed by RemoveExtras
        """
        preprocessed = self.apply_preprocessors([RemoveExtras(names=False)])
        nlp = spacy.load(model)
        tokens = [token for token in nlp(preprocessed)]

        return len(tokens)

    def get_proportion_in_speech(self, list_of_words: list, model: str = "en_core_web_md", preprocessors: list = None,
                                 absolute=False) -> float:
        """
        :param preprocessors: list of preprocessors needed for a specific model
        :param list_of_words: list of words to check if speech contains one
        :param model: model name from spacy to be used for tokenization of speech content
        :param absolute: if true retains an absolute count else proportion
        :return: proportion of list_of_words in content
        """
        if preprocessors is None:
            preprocessors = [RemoveExtras(names=True),
                             Lemmatize(),
                             RemovePunctuation(),
                             Lower()]
        preprocessed = self.apply_preprocessors(preprocessors)
        count = 0
        nlp = spacy.load(model)
        tokens_text = [token.text for token in nlp(preprocessed)]
        list_of_words = [word.strip().lower() for word in list_of_words]

        for word in list_of_words:
            if word in tokens_text:
                count += 1

        if absolute:
            return count
        else:
            return count / len(tokens_text)

    def get_mean_sentence_length(self):
        preprocessed = self.apply_preprocessors([RemoveExtras(names=False),
                                                 Sentensize()])
        sentence_lengths = [len(sen) for sen in preprocessed]
        return np.mean(np.array(sentence_lengths))

    def get_subjectivity(self) -> float:
        """
        uses TextBlob
        requires self.content to be preprocessed
        :return: output range: [0-1] - higher value means higher subjectivity
        """
        preprocessors = [RemoveExtras(names=False),
                         RemoveStopwords()]
        preprocessed = self.apply_preprocessors(preprocessors)
        return TextBlob(preprocessed).sentiment.subjectivity

    def get_polarity(self) -> float:
        """
        uses TextBlob
        requires self.content to be preprocessed
        :return: output range: [-1, 1]
        """
        preprocessors = [RemoveExtras(names=False),
                         RemoveStopwords()]
        preprocessed = self.apply_preprocessors(preprocessors)
        return TextBlob(preprocessed).sentiment.polarity

    def get_complexity(self, model: str = "en_core_web_md", preprocessors: list = None) -> float:
        """
        uses textdescriptives
        requires self.content to be preprocessed
                RemoveExtras(names=False),
                RemoveStopwords(),
        :param preprocessors: list of preprocessors needed for a specific model
        :param model: model name from spacy
        :return: Flesch reading ease
                90-100	very easy to read, easily understood by an average 11-year-old student
                80-90	easy to read
                70-80	fairly easy to read
                60-70	easily understood by 13- to 15-year-old students
                50-60	fairly difficult to read
                30-50	difficult to read, best understood by college graduates
                0-30	very difficult to read, best understood by university graduates
        """
        if preprocessors is None:
            preprocessors = [RemoveExtras(names=False),
                             RemoveStopwords()]
        preprocessed = self.apply_preprocessors(preprocessors)
        nlp = spacy.load(model)
        nlp.add_pipe('textdescriptives')
        doc = nlp(preprocessed)

        return doc._.readability["flesch_reading_ease"]

    def get_lexical_richness(self, model: str = "en_core_web_md", preprocessors: list = None) -> float:
        """
        requires self.content to be preprocessed
                RemoveExtras(names=False),
                RemoveStopwords(),
        :param preprocessors: list of preprocessors needed for a specific model
        :param model: model name from spacy
        :return: Hapax richness
        """
        if preprocessors is None:
            preprocessors = [RemoveExtras(names=False),
                             RemoveStopwords()]
        preprocessed = self.apply_preprocessors(preprocessors)
        nlp = spacy.load(model)
        tokens_text = [token.text.strip().lower() for token in nlp(preprocessed)]
        counter = Counter(tokens_text)
        return sum(np.array(list(counter.values())) == 1) / len(tokens_text)

    def apply_preprocessors(self, preprocessors: list):
        """
        :param preprocessors: a list of preprocessors to be applied to content
        :return: preprocessed speech
        """
        preprocessed = self.content
        if preprocessors is None or len(preprocessors) == 0:
            raise Exception("preprocessors should not be none or empty!")

        for preprocessor in preprocessors:
            preprocessed = preprocessor(preprocessed)

        return preprocessed
