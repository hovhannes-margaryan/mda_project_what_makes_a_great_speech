from transformers import pipeline
from preprocessors import Sentensize, RemoveStopwords, Lower
import spacy
import numpy as np
import re
from textblob import TextBlob
from collections import Counter


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

    def get_subjectivity(self) -> float:
        """
        uses TextBlob
        requires self.content to be preprocessed
        :return: output range: [0-1] - higher value means higher subjectivity
        """

        return TextBlob(self.content).sentiment.subjectivity

    def get_polarity(self) -> float:
        """
        uses TextBlob
        requires self.content to be preprocessed
        :return: output range: [-1, 1]
        """
        return TextBlob(self.content).sentiment.polarity

    def get_complexity(self, model: str = "en_core_web_md") -> float:
        """
        uses textdescriptives
        requires self.content to be preprocessed
                RemoveExtras(names=False),
                RemoveStopwords(),
                RemovePunctuation(),
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
        nlp = spacy.load(model)
        nlp.add_pipe("textdescriptives")
        doc = nlp(self.content)

        return doc._.readability["flesch_reading_ease"]

    def get_lexical_richness(self, model: str = "en_core_web_md") -> float:
        """
        requires self.content to be preprocessed
                RemoveExtras(names=False),
                RemoveStopwords(),
                RemovePunctuation(),
        :param model: model name from spacy
        :return: Hapax richness
        """
        nlp = spacy.load(model)
        tokens_text = [token.text.strip().lower() for token in nlp(self.content)]
        counter = Counter(tokens_text)
        return sum(np.array(list(counter.values())) == 1) / len(tokens_text)




