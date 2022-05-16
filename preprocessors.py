import spacy
import re


class BasicPreprocess:
    def __init__(self, model: str = "en_core_web_md", **params):
        """

        :param model: model name from spacy
        :param params:
        """
        self.params = params
        self.nlp = spacy.load(model)

    def __call__(self, text) -> str:
        raise NotImplemented


class AdvancedPreprocess:
    def __init__(self, model: str = "en_core_web_md", **params):
        """

        :param model: model name from spacy
        :param params:
        """
        self.params = params
        self.nlp = spacy.load(model)

    def __call__(self, text) -> list:
        raise NotImplemented


class RemoveStopwords(BasicPreprocess):
    def __init__(self, model: str = "en_core_web_md", **params):
        super(RemoveStopwords, self).__init__(model, **params)

    def __call__(self, text: str, **params) -> str:
        return " ".join(str(item) for item in [token for token in self.nlp(text) if not token.is_stop])


class Lemmatize(BasicPreprocess):
    def __init__(self, model: str = "en_core_web_md", **params):
        super(Lemmatize, self).__init__(model, **params)

    def __call__(self, text: str, **params) -> str:
        return " ".join(str(item) for item in [token.lemma_ for token in self.nlp(text)])


class RemovePunctuation(BasicPreprocess):
    def __init__(self, model: str = "en_core_web_md", **params):
        super(RemovePunctuation, self).__init__(model, **params)

    def __call__(self, text: str, **params) -> str:
        return " ".join(str(item) for item in [token for token in self.nlp(text) if not token.is_punct])


class Lower(BasicPreprocess):
    def __init__(self, model: str = "en_core_web_md", **params):
        super(Lower, self).__init__(model, **params)

    def __call__(self, text: str, **params) -> str:
        return " ".join(str(item) for item in [token.lower_ for token in self.nlp(text)])


class Sentensize(AdvancedPreprocess):
    def __init__(self, model: str = "en_core_web_md", **params):
        super(Sentensize, self).__init__(model, **params)

    def __call__(self, text: str, **params) -> list:
        return list(self.nlp(text).sents)


class RemoveExtras(BasicPreprocess):
    def __init__(self, model: str = "en_core_web_md", **params):
        super(RemoveExtras, self).__init__(model, **params)

    def __call__(self, text: str) -> str:
        result = text.strip()
        result = result.replace("AAm\nmeerriiccaannRRhheettoorriicc..ccoom\nm", "").replace(
            "Property of AmericanRhetoric.com", "")

        result = re.sub("Transcription by [a-zA-Z]+ [a-zA-Z]\. [a-zA-Z]+\.", "", result)
        result = re.sub("Page [0-9]", "", result)
        result = re.sub("Updated [0-9]+/[0-9]+/[0-9]+", "", result)
        result = re.sub("AUTHENTICITY CERTIFIED: ([a-zA-Z]+( [a-zA-Z]+)+)", "", result)
        result = result.strip()

        if len(self.params.keys()) != 0:
            if self.params["names"]:
                return "\n".join(str(item) for item in result.split("\n")[3:])
        else:
            return result
