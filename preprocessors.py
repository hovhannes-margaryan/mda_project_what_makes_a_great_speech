import spacy

class BasicPreprocess:
    def __init__(self, model: str = "en_core_web_md"):
        self.nlp = spacy.load(model)

    def __call__(self, text: str, **params) -> str:
        raise NotImplemented

class AdvancedPreprocess:
    def __init__(self, model: str = "en_core_web_md"):
        self.nlp = spacy.load(model)

    def __call__(self, text: str, **params) -> list:
        raise NotImplemented

class RemoveStopwords(BasicPreprocess):
    def __init__(self, model: str = "en_core_web_md"):
        super(RemoveStopwords, self).__init__(model)

    def __call__(self, text: str, **params) -> str:
        return " ".join(str(item) for item in [token for token in self.nlp(text) if not token.is_stop])


class Lemmatize(BasicPreprocess):
    def __init__(self, model: str = "en_core_web_md"):
        super(Lemmatize, self).__init__(model)

    def __call__(self, text: str, **params) -> str:
        return " ".join(str(item) for item in [token.lemma_ for token in self.nlp(text)])


class RemovePunctuation(BasicPreprocess):
    def __init__(self, model: str = "en_core_web_md"):
        super(RemovePunctuation, self).__init__(model)

    def __call__(self, text: str, **params) -> str:
        return " ".join(str(item) for item in [token for token in self.nlp(text) if not token.is_punct])


class Lower(BasicPreprocess):
    def __init__(self, model: str = "en_core_web_md"):
        super(Lower, self).__init__(model)

    def __call__(self, text: str, **params) -> str:
        return " ".join(str(item) for item in [token.lower_ for token in self.nlp(text)])


class Sentensize(AdvancedPreprocess):
    def __init__(self, model: str = "en_core_web_md"):
        super(Sentensize, self).__init__(model)

    def __call__(self, text: str, **params) -> list:
        return list(self.nlp(text).sents)


class RemoveExtras(BasicPreprocess):
    def __init__(self,  model: str = "en_core_web_md"):
        super(RemoveExtras, self).__init__(model)

    def __call__(self, *args, **kwargs):
        raise NotImplemented


