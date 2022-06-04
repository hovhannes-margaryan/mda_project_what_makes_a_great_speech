import os
import glob
import fitz
from Speech import Speech


class BasicDataset:
    def __init__(self, path: str, preprocessors: list):
        self.paths = glob.glob(os.path.join(path, "*"))
        self.preprocessors = preprocessors

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> str:
        with fitz.open(self.paths[index]) as doc:
            data = ""
            for page in doc:
                data += page.get_text()

        for preprocessor in self.preprocessors:
            data = preprocessor(data)

        return data


class SpeechDataset:
    def __init__(self, path: str):
        self.paths = glob.glob(os.path.join(path, "*"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Speech:
        with fitz.open(self.paths[index]) as doc:
            data = ""
            for page in doc:
                data += page.get_text()
        speech = Speech(data)

        return speech
