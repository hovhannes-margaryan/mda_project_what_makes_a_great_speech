import os
import glob
import fitz
from DataLoader import DataLoader


class Dataset:
    def __init__(self, path: str, preprocessors: list):
        self.paths = glob.glob(os.path.join(path, "*"))
        self.preprocessors = preprocessors

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with fitz.open(self.paths[index]) as doc:
            data = ""
            for page in doc:
                data += page.get_text()

        for preprocessor in self.preprocessors:
            data = preprocessor(data)

        return data