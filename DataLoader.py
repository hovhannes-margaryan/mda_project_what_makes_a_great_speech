from Dataset import SpeechDataset


class DataLoader:
    def __init__(self, dataset: SpeechDataset):
        self.dataset = dataset
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < self.dataset.__len__():
            data = self.dataset.__getitem__(self._index)
            self._index += 1
            return data
        raise StopIteration