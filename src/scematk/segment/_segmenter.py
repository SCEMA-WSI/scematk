from abc import ABC, abstractmethod
from dask.array import Array

class Segmenter(ABC):
    def __init__(self) -> None:
        self.fitted = False

    @abstractmethod
    def fit(self, image: Array) -> None:
        pass

    @abstractmethod
    def transform(self, image: Array) -> Array:
        pass

    @abstractmethod
    def fit_and_transform(self, image: Array) -> Array:
        pass