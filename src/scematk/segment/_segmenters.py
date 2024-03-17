from abc import ABC, abstractmethod
from dask.array import Array

class Segmenter(ABC):
    def __init__self(self, name: str) -> None:
        assert isinstance(name, str)
        self.name = name
        self.fitted = False

    @abstractmethod
    def fit(self, image: Array) -> None:
        pass

class PrimarySegmenter(Segmenter):
    def __init__(self, name: str) ->  None:
        super().__init__(name)

    @abstractmethod
    def segment(self, image: Array) -> Array:
        pass

    @abstractmethod
    def fit_and_segment(self, image: Array) -> Array:
        pass

class SecondarySegmenter(Segmenter):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @abstractmethod
    def segment(self, image: Array, mask: Array) -> Array:
        pass

    @abstractmethod
    def fit_and_segment(self, image: Array, mask: Array) -> Array:
        pass

class TertiarySegmenter(Segmenter):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @abstractmethod
    def segment(self, image: Array, mask_primary: Array, mask_secondary: Array) -> Array:
        pass

    @abstractmethod
    def fit_and_segment(self, image: Array, mask_primary: Array, mask_secondary: Array) -> Array:
        pass