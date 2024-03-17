from ..process._processor import Processor
from abc import ABC, abstractmethod
from dask.array import Array

class Segmenter(ABC):
    def __init__self(self, name: str, preprocessor: Processor, postprocessor: Processor) -> None:
        assert isinstance(name, str), "Name must be a string"
        assert isinstance(preprocessor, Processor), "Preprocessor must be an instance of Processor"
        assert isinstance(postprocessor, Processor), "Postprocessor must be an instance of Processor"
        self.name = name
        self.fitted = False
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @abstractmethod
    def fit(self, image: Array) -> None:
        pass

class PrimarySegmenter(Segmenter):
    def __init__(self, name: str, preprocessor: Processor, postprocessor: Processor) -> None:
        super().__init__(name, preprocessor, postprocessor)

    @abstractmethod
    def segment(self, image: Array) -> Array:
        pass

    @abstractmethod
    def fit_and_segment(self, image: Array) -> Array:
        pass

class SecondarySegmenter(Segmenter):
    def __init__(self, name: str, preprocessor: Processor, postprocessor: Processor) -> None:
        super().__init__(name, preprocessor, postprocessor)

    @abstractmethod
    def segment(self, image: Array, mask: Array) -> Array:
        pass

    @abstractmethod
    def fit_and_segment(self, image: Array, mask: Array) -> Array:
        pass

class TertiarySegmenter(Segmenter):
    def __init__(self, name: str, preprocessor: Processor, postprocessor: Processor) -> None:
        super().__init__(name, preprocessor, postprocessor)

    @abstractmethod
    def segment(self, image: Array, mask_primary: Array, mask_secondary: Array) -> Array:
        pass

    @abstractmethod
    def fit_and_segment(self, image: Array, mask_primary: Array, mask_secondary: Array) -> Array:
        pass