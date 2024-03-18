from ..process._processor import Processor
from abc import ABC, abstractmethod
from dask.array import Array

class Segmenter(ABC):
    def __init__(self, name: str, preprocessor: Processor, postprocessor: Processor) -> None:
        assert isinstance(name, str), "Name must be a string"
        if preprocessor is None:
            preprocessor = self._default_preprocessor()
        if postprocessor is None:
            postprocessor = self._default_postprocessor()
        assert isinstance(preprocessor, Processor), "Preprocessor must be a Processor"
        assert isinstance(postprocessor, Processor), "Postprocessor must be a Processor"
        self.name = name
        self.fitted = False
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @abstractmethod
    def fit(self, image: Array) -> None:
        pass

    @abstractmethod
    def _default_preprocessor(self) -> Processor:
        pass

    @abstractmethod
    def _default_postprocessor(self) -> Processor:
        pass

class PrimarySegmenter(Segmenter):
    def __init__(self, name: str, preprocessor: Processor = None, postprocessor: Processor = None) -> None:
        super().__init__(name, preprocessor, postprocessor)

    @abstractmethod
    def segment(self, image: Array) -> Array:
        pass

    @abstractmethod
    def fit_and_segment(self, image: Array) -> Array:
        pass

class SecondarySegmenter(Segmenter):
    def __init__(self, name: str, preprocessor: Processor = None, postprocessor: Processor = None) -> None:
        super().__init__(name, preprocessor, postprocessor)

    @abstractmethod
    def segment(self, image: Array, mask: Array) -> Array:
        pass

    @abstractmethod
    def fit_and_segment(self, image: Array, mask: Array) -> Array:
        pass

class TertiarySegmenter(Segmenter):
    def __init__(self, name: str, preprocessor: Processor = None, postprocessor: Processor = None) -> None:
        super().__init__(name, preprocessor, postprocessor)

    @abstractmethod
    def segment(self, image: Array, mask_primary: Array, mask_secondary: Array) -> Array:
        pass

    @abstractmethod
    def fit_and_segment(self, image: Array, mask_primary: Array, mask_secondary: Array) -> Array:
        pass