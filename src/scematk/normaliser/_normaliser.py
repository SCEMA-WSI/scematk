from ..process._processor import Processor
from abc import ABC, abstractmethod
from dask.array import Array

class Normaliser(ABC):
    def __init__(self, name, preprocessor: Processor = None, postprocessor: Processor = None) -> None:
        self.name = name
        if preprocessor is None:
            preprocessor = self._default_preprocessor()
        if postprocessor is None:
            postprocessor = self._default_postprocessor()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.fitted = False

    @abstractmethod
    def fit(self, image: Array) -> None:
        pass

    @abstractmethod
    def run(self, image: Array) -> Array:
        pass

    @abstractmethod
    def fit_and_run(self, image: Array) -> Array:
        pass

    @abstractmethod
    def _default_preprocessor(self) -> Processor:
        pass

    @abstractmethod
    def _default_postprocessor(self) -> Processor:
        pass