from ..process._processor import Processor
from abc import ABC, abstractmethod
from dask.array import Array

class QCStep(ABC):
    def __init__(self, name, preprocessor: Processor = None, postprocessor: Processor = None) -> None:
        self.name = name
        preprocessor = preprocessor or self._default_preprocessor()
        postprocessor = postprocessor or self._default_postprocessor()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

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
        return Processor()

    @abstractmethod
    def _default_postprocessor(self) -> Processor:
        return Processor()