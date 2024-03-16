from abc import ABC, abstractmethod
from dask.array import Array

class Process(ABC):
    def __init__(self, name: str) -> None:
        assert isinstance(name, str), f"Expected name to be str, got {type(name)}"
        self.name = name

    @abstractmethod
    def process(self, image: Array) -> Array:
        pass


    def __str__(self) -> str:
        return f"Process: {self.name}"