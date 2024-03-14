from ._process import Process
from dask.array import Array

class Preprocessor():
    def __init__(self):
        self.processes = []

    def add_process(self, process: Process) -> None:
        assert isinstance(process, Process), f"Expected process to be Process, got {type(process)}"
        self.processes.append(process)

    def process(self, image: Array) -> Array:
        for process in self.processes:
            image = process.process(image)
        return image