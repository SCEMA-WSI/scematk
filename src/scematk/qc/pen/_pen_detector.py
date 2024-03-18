from .._qc_step import QCStep
from ...process._processor import Processor

class PenDetector(QCStep):
    def __init__(self, name: str, preprocessor: Processor = None, postprocessor: Processor = None) -> None:
        super().__init__(name, preprocessor, postprocessor)