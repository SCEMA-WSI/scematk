from ._fold_detector import FoldDetector
from ...process._processor import Processor

class RulesBasedFoldDetector(FoldDetector):
    def __init__(self, preprocessor: Processor = None, postprocessor: Processor = None) -> None:
        super().__init__("Rules Based Tissue Fold Detector", preprocessor, postprocessor)
