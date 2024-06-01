from scematk.process._process import Processor

from .._segmenter import Segmenter


class TissueSegmenter(Segmenter):
    def __init__(
        self,
        name: str,
        preprocessor: Processor | None = None,
        postprocessor: Processor | None = None,
    ):
        """Constructor for the TissueSegmenter class

        Args:
            name (str): Name of the segmenter
            preprocessor (Processor | None, optional): Preprocessor. Defaults to None.
            postprocessor (Processor | None, optional): Postprocessor. Defaults to None.
        """
        super().__init__(name, preprocessor, postprocessor)
