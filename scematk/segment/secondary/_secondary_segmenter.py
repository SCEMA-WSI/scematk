from typing import Optional

from ...process._process import Processor
from .._segmenter import Segmenter


class SecondarySegmenter(Segmenter):
    def __init__(
        self,
        name: str,
        preprocessor: Optional[Processor] = None,
        postprocessor: Optional[Processor] = None,
    ):
        """Secondary segmenter class.

        Args:
            name (str): Name of the segmenter.
            preprocessor (Processor, optional): Processor to apply to the image before running the segmenter. Defaults to None.
            postprocessor (Processor, optional): Processor to apply to the image after running the segmenter. Defaults to None.
        """
        super().__init__(name, preprocessor, postprocessor)
