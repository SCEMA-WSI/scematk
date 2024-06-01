from ..module._module import Module
from ..process._process import Processor


class Segmenter(Module):
    def __init__(
        self,
        name: str,
        preprocessor: Processor | None = None,
        postprocessor: Processor | None = None,
    ):
        """Constructor for the Segmenter class

        Args:
            name (str): Name of the segmenter
            preprocessor (Processor | None, optional): Preprocessor. Defaults to None.
            postprocessor (Processor | None, optional): Postprocessor. Defaults to None.
        """
        super().__init__(name, preprocessor, postprocessor)
