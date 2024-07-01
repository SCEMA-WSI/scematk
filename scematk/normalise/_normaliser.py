from typing import Optional

from ..module._module import Module
from ..process._process import Processor


class Normaliser(Module):
    def __init__(
        self,
        name: str,
        preprocessor: Optional[Processor] = None,
        postprocessor: Optional[Processor] = None,
    ):
        """Constructor for normaliser class

        Args:
            name (str): Name of the normaliser
            preprocessor (Optional[Processor]): Preprocessor for the module. Defaults to None.
            postprocessor (Optional[Processor]): Postprocessor for the module. Defaults to None.
        """
        super().__init__(name, preprocessor, postprocessor)
