from typing import List

from ..module._module import Module
from ..process._process import Processor


class StainDeconvolver(Module):
    def __init__(
        self,
        name: str,
        out_stains: List[str],
        preprocessor: Processor | None = None,
        postprocessor: Processor | None = None,
    ) -> None:
        """Constructor for StainDeconvolver

        Args:
            name (str): Name of the module
            out_stains (List[str]): List of output stains
            preprocessor (Processor, optional): Preprocessor. Defaults to None.
            postprocessor (Processor, optional): Postprocessor. Defaults to None.
        """
        super().__init__(name, preprocessor, postprocessor)
        assert isinstance(out_stains, list)
        assert all(isinstance(stain, str) for stain in out_stains)
        self.out_stains = out_stains
