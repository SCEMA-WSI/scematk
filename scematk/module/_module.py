from abc import ABC, abstractmethod

from ..image._image import Image
from ..process._process import Processor


class Module(ABC):
    def __init__(
        self,
        name: str,
        preprocessor: Processor | None = None,
        postprocessor: Processor | None = None,
    ) -> None:
        """Constructor for Module class

        Args:
            name (str): name of the module
            preprocessor (Processor | None, optional): Preprocessor. Defaults to None.
            postprocessor (Processor | None, optional): Postprocessor. Defaults to None.
        """
        preprocessor = preprocessor if preprocessor is not None else self._default_preprocessor()
        postprocessor = (
            postprocessor if postprocessor is not None else self._default_postprocessor()
        )
        assert isinstance(name, str), "name must be a string"
        assert isinstance(preprocessor, Processor), "preprocessor must be a Processor instance"
        assert isinstance(postprocessor, Processor), "postprocessor must be a Processor instance"
        self.name = name
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.fitted = False

    @abstractmethod
    def fit(self, image: Image) -> None:
        """Fit the module to the image

        Args:
            image (Image): Image to fit the module to
        """
        pass

    @abstractmethod
    def run(self, image: Image) -> Image:
        """Run the module on the image

        Args:
            image (Image): Image to run the module on

        Returns:
            Image: Image after running the module
        """
        pass

    @abstractmethod
    def fit_and_run(self, image: Image) -> Image:
        """Fit the module to the image and run the module on the image

        Args:
            image (Image): Image to fit the module to and run the module on

        Returns:
            Image: Image after running the module
        """
        pass

    @abstractmethod
    def _default_preprocessor(self) -> Processor:
        """Default preprocessor for the module

        Returns:
            Processor: Default preprocessor
        """
        pass

    @abstractmethod
    def _default_postprocessor(self) -> Processor:
        """Default postprocessor for the module

        Returns:
            Processor: Default postprocessor
        """
        pass
