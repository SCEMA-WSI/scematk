from .._process import Process
from dask.array import Array
from dask_image.ndmorph import binary_closing, binary_dilation, binary_erosion, binary_opening

class BinaryClosing(Process):
    def __init__(
        self,
        structure = None,
        iterations=1,
        origin=0,
        mask=None,
        border_value=0,
        brute_force=False
    ) -> None:
        super().__init__("Perform binary closing on a 2D array.")
        self.structure = structure
        self.iterations = iterations
        self.origin = origin
        self.mask = mask
        self.border_value = border_value
        self.brute_force = brute_force

    def process(self, image: Array) -> Array:
        assert isinstance(image, Array), "image must be a dask array"
        assert image.ndim == 2, "image must be a 2D array"
        return binary_closing(
            image,
            structure=self.structure,
            iterations=self.iterations,
            origin=self.origin,
            mask=self.mask,
            border_value=self.border_value,
            brute_force=self.brute_force
        )

class BinaryDilation(Process):
    def __init__(
        self,
        structure = None,
        iterations=1,
        origin=0,
        mask=None,
        border_value=0,
        brute_force=False
    ) -> None:
        super().__init__("Perform binary dilation on a 2D array.")
        self.structure = structure
        self.iterations = iterations
        self.origin = origin
        self.mask = mask
        self.border_value = border_value
        self.brute_force = brute_force

    def process(self, image: Array) -> Array:
        assert isinstance(image, Array), "image must be a dask array"
        assert image.ndim == 2, "image must be a 2D array"
        return binary_dilation(
            image,
            structure=self.structure,
            iterations=self.iterations,
            origin=self.origin,
            mask=self.mask,
            border_value=self.border_value,
            brute_force=self.brute_force
        )

class BinaryErosion(Process):
    def __init__(
        self,
        structure = None,
        iterations=1,
        origin=0,
        mask=None,
        border_value=0,
        brute_force=False
    ) -> None:
        super().__init__("Perform binary erosion on a 2D array.")
        self.structure = structure
        self.iterations = iterations
        self.origin = origin
        self.mask = mask
        self.border_value = border_value
        self.brute_force = brute_force

    def process(self, image: Array) -> Array:
        assert isinstance(image, Array), "image must be a dask array"
        assert image.ndim == 2, "image must be a 2D array"
        return binary_erosion(
            image,
            structure=self.structure,
            iterations=self.iterations,
            origin=self.origin,
            mask=self.mask,
            border_value=self.border_value,
            brute_force=self.brute_force
        )

class BinaryOpening(Process):
    def __init__(
        self,
        structure = None,
        iterations=1,
        origin=0,
        mask=None,
        border_value=0,
        brute_force=False
    ) -> None:
        super().__init__("Perform binary opening on a 2D array.")
        self.structure = structure
        self.iterations = iterations
        self.origin = origin
        self.mask = mask
        self.border_value = border_value
        self.brute_force = brute_force

    def process(self, image: Array) -> Array:
        assert isinstance(image, Array), "image must be a dask array"
        assert image.ndim == 2, "image must be a 2D array"
        return binary_opening(
            image,
            structure=self.structure,
            iterations=self.iterations,
            origin=self.origin,
            mask=self.mask,
            border_value=self.border_value,
            brute_force=self.brute_force
        )