from dask_image.ndfilters import gaussian_filter

from ...image._image import Image
from ...image._ubyte_image import UByteImage
from .._process import Process


class GaussianBlur(Process):
    def __init__(self, sigma: float, metric: str = "micron", truncate: float = 3.0) -> None:
        """Constructor for GaussianBlur

        Args:
            sigma (float): Sigma of the Gaussian filter
            metric (str, optional): Units of the sigma. Defaults to 'micron'.
            truncate (float, optional): Truncate the filter at this many standard deviations. Defaults to 3.0.
        """
        assert isinstance(sigma, (int, float)), "sigma must be a number"
        assert sigma > 0, "sigma must be positive"
        assert isinstance(metric, str), "metric must be a string"
        assert metric in ["micron", "pixel"], 'metric must be either "micron" or "pixel"'
        assert isinstance(truncate, (int, float)), "truncate must be a number"
        assert truncate > 0, "truncate must be positive"
        self.sigma = sigma
        self.metric = metric
        self.truncate = truncate
        super().__init__(name=f"Gaussian Blur with a sigma of {sigma} {metric}s")

    def run(self, image: Image) -> Image:
        """Run the GaussianBlur process

        Args:
            image (Image): SCEMATK Image object

        Raises:
            NotImplementedError: GaussianBlur only supports UByteImage objects at this time

        Returns:
            Image: Image object with the GaussianBlur applied
        """
        assert isinstance(image, Image), "image must be an Image"
        sigma = image.pixel_from_micron(self.sigma) if self.metric == "micron" else self.sigma
        truncate = self.truncate
        img = image.image
        if isinstance(image, UByteImage):
            img = img.astype("float32")
        img = gaussian_filter(img, sigma=(sigma, sigma, 0), mode="mirror", truncate=truncate)
        if isinstance(image, UByteImage):
            img = img.astype("uint8")
            return UByteImage(img, image.info, image.channel_names)
        else:
            raise NotImplementedError("GaussianBlur only supports UByteImage objects at this time")
