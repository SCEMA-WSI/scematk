from ...image._image import Image
from .._process import Process


class Rechunk(Process):
    def __init__(self, chunks: tuple) -> None:
        """Rechunk image to new chunks

        Args:
            chunks (tuple): New chunks
        """
        super().__init__(f"Rechunk image to {str(chunks)}")
        self.chunks = chunks

    def run(self, image: Image) -> Image:
        """Rechunk image to new chunks

        Args:
            image (Image): SCEMATK Image object

        Returns:
            Image: SCEMATK Image object
        """
        image.rechunk(self.chunks)
        return image
