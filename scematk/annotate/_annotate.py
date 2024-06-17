from abc import ABC
from typing import List, Optional, Tuple

from shapely.geometry import LineString


class Annotation(ABC):
    def __init__(self, contours: List[LineString], colour: str = "red") -> None:
        """Initialise the Annotation object

        Args:
            contours (List[LineString]): A list of LineString objects representing the contours of the annotation.
            colour (str, optional): The colour of the annotation. Defaults to "red".
        """
        assert isinstance(contours, list)
        if len(contours) > 0:
            assert all(isinstance(contour, LineString) for contour in contours)
        assert isinstance(colour, str)
        self.contours = contours
        self.colour = colour

    def _get_annotation_region(
        self, y_start: int, x_start: int, y_length: int, x_length: int
    ) -> Optional[List[LineString]]:
        """Get the annotation for a region

        Args:
            y_start (int): The y-coordinate of the top-left corner of the region
            x_start (int): The x-coordinate of the top-left corner of the region
            y_length (int): The height of the region
            x_length (int): The width of the region

        Returns:
            List[LineString]: A list of LineString objects representing the contours of the annotation
        """
        return None

    def _get_annotation_thumb(self, coarsen_factor: int) -> List[LineString]:
        """Get the annotation for a thumbnail

        Args:
            coarsen_factor (int): The coarsening factor of the thumbnail

        Returns:
            List[LineString]: A list of LineString objects representing the contours of the annotation
        """
        assert isinstance(coarsen_factor, int), "coarsen_factor must be an integer"
        assert coarsen_factor > 0, "coarsen_factor must be greater than 0"
        contours = self.contours
        return_contours = []
        for contour in contours:
            return_contours.append(
                LineString([(x // coarsen_factor, y // coarsen_factor) for x, y in contour.coords])
            )
        return return_contours

    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the bounding box of the annotation

        Returns:
            Tuple[int, int, int, int]: A tuple representing the bounding box of the annotation
        """
        contours = self.contours
        if len(contours) == 0:
            return None
        bounds = [contour.bounds for contour in contours]
        x_min = int(min([bound[0] for bound in bounds]))
        y_min = int(min([bound[1] for bound in bounds]))
        x_max = int(max([bound[2] for bound in bounds]))
        y_max = int(max([bound[3] for bound in bounds]))
        return (y_min, x_min, y_max - y_min, x_max - x_min)
