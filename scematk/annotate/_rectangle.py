from shapely.geometry import LineString

from ._annotate import Annotation


class Rectangle(Annotation):
    def __init__(
        self, y_start: int, x_start: int, y_length: int, x_length: int, colour: str = "red"
    ) -> None:
        assert isinstance(y_start, int), "y_start must be an integer"
        assert isinstance(x_start, int), "x_start must be an integer"
        assert isinstance(y_length, int), "y_length must be an integer"
        assert isinstance(x_length, int), "x_length must be an integer"
        rectangle = LineString(
            [
                (x_start, y_start),
                (x_start + x_length, y_start),
                (x_start + x_length, y_start + y_length),
                (x_start, y_start + y_length),
                (x_start, y_start),
            ]
        )
        super().__init__([rectangle], colour=colour)
