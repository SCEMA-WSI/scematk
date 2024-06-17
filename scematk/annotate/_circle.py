import numpy as np
from shapely.geometry import LineString

from ._annotate import Annotation


class Circle(Annotation):
    def __init__(self, y_center: int, x_center: int, radius: int, colour: str = "red") -> None:
        assert isinstance(y_center, int), "y_center must be an integer"
        assert isinstance(x_center, int), "x_center must be an integer"
        assert isinstance(radius, int), "radius must be an integer"
        circle = [
            (x_center + int(radius * np.cos(theta)), y_center + int(radius * np.sin(theta)))
            for theta in np.linspace(0, 2 * np.pi, 100)
        ]
        circle.append(circle[0])
        circle = LineString(circle)
        super().__init__([circle], colour=colour)
