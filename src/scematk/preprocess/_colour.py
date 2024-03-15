from ._process import Process
from ..colour._grey_transform import (
    rgb_to_grey,
    rgb_to_gray
)
from ..colour._od_transform import (
    grey_to_od,
    gray_to_od,
    rgb_to_od
)

class RGBToGrey(Process):
    def __init__(self):
        pass

    def process(self, image):
        return rgb_to_grey(image)

class RGBToGray(Process):
    def __init__(self):
        pass

    def process(self, image):
        return rgb_to_gray(image)

class GreyToOD(Process):
    def __init__(self):
        pass

    def process(self, image):
        return grey_to_od(image)

class GrayToOD(Process):
    def __init__(self):
        pass

    def process(self, image):
        return gray_to_od(image)

class RGBToOD(Process):
    def __init__(self):
        pass

    def process(self, image):
        return rgb_to_od(image)
