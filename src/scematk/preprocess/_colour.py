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
    def __init__(self, as_ubyte: bool = False):
        self.as_ubyte = as_ubyte

    def process(self, image):
        return rgb_to_grey(image, self.as_ubyte)

class RGBToGray(Process):
    def __init__(self, as_ubyte: bool = False):
        self.as_ubyte = as_ubyte

    def process(self, image):
        return rgb_to_gray(image, self.as_ubyte)

class GreyToOD(Process):
    def __init__(self, clip: bool = False):
        self.clip = clip

    def process(self, image):
        return grey_to_od(image, self.clip)

class GrayToOD(Process):
    def __init__(self, clip: bool = False):
        self.clip = clip

    def process(self, image):
        return gray_to_od(image, self.clip)

class RGBToOD(Process):
    def __init__(self, clip: bool = False):
        self.clip = clip

    def process(self, image):
        return rgb_to_od(image, self.clip)
