from dask.array import Array

class WSI():
    def __init__(self, img: Array, info: dict) -> None:
        assert isinstance(img, Array), 'img must be a dask array'
        assert img.ndim == 3, 'img must be a 3D array'
        assert isinstance(info, dict), 'info must be a dictionary'
        self.img = img
        self.shape = img.shape
        self.ndim = img.ndim
        self.info = info
        self._set_mpp()

    def _set_mpp(self) -> None:
        if "mpp" in self.info:
            self.mpp = self.info["mpp"]
        elif "mpp-x" in self.info and "mpp-y" in self.info:
            if self.info["mpp-x"] == self.info["mpp-y"]:
                self.mpp = self.info["mpp-x"]
            else:
                self.mpp = None
        else:
            self.mpp = None