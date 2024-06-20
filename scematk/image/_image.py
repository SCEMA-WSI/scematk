import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from dask.array import Array
from matplotlib_scalebar.scalebar import ScaleBar
from numpy import ndarray
from stardist import random_label_cmap

from ..annotate._annotate import Annotation


class Image(ABC):
    def __init__(self, image: Array, info: dict, channel_names: List[str]) -> None:
        """Constructor for the base Image class

        Args:
            image (Array): The WSI image as a dask array
            info (dict): A dictionary containing metadata about the image
            channel_names (List[str]): A list of strings containing the names of the channels in the image
        """
        assert isinstance(image, Array), "image must be a dask array"
        assert image.ndim in [2, 3], "image must be 2D or 3D"
        assert isinstance(info, dict), "info must be a dictionary"
        if not isinstance(channel_names, list):
            channel_names = [channel_names]
        assert all(
            isinstance(name, str) for name in channel_names
        ), "All channel names must be strings"
        if image.ndim == 3:
            assert (
                len(channel_names) == image.shape[2]
            ), "number of channel names must match number of channels in image"
        else:
            assert (
                len(channel_names) == 1
            ), "number of channel names must match number of channels in image"
        self.image = image
        self.info = info
        self.ndim = image.ndim
        self.shape = image.shape
        self.dtype = str(image.dtype)
        self.name = info.get("name", None)
        if "mpp" in self.info:
            self.mpp = float(self.info["mpp"])
        elif "mpp-x" in self.info and "mpp-y" in self.info:
            if float(self.info["mpp-x"]) == float(self.info["mpp-y"]):
                self.mpp = float(self.info["mpp-x"])
            else:
                self.mpp = 0.0
        else:
            self.mpp = 0.0
        self.channel_names = channel_names
        self.interpolation_strat = "antialiased"

    def set_channel_names(self, channel_names: List[str] | str) -> None:
        if isinstance(channel_names, str):
            channel_names = [channel_names]
        assert isinstance(channel_names, list), "channel_names must be a list of strings"
        assert all(
            isinstance(name, str) for name in channel_names
        ), "all elements of channel_names must be strings"
        if self.ndim == 3:
            assert (
                len(channel_names) == self.shape[2]
            ), "number of channel names must match number of channels in image"
        else:
            assert (
                len(channel_names) == 1
            ), "number of channel names must match number of channels in image"
        self.channel_names = channel_names

    def save_image(self, path: str, overwrite: bool = False) -> None:
        """Save the WSI image to a zarr file

        Args:
            path (str): The path to save the zarr file
            overwrite (bool, optional): Whether to overwrite a preexisting file if it exists. Defaults to False.
        """
        assert isinstance(path, str), "path must be a string"
        assert isinstance(overwrite, bool), "overwrite must be a boolean"
        assert overwrite or not os.path.exists(path), "zarr file already exists"
        da.to_zarr(self.image, path, overwrite=True)

    def save_info(self, path: str, overwrite: bool = False) -> None:
        """Save the WSI metadata to a json file

        Args:
            path (str): The path to save the json file
            overwrite (bool, optional): Whether to overwrite a preexisting file if it exists. Defaults to False.
        """
        assert isinstance(path, str), "path must be a string"
        assert isinstance(overwrite, bool), "overwrite must be a boolean"
        assert overwrite or not os.path.exists(path), "info file already exists"
        with open(path, "w") as f:
            json.dump(self.info, f)

    def save(self, image_path: str, info_path: str, overwrite: bool = False) -> None:
        """Save the WSI image and metadata to files

        Args:
            image_path (str): The path to save the zarr file
            info_path (str): The path to save the json file
            overwrite (bool, optional): Whether to overwrite a preexisting file if it exists. Defaults to False.
        """
        self.save_image(image_path, overwrite)
        self.save_info(info_path, overwrite)

    def rechunk(self, chunks: tuple) -> None:
        """Rechunk the WSI image

        Args:
            chunks (tuple): The new chunk shape
        """
        assert isinstance(chunks, tuple), "chunks must be a tuple"
        assert all(
            isinstance(chunk, int) for chunk in chunks
        ), "all elements of chunks must be integers"
        assert len(chunks) == self.ndim, "length of chunks must match number of dimensions in image"
        assert all(chunk > 0 for chunk in chunks), "all elements of chunks must be positive"
        if self.ndim == 3:
            assert chunks[2] == len(
                self.channel_names
            ), "third element of chunks must match number of channels in image"
        self.image = self.image.rechunk(chunks)

    def pixel_from_micron(self, micron: float) -> float:
        """Convert microns to pixels according to the microns per pixel (mpp) of the image

        Args:
            micron (float): The number of microns to convert to pixels

        Raises:
            ValueError: Microns per pixel (mpp) not available

        Returns:
            float: The number of pixels equivalent to the given number of microns
        """
        assert isinstance(micron, (int, float)), "micron must be a number"
        assert micron > 0, "micron must be positive"
        if not self.mpp or not isinstance(self.mpp, (int, float)) or self.mpp <= 0:
            raise ValueError("Microns per pixel (mpp) not available")
        return micron / self.mpp

    def read_region(
        self,
        y_min: int,
        x_min: int,
        y_len: int,
        x_len: int,
        pad: bool = True,
        channel: str | None = None,
    ) -> ndarray:
        """Read a region of the WSI image

        Args:
            y_min (int): The minimum y-coordinate of the region
            x_min (int): The minimum x-coordinate of the region
            y_len (int): The length of the region in the y-direction
            x_len (int): The length of the region in the x-direction
            pad (bool, optional): Whether to pad regions outside the image bounds with zeros. Defaults to True.
            channel (str, optional): A channel name. If specified, only the requested channel will be returned. Defaults to None.

        Raises:
            ValueError: Cannot specify channel for single channel image

        Returns:
            ndarray: The region of the image as a numpy array
        """
        assert isinstance(y_min, int), "y_min must be an integer"
        assert isinstance(x_min, int), "x_min must be an integer"
        assert isinstance(y_len, int), "y_len must be an integer"
        assert isinstance(x_len, int), "x_len must be an integer"
        assert isinstance(pad, bool), "pad must be a boolean"
        assert isinstance(channel, (str, type(None))), "channel must be a string or None"
        assert y_min >= 0 or pad, "y_min must be non-negative if no padding is being applied"
        assert x_min >= 0 or pad, "x_min must be non-negative if no padding is being applied"
        assert y_len > 0, "y_len must be positive"
        assert x_len > 0, "x_len must be positive"
        assert (
            y_min + y_len <= self.shape[0] or pad
        ), "y_min + y_len must be less than or equal to the height of the image if no padding is being applied"
        assert (
            x_min + x_len <= self.shape[1] or pad
        ), "x_min + x_len must be less than or equal to the width of the image if no padding is being applied"
        assert (
            channel is None or channel in self.channel_names
        ), "channel must be one of the channel names"
        y_pad = [0, 0]
        x_pad = [0, 0]
        y_pad[0] = max(0, -y_min) if y_min < 0 else 0
        y_pad[1] = max(0, y_min + y_len - self.shape[0]) if y_min + y_len > self.shape[0] else 0
        x_pad[0] = max(0, -x_min) if x_min < 0 else 0
        x_pad[1] = max(0, x_min + x_len - self.shape[1]) if x_min + x_len > self.shape[1] else 0
        y_max = min(self.shape[0], y_min + y_len)
        x_max = min(self.shape[1], x_min + x_len)
        y_min = max(0, y_min)
        x_min = max(0, x_min)
        region = self.image[y_min:y_max, x_min:x_max].compute()
        if pad:
            if self.ndim == 2:
                region = np.pad(
                    region,
                    ((y_pad[0], y_pad[1]), (x_pad[0], x_pad[1])),
                    mode="constant",
                    constant_values=0,
                )
            else:
                region = np.pad(
                    region,
                    ((y_pad[0], y_pad[1]), (x_pad[0], x_pad[1]), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
        if channel:
            if self.ndim == 2:
                raise ValueError("Cannot specify channel for single channel image")
            channel_index = self.channel_names.index(channel)
            region = region[..., channel_index]
        return region

    def show_region(
        self,
        y_min: int,
        x_min: int,
        y_len: int,
        x_len: int,
        pad: bool = True,
        channel: str | None = None,
        scalebar: bool = True,
        scalebar_location: str = "lower right",
        overlay: Optional["Image"] = None,
        invert_overlay: bool = False,
        overlay_cmap: str | None = None,
    ) -> None:
        """Display a region of the WSI image

        Args:
            y_min (int): The minimum y-coordinate of the region
            x_min (int): The minimum x-coordinate of the region
            y_len (int): The length of the region in the y-direction
            x_len (int): The length of the region in the x-direction
            pad (bool, optional): Whether to pad regions outside the image bounds with zeros. Defaults to True.
            channel (str, optional): A channel name. If specified, only the requested channel will be displayed. Defaults to None.
            scalebar (bool, optional): Whether to include a scalebar. Defaults to True.
            scalebar_location (str, optional): The location of the scalebar, if it is to be included. Defaults to "lower right".
            overlay (Image, optional): An image object to overlay on the region. Defaults to None.
            invert_overlay (bool, optional): Whether to invert the overlay image. Defaults to False.
            overlay_cmap (str, optional): The colour map to use for the overlay image. Defaults to None.

        Raises:
            NotImplementedError: Only 1, 2, or 3 channels supported
        """
        assert isinstance(scalebar, bool), "scalebar must be a boolean"
        assert isinstance(scalebar_location, str), "scalebar_location must be a string"
        assert isinstance(overlay, (Image, type(None))), "overlay must be an Image object or None"
        assert isinstance(invert_overlay, bool), "invert_overlay must be a boolean"
        assert isinstance(overlay_cmap, (str, type(None))), "overlay_cmap must be a string or None"
        if "mpp" not in self.info:
            scalebar = False
        region = self.read_region(y_min, x_min, y_len, x_len, pad, channel)
        channel_names = self.channel_names
        if len(channel_names) == 1 or channel:
            region = np.squeeze(region)
            cmap = "gray"
        elif len(channel_names) == 2 and not channel:
            region = np.pad(region, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
            cmap = None
        elif len(channel_names) == 3 and not channel:
            cmap = None
        else:
            raise NotImplementedError("Only 1, 2, or 3 channels supported")
        if self.ndim == 2:
            if self.dtype == "bool":
                cmap = "gray"
            elif self.dtype in ["int", "int32", "int64"]:
                cmap = random_label_cmap()
        plt.imshow(region, interpolation=self.interpolation_strat, cmap=cmap)
        if overlay:
            img_overlay, alpha_img, cmap = overlay._get_region_overlay(
                y_min, x_min, y_len, x_len, pad, invert_overlay=invert_overlay
            )
            if img_overlay is not None:
                cmap = overlay_cmap if overlay_cmap else cmap
                plt.imshow(img_overlay, alpha=alpha_img, cmap=cmap, interpolation="nearest")
        if scalebar:
            scalebar = ScaleBar(
                self.mpp,
                units="µm",
                location=scalebar_location,
                length_fraction=0.1,
                border_pad=0.5,
            )
            plt.gca().add_artist(scalebar)
        plt.axis("off")
        plt.show()

    @abstractmethod
    def get_thumb(self, target_size: int = 512) -> ndarray:
        """Get a thumbnail of the WSI image

        Args:
            target_size (int, optional): The target size for the longest dimension of the image. Defaults to 512.

        Returns:
            ndarray: The thumbnail image as a numpy array
        """
        pass

    def show_thumb(
        self,
        target_size: int = 512,
        scalebar: bool = True,
        scalebar_location: str = "lower right",
        overlay: Optional["Image"] = None,
        invert_overlay: bool = False,
        overlay_cmap: str | None = None,
        grid_lines: str | None = None,
        annotate: Optional[Annotation] = None,
    ) -> None:
        """Display a thumbnail of the WSI image

        Args:
            target_size (int, optional): The target size for the longest dimension of the image. Defaults to 512.
            scalebar (bool, optional): Whether to include a scalebar. Defaults to True.
            scalebar_location (str, optional): The location of the scalebar, if it is to be included. Defaults to "lower right".
            grid_lines (str, optional): Whether to plot grid lines at every 10,000 pixels. To plot, specify the colour you would like the lines to be. Defaults to None.
            annotate (Annotation, optional): An annotation object to overlay on the thumbnail. Defaults to None.
        """
        assert isinstance(target_size, int), "target_size must be an integer"
        assert target_size > 0, "target_size must be positive"
        assert isinstance(scalebar, bool), "scalebar must be a boolean"
        assert isinstance(scalebar_location, str), "scalebar_location must be a string"
        assert scalebar_location in [
            "lower right",
            "lower left",
            "upper right",
            "upper left",
        ], "scalebar_location must be one of 'lower right', 'lower left', 'upper right', or 'upper left'"
        assert isinstance(overlay, (Image, type(None))), "overlay must be an Image object or None"
        assert isinstance(invert_overlay, bool), "invert_overlay must be a boolean"
        assert isinstance(overlay_cmap, (str, type(None))), "overlay_cmap must be a string or None"
        assert isinstance(grid_lines, (str, type(None))), "grid_lines must be a string or None"
        assert isinstance(
            annotate, (Annotation, type(None))
        ), "annotate must be an Annotation object or None"
        if not self.mpp:
            scalebar = False
        thumb = self.get_thumb(target_size)
        if len(thumb.shape) == 2:
            cmap = "gray"
        else:
            cmap = None
        plt.imshow(thumb, cmap=cmap)
        coarsen_factor = max([s // target_size for s in self.shape])
        if coarsen_factor == 0:
            coarsen_factor = 1
        if overlay:
            img_overlay, alpha_img, overlay_cmap = overlay._get_thumb_overlay(
                coarsen_factor, invert_overlay=invert_overlay
            )
            if img_overlay is not None:
                plt.imshow(img_overlay, alpha=alpha_img, cmap=overlay_cmap)
        if grid_lines:
            spacing = 10000 // coarsen_factor
            y_spacing = range(spacing, thumb.shape[0], spacing)
            x_spacing = range(spacing, thumb.shape[1], spacing)
            grid_line_col = grid_lines if isinstance(grid_lines, str) else "red"
            for y in y_spacing:
                plt.axhline(y, color=grid_line_col, linestyle="--")
            for x in x_spacing:
                plt.axvline(x, color=grid_line_col, linestyle="--")
        if annotate:
            colour = annotate.colour
            annotation = annotate._get_annotation_thumb(coarsen_factor)
            if annotation:
                for contour in annotation:
                    plt.plot(*contour.xy, color=colour)
        if scalebar:
            coarsen_factor = max([s // target_size for s in self.shape])
            if coarsen_factor == 0:
                coarsen_factor = 1
            scalebar = ScaleBar(
                self.mpp * coarsen_factor,
                units="µm",
                location=scalebar_location,
                length_fraction=0.1,
                border_pad=0.5,
            )
            plt.gca().add_artist(scalebar)
        plt.axis("off")
        plt.show()

    def _get_region_overlay(
        self,
        y_min: int,
        x_min: int,
        y_len: int,
        x_len: int,
        pad: bool = True,
        invert_overlay: bool = False,
    ) -> Tuple[ndarray | None, ndarray | None, str | None]:
        """Get the overlay of an image region to superimpose on another image

        Args:
            y_min (int): The minimum y-coordinate of the region
            x_min (int): The minimum x-coordinate of the region
            y_len (int): The length of the region in the y direction
            x_len (int): The length of the region in the x direction
            pad (bool, optional): Whether to pad the regions outside the image with zeros. Defaults to True.
            invert_overlay (bool, optional): Whether to invert the overlay region. Defaults to False.

        Returns:
            Tuple[ndarray | None, ndarray | None, str | None]: The overlay to superimpose on the image, the alpha channel of the overlay, and the colour map to use for the overlay
        """
        return None, None, None

    def _get_thumb_overlay(
        self, coarsen_factor: int, invert_overlay: bool = False
    ) -> Tuple[ndarray | None, ndarray | None, str | None]:
        """Get the overlay of a thumbnail image to superimpose on another image

        Args:
            coarsen_factor (int): The coarsen factor to use to generate the thumbnail
            invert_overlay (bool, optional): Whether to invert the overlay thumbnail. Defaults to False.

        Returns:
            Tuple[ndarray | None, ndarray | None, str | None]: The overlay to superimpose on the image, the alpha channel of the overlay, and the colour map to use for the overlay
        """
        return None, None, None

    def _repr_html_(self) -> str:
        """Generate an HTML representation of the Image object

        Returns:
            str: The HTML representation of the Image object
        """
        colour_map = {
            "Red": "#FF0000",
            "Green": "#00FF00",
            "Blue": "#0000FF",
            "Hematoxylin": "#2A2670",
            "Eosin": "#E63DB7",
            "DAB": "#813723",
        }
        channel_names = []
        for colour in self.channel_names:
            if colour in colour_map:
                channel_names.append(f'<span style="color: {colour_map[colour]};">{colour}</span>')
            else:
                channel_names.append(colour)
        total_width = 400
        html = f"""
        <div style="width: {total_width}px; background-color: #202020; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
            <h1>SCEMATK Image Object</h1>
            <p>Name: {self.name if self.name else "SCEMATK Image"}</p>
            <p>Format: {self.info.get('format', 'Unknown').title()}</p>
            <p>Channels: {", ".join(channel_names)}</p>
            <p>Dimensions: {" x ".join([f"{i:,}" for i in self.shape])}</p>
            <p>Chunks: {" x ".join([f"{i:,}" for i in self.image.chunksize[:2]])}
            <p>Data Type: {self.dtype}</p>
        </div>
        """
        return html
