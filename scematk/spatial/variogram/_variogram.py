import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys
from typing import Optional

class Variogram():
    def __init__(self,
        bin_width: float,
        prune: float = 1.0,
        chunk_size: int = 4096,
        count_threshold: int = 10000,
        model: str = "empirical",
        use_nugget: bool = False,
        restrict_range: bool = False,
        range_min: float = 0.0,
        local_exclusion: float = 0.0,
        max_distance: float = 0.0
    ):
        if isinstance(bin_width, int): bin_width = float(bin_width)
        assert isinstance(bin_width, float), "bin_width should be a float"
        assert isinstance(prune, float), "prune should be a float"
        assert isinstance(chunk_size, int), "chunk_size should be an integer"
        assert chunk_size > 0, "chunk_size should be positive"
        assert isinstance(count_threshold, int), "count_threshold should be an integer"
        assert count_threshold >= 0, "count threshold can't be negative"
        assert isinstance(model, str), "model should be a string"
        assert model in [
            "spherical",
            "exponential",
            "gaussian",
            "empirical",
        ], "model should be one of spherical, exponential, gaussian or empirical"
        assert isinstance(use_nugget, bool), "use_nugget should be a boolean"
        assert isinstance(restrict_range, bool), "restrict_range should be a boolean"
        assert isinstance(range_min, (float, int)), "range_min should be a number"
        assert range_min >= 0, "range_min can't be negative"
        assert isinstance(local_exclusion, (int, float)), "Local exclusion must be a number"
        assert local_exclusion >= 0, "Local exclusion can't be negative"
        if isinstance(max_distance, int): max_distance = float(max_distance)
        assert isinstance(max_distance, float), "max_distance should be a float"
        self.bin_width = bin_width
        self.prune = 1.0
        self.chunk_size = 4096
        self.count_threshold = count_threshold
        self.model = model
        self.use_nugget = use_nugget
        self.restrict_range = restrict_range
        self.range_min = range_min
        self.local_exclusion = local_exclusion
        self.max_distance = max_distance
        self.fitted = False
        
    def _spherical_variogram(self, h, nugget, sill, range_):
        return np.where(
            h <= range_,
            nugget + (sill - nugget) * (1.5 * (h / range_) - 0.5 * (h / range_) ** 3),
            sill,
        )

    def _exponential_variogram(self, h, nugget, sill, range_):
        return nugget + (sill - nugget) * (1 - np.exp(-3 * h / range_))

    def _gaussian_variogram(self, h, nugget, sill, range_):
        return nugget + (sill - nugget) * (1 - np.exp(-1 * (h**2 / range_**2)))
        
    def _fit_structured_variogram(self) -> None:
        if not self.model == "empirical":    
            assert self.fitted, "This model should be fitted before computing structural variogram"
            lags = self.lags
            means = self.means
            counts = self.counts
            if self.max_distance > 0:
                idx_keep = (counts > self.count_threshold) & ((lags * self.bin_width) <= self.max_distance)
            else:
                idx_keep = counts > self.count_threshold
            lags = lags[idx_keep]
            means = means[idx_keep]
            counts = counts[idx_keep]
            model = None
            if self.model == "spherical":
                model = self._spherical_variogram
            elif self.model == "exponential":
                model = self._spherical_variogram
            elif self.model == "gaussian":
                model = self._gaussian_variogram
            if model is None:
                ValueError(f"Variogram model '{self.model}' is not implemented.")
            min_bounds = [0.0, 0.0, self.range_min]
            max_bounds = [sys.float_info.min, np.inf, np.inf]
            if self.use_nugget:
                max_bounds[0] = np.inf
            if self.restrict_range:
                max_bounds[2] = max(lags)
            params, _ = curve_fit(model, lags, means, bounds=(min_bounds, max_bounds))
            self.nugget = params[0]
            self.sill = params[1]
            self.range = params[2]
        else:
            self.nugget = None
            self.sill = None
            self.range = None
        
    def fit(self, data: dd.DataFrame, value_col: str, x_col: str, y_col: str) -> None:
        assert not self.fitted, "This model has always been fitted"
        assert isinstance(data, dd.DataFrame), "data should be a dask dataframe"
        column_names = data.columns
        assert isinstance(value_col, str), "value_col should be a string"
        assert value_col in column_names, "value_col should be a column in the dataframe"
        assert isinstance(x_col, str), "x_col should be a string"
        assert x_col in column_names, "x_col should be a column in the dataframe"
        assert isinstance(y_col, str), "y_col should be a string"
        assert y_col in column_names, "y_col should be a column in the dataframe"
        self.value_col = value_col
        self.x_col = x_col
        self.y_col = y_col
        data = data.copy()
        if self.prune < 1:
            data = data.sample(frac = self.prune)
        coords = data[[x_col, y_col]].to_dask_array(lengths=True)
        coords = coords.rechunk((self.chunk_size, 2))
        x_diff = coords[:, None, 0] - coords[None, :, 0]
        y_diff = coords[:, None, 1] - coords[None, :, 1]
        dist_mat = da.sqrt(x_diff ** 2 + y_diff ** 2)
        dist_mat[da.eye(dist_mat.shape[0], dtype=bool)] = 0
        dist_mat = dist_mat // self.bin_width
        y = data[value_col].to_dask_array(lengths=True)
        y = y.rechunk(self.chunk_size)
        semivariances = (y[:,None] - y[None,:]) ** 2
        dist_mat = dist_mat.flatten()
        semivariances = semivariances.flatten()
        df = dd.from_dask_array(da.stack([dist_mat, semivariances], axis=1), columns=['group', 'value'])
        df = df[df['group'] != 0]
        grouped = df.groupby('group')['value'].agg(['mean', 'count'])
        grouped = grouped.reset_index().sort_values('group').compute()
        self.lags = np.array(grouped['group'].to_list())
        self.means = np.array(grouped['mean'].to_list())
        self.counts = np.array(grouped['count'].to_list())
        self.fitted = True
        
    def refit_structure(
        self,
        model: Optional[str] = None,
        use_nugget: Optional[bool] = None,
        count_threshold: Optional[int] = None,
        restrict_range: Optional[bool] = None,
        range_min: Optional[float] = None,
        local_exclusion: Optional[float] = None,
        max_distance: Optional[float] = None
    ) -> None:
        assert self.fitted, "Variogram must be fitted first by calling .fit() method."
        if model is not None:
            assert isinstance(model, str), "model should be a string"
            assert model in [
                "spherical",
                "exponential",
                "gaussian",
                "empirical",
            ], "model should be one of spherical, exponential, gaussian or empirical"
            self.model = model
        if use_nugget is not None:
            assert isinstance(use_nugget, bool), "use_nugget should be a boolean"
            self.use_nugget = use_nugget
        if count_threshold is not None:
            assert isinstance(count_threshold, int), "count_cut_off should be an integer"
            assert count_threshold >= 0, "count_cut_off can't be negative"
            self.count_threshold = count_threshold
        if restrict_range is not None:
            assert isinstance(restrict_range, bool), "restrict_range should be a boolean"
            self.restrict_range = restrict_range
        if range_min is not None:
            assert isinstance(range_min, (float, int)), "range_min should be a number"
            assert range_min >= 0, "range_min can't be negative"
            self.range_min = range_min
        if local_exclusion is not None:
            assert isinstance(local_exclusion, (float, int)), "local_exclusion must be a number"
            assert local_exclusion >= 0, "local_exclusion must not be negative"
            self.local_exclusion = local_exclusion
        if max_distance is not None:
            if isinstance(max_distance, int): max_distance = float(max_distance)
            assert isinstance(max_distance, float), "max_distance should be a float"
            self.max_distance = max_distance
        self._fit_structured_variogram()
    
    def plot(
        self,
        empirical_params: dict = {},
        structure_params: dict = {},
        x_title: str = "Lag Distances",
        y_title: str = "Semivariance",
        title: Optional[str] = None,
        include_y_zero: bool = True,
    ):
        assert self.fitted, "Variogram must be fitted first buy calling .fit() method."
        assert isinstance(empirical_params, dict), "empirical_params should be a dictionary"
        assert isinstance(structure_params, dict), "structure_params should be a dictionary"
        assert isinstance(include_y_zero, bool), "include_y_zero should be a boolean"
        if "s" not in empirical_params:
            empirical_params["s"] = 1
        if "c" not in structure_params:
            structure_params["c"] = "red"
        if self.max_distance > 0:
            valid_idx = (self.counts > self.count_threshold) & ((self.lags * self.bin_width) <= self.max_distance)
        else:
            valid_idx = (self.counts > self.count_threshold)
        lags = self.lags[valid_idx]
        semivariances = self.means[valid_idx]
        plt.scatter(lags, semivariances, **empirical_params)
        if self.model in ["exponential", "spherical", "gaussian"]:
            if self.model == "exponential":
                model = self._exponential_variogram
            elif self.model == "spherical":
                model = self._spherical_variogram
            elif self.model == "gaussian":
                model = self._gaussian_variogram
            x_vals = np.linspace(0, max(lags), 1000)
            y_vals = model(x_vals, self.nugget, self.sill, self.range)
            plt.plot(x_vals, y_vals, **structure_params)
        if title is not None:
            plt.title(title)
        if x_title is not None:
            plt.xlabel(x_title)
        if y_title is not None:
            plt.ylabel(y_title)
        if include_y_zero:
            plt.ylim(0)
        plt.show()