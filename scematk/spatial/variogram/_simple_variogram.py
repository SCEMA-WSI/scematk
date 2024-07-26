import sys
from math import ceil
from typing import Optional, Tuple

import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.delayed import delayed
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix


class SimpleVariogram:
    def __init__(
        self,
        max_distance: float,
        num_bins: int,
        prune: float = 0.0,
        num_marginal_bins: Tuple[int, int] | int = 100,
        count_cut_off: int = 0,
        model: str = "empirical",
        use_nugget: bool = False,
        restrict_range: bool = False,
        range_min: float = 0.0,
        local_exclusion: float = 0.0
    ) -> None:
        assert isinstance(max_distance, (float, int)), "max_distance should be a number"
        assert max_distance > 0, "max_distance should be positive"
        assert isinstance(num_bins, int), "num_bins should be an integer"
        assert num_bins > 0, "num_bins should be positive"
        assert isinstance(prune, float), "prune should be a float"
        assert (prune >= 0) and (prune < 1), "prune should be between 0 and 1"
        if isinstance(num_marginal_bins, int):
            num_marginal_bins = (num_marginal_bins, num_marginal_bins)
        assert isinstance(
            num_marginal_bins, tuple
        ), "num_marginal_bins should be an integer or a tuple"
        assert all(
            isinstance(entry, int) for entry in num_marginal_bins
        ), "All entries of num_marginal_bins should be an integer"
        assert len(num_marginal_bins) == 2, "num_marginal_bins should only have 2 entries"
        assert isinstance(count_cut_off, int), "count_cut_off should be an integer"
        assert count_cut_off >= 0, "count_cut_off can't be negative"
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
        self.max_distance = max_distance
        self.num_bins = num_bins
        self.prune = prune
        self.num_marginal_bins = num_marginal_bins
        self.count_cut_off = count_cut_off
        self.model = model
        self.use_nugget = use_nugget
        self.restrict_range = restrict_range
        self.range_min = range_min
        self.local_exclusion = local_exclusion
        self.nugget: Optional[float] = None
        self.sill: Optional[float] = None
        self.range: Optional[float] = None
        self.fitted = False

    def _prune_df(self, df):
        nrows = df.shape[0]
        keep = 1 - self.prune
        nkeep = ceil(keep * nrows)
        return df.sample(frac=nkeep / nrows, replace=False)

    @delayed(nout=2)
    def _fit_empirical(
        self,
        tile: dd.DataFrame,
        edge: dd.DataFrame,
        overlap: dd.DataFrame,
        value_col: str,
        x_col: str,
        y_col: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        semivariances = np.zeros(self.num_bins)
        counts = np.zeros(self.num_bins, dtype=int)
        if not isinstance(tile, pd.DataFrame):
            return semivariances, counts
        if tile.empty:
            return semivariances, counts
        if self.prune > 0:
            tile = self._prune_df(tile)
        tile_coords = tile[[x_col, y_col]].to_numpy()
        bin_width = self.max_distance / self.num_bins
        dist_mat = distance_matrix(tile_coords, tile_coords) // bin_width
        dist_mat = dist_mat.astype(int)
        for i in range(dist_mat.shape[0] - 1):
            for j in range(i + 1, dist_mat.shape[1]):
                if dist_mat[i, j] < self.num_bins:
                    semivariances[dist_mat[i, j]] += (
                        0.5 * (tile.iloc[i][value_col] - tile.iloc[j][value_col]) ** 2
                    )
                    counts[dist_mat[i, j]] += 1
        if edge.empty | overlap.empty:
            return semivariances, counts
        if not isinstance(edge, pd.DataFrame):
            return semivariances, counts
        if not isinstance(overlap, pd.DataFrame):
            return semivariances, counts
        if self.prune > 0:
            edge = self._prune_df(edge)
            overlap = self._prune_df(overlap)
        edge_coords = edge[[x_col, y_col]].to_numpy()
        overlap_coords = overlap[[x_col, y_col]].to_numpy()
        dist_mat = distance_matrix(edge_coords, overlap_coords) // bin_width
        dist_mat = dist_mat.astype(int)
        for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[1]):
                if dist_mat[i, j] < self.num_bins:
                    semivariances[dist_mat[i, j]] += (
                        0.5 * (edge.iloc[i][value_col] - overlap.iloc[j][value_col]) ** 2
                    )
                    counts[dist_mat[i, j]] += 1
        return semivariances, counts

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

    def _fit_structured_variogram(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if self.model == "empirical":
            return 0.0, 0.0, 0.0
        valid_idx = (self.counts > self.count_cut_off) & (self.lags > self.local_exclusion)
        lags = self.lags[valid_idx]
        semivariances = self.semivariances[valid_idx]
        if self.model == "spherical":
            model = self._spherical_variogram
        elif self.model == "exponential":
            model = self._spherical_variogram
        elif self.model == "gaussian":
            model = self._gaussian_variogram
        else:
            ValueError(f"Variogram model '{self.model}' is not implemented.")
        min_bounds = [0.0, 0.0, self.range_min]
        max_bounds = [sys.float_info.min, np.inf, np.inf]
        if self.use_nugget:
            max_bounds[0] = np.inf
        if self.restrict_range:
            max_bounds[2] = max(lags)
        params, _ = curve_fit(model, lags, semivariances, bounds=(min_bounds, max_bounds))
        return params

    def fit(self, data: dd.DataFrame, value_col: str, x_col: str, y_col: str) -> None:
        assert isinstance(data, dd.DataFrame), "data should be a dask dataframe"
        column_names = data.columns
        assert isinstance(value_col, str), "value_col should be a string"
        assert value_col in column_names, "value_col should be a column in the dataframe"
        assert isinstance(x_col, str), "x_col should be a string"
        assert x_col in column_names, "x_col should be a column in the dataframe"
        assert isinstance(y_col, str), "y_col should be a string"
        assert y_col in column_names, "y_col should be a column in the dataframe"
        x_percs = np.linspace(0, 1, self.num_marginal_bins[0] + 1)
        y_percs = np.linspace(0, 1, self.num_marginal_bins[1] + 1)
        x_perc_vals = data[x_col].quantile(x_percs).compute().tolist()
        y_perc_vals = data[y_col].quantile(y_percs).compute().tolist()
        x_perc_vals = list(set(x_perc_vals))
        y_perc_vals = list(set(y_perc_vals))
        x_perc_vals.sort()
        y_perc_vals.sort()
        x_perc_vals[0] -= 1
        x_perc_vals[-1] += 1
        y_perc_vals[0] -= 1
        y_perc_vals[-1] += 1
        x_perc_vals_start = x_perc_vals[:-1]
        x_perc_vals_end = x_perc_vals[1:]
        y_perc_vals_start = y_perc_vals[:-1]
        y_perc_vals_end = y_perc_vals[1:]
        local_semivariances = []
        local_counts = []
        for x_start, x_end in zip(x_perc_vals_start, x_perc_vals_end):
            for y_start, y_end in zip(y_perc_vals_start, y_perc_vals_end):
                tile = data[
                    (
                        (data[x_col] >= x_start)
                        & (data[x_col] < x_end)
                        & (data[y_col] >= y_start)
                        & (data[y_col] < y_end)
                    )
                ]
                xe_start = x_end - self.max_distance
                ye_start = y_end - self.max_distance
                edge = data[
                    (
                        (data[x_col] < x_end)
                        & (data[y_col] < y_end)
                        & (data[x_col] > x_start)
                        & (data[y_col] < y_end)
                        & ((data[x_col] >= xe_start) | (data[y_col] >= ye_start))
                    )
                ]
                overlap = data[
                    (
                        (data[x_col] >= x_start)
                        & (data[x_col] < x_end + self.max_distance)
                        & (data[y_col] >= y_start - self.max_distance)
                        & (data[y_col] < y_end + self.max_distance)
                        & ((data[x_col] >= x_end) | (data[y_col] >= y_end))
                    )
                ]
                semivariances, counts = self._fit_empirical(
                    tile, edge, overlap, value_col, x_col, y_col
                )
                semivariances = da.from_delayed(semivariances, shape=(self.num_bins,), dtype=float)
                counts = da.from_delayed(counts, shape=(self.num_bins,), dtype=int)
                local_semivariances.append(semivariances)
                local_counts.append(counts)
        semivariances = da.stack(local_semivariances, axis=1).sum(axis=1)
        counts = da.stack(local_counts, axis=1).sum(axis=1)
        avg_semivariances = semivariances / counts.clip(1)
        combined = da.stack([avg_semivariances, counts], axis=1).compute()
        avg_semivariances = combined[:, 0]
        counts = combined[:, 1].astype(int)
        bin_width = self.max_distance / self.num_bins
        lags = [(i * bin_width) + (bin_width / 2) for i in range(self.num_bins)]
        self.semivariances = avg_semivariances
        self.counts = counts
        self.lags = np.array(lags)
        self.empirical_fitted = True
        nugget, sill, range_ = self._fit_structured_variogram()
        if nugget == sill == range_ == 0:
            self.nugget = None
            self.sill = None
            self.range = None
        else:
            self.nugget = nugget
            self.sill = sill
            self.range = range_
        self.fitted = True

    def refit_structure(
        self,
        model: Optional[str] = None,
        use_nugget: Optional[bool] = None,
        count_cut_off: Optional[int] = None,
        restrict_range: Optional[bool] = None,
        range_min: Optional[float] = None,
        local_exclusion: Optional[float] = None,
    ) -> None:
        assert self.fitted, "Variogram must be fitted first buy calling .fit() method."
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
        if count_cut_off is not None:
            assert isinstance(count_cut_off, int), "count_cut_off should be an integer"
            assert count_cut_off >= 0, "count_cut_off can't be negative"
            self.count_cut_off = count_cut_off
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
        self.nugget, self.sill, self.range = self._fit_structured_variogram()

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
        valid_idx = self.counts > self.count_cut_off
        lags = self.lags[valid_idx]
        semivariances = self.semivariances[valid_idx]
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
