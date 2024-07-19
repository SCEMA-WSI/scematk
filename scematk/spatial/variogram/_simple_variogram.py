import sys
from math import ceil, floor
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
        model: str = "empirical",
        use_nugget: bool = False,
        restrict_range: bool = False,
        local_cut_off: float = 0.0,
        n_tiles: Tuple[int, int] | int = 100
    ):
        assert isinstance(max_distance, (float, int)), "max_distance must be a number"
        assert max_distance > 0, "max_distance is greater than 0"
        assert isinstance(num_bins, int), "num_bins must be an integer"
        assert num_bins > 0, "num_bins must be greater than 0"
        assert isinstance(prune, (float, int)), "prune must be a number"
        assert isinstance(model, str), "model must be a string"
        assert model in ['spherical', 'exponential', 'gaussian', 'empirical'], "model must be one of empirical, spherical, exponential or gaussian"
        assert isinstance(use_nugget, bool), "use_nugget must be True or False"
        assert isinstance(restrict_range, bool), "restrict_range must be True or False"
        assert isinstance(local_cut_off, (float, int)), "local_cut_off must be a number"
        if isinstance(n_tiles, int):
            n_tiles = (n_tiles, n_tiles)
        assert isinstance(n_tiles, tuple), "n_tiles should be an integer or a tuple of integers"
        assert len(n_tiles) == 2, "n_tiles should be an integer or a tuple of integers"
        assert all(isinstance(var, int) for var in n_tiles), "n_tiles should be an integer or a tuple of integers"
        self.max_distance = max_distance
        self.num_bins = num_bins
        self.prune = prune
        self.model = model
        self.use_nugget = use_nugget
        self.restrict_range = restrict_range
        self.local_cut_off = local_cut_off
        self.n_tiles = n_tiles
        self.empirical_fitted = False
        self.structured_fitted = False
        self.fitted = False
    
    def _prune_df(self, df: dd.DataFrame):
        keep = 1 - self.prune
        nrows = df.shape[0]
        nkeep = ceil(nrows * keep)
        df = df.sample(nkeep)
        return df
        
    @delayed(nout=2)
    def _local_semivariance(self, data_tile: dd.DataFrame, data_overlap: dd.DataFrame, value_col: str, x_col: str, y_col: str):
        semivariances = np.zeros(self.num_bins)
        counts = np.zeros(self.num_bins, dtype=int)
        if not isinstance(data_tile, pd.DataFrame):
            return semivariances, counts
        if not isinstance(data_overlap, pd.DataFrame):
            return semivariances, counts
        assert isinstance(data_tile, pd.DataFrame)
        assert isinstance(data_overlap, pd.DataFrame)
        assert isinstance(value_col, str)
        assert value_col in data_tile.columns
        assert isinstance(x_col, str)
        assert x_col in data_tile.columns
        assert isinstance(y_col, str)
        assert y_col in data_tile.columns
        if data_tile.empty:
            return semivariances, counts
        if self.prune > 0:
            data_tile = self._prune_df(data_tile)
        tile_coords = data_tile[[x_col, y_col]].to_numpy()
        bin_width = self.max_distance / self.num_bins
        dist_mat = distance_matrix(tile_coords, tile_coords) // bin_width
        dist_mat = dist_mat.astype(int)
        nrows = dist_mat.shape[0]
        for i in range(nrows - 1):
            for j in range(i + 1, nrows):
                if dist_mat[i, j] < self.num_bins:
                    semivariances[dist_mat[i, j]] += 0.5 * ((data_tile.iloc[i][value_col] - data_tile.iloc[j][value_col]) ** 2)
                    counts[dist_mat[i, j]] += 1
        if data_overlap.empty:
            return semivariances, counts
        if self.prune > 0:
            data_overlap = self._prune_df(data_overlap)
        overlap_coords = data_overlap[[x_col, y_col]].to_numpy()
        dist_mat = distance_matrix(tile_coords, overlap_coords) // bin_width
        dist_mat = distance_matrix(tile_coords, tile_coords) // bin_width
        dist_mat = dist_mat.astype(int)
        nrows = dist_mat.shape[0]
        ncols = dist_mat.shape[1]
        for i in range(nrows):
            for j in range(ncols):
                if dist_mat[i, j] < self.num_bins:
                    semivariances[dist_mat[i, j]] += 0.25 * ((data_tile.iloc[i][value_col] - data_tile.iloc[j][value_col]) ** 2)
                    counts[dist_mat[i, j]] += 1
        return semivariances, counts
        
    def _fit_empirical_variogram(self, data: dd.DataFrame | pd.DataFrame, value_col: str, x_col: str, y_col: str):
        if isinstance(data, pd.DataFrame):
            data = dd.from_pandas(data, npartitions=1)
        assert isinstance(data, dd.DataFrame)
        assert isinstance(value_col, str)
        assert value_col in data.columns
        assert isinstance(x_col, str)
        assert x_col in data.columns
        assert isinstance(y_col, str)
        assert y_col in data.columns
        n_tiles = self.n_tiles
        max_distance = self.max_distance
        x_percs = np.linspace(0, 1, n_tiles[0] + 1).tolist()
        y_percs = np.linspace(0, 1, n_tiles[1] + 1).tolist()
        x_perc_vals = data[x_col].quantile(x_percs).compute().tolist()
        y_perc_vals = data[y_col].quantile(y_percs).compute().tolist()
        x_perc_vals = list(set(x_perc_vals))
        y_perc_vals = list(set(y_perc_vals))
        x_perc_vals.sort()
        y_perc_vals.sort()
        x_perc_vals_start = x_perc_vals[:-1]
        x_perc_vals_end = x_perc_vals[1:]
        y_perc_vals_start = y_perc_vals[:-1]
        y_perc_vals_end = y_perc_vals[1:]
        local_semivariances = []
        local_counts = []
        for x_start, x_end in zip(x_perc_vals_start, x_perc_vals_end):
            for y_start, y_end in zip(y_perc_vals_start, y_perc_vals_end):
                data_tile = data[
                    (data[x_col] >= x_start)
                    & (data[x_col] < x_end)
                    & (data[y_col] >= y_start)
                    & (data[y_col] < y_end)
                ]
                data_overlap = data[
                    (data[x_col] >= x_start - max_distance)
                    & (data[x_col] <= x_end + max_distance)
                    & (data[y_col] >= y_start - max_distance)
                    & (data[y_col] <= y_end + max_distance)
                    & (
                        (data[x_col] > x_end)
                        | (data[x_col] < x_start)
                        | (data[y_col] < y_start)
                        | (data[y_col] > y_end)
                    )
                ]
                semivariances, counts = self._local_semivariance(data_tile, data_overlap, value_col, x_col, y_col)
                semivariances = da.from_delayed(semivariances, shape=(self.num_bins,), dtype=float)
                counts = da.from_delayed(counts, shape=(self.num_bins,), dtype=int)
                local_semivariances.append(semivariances)
                local_counts.append(counts)
        semivariances = da.stack(local_semivariances, axis=1).sum(axis=1)
        counts = da.stack(local_counts, axis=1).sum(axis=1)
        avg_semivariances = semivariances / counts.clip(1)
        grouped_output = da.stack([avg_semivariances, counts], axis=1).compute()
        avg_semivariances = grouped_output[:,0]
        counts = grouped_output[:,1].astype(int)
        nz = counts > 0
        bin_width = self.max_distance / self.num_bins
        lags = [(i * bin_width) + (bin_width/2) for i in range(self.num_bins)]
        self.semivariances = avg_semivariances[nz]
        self.counts = counts[nz]
        self.lags = np.array(lags)[nz]
        self.empirical_fitted = True
    
    def _spherical_variogram(self, h, nugget, sill, range_):
        return np.where(
            h <= range_,
            nugget + (sill -  nugget) * (1.5 * (h / range_) - 0.5 * (h / range_)**3),
            sill
        )

    def _exponential_variogram(self, h, nugget, sill, range_):
        return nugget + (sill - nugget) * (1 - np.exp(-3 * h / range_))

    def _gaussian_variogram(self, h, nugget, sill, range_):
        return nugget + (sill - nugget) * (1 - np.exp(-1 * (h ** 2 / range_ ** 2)))

    def _fit_structured_variogram(self):
        if self.model == 'empirical':
            return None, None, None
        semivariances = self.semivariances
        lags = self.lags
        local_cut_off = self.local_cut_off
        semivariances = semivariances[lags >= local_cut_off]
        lags = lags[lags >= local_cut_off]
        if self.model == 'spherical':
            model = self._spherical_variogram
        elif self.model == 'exponential':
            model = self._exponential_variogram
        elif self.model == 'gaussian':
            model = self._gaussian_variogram
        else:
            raise NotImplementedError(f"Variogram model '{self.model}' is not implemented.")
        min_bounds = [0, 0, 0]
        max_bounds = [sys.float_info.min, np.inf, np.inf]
        if self.use_nugget:
            max_bounds[0] = np.inf
        if self.restrict_range:
            min_bounds[2] = local_cut_off
            max_bounds[2] = max(lags)
        params, _ = curve_fit(model, lags, semivariances, bounds=(min_bounds, max_bounds))
        return params
        
    def fit(self, data: dd.DataFrame | pd.DataFrame, value_col: str, x_col: str, y_col: str):
        if isinstance(data, pd.DataFrame):
            data = dd.from_pandas(data, npartitions=1)
        assert isinstance(data, dd.DataFrame)
        assert isinstance(value_col, str)
        assert value_col in data.columns
        assert isinstance(x_col, str)
        assert x_col in data.columns
        assert isinstance(y_col, str)
        assert y_col in data.columns
        self._fit_empirical_variogram(data, value_col, x_col, y_col)
        nugget, sill, range_ = self._fit_structured_variogram()
        self.nugget = nugget
        self.sill = sill
        self.range = range_
        self.fitted = True
    
    def refit_structure(self, model: str = None, use_nugget: bool = None, restrict_range: bool = None, local_cut_off: int = None):
        assert self.fitted, "Variogram must be fitted first buy calling .fit() method."
        if model is not None:
            assert isinstance(model, str)
            assert model in ['spherical', 'exponential', 'gaussian', 'empirical']
            self.model = model
        if use_nugget is not None:
            assert isinstance(use_nugget, bool)
            self.use_nugget = use_nugget
        if restrict_range is not None:
            assert isinstance(restrict_range, bool)
            self.restrict_range = restrict_range
        if local_cut_off is not None:
            assert isinstance(local_cut_off, int)
            self.local_cut_off = local_cut_off
        self.nugget, self.sill, self.range = self._fit_structured_variogram()

    def plot(self, empirical_params: dict = {}, structure_params: dict = {}, x_title: str = "Lag Distances", y_title: str = "Semivariance", title: str = None, include_y_zero: bool = True):
        assert self.fitted, "Variogram must be fitted first buy calling .fit() method."
        assert isinstance(empirical_params, dict)
        assert isinstance(structure_params, dict)
        assert isinstance(include_y_zero, bool)
        if 's' not in empirical_params:
            empirical_params['s'] = 1
        if 'c' not in structure_params:
            structure_params['c'] = 'red'
        if self.model == 'exponential':
            model = self._exponential_variogram
        elif self.model == 'spherical':
            model = self._spherical_variogram
        elif self.model == 'gaussian':
            model = self._gaussian_variogram
        plt.scatter(self.lags, self.semivariances, **empirical_params)
        if self.model in ['exponential', 'spherical', 'gaussian']:
            x_vals = np.linspace(0, max(self.lags), 1000)
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