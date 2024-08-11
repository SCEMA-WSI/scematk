import dask.array as da
import dask.dataframe as dd
from math import sqrt
import numpy as np
from scipy.stats import norm
from typing import Tuple

class GlobalMoransI():
    def __init__(
        self,
        distance_threshold: float = None,
        normalisation: str = "row",
        chunk_size: int = 1000
    ) -> None:
        assert isinstance(distance_threshold, (float, int)), "distance_threshold must be numeric"
        assert isinstance(normalisation, str), "normalisation must be a string"
        assert normalisation in ['row', 'whole', 'none'], "normalisation must be one of row, whole or none"
        assert isinstance(chunk_size, int), "chunk_size must be an integer"
        self.distance_threshold = distance_threshold
        self.normalisation = normalisation
        self.chunk_size = chunk_size
        self.fitted = False
        
    def fit(self, data: dd.DataFrame, value_col: str, x_col: str, y_col: str) -> None:
        assert not self.fitted, "This model has already been fitted"
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
        coords = data[[x_col, y_col]].to_dask_array(lengths=True)
        coords = coords.rechunk((self.chunk_size, 2))
        x_diff = coords[:, None, 0] - coords[None, :, 0]
        y_diff = coords[:, None, 1] - coords[None, :, 1]
        W = da.less_equal(da.sqrt(x_diff ** 2 + y_diff ** 2), self.distance_threshold).astype(float)
        W[da.eye(W.shape[0], dtype=bool)] = 0
        if self.normalisation == 'row':
            row_sum = W.sum(axis=1, keepdims=True)
            row_sum = da.where(row_sum == 0, 1, row_sum)
            W = W / row_sum
        elif self.normalisation == 'whole':
            total = W.sum()
            total = total if total > 0 else 1
            W = W / total
        s_0 = W.sum()
        s_1 = (W + W.T) ** 2
        s_1 = 0.25 * s_1.sum()
        s_2 = (W.sum(axis=0) + W.sum(axis=1)) ** 2
        s_2 = s_2.sum()
        y = data[value_col].to_dask_array(lengths=True)
        y = y.rechunk(self.chunk_size)
        z = y - y.mean()
        auto_corr = W * (z[:,None] * z[None,:])
        auto_corr = auto_corr.sum()
        z2 = z ** 2
        z2 = z2.sum()
        s_0, s_1, s_2, auto_corr, z2 = da.stack([s_0, s_1, s_2, auto_corr, z2]).compute()
        n = W.shape[0]
        self.i = (n/s_0) * (auto_corr/z2)
        var_num = n * n * (n-1) * s_1 - n * (n-1) * s_2 - 2 * s_0 * s_0
        var_den = (n+1) *  (n-1) * (n-1) * s_0 * s_0
        self.var = np.clip(var_num/var_den, 0, np.inf)