import dask.array as da
import dask.dataframe as dd
import numpy as np

class GlobalMoran():
    def __init__(
        self,
        distance_threshold: float,
        normalisation: str = "row",
        chunk_size: int = 1000,
        prune: float = 1.0
    ) -> None:
        if isinstance(distance_threshold, int): distance_threshold = float(distance_threshold)
        assert isinstance(distance_threshold, float), 'distance_threshold should be a float'
        assert isinstance(normalisation, str), "normalisation should be a string"
        assert normalisation in ["row", "whole", "none"], "normalisation should be one of row, whole or none"
        assert isinstance(chunk_size, int), "chunk_size should be an integer"
        assert isinstance(prune, float), "prune should be a float"
        assert (0 < prune) and (prune <= 1), "prune should be between 0 and 1"
        self.distance_threshold = distance_threshold
        self.normalisation = normalisation
        self.chunk_size = chunk_size
        self.prune = prune
        self.fitted = False
    
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
        y = data[value_col].to_dask_array(lengths=True)
        y = y.rechunk(self.chunk_size)
        z = y - y.mean()
        auto_corr = W * (z[:,None] * z[None,:])
        auto_corr = auto_corr.sum()
        z2 = z ** 2
        z2 = z2.sum()
        s_0, auto_corr, z2 = da.stack([s_0, auto_corr, z2]).compute()
        n = W.shape[0]
        self.i = (n/s_0) * (auto_corr/z2)