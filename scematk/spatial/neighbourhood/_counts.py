import dask.array as da
import dask.dataframe as dd

def neighbourhood_count(
    data: dd.DataFrame,
    distance_threshold: float,
    x_col: str,
    y_col: str,
    label_col: str = 'Meta_Global_Mask_Label',
    chunk_size: int = 4096
) -> dd.DataFrame:
    assert isinstance(data, dd.DataFrame), "data should be a dask dataframe"
    if isinstance(distance_threshold, int): distance_threshold = float(distance_threshold)
    assert isinstance(distance_threshold, float), "distance_threshold should be a float"
    column_names = data.columns
    assert isinstance(x_col, str), "x_col should be a string"
    assert x_col in column_names, "x_col should be a column of data"
    assert isinstance(y_col, str), "y_col should be a string"
    assert y_col in column_names, "y_col should be a column of data"
    assert isinstance(label_col, str), "label_col should be a string"
    assert label_col in column_names, "label_col should be a column of data"
    assert isinstance(chunk_size, int), "chunk_size should be an integer"
    assert chunk_size > 0, "chunk_size should be positive"
    data = data.copy()
    coords = data[[x_col, y_col]].to_dask_array(lengths=True)
    coords = coords.rechunk((chunk_size, 2))
    x_diff = coords[:, None, 0] - coords[None, :, 0]
    y_diff = coords[:, None, 1] - coords[None, :, 1]
    dist_mat = da.less_equal(da.sqrt(x_diff ** 2 + y_diff ** 2), distance_threshold).astype(int)
    dist_mat[da.eye(dist_mat.shape[0], dtype=bool)] = 0
    counts = dist_mat.sum(axis=1).to_dask_dataframe().repartition(npartitions=data.npartitions)
    df = data[[label_col]]
    df[f'SpatialNeighbourCounts'] = counts.values
    return df