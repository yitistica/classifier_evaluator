from typing import Union, Optional
import numpy as np
import pandas as pd


def data_type_converter(series: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
    """
    convert list or pandas.Series to numpy.ndarray;
    :param series: [list, pandas.Series], a series;
    :return: numpy.ndarray(shape=(m), ), the converted array;
    """
    if isinstance(series, list):
        series = np.array(series)
    elif isinstance(series, np.ndarray):
        pass
    elif isinstance(series, pd.Series):
        series = series.values()

    return series