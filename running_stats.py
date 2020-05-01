import numpy as np
from numpy.lib.stride_tricks import as_strided


def running_average_simple(values, window=10):
    if (window>=values.shape[0]):
        return np.array([np.nan])
    result = np.zeros(values.shape[0] - window + 1)
    for i in range(len(result)):
        result[i] = np.nanmean(values[i:i + window,:])
    return result

def running_variance_simple(values, window=10):
    if (window>=values.shape[0]):
        return np.array([np.nan])
    result = np.zeros(values.shape[0] - window + 1)
    for i in range(len(result)):
        result[i] = np.nanvar(values[i:i + window,:])
    return result

def running_z_simple(values, means, window=10):
    if (window>=values.shape[0]):
        return np.array([np.nan])
    nReps = values.shape[1]
    result = np.zeros(values.shape[0] - window + 1)
    assert len(means) == len(result)
    for i in range(len(result)):
        result[i] = np.nansum(values[i:i + window,:])*means[i]/(np.count_nonzero(~np.isnan(values[i:i + window,:])))
    return result
