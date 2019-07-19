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



if __name__ == "__main__":
    test_array = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
     
    print "array:"
    print test_array    
    print "shape:"
    print test_array.shape
    print "means:"
    print running_average_simple(test_array, window=1)
    print running_average_simple(test_array, window=2)
    print running_average_simple(test_array, window=3)
    print running_average_simple(test_array, window=4)
    print "variances:"
    print running_variance_simple(test_array, window=1)
    print running_variance_simple(test_array, window=2)
    print running_variance_simple(test_array, window=3)
    print running_variance_simple(test_array, window=4)
    print "z:"
    print running_z_simple(test_array, running_average_simple(test_array, window=1), window=1)
    print running_z_simple(test_array, running_average_simple(test_array, window=2), window=2)
    print running_z_simple(test_array, running_average_simple(test_array, window=3), window=3)
    print running_z_simple(test_array, running_average_simple(test_array, window=4), window=4)
