import h5py
from scipy.ndimage.filters import gaussian_filter1d

def smoothen(filename):
    f = h5py.File(filename+".hdf5", "r")
    data = f['data']
    return gaussian_filter1d(data, sigma=2, axis=2)