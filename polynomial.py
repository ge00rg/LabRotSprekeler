import numpy as np
import h5py

def fit_polynomial(filename, deg=6, exclude_mask=None):
    '''
    returns: ndarray of shape (deg+1, n_trials, n_dendrites)
    '''
    f = h5py.File(filename+'.hdf5', 'r')
    
    data = f['data']
    meta = f['meta']
    
    n_trials = data.shape[0]
    n_dendrites = data.shape[1]
    
    if exclude_mask:
        #remove unwanted trials
        pass
    
    coef_mat = np.zeros((deg+1, n_trials, n_dendrites))
    for i in range(n_trials):
        x = np.arange(0,180)
        y = data[i,:,:]

        coef = np.polynomial.polynomial.polyfit(x,y.T,deg)
        coef_mat[:, i, :] = coef
        
    return coef_mat

def get_poly_time(coef):
    '''
    returns: ndarray of shape (n_trials, n_dendrites, n_times)
    '''
    x = np.arange(0, 180)
    p = np.polynomial.polynomial.polyval(x, coef)
    
    return p

def get_feature_matrix(coef):
    '''
    returns: ndarray of shape (n_trials, (n_coef*n_dendrites)), with every row being (p0_all_dends, ..., p_n_all_dends)
    '''
    n_coef = coef.shape[0]
    n_trials = coef.shape[1]
    n_dendrites = coef.shape[2]
    
    X = np.zeros((n_trials, n_dendrites*n_coef))
    for i in range(n_trials):
        mat = coef[:,i,:]
        #print(mat.shape)
        for j in range(n_dendrites):
            #print(X[i, j*n_coef:(j+1)*n_coef].shape)
            X[i, j*n_coef:(j+1)*n_coef] = mat[:,j]
            
    return X

def get_orig_matrix(X, deg):
    n_trials = X.shape[0]
    print(X.shape)
    n_dendrites = X.shape[1]/(deg+1)
    assert(n_dendrites%1 == 0)
    n_dendrites = int(n_dendrites)
    
    data = np.zeros((deg+1, n_trials, n_dendrites))
    for i in range(n_trials):
        for j in range(n_dendrites):
            data[:, i, j] = X[i,j*(deg+1):(j+1)*(deg+1)]
    
    return data

def get_f_matrix(coef):
    n_coef = coef.shape[0]
    n_trials = coef.shape[1]
    n_dendrites = coef.shape[2]
    
    X = np.zeros((n_trials, n_dendrites*n_coef))
    for i in range(n_trials):
        for j in range(n_coef):
            X[i, j*n_dendrites:(j+1)*n_dendrites] = coef[j, i, :]
            
    return X