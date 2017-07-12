import numpy as np

def rW(R, Q):
    n = R.shape[0]
    return 1 - (6*np.sum((R-Q)**2 * ((np.ones(n)*n-R+np.ones(n)) + (np.ones(n)*n-Q+np.ones(n)))))/(n**4+n**3-n**2-n)

def rFree(R, Q, W):
    n = R.shape[0]
    A = 1
    B = -2/(np.sum(W*(np.ones(n)*n-2*np.arange(n)+np.ones(n))**2))
    D = R-Q
    
    return A + B*np.sum(W*D**2)

def rFree_sym(R, Q, W_r, W_q):
    n = R.shape[0]
    A = 1
    B = -2/(np.sum(W_r*W_q*(np.ones(n)*n-2*np.arange(n)+np.ones(n))**2))
    D = R-Q
    
    return A + B*np.sum(W_r*W_q*D**2)