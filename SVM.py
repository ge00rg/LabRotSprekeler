from scipy.io import loadmat
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def SVM_presence_single(filename, amp, svm_kernel='linear', cv=5):
    '''
    Trains an SVM on the Ca2+ data of single dendrites to detect stimulus presence.
    
    filename: string, name of experiment file
    amp: float, stimulus strength, must exist
    svm_kernel: string, specifies svm kernel to use for training
    cv: int, number of folds for cv
    
    returns: n_dendrites x cv ndarray. Each row holds the accuracy values for each fold for one dendrite.
    '''
    f = h5py.File(filename+".hdf5", "r")
    data = f['data']
    meta = f['meta']
    #load data

    stims = np.unique(meta[:,1])[1:]        #exclude zero-stim-trials
    assert(amp in list(stims)), "this stimAmp was not used in chosen experiment. Cose one from {}.".format(stims)
    #load stimAmps and check whether the chosen amp matches one of them

    clf = svm.SVC(kernel=svm_kernel)
    #create classifier
    
    n_dendrites = data.shape[1]
    #number of dendrites, loop over these
    
    sc = np.zeros((n_dendrites, cv))
    #will later hold the cv-scores
    for site in range(n_dendrites):

        baseline = np.mean(data[:,:,:58], axis=2).reshape(data.shape[0], data.shape[1], 1)
        #baselibe for every trial is the mean over the first second - 58 frames
        
        mn_dnd_chng = np.mean(data[:,:,58:116]-baseline, axis=2)
        #mean dendritic change = second second minus baseline, mean

        present_mask = meta[:,1]==amp
        absent_mask = meta[:,1]==0
        #boolean masks of the trials where stimAmp==amp and stimAmp==0

        trials_mask = np.logical_or(present_mask, absent_mask)
        #boolean mask for all trials where the stimulus is either our chosen one or absent (0)

        y_score = mn_dnd_chng[trials_mask, site]
        #score that we use for classification - only stimulus present - and absent trials at one dendrite

        amp_mask = meta[:,1][trials_mask]==amp
        y_true = (amp_mask-0.5)*2
        #take all trials used for making y_score. Mark the present ones with a '1', absent ones with a '-1'

        scores = cross_val_score(clf, y_score.reshape(y_score.shape[0], 1), y_true.reshape(y_true.shape[0]), cv=cv)
        #compute accuracy scores using crossvalidation
        
        sc[site, :] = scores
        
    return sc

def SVM_hitmiss_single(filename, amp, svm_kernel='linear', cv=5):
    '''
    Trains an SVM on the Ca2+ data of single dendrites to detect/predict behavior.
    
    filename: string, name of experiment file
    amp: float, stimulus strength, must exist
    svm_kernel: string, specifies svm kernel to use for training
    cv: int, number of folds for cv
    
    returns: n_dendrites x cv ndarray. Each row holds the accuracy values for each fold for one dendrite.
    '''
    f = h5py.File(filename+".hdf5", "r")
    data = f['data']
    meta = f['meta']
    #load data
    
    stims = np.unique(meta[:,1])[1:]        #exclude zero-stim-trials
    assert(amp in list(stims)), "this stimAmp was not used in chosen experiment. Cose one from {}.".format(stims)
    #load stimAmps and check whether the chosen amp matches one of them

    n_dendrites = data.shape[1]
    #get number of dendrites

    clf = svm.SVC(kernel=svm_kernel, class_weight='balanced')
    #classifier. balalced class weights because classes can be very unbalanced

    bals = []
    ns = []    
    sc = np.zeros((n_dendrites, cv))
    #will hold balances, number of samples and scores
    
    for site in range(n_dendrites):

        baseline = np.mean(data[:,:,:58], axis=2).reshape(data.shape[0], data.shape[1], 1)
        #compute baseline as mean over first second for each trial

        mn_dnd_chng = np.mean(data[:,:,58:116]-baseline, axis=2)
        #mean dendritic change is mean second second activity minus baseline

        trials_mask = meta[:,1]==amp
        #get mask for trials with chosen amp

        y_score = mn_dnd_chng[trials_mask, site]
        #scores used are mean dendritic changes on trials with chosen stimulus at chosen dendrite

        hit_mask = meta[:, 2]==1
        #mask, true when hit

        end_mask = hit_mask[trials_mask]
        #only use trials with chosen amp
        
        y_true = (end_mask-0.5)*2
        #convert to 1 and -1 values

        balance = np.sum(end_mask)/end_mask.shape[0]
        n_y = end_mask.shape[0]

        bals.append(balance)
        ns.append(n_y)
        #compute and append balances and sample number

        scores = cross_val_score(clf, y_score.reshape(y_score.shape[0], 1), y_true.reshape(y_true.shape[0]), cv=cv)
        sc[site, :] = scores
        
    return sc, bals, ns

def SVM_presence_combined(filename, svm_kernel='linear', cv=10):
    '''
    Trains an SVM on the Ca2+ data of all dendrites to detect stimulus presence.
    
    filename: string, name of experiment file
    svm_kernel: string, specifies svm kernel to use for training
    cv: int, number of folds for cv
    
    returns: n_stims x cv ndarray. Each row holds the accuracy values for each fold for one stimulus strength.
    '''
    f = h5py.File(filename+".hdf5", "r")
    data = f['data']
    meta = f['meta']
    stims = np.unique(meta[:,1])[1:]        #exclude zero
    #load data and stimulia
    
    sc = np.zeros((stims.shape[0], cv))
    #will hold scores
    
    for i, amp in enumerate(stims):
        baseline = np.mean(data[:,:,:58], axis=2).reshape(data.shape[0], data.shape[1], 1)
        #compute baseline as average over first second for every trial
        
        mn_dnd_chng = np.mean(data[:,:,58:116]-baseline, axis=2)
        #mean dendritic change with baseline subtracted, second second

        present_mask = meta[:,1]==amp
        absent_mask = meta[:,1]==0
        #masks for present and absent trials

        trials_mask = np.logical_or(present_mask, absent_mask)
        #combine these to mask of all trials with stimulus zero or the chosen stimulus

        y_score = mn_dnd_chng[trials_mask, :]
        #scores used are mean dendritic changes in chosen trials

        amp_mask = meta[:,1][trials_mask]==amp
        y_true = (amp_mask-0.5)*2
        #stimulus present trials are marked as '1', absent ones as '-1'

        clf = svm.SVC(kernel=svm_kernel)
        scores = cross_val_score(clf, y_score, y_true.reshape(y_true.shape[0]), cv=cv)
        #make classifier and compute accuracy scores using cross validation
        
        sc[i, :] = scores
    return sc

def SVM_hitmiss_combined(filename, svm_kernel='linear', cv=4):
    '''
    Trains an SVM on the Ca2+ data of all dendrites to detect/predict hits/misses.
    
    filename: string, name of experiment file
    svm_kernel: string, specifies svm kernel to use for training
    cv: int, number of folds for cv
    
    returns: n_stims x cv ndarray. Each row holds the accuracy values for each fold for one stimulus strength.
    '''
    f = h5py.File(filename+".hdf5", "r")
    data = f['data']
    meta = f['meta']
    stims = np.unique(f['meta'][:,1])[1:]        #exclude zero
    #load data and stims
    
    sc = np.zeros((stims.shape[0], cv))
    bals = []
    ns = []
    #will hold scores, balances and sample numbers
    
    for k, amp in enumerate(stims):
        baseline = np.mean(data[:,:,:58], axis=2).reshape(data.shape[0], data.shape[1], 1)
        #baseline as average over first second for each trial
        
        mn_dnd_chng = np.mean(data[:,:,58:116]-baseline, axis=2)
        #mean dendritic change as mean over second second minus baseline

        trials_mask = meta[:,1]==amp
        #we use only trials with a given stimulus
        
        y_score = mn_dnd_chng[trials_mask, :]
        #scores are mean dendritic changes in these trials

        hit_mask = meta[:, 2]==1
        #mask of hit-trials

        end_mask = hit_mask[trials_mask]
        #apply trials_mask to hit-mask
        
        y_true = (end_mask-0.5)*2
        #if hit, 1, else -1

        balance = np.sum(end_mask)/end_mask.shape[0]
        n_y = end_mask.shape[0]

        bals.append(balance)
        ns.append(n_y)
        #compute sample numbers and balances

        clf = svm.SVC(kernel=svm_kernel, class_weight='balanced')
        scores = cross_val_score(clf, y_score, y_true.reshape(y_true.shape[0]), cv=cv)
        sc[k,:] = scores
        #classifier and scores
        
    return sc, bals, ns

def SVM_hitmiss_combined_window(filename, start, stop, base='frac', svm_kernel='linear', cv=4):
    '''
    Trains an SVM on the Ca2+ data of all dendrites to detect/predict hits/misses.
    
    filename: string, name of experiment file
    start: int, starting frame of averaging window
    stop: int, final frame of averaging window
    base: string, mode of subtracting baseline.
        'normal': always subtract full baseline
        'frac': subtract baseline proportional to fraction of post-stiumulus frames
    svm_kernel: string, specifies svm kernel to use for training
    cv: int, number of folds for cv
    
    returns: n_stims x cv ndarray. Each row holds the accuracy values for each fold for one stimulus strength.
    '''
    assert(stop > start), "window needs to have positive size."
    assert(start >= 0), "start needs to be larger or equal to zero."
    
    f = h5py.File(filename+".hdf5", "r")
    data = f['data']
    meta = f['meta']
    stims = np.unique(f['meta'][:,1])
    #load data and stims
    
    assert(stop <= data.shape[2]), "Stop has to be smaller or equal to the number of frames"
    
    sc = np.zeros((stims.shape[0], cv))
    #will hold scores
    
    for k, amp in enumerate(stims):
        baseline = np.mean(data[:,:,:58], axis=2).reshape(data.shape[0], data.shape[1], 1)
        #baseline as average over first second for each trial
        
        #for 'frac' mode, make the weighting of the baseline proportional the the fraction of post stimulus frames
        if base == 'normal':
            bl_factor = 1
        if base == 'frac':
            n_pre = 58 - start
            n_post = stop - 58
            
            bl_factor = n_post/(n_pre + n_post)
            #compute baseline weight
            
            if n_pre <= 0:
                #start >= 58 -> everything is post stimulus
                bl_factor = 1
            if n_post <= 0:
                #stop <= 50 -> everything is pre-stimulus
                bl_factor = 0
        
        mn_dnd_chng = np.mean(data[:,:,start:stop]-baseline*bl_factor, axis=2)
        #mean dendritic change as mean over second second minus baseline

        trials_mask = meta[:,1]==amp
        #we use only trials with a given stimulus
        
        y_score = mn_dnd_chng[trials_mask, :]
        #scores are mean dendritic changes in these trials

        hit_mask = meta[:, 2]==1
        #mask of hit-trials

        end_mask = hit_mask[trials_mask]
        #apply trials_mask to hit-mask
        
        y_true = (end_mask-0.5)*2
        #if hit, 1, else -1

        clf = svm.SVC(kernel=svm_kernel, class_weight='balanced')
        try:
            scores = cross_val_score(clf, y_score, y_true.reshape(y_true.shape[0]), cv=cv)
        except ValueError:
            scores = np.zeros(cv)
        sc[k,:] = scores
        #classifier and scores. If not enough instances of either class, return all zeros and continue loop.
        
    return sc