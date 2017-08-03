import h5py
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from smooth import smoothen
from sklearn.decomposition import PCA

def SVM_presence_single(filename, amp, svm_kernel='linear', cv=5, scoring='accuracy'):
    '''
    Trains an SVM on the Ca2+ data of single dendrites to detect stimulus presence.
    
    filename: string, name of experiment file
    amp: float, stimulus strength, must exist
    svm_kernel: string, specifies svm kernel to use for training
    cv: int, number of folds for cv
    scoring: only use accuracy!
    
    returns: n_dendrites x cv ndarray. Each row holds the accuracy values for each fold for one dendrite.
    '''
    f = h5py.File(filename+".hdf5", "r")
    data = f['data']
    meta = f['meta']
    g = h5py.File(filename[:-6]+"roi.hdf5", "r")
    motion_mask = (g['inFrameDend'][:].astype(bool)).reshape(g['inFrameDend'].shape[0])
    #load data

    stims = np.unique(meta[:,1])[1:]        #exclude zero-stim-trials
    assert(amp in list(stims)), "this stimAmp was not used in chosen experiment. Cose one from {}.".format(stims)
    #load stimAmps and check whether the chosen amp matches one of them

    clf = svm.SVC(kernel=svm_kernel)
    #create classifier
    
    #n_dendrites = data.shape[1]
    n_dendrites = np.sum(motion_mask)
    #number of dendrites, loop over these
    
    sc = np.zeros((n_dendrites, cv))
    #will later hold the cv-scores
    
    for site in range(n_dendrites):
        baseline = np.mean(data[:,motion_mask,:58], axis=2).reshape(data.shape[0], data[:,motion_mask,:].shape[1], 1)
        #baselibe for every trial is the mean over the first second - 58 frames
        
        mn_dnd_chng = np.mean(data[:,motion_mask,58:116]-baseline, axis=2)
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
        
        if scoring == 'accuracy':
            scores = cross_val_score(clf, y_score.reshape(y_score.shape[0], 1), y_true.reshape(y_true.shape[0]), cv=cv)
            #compute accuracy scores using crossvalidation
        else:
            print('Use only accuracy for scoring!')
        
        sc[site, :] = scores
        
    return sc

def SVM_single_all(filename):
    '''
    Computes SVM-performances of all dendrites for all stimuli, sorted from best to worst
    
    filename: string, name of file used
    
    returns: results matrix stimuli x dendrites x (dendrite index, mean accuracy, standard deviation)
    '''
    n_dendrites = h5py.File(filename+'.hdf5', 'r')['data'].shape[1]
    g = h5py.File(filename[:-6]+"roi.hdf5", "r")
    motion_mask = (g['inFrameDend'][:].astype(bool)).reshape(g['inFrameDend'].shape[0])
    #get number of dendrites pre motion correction and motion mask

    n_out = np.sum(motion_mask)
    #number of dendrites after motion correction

    res = np.zeros((6, n_out, 3))
    #will hold results - stimuli x dendrites x (dendrite_index, accurady, standard deviation)

    stims = np.unique(h5py.File(filename+'.hdf5', 'r')['meta'][:,1])[1:]
    #get nonzero stimuli

    for k, amp in enumerate(stims):
        scs = SVM_presence_single(filename, amp, scoring='accuracy')
        #crossval-scores

        means = np.mean(scs, axis=1)
        sdvs = np.std(scs, axis=1)
        #mean and standard deviation

        #correct for discrepancy between pre and post motion mask, save results in res
        inds = np.argsort(means)[::-1]
        for i in range(n_out):
            c = -1
            for j in range(n_dendrites):
                c += motion_mask[j]
                d = j
                if c == inds[i]:
                    break
                    
            res[k,i,0] = d
            res[k,i,1] = means[inds[i]]
            res[k,i,2] = sdvs[inds[i]]
            
    return res

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
    g = h5py.File(filename[:-6]+"roi.hdf5", "r")
    motion_mask = (g['inFrameDend'][:].astype(bool)).reshape(g['inFrameDend'].shape[0])
    #load data
    
    stims = np.unique(meta[:,1])[:]        #exclude zero-stim-trials
    assert(amp in list(stims)), "this stimAmp was not used in chosen experiment. Cose one from {}.".format(stims)
    #load stimAmps and check whether the chosen amp matches one of them

    n_dendrites = np.sum(motion_mask)
    #get number of dendrites

    clf = svm.SVC(kernel=svm_kernel, class_weight='balanced')
    #classifier. balalced class weights because classes can be very unbalanced

    bals = []
    ns = []    
    sc = np.zeros((n_dendrites, cv))
    #will hold balances, number of samples and scores
    
    for site in range(n_dendrites):

        baseline = np.mean(data[:,motion_mask,:58], axis=2).reshape(data.shape[0], data[:,motion_mask,:].shape[1], 1)
        #baselibe for every trial is the mean over the first second - 58 frames
        
        mn_dnd_chng = np.mean(data[:,motion_mask,58:116]-baseline, axis=2)
        #mean dendritic change = second second minus baseline, mean

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
        #balance - fraction of hit trials, n_y number of hit-trials

        bals.append(balance)
        ns.append(n_y)
        #compute and append balances and sample number

        scores = cross_val_score(clf, y_score.reshape(y_score.shape[0], 1), y_true.reshape(y_true.shape[0]), cv=cv)
        sc[site, :] = scores
        
    return sc, bals, ns

def SVM_hitmiss_all(filename):
    '''
    Computes SVM-performancce for all stimuli and all valid dendrites for hit/miss scenario, sorted from best to worst
    
    filename: string, name of file used
    
    returns: result matrix stims x dendrites x (dendrite_index, mean accuracy, standard deviation)
    '''
    g = h5py.File(filelist[0][:-6]+"roi.hdf5", "r")
    motion_mask = (g['inFrameDend'][:].astype(bool)).reshape(g['inFrameDend'].shape[0])
    n_0 = np.sum(motion_mask)
    #get motion mask and n_0, number of dendrites after motion correction

    n_dendrites = h5py.File(filename+'.hdf5', 'r')['data'].shape[1]
    #number of dendrites before motion correction

    res = np.zeros((7, n_0, 3))
    #result matrix

    stims = np.unique(h5py.File(filename+'.hdf5', 'r')['meta'][:,1])
    #stimuli
    
    #compute performance for all stimuli and dendrites, do correction for motion mask
    for l, amp in enumerate(stims):

        scs, _, _ = SVM_hitmiss_single(filename, amp)

        means = np.mean(scs, axis=1)
        sdvs = np.std(scs, axis=1)

        inds = np.argsort(means)[::-1]

        for i in range(n_0):
            c = -1
            for j in range(n_dendrites):
                c += motion_mask[j]
                d = j
                if c == inds[i]:
                    break

            res[l,i,0] = d
            res[l,i,1] = means[inds[i]]
            res[l,i,2] = sdvs[inds[i]]
            #store results
    return res

def do_PCA(X, n_comps=5):
    pca = PCA(n_components=n_comps)
    
    #denoising, but staying in original space!
    X_pca = pca.fit_transform(X)
    
    #return pca.inverse_transform(X_pca), pca
    return X_pca, pca

def SVM_presence_combined(filename, mask=None, svm_kernel='linear', cv=10, z_score=True, C=1, pca=False):
    '''
    Trains an SVM on the Ca2+ data of all dendrites to detect stimulus presence.
    
    filename: string, name of experiment file
    mask: custom boolean mask of shape n_dendrites that enabels the user to select a subset of dendrites to use
    svm_kernel: string, specifies svm kernel to use for training
    cv: int, number of folds for cv
    z-score: bool, if true the response gets normalized. Should be set at true.
    C: float, C-value for SVM
    pca: bool, if true do PCA on the data first. Should be left at false.
    
    returns: n_stims x cv ndarray. Each row holds the accuracy values for each fold for one stimulus strength.
    '''
    f = h5py.File(filename+".hdf5", "r")
    data = f['data']
    meta = f['meta']
    g = h5py.File(filename[:-6]+"roi.hdf5", "r")
    motion_mask = (g['inFrameDend'][:].astype(bool)).reshape(g['inFrameDend'].shape[0])
    stims = np.unique(meta[:,1])[1:]        #exclude zero
    #load data and stimulia
    
    if mask is not None:
        motion_mask = np.logical_and(motion_mask, mask)
    #using additional dendrite mask if given
    
    sc = np.zeros((stims.shape[0], cv))
    #will hold scores
    
    for i, amp in enumerate(stims):
        baseline = np.mean(data[:,motion_mask,:58], axis=2).reshape(data.shape[0], data[:,motion_mask,:].shape[1], 1)
        #baselibe for every trial is the mean over the first second - 58 frames
        
        mn_dnd_chng_nz = np.mean(data[:,motion_mask,58:116]-baseline, axis=2)
        #mean dendritic change = second second minus baseline, mean
        
        if z_score:
            mn_dnd_chng = preprocessing.scale(mn_dnd_chng_nz)
        #normalize data

        else:
            mn_dnd_chng = mn_dnd_chng_nz
            
        if pca:
            mn_dnd_chng = do_PCA(mn_dnd_chng)[0]
        #use PCA on data

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

        clf = svm.SVC(kernel=svm_kernel, C=C)
        scores = cross_val_score(clf, y_score, y_true.reshape(y_true.shape[0]), cv=cv)
        #make classifier and compute accuracy scores using cross validation
        
        sc[i, :] = scores
    return sc

def SVM_hitmiss_combined(filename, mask=None, svm_kernel='linear', cv=5, z_score=True, C=1, pca=False):
    '''
    Trains an SVM on the Ca2+ data of all dendrites to detect/predict hits/misses.
    
    filename: string, name of experiment file
    mask: boolean mask of size n_dendrites, allows to select dendrites
    svm_kernel: string, specifies svm kernel to use for training
    cv: int, number of folds for cv
    z-score: bool, if true, data is normalized
    C: float, C-parameter of SVM
    pca: bool, if true, apply PCA to data. Should be left false
    
    returns: n_stims x cv ndarray. Each row holds the accuracy values for each fold for one stimulus strength.
    '''
    f = h5py.File(filename+".hdf5", "r")
    data = f['data']
    meta = f['meta']
    g = h5py.File(filename[:-6]+"roi.hdf5", "r")
    motion_mask = (g['inFrameDend'][:].astype(bool)).reshape(g['inFrameDend'].shape[0])
    stims = np.unique(f['meta'][:,1])        #exclude zero
    #load data and stims

    if mask is not None:
        motion_mask = np.logical_and(motion_mask, mask)
    #apply mask if existing to dendrites
    
    sc = np.zeros((stims.shape[0], cv))
    bals = []
    ns = []
    #will hold scores, balances and sample numbers
    
    for k, amp in enumerate(stims):
        baseline = np.mean(data[:,motion_mask,:58], axis=2).reshape(data.shape[0], data[:,motion_mask,:].shape[1], 1)
        #baselibe for every trial is the mean over the first second - 58 frames
        
        mn_dnd_chng_nz = np.mean(data[:,motion_mask,58:116]-baseline, axis=2)
        #mean dendritic change = second second minus baseline, mean
        
        if z_score:
            mn_dnd_chng = preprocessing.scale(mn_dnd_chng_nz)
        #normalize data

        else:
            mn_dnd_chng = mn_dnd_chng_nz
            
        if pca:
            mn_dnd_chng = do_PCA(mn_dnd_chng)[0]
        #PCA if applicable

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
        #balance - fraction of hit-trials, n_y number of hit-trials

        bals.append(balance)
        ns.append(n_y)
        #compute sample numbers and balances

        clf = svm.SVC(kernel=svm_kernel, class_weight='balanced', C=C)
        #clf = svm.SVC(kernel='linear', class_weight='balanced', C=1)

        try:
            scores = cross_val_score(clf, y_score, y_true.reshape(y_true.shape[0]), cv=cv)
        except ValueError:
            scores = np.zeros(cv)
        sc[k,:] = scores

        #classifier and scores
        
    return sc, bals, ns