""" Apply the FORCe algorithm to an EEG sample 
Inputs:
    - EEGdata : np array of (channels, samples)
    - fs : sample rate in Hz
    - electrodes : list of electrode labels e.g. ['AF3','F7','F3', ...]
Returns:
    - cleanEEG : np array of cleaned EEG (channels, samples)
"""

import numpy as np


from channel_threshold import channel_threshold
from decomposition import apply_wavelet_ICA, reconstruct
from ic_removal import remove_contaminated
from spike_zone_thresholding import apply_soft_threshold


import pdb 

def check_valid(EEGdata):
    """ perform validation checks on EEGdata """
    if np.isnan(EEGdata).any():
        raise Exception('EEGdata contains NaNs')
    if np.isinf(EEGdata).any():
        raise Exception('EEGdata contains Infs')

def FORCe(EEGdata, fs, electrodes):
    """ Run FORCe on EEGdata """
    check_valid(EEGdata)
    
    EEGdata = channel_threshold(EEGdata, electrodes)
    
    ICs, mixMat, wavePacks, tNodes = apply_wavelet_ICA(EEGdata)
    
    ICs = remove_contaminated(ICs, mixMat, electrodes, fs)
    
    newSigNode = ICs.copy()
    newSigNode[0] = np.matmul(mixMat, ICs[0])

    newSigNode = apply_soft_threshold(newSigNode, tNodes)
    
    cleanEEG = reconstruct(wavePacks, newSigNode, tNodes)

    return cleanEEG