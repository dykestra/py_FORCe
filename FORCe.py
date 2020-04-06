""" Apply the FORCe algorithm to an EEG sample 
Inputs:
    - EEGdata : np array of (channels, samples)
    - fs : sample rate in Hz
    - electrodes : list of electrode labels e.g. ['AF3','F7','F3', ...]
Returns:
    - cleanEEG : np array of cleaned EEG (channels, samples)
"""

import numpy as np
from pywt import wavedec, WaveletPacket

from electrode_locations import electrode_locations
from sobi import sobi_fast

import pdb 

def check_valid(EEGdata):
    """ perform validation checks on EEGdata """
    if np.isnan(EEGdata).any():
        raise Exception('EEGdata contains NaNs')
    if np.isinf(EEGdata).any():
        raise Exception('EEGdata contains Infs')
    
def get_electrode_locations(electrodes):
    """ get relevant electrode locations """
    locs = {k:np.array(v) for (k,v) in electrode_locations.items() if k in electrodes}
    if len(locs) < len(electrodes): 
        unknown = [e for e in electrodes if e not in locs]
        raise Exception('\nUnknown electrodes: {}, please add to electrode_locations'.format(unknown))
    return locs
    
def estimate_removed_channels(EEGdata, remChs, electrodes):
    """ estimate values for removed channels from neighbouring channel signals """
    locs = get_electrode_locations(electrodes)
    keepChs = [i for i in range(len(electrodes)) if i not in remChs]

    for rc in remChs:
        # calculate euclidean distance from keepChs
        dist = np.zeros(len(keepChs))
        for i,kc in enumerate(keepChs):
            dist[i] = np.linalg.norm(locs[electrodes[rc]] - locs[electrodes[kc]])
        # set new data values to mean of keepChs signals, weighted by 1/dist
        for s in range(EEGdata.shape[1]): 
            EEGdata[rc,i] = np.mean( np.divide( EEGdata[keepChs,i], dist ) )
        
    return EEGdata

def applyWaveletICA(EEGdata):
    """ apply wavelet decomposition and ICA to EEGdata """
    terminal_nodes = ['aa','ad','da','dd'] # labels for level 2 nodes in WaveletPacket tree
    T = len(terminal_nodes)
    N, _ = EEGdata.shape # no. channels
    
    wavePacks = []
    waveData = [[] for i in range(T)]
    for c in range(N):
        wavePacket = WaveletPacket(data=EEGdata[c,:], wavelet='sym4', maxlevel=2)
        wavePacks.append(wavePacket)
        for i,n in enumerate(terminal_nodes):
            waveData[i].append(wavePacket[n].data)
    
    mixMat, V, X = sobi_fast(np.array(waveData[0]))
    ICs = np.transpose(V) * X
    
    ICs += [wp for wp in waveData[1:]]
    
    return ICs, mixMat, wavePacks

def FORCe(EEGdata, fs, electrodes):
    """ Run FORCe on EEGdata """
    check_valid(EEGdata)
    
    N, M = EEGdata.shape
    
    # channel threshold 200uV
    remCh = [i for i,c in enumerate(EEGdata) if np.max(c) > 200 ]
    if len(remCh) == N: 
        raise Exception('All channels > 200uV')
    
    EEGdata = estimate_removed_channels(EEGdata, remCh, electrodes)
    
    ICs, mixMat, wavePacks = applyWaveletICA(EEGdata)
    
    # TODO identify contaminated ICs for approximation coefficients
    
    # TODO soft thresholding on approximation and detail
    
    # TODO reconstruct EEG from wavelet decomposition

    return EEGdata