""" Apply the FORCe algorithm to an EEG sample 
Inputs:
    - EEGdata : np array of (channels, samples)
    - fs : sample rate in Hz
    - electrodes : list of electrode labels e.g. ['AF3','F7','F3', ...]
Returns:
    - cleanEEG : np array of cleaned EEG (channels, samples)
"""

import numpy as np
from numpy.fft import fft
from scipy.stats import kurtosis

from electrode_locations import electrode_locations
from decomposition import applyWaveletICA
from automutual_information import extractFeaturesMultiChsWaveAMI

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

def powerspectrum(data, Fs):
    """ compute power spectrum of data """
    L = data.shape[0]
    NFFT = 2**np.ceil(np.log2(L)) 
    Y = fft(data, int(NFFT)) / L
    f = Fs/2*np.linspace(0,1,NFFT/2)
    power = 2 * abs(Y[:int(NFFT/2)])
    p = np.array([f,power])
    return p

def FORCe(EEGdata, fs, electrodes):
    """ Run FORCe on EEGdata """
    check_valid(EEGdata)
    
    N, M = EEGdata.shape
    
    # channel threshold 200uV
    remCh = [i for i,c in enumerate(EEGdata) if np.max(c) > 200 ]
    if len(remCh) == N: 
        raise Exception('All channels > 200uV')
    
    EEGdata = estimate_removed_channels(EEGdata, remCh, electrodes)
    
    ICs, mixMat, wavePacks, tNodes = applyWaveletICA(EEGdata)
    
    # TODO identify contaminated ICs for approximation coefficients
    tN = 0 # approximation coefficients only
    
    remICsPT = [] # projection threshold
    remICsP2P = [] # peak-to-peak
    
    projIC = []
    entropy = []
    specDist = []
    gammaPSD = []
    stdProj = []
    stdRatio = []
    featsClust = []
    for iNo in range(ICs[tN].shape[0]):
        
        # estimate scalp projections
        ICtemp = ICs[tN].copy()
        for iT in range(ICtemp.shape[0]):
            if iT != iNo:
                ICtemp[iT,:] = np.zeros((1,ICtemp.shape[1]))
        projIC.append( np.matmul( mixMat, ICtemp ) )

        # projection thresholds
        pthresh = []
        p2p = []
        for pNo in range(projIC[iNo].shape[0]):
            signal = projIC[iNo][pNo,:]
            pthresh.append( min(signal) < -100 or max(signal) > 100 )
            p2p.append( max(signal) - min(signal) > 60 )

        if any(pthresh): remICsPT.append(iNo)
        if any(p2p): remICsP2P.append(iNo)

        # get kurtosis value
        entropy.append(np.mean( kurtosis( np.transpose(projIC[iNo]), fisher=False ) ))
        
        # power spectrum
        specDistT = []
        gammaPSDT = []
        for pNo in range(projIC[iNo].shape[0]):
            psT = powerspectrum( projIC[iNo][pNo,:], fs )
            idealDistro = np.vstack((psT[0,:], 1./psT[0,:]))
            idealDistro[1,0] = 0 # remove inf val for 1/0
            # normalise
            idealDistro[1,1:] = idealDistro[1,1:] / max(idealDistro[1,1:])
            psCheck = psT[1,1:] / max(psT[1,1:])
            
            diff = psCheck - idealDistro[1,1:]
            specDistT.append( np.sqrt( np.matmul(diff, np.transpose(diff)) ) )
            gammaPSDT.append( np.mean( psT[1, np.where(psT[0,:] > 30)] ) )
            
        specDist.append(np.mean(specDistT))
        gammaPSD.append(max(gammaPSDT))

        # std and std ratio
        stdProjT = [np.std(signal,ddof=1) for signal in projIC[iNo]]
        stdProj.append(max(stdProjT))
        frontChans = [i for i,c in enumerate(electrodes) if 'F' in c]
        otherChans = [i for i,c, in enumerate(electrodes) if not 'F' in c]
        stdRatio.append(np.mean([stdProjT[i] for i in frontChans]) / np.mean([stdProjT[i] for i in otherChans]))
        
        # get AMI
        featsClust.append( extractFeaturesMultiChsWaveAMI(projIC[iNo],fs) )
    
    # TODO soft thresholding on approximation and detail
    
    # TODO reconstruct EEG from wavelet decomposition

    return EEGdata