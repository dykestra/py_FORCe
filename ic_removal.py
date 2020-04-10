""" Identify and remove contaminated ICs """

import numpy as np
from numpy.fft import fft
from scipy.stats import kurtosis

from automutual_information import extractFeaturesMultiChsWaveAMI
from constants import *

import pdb

def powerspectrum(data, Fs):
    """ compute power spectrum of data """
    L = data.shape[0]
    NFFT = 2**np.ceil(np.log2(L)) 
    Y = fft(data, int(NFFT)) / L
    f = Fs/2*np.linspace(0,1,int(NFFT/2))
    power = 2 * abs(Y[:int(NFFT/2)])
    p = np.array([f,power])
    return p

def stdm(X):
    """ equivalent of Matlab std() """
    return np.std(X, ddof=1)

def estimate_scalp_projections(ICs, mixMat):
    projIC = []
    for iNo in range(ICs.shape[0]):
        ICtemp = ICs.copy()
        for iT in range(ICtemp.shape[0]):
            if iT != iNo:
                ICtemp[iT,:] = np.zeros((1,ICtemp.shape[1]))
        projIC.append( np.matmul( mixMat, ICtemp ) )
    return projIC

def projection_thresholds(projIC):
    remICsPT = []
    remICsP2P = []
    for iNo in range(len(projIC)):
        pthresh = []
        p2p = []
        for pNo in range(projIC[iNo].shape[0]):
            signal = projIC[iNo][pNo,:]
            pthresh.append( min(signal) < PROJECTION_MIN or max(signal) > PROJECTION_MAX )
            p2p.append( max(signal) - min(signal) > P2P_MAX )

        if any(pthresh): remICsPT.append(iNo)
        if any(p2p): remICsP2P.append(iNo)
        
    return remICsPT, remICsP2P

def kurtosis_threshold(projIC):
    entropy = [np.mean( kurtosis( np.transpose(proj), fisher=False ) ) for proj in projIC]
    mu_e = np.mean(entropy)
    sig_e = stdm(entropy)
    remICsKurt = [i for i,e in enumerate(entropy) if e > mu_e + (KURTOSIS_FACTOR*sig_e) or e < mu_e - (KURTOSIS_FACTOR*sig_e)]      
    return remICsKurt
    
def power_spectrum_thresholds(projIC, fs):
    specDist = []
    gammaPSD = []
    for iNo in range(len(projIC)):
        specDistT = []
        gammaPSDT = []
        for pNo in range(projIC[iNo].shape[0]):
            psT = powerspectrum( projIC[iNo][pNo,:], fs )
            idealDistro = np.vstack((psT[0,1:], 1./psT[0,1:]))
            # normalise
            idealDistro[1,:] = idealDistro[1,:] / max(idealDistro[1,:])
            psCheck = psT[1,1:] / max(psT[1,1:])
            
            diff = psCheck - idealDistro[1,:]
            specDistT.append( np.sqrt( np.matmul(diff, np.transpose(diff)) ) )
            gammaPSDT.append( np.mean( psT[1, np.where(psT[0,:] > 30)] ) )
        specDist.append(np.mean(specDistT))
        gammaPSD.append(max(gammaPSDT))

    # psd distance from 1/F distribution threshold
    remICsSpecDist = [i for i,s in enumerate(specDist) if s > SPEC_DIST_MAX]     
    # gamma psd threshold
    remICsGamma = [i for i,g in enumerate(gammaPSD) if g > GAMMA_MAX]
    
    return remICsSpecDist, remICsGamma

def power_spectrum_ratio_threshold(ICs, fs):
    specRatio = []
    for iNo in range(ICs.shape[0]):
        ps = powerspectrum(ICs[iNo,:], fs)
        muLow = np.mean(ps[1, np.where(ps[0,:]<20)])
        muHigh = np.mean(ps[1, np.where(ps[0,:]>20)])
        specRatio.append( muHigh / muLow )

    remICsSpecRatio = [i for i,r in enumerate(specRatio) if r > SPEC_RATIO_MAX]
    return remICsSpecRatio
 
def std_thresholds(projIC, electrodes):
    frontChans = [i for i,c in enumerate(electrodes) if 'F' in c]
    otherChans = [i for i,c, in enumerate(electrodes) if not 'F' in c]
    stdProj = []
    stdRatio = []
    for iNo in range(len(projIC)):
        stdProjT = [stdm(signal) for signal in projIC[iNo]]
        stdProj.append(max(stdProjT))

        stdRatio.append(np.mean([stdProjT[i] for i in frontChans]) / np.mean([stdProjT[i] for i in otherChans]))      
    
    # std threshold
    remICsStd = [i for i,s in enumerate(stdProj) if s > np.mean(stdProj) + STD_FACTOR * stdm(stdProj)]
    # std ratio threshold
    remICsStdRatio = [i for i,s in enumerate(stdRatio) if s > np.mean(stdRatio + STD_RATIO_FACTOR * stdm(stdRatio))] 
    
    return remICsStd, remICsStdRatio

def automutual_information_threshold(projIC, fs):               
    featsClust = [extractFeaturesMultiChsWaveAMI(proj,fs) for proj in projIC]
    remICsAMI = [i for i,f in enumerate(featsClust) if f < AMI_MIN or f > AMI_MAX]
    return remICsAMI
    
def projection_spiking_threshold(projIC):
    noSpikes = []
    tN = 0
    for iNo in range(len(projIC)):
        # find spike zones in projection
        muSig = abs(np.mean(projIC[tN], 0)) # mean signal across channels
        A1 = np.where(muSig[1:-1] > muSig[2:])[0] + 1
        spikePos = A1[np.where(muSig[A1] > muSig[A1-1])]
        # calculate coefficients
        coefVar = np.zeros(len(spikePos))
        for i in range(len(spikePos)):
            sig_spike = stdm(abs(np.mean(projIC[tN][:,spikePos[i]-1:spikePos[i]+2],0)))
            mu_spike = np.mean(abs(np.mean(projIC[tN][iNo,spikePos[i]-1:spikePos[i]+2],0)))
            coefVar[i] = sig_spike / mu_spike
        # soft thresholding
        T = 0.1 * (np.mean(coefVar) + stdm(coefVar))
        noSpikes.append(len([c for c in coefVar if c > T]) / len(coefVar))
        
    remICsNoSpikes = [i for i,n in enumerate(noSpikes) if n >= PROJECTION_SPIKE_THRESH]
    return remICsNoSpikes

def IC_spiking_threshold(ICs):
    remICsSpike = []
    for iNo in range(ICs.shape[0]):
        if max(abs(ICs[iNo,:])) > np.mean(abs(ICs[iNo,:])) + IC_SPIKE_FACTOR * stdm(ICs[iNo,:]):
            remICsSpike.append(iNo)
    return remICsSpike

def remove_contaminated(ICs, mixMat, electrodes, fs):
    tN = 0 # approximation coefficients only
    
    projIC = estimate_scalp_projections(ICs[tN], mixMat)
    
    remICsPT, remICsP2P = projection_thresholds(projIC)
    remICsKurt = kurtosis_threshold(projIC)
    remICsSpecDist, remICsGamma = power_spectrum_thresholds(projIC, fs)
    remICsSpecRatio = power_spectrum_ratio_threshold(ICs[tN], fs)
    remICsStd, remICsStdRatio = std_thresholds(projIC, electrodes)
    remICsAMI = automutual_information_threshold(projIC, fs)
    remICsNoSpikes = projection_spiking_threshold(projIC)
    remICsSpike = IC_spiking_threshold(ICs[tN])
     
    # all ICs marked for removal
    remICs_TOTAL = [remICsPT, remICsP2P, remICsSpike, remICsKurt, remICsSpecDist, remICsGamma,
                     remICsSpecRatio, remICsStd, remICsStdRatio, remICsAMI, remICsNoSpikes]
    # number of thresholds each IC has exceeded
    len_iNos = [len([r for r in remICs_TOTAL if iNo in r]) for iNo in range(ICs[tN].shape[0])] 
    remICs = [i for i,l in enumerate(len_iNos) if l > 3] # ICs to remove
    
    if len(remICs) == ICs[tN].shape[0]:
        raise Exception('All ICs removed')
    
    # remove ICs
    ICs[tN][remICs,:] = np.zeros((len(remICs), ICs[tN].shape[1]))
    
    return ICs