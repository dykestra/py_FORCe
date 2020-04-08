""" Identify and remove contaminated ICs """

import numpy as np
from numpy.fft import fft
from scipy.stats import kurtosis

from automutual_information import extractFeaturesMultiChsWaveAMI

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

def remove_contaminated(ICs, mixMat, electrodes, fs):
    tN = 0 # approximation coefficients only
    
    remICsPT = [] # projection threshold
    remICsP2P = [] # peak-to-peak
    remICsSpike = [] # spikedness of ICs
    
    projIC = []
    entropy = []
    specDist = []
    gammaPSD = []
    specRatio = []
    stdProj = []
    stdRatio = []
    featsClust = []
    noSpikes = []
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

        # power spectrum ratio
        ps = powerspectrum(ICs[tN][iNo,:], fs)
        muLow = np.mean(ps[1, np.where(ps[0,:]<20)])
        muHigh = np.mean(ps[1, np.where(ps[0,:]>20)])
        specRatio.append( muHigh / muLow )

        # std and std ratio
        stdProjT = [np.std(signal,ddof=1) for signal in projIC[iNo]]
        stdProj.append(max(stdProjT))
        frontChans = [i for i,c in enumerate(electrodes) if 'F' in c]
        otherChans = [i for i,c, in enumerate(electrodes) if not 'F' in c]
        stdRatio.append(np.mean([stdProjT[i] for i in frontChans]) / np.mean([stdProjT[i] for i in otherChans]))
        
        # get AMI
        featsClust.append( extractFeaturesMultiChsWaveAMI(projIC[iNo],fs) )
        
        # find spike zones in projection
        muSig = abs(np.mean(projIC[tN], 0)) # mean signal across channels
        A1 = np.where(muSig[1:-1] > muSig[2:])[0] + 1
        spikePos = A1[np.where(muSig[A1] > muSig[A1-1])]
        # calculate coefficients
        coefVar = np.zeros(len(spikePos))
        for i in range(len(spikePos)):
            sig_spike = np.std(abs(np.mean(projIC[tN][:,spikePos[i]-1:spikePos[i]+2],0)),ddof=1)
            mu_spike = np.mean(abs(np.mean(projIC[tN][iNo,spikePos[i]-1:spikePos[i]+2],0)))
            coefVar[i] = sig_spike / mu_spike
        # soft thresholding
        T = 0.1 * (np.mean(coefVar) + np.std(coefVar,ddof=1))
        noSpikes.append(len([c for c in coefVar if c > T]) / len(coefVar))
        
        # spikedness of ICs
        if max(abs(ICs[tN][iNo,:])) > np.mean(abs(ICs[tN][iNo,:])) + 3*np.std(ICs[tN][iNo,:],ddof=1):
            remICsSpike.append(iNo)

    # kurtosis threshold
    mu_e = np.mean(entropy)
    sig_e = np.std(entropy,ddof=1)
    remICsKurt = [i for i,e in enumerate(entropy) if e > mu_e + (0.5*sig_e) or e < mu_e - (0.5*sig_e)]      
      
    # psd distance from 1/F distribution threshold
    remICsSpecDist = [i for i,s in enumerate(specDist) if s > 3.5]  
       
    # gamma psd threshold
    remICsGamma = [i for i,g in enumerate(gammaPSD) if g > 1.7]
    
    # psd ratio threshold
    remICsSpecRatio = [i for i,r in enumerate(specRatio) if r > 1.0]
    
    # std threshold
    remICsStd = [i for i,s in enumerate(stdProj) if s > np.mean(stdProj) + 2*np.std(stdProj,ddof=1)]
    
    # std ratio threshold
    remICsStdRatio = [i for i,s in enumerate(stdRatio) if s > np.mean(stdRatio + np.std(stdRatio,ddof=1))]
    
    # AMI threshold
    remICsAMI = [i for i,f in enumerate(featsClust) if f < 2.0 or f > 3.0]
            
    # no. spikes threshold
    remICsNoSpikes = [i for i,n in enumerate(noSpikes) if n >= 0.25]
            
    # all ICs marked for removal by threshold
    remICs_TOTAL = [remICsPT, remICsP2P, remICsSpike, remICsKurt, remICsSpecDist, remICsGamma, remICsSpecRatio, remICsStd, remICsStdRatio, remICsAMI, remICsNoSpikes]
    
    len_iNos = [len([r for r in remICs_TOTAL if iNo in r]) for iNo in range(ICs[tN].shape[0])] # number of thresholds each IC has exceeded
    remICs = [i for i,l in enumerate(len_iNos) if l > 3] # ICs to remove
    
    if len(remICs) == ICs[tN].shape[0]:
        raise Exception('All ICs removed')
    
    # remove ICs
    ICs[tN][remICs,:] = np.zeros((len(remICs), ICs[tN].shape[1]))
    
    return ICs