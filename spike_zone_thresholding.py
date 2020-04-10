""" Spike Zone Thresholding for both approximation and detail coefficients """
import numpy as np 

from constants import WEIGHT_APPROX, WEIGHT_DETAIL, GAIN_APPROX, GAIN_DETAIL

def apply_soft_threshold(newSigNode, tNodes):
    for tN in range(len(tNodes)):
        adjustVal = GAIN_APPROX if tN == 0 else GAIN_DETAIL
        checkVal = WEIGHT_APPROX if tN == 0 else WEIGHT_DETAIL
            
        spikePos = []
        coefVar = []
        for iNo in range(newSigNode[tN].shape[0]):
            muSig = newSigNode[tN][iNo,:]
            A1up = np.where(muSig[1:-1] > muSig[2:])[0] +1
            A1lo = np.where(muSig[1:-1] < muSig[2:])[0] +1
            sp_temp = np.concatenate((A1up[ np.where(muSig[A1up] > muSig[A1up-1]) ], A1lo[ np.where(muSig[A1lo] < muSig[A1lo-1]) ] ))
            indUp = np.where(sp_temp > 2)
            indUse = indUp[0][ np.where(sp_temp[indUp] < (len(muSig)-3)) ] 
            spikePos.append(sp_temp[indUse])
            
            # segments of +- 3 samples
            upperVals = np.var([ newSigNode[tN][iNo, spikePos[iNo] + x] for x in range(-3,4) ],axis=0, ddof=1)
            lowerVals = np.std([ newSigNode[tN][iNo, spikePos[iNo] + x] for x in range(-3,4) ],axis=0, ddof=1)
            
            coefVar.append( upperVals / lowerVals )            
        allCoefVar = np.concatenate(coefVar)

        # apply soft thresholding
        for iNo in range(newSigNode[tN].shape[0]):
            T = checkVal * (np.mean(allCoefVar) + np.std(allCoefVar, ddof=1))
            for i in range(len(spikePos[iNo])):
                if abs(coefVar[iNo][i]) > T:
                    newSigNode[tN][iNo, spikePos[iNo][i]] *= adjustVal               
        
    return newSigNode  
                
       
    