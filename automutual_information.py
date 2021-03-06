""" Extract features for Auto-Mutual Information """
import numpy as np

def minf(pab, papb, L):
    eps = 1e-12    
    rows = []
    cols = []
    for i in range(L):
        for j in range(L):
            if pab[i,j] > eps and papb[i,j] > eps:
                rows.append(i)
                cols.append(j)
    I = (np.array(rows), np.array(cols))            
    y = pab[I] * np.log2(pab[I] / papb[I]) # same values, different order from Matlab
    return y

def hist2a(A,B,L):
    ma = min(A)
    MA = max(A)
    mb = min(B)
    MB = max(B)
    eps = 2.2204e-16
    A = np.round((A-ma)*(L-1)/(MA-ma+eps))
    B = np.round((B-mb)*(L-1)/(MB-mb+eps))
    n = np.zeros((L,L))
    for i in range(L):
        inds = np.where(A==i)
        a = np.digitize(B[inds], [*range(L)])
        for k in range(len(a)):
            n[i,a[k]-1] += 1
    return n        

def mi(A, B):
    L = 32
    na,_ = np.histogram(A[:],L)
    na = na/sum(na)
    nb,_ = np.histogram(B[:],L)
    nb = nb/sum(nb)
    n2 = hist2a(A,B,L)
    n2 = n2/np.sum(n2)

    t = minf( n2, np.outer(na,nb), L )
    return sum(t)

def extractFeaturesMultiChsWaveAMI(signalSet, Fs):
    """ calculate auto-mutual information (AMI) """
    lagOffsets = [60]
    features = np.zeros((1,len(lagOffsets)))
    for lagNo in range(len(lagOffsets)):
        endP = signalSet.shape[1]
        lags = range(0, int(np.floor(endP/2)), lagOffsets[lagNo])
        AMIs = np.zeros((signalSet.shape[0],len(lags)))
        for chNo in range(signalSet.shape[0]):
            for j in range(len(lags)):
                AMIs[chNo,j] = mi( signalSet[chNo, :endP-lags[j]-1], signalSet[chNo, 1+lags[j]:endP] )
        features[lagNo] = max(np.mean(AMIs,1))
    
    return features
