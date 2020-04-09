""" Wavelet and ICA decomposition using Second Order Blind Identification (SOBI) 
    - sobi_fast() : Equivalent of sobi_FAST.m from original Matlab with ntrials=1
    - apply_wavelet_ICA() : decompose EEG data into components
    - reconstruct() : reconstruct the EEG data from components
"""
import numpy as np 
from pywt import wavedec, WaveletPacket
from numpy.linalg import svd, pinv, norm, eig

import pdb

def sobi_fast(X):
    """ Apply SOBI
    
        X = H * Sources
        ICs = V * X
    Input:
        - X : input data (channels, samples)
    Returns:
        - H : estimated mixing matrix (channels, channels)
        - V : inverse mixing matrix (channels, channels)
        - X : processed input data (channels, samples)
    """
    DEFAULT_LAGS = 100
    m, N = X.shape
    
    p = min([DEFAULT_LAGS, np.ceil(N/3)])

    kronval = np.kron(np.mean(X,1), np.ones((1,N))).reshape((m,N))
    X = X - kronval

    _, S, VV = np.linalg.svd(np.transpose(X))
    S = np.diag(S)
    Q = np.matmul(np.linalg.pinv(S), VV)
    X = np.matmul(Q,X)

    k = 0
    pm = int(p*m)
    M = np.zeros((m,pm))
    for u in range(0, pm, m):
        k+=1
        Rxp = np.matmul(X[:,k:N], np.transpose(X[:,:N-k])) / (N-k)
        M[:,u:u+m] = np.linalg.norm(Rxp, 'fro') * Rxp

    eps = 1/np.sqrt(N)/100
    encore = True
    V = np.eye(m)

    while encore:
        encore = False
        for p in range(m-1):
            for q in range(p+1,m):
                g = np.array( [ M[p,p:pm:m]-M[q,q:pm:m], 
                               M[p,q:pm:m]+M[q,p:pm:m], 
                               1j*(M[q,p:pm:m]-M[p,q:pm:m]) ] ) 
                D, vcp = np.linalg.eig(np.real(np.matmul(g,np.transpose(g))))
                K = np.argsort(abs(D))
                
                angles = vcp[:,K[2]]
                angles = np.sign(angles[0]) * angles
                c = np.sqrt(0.5 + angles[0]/2)
                
                sr = 0.5 * (angles[1] - 1j * angles[2]) / c
                sc = np.conj(sr)
                
                oui = abs(sr) > eps
                encore = encore or oui

                if oui:
                    colp = M[:,p:pm:m].copy()
                    colq = M[:,q:pm:m].copy()
                    M[:,p:pm:m] = np.real(c * colp + sr * colq)
                    M[:,q:pm:m] = np.real(c * colq - sc * colp)
                    rowp = M[p,:].copy()
                    rowq = M[q,:].copy()
                    M[p,:] = np.real(c * rowp + sc * rowq)
                    M[q,:] = np.real(c * rowq - sr * rowp)
                    temp = V[:,p].copy()
                    V[:,p] = np.real(c * V[:,p] + sr * V[:,q])
                    V[:,q] = np.real(c * V[:,q] - sc * temp)
            
    H = np.matmul( np.linalg.pinv(Q), V )

    return H, V, X

def apply_wavelet_ICA(EEGdata):
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
    ICs = [np.matmul(np.transpose(V), X)]
    
    ICs += [np.array(wp) for wp in waveData[1:]]
    
    return ICs, mixMat, wavePacks, terminal_nodes

def reconstruct(wavePacks, newSigNode, tNodes):
    """ Reconstruct EEG data from wavePacks with updated data """
    nC = len(wavePacks)
    nT = len(tNodes)
    for chNo in range(nC):
        for tN in range(nT):
            wavePacks[chNo][tNodes[tN]].data = newSigNode[tN][chNo]
            
    # reconstruct EEG from wavePacks
    cleanEEG = []
    for chNo in range(nC):
        cleanEEG.append(wavePacks[chNo].reconstruct())
        
    return np.array(cleanEEG)