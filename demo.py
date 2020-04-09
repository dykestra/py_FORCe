import pickle
import numpy as np
import matplotlib.pyplot as plt

from FORCe import FORCe

SAMPLE_RATE = 256
ELECTRODES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

def run_FORCe(raw):
    """ Applies FORCe to clip in 1s sliding window """
    win_length_s = 1.0
    win_length = int(win_length_s * SAMPLE_RATE)
    segment_length = raw.shape[1]
    EEG_clean = []

    for iw in range(0, segment_length, win_length):
        window = raw[:,iw : (iw + win_length)]
        EEG_clean.append(FORCe(window, SAMPLE_RATE, ELECTRODES))
        
    return np.concatenate([win for win in EEG_clean],1)

def plot_EEG(eeg, title):
    """ Plot visualisation of EEG data """
    N, S = eeg.shape # no. channels, no. samples
    sep = 150 # separation between channel signals
    
    ax = plt.subplot(111)
    ax.set_title(title)
    for i in range(N):
        ax.plot(eeg[i]-(i*sep), linewidth=1, label=ELECTRODES[i])
        
    ax.axis([0, S, -N*sep, sep])
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8))

if __name__ == '__main__':
    raw = pickle.load(open('example_EEG.p', 'rb'))
    plt.figure(1)
    plot_EEG(raw, 'Raw EEG')
    
    clean = run_FORCe(raw)
    plt.figure(2)
    plot_EEG(clean, 'Clean EEG')
    
    plt.show()
    
    print('CLEANED')