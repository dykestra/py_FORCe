# FORCe python version

Python implementation of 'Fully Online and automated artifact Remove for brain-Computer interfacing' (FORCe) 

Information about original Matlab code can be found [here](https://www.iandaly.org/force/).
> "FORCe attempts to perform fully automated EEG artifact removal for brain-computer interfaces (BCIs). It is able to remove blinks, ECG, movement, and a large amount of EMG artifact from the EEG very quickly. Therefore, it can be used during online BCI. FORCe operates by first taking a Wavelet decomposition of a short window of the EEG and then applying a combination of soft and hard thresholding to the detail coefficients of the decompositions. The approximation coefficients are further processed by independent component analysis and combinations of various statistical thresholds are used to automatically identify components which contain artifacts are remove them."


## Dependencies
- numpy
- scipy
- pywavelets v1.0.1

## How to Use
To import:
```
from FORCe import FORCe
```

To run:
```
clean_EEG = FORCe(EEGdata, fs, electrodes)
```

Where     
- `EEGdata` : numpy array of (channels, samples)
- `fs` : sample rate in Hz
- `electrodes` : list of electrode labels e.g. ['AF3', 'F7', 'F3', ...]

## Electrode data
This artifact removal method requires the 3D positions of each electrode. Known electrode locations are stored in `electrode_locations.py`. If your electrodes are not listed here please add their coordinates to the dict.

## References

Daly, I. et al., 2014. FORCe: Fully Online and automated artifact Removal for brain-Computer interfacing. IEEE transactions on neural systems and rehabilitation engineering?: a publication of the IEEE Engineering in Medicine and Biology Society. Available at: http://www.ncbi.nlm.nih.gov/pubmed/25134085