""" Functions to perform channel threshold and interpolate signal for removed channels """
import numpy as np
from electrode_locations import electrode_locations
from constants import CHANNEL_THRESH

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
        for i in range(EEGdata.shape[1]): 
            EEGdata[rc,i] = np.mean( np.divide( EEGdata[keepChs,i], dist ) )
        
    return EEGdata

def channel_threshold(EEGdata, electrodes):
    """ perform channel threshold """
    remCh = [i for i,c in enumerate(EEGdata) if np.max(c) > CHANNEL_THRESH ]
    if len(remCh) == EEGdata.shape[0]: 
        raise Exception('All channels > {}uV'.format(CHANNEL_THRESH))
    
    EEGdata = estimate_removed_channels(EEGdata, remCh, electrodes)
    return EEGdata
    