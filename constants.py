""" Config file for all thresholds """

CHANNEL_THRESH = 200 # uV

""" Contaminated IC Detection/Removal """
PROJECTION_MIN = -100 # uV
PROJECTION_MAX = 100 # uV
P2P_MAX = 60 # uV

KURTOSIS_FACTOR = 0.5 # multiplied by std of entropy

SPEC_DIST_MAX = 3.5 # max distance from ideal distribution
GAMMA_MAX = 1.7

SPEC_RATIO_MAX = 1.0 # max ratio of high/low freqs in ICs

STD_FACTOR = 2 # multiplied by std of std of projection
STD_RATIO_FACTOR = 1 # multiplied by std of frontal/other channel std ratio

AMI_MIN = 2 # auto-mutual information threshold
AMI_MAX = 3

PROJECTION_SPIKE_THRESH = 0.25 # max value for no. identified spikes / total no. spike zone coefficients

IC_SPIKE_FACTOR = 3 # multiplied by std of IC

""" Soft Thesholding """
WEIGHT_APPROX = 0.7
WEIGHT_DETAIL = 0.2
GAIN_APPROX = 0.8
GAIN_DETAIL = 0.07
