import mne
import pandas as pd
from mne.io import read_raw_edf
import numpy as np
from scipy.signal import butter, lfilter
from scipy import signal 
from scipy.signal import welch

# Sample rate and desired cutoff frequencies (in Hz).
FS = 500.0
LOWCUT = 0.5
HIGHCUT = 45.0

# https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

# Calculate Representative Rest1 mean:
def baseline_calc(a):
    e = np.mean(a)
    return a - e

# Banpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut=.5, highcut=45., fs=500, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def calc_psd(s, _fs = FS, _avg='median', fmax = 100):
    _nperseg = 4*_fs
    x, y = welch(s, fs=_fs, average=_avg, nperseg=_nperseg)
    x, y = x[np.where(x<fmax)], y.T[np.where(x<fmax)]

    return x, y 

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n + 1]