from matplotlib import pyplot as plt
import numpy as np
from figure_v3 import plot_sig
from eeg_v2 import EEG
from utils import chunks, baseline_calc, butter_bandpass_filter
import joblib
from mne.preprocessing import ICA
from utils import calc_psd

_scale_c = 9.0175556274876e-05
_path = "./figures/"

def _get_ica_map(ica, components=None):
    """Get ICA topomap for components"""
    fast_dot = np.dot
    if components is None:
        components = list(range(ica.n_components_))
    maps = fast_dot(ica.mixing_matrix_[:, components].T,
                    ica.pca_components_[:ica.n_components_])
    return maps

def get_chunks(sig):
    i = 0
    for task in sig.tasks:
        for c in chunks(task, 15):
            i+=1
            yield i, c
            
def count_chunks(sig):
    i = 0
    for task in sig.tasks:
        for c in chunks(task, 15):
            i+=1
    return i

# Reading raw data
path_edf="./edf/1578_alice/edf/A0001578.edf"
path_stage="./edf/1578_alice/csv/STAGE.csv"

eeg = EEG(path_edf=path_edf, path_stage=path_stage)

# Preprocessing pipeline
def baseline_bandpass(chunk):
    sample_raw = eeg.raw.copy().crop(np.min(chunk),np.max(chunk),include_tmax=False)
    sample_raw_eog = eeg.eog_channels.copy().crop(np.min(chunk),np.max(chunk),include_tmax=False)


    # Baseline
    sample_raw_baseline = sample_raw.copy()
    sample_raw_baseline = sample_raw_baseline.apply_function(baseline_calc)

    # Bandpass
    sample_raw_bandpass = sample_raw_baseline.copy()
    sample_raw_bandpass = sample_raw_bandpass.apply_function(butter_bandpass_filter)
    
    return sample_raw, sample_raw_baseline, sample_raw_bandpass, sample_raw_eog

# ICA pipeline
def ica_pipe(sample_raw_bandpass):    
    clf = joblib.load("./models/eog_classifier_v2.joblib")


    sample_raw_train = sample_raw_bandpass.copy()
    sample_raw_corrected = sample_raw_bandpass.copy()

    # Fitting ICA
    ica = ICA(method="extended-infomax", random_state=1)
    ica.fit(sample_raw_corrected)

    maps = _get_ica_map(ica).T
    scalings = np.linalg.norm(maps, axis=0)
    maps /= scalings[None, :]
    X = maps.T

    # Predict EOG
    eog_preds = clf.predict(X)
    list_of_eog = np.where(eog_preds == 1)[0]
    
    # ica.plot_sources(inst=sample_raw_train) 
    # ica.plot_components(inst=sample_raw_train)

    ica.exclude = list_of_eog
    ica.apply(sample_raw_corrected)
    
    return ica, sample_raw_train, sample_raw_corrected

plt.rcParams["figure.figsize"] = [10,10]

if __name__ == "__main__":   
    # fig2 
    i = 0
    for task in eeg.tasks:
        for c in chunks(task, 15):
            i+=1
            
            if i!=7:
                continue
            
            if i>7:
                break
            
            sample_raw, sample_raw_baseline, sample_raw_bandpass, sample_raw_eog = baseline_bandpass(c)
            

            # Raw
            plot_sig(
                X = sample_raw.get_data() * 1e3,
                fs = 500,
                dpi = 1200,
                unit = "μV",
                ch_names = sample_raw.ch_names,
                X_EOG = sample_raw_eog.get_data() * 1e3,
                eog_ch_names = sample_raw_eog.ch_names,
                scale = _scale_c*1e3,
                fname=f"{_path}fig2_raw"
            )
            

            # Baseline + Bandpass
            plot_sig(
                X = sample_raw_bandpass.get_data() * 1e3,
                fs = 500,
                dpi = 1200,
                unit = "μV",
                ch_names = sample_raw.ch_names,
                X_EOG = sample_raw_eog.get_data() * 1e3,
                eog_ch_names = sample_raw_eog.ch_names,
                markers = [.3,2.2,3.3,4.9],
                scale = _scale_c*1e3,
                fname=f"{_path}fig2_bandpass"

            )


    