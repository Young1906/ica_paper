import configparser
from eeg import EEG
from utils import chunks, baseline_calc, butter_bandpass_filter
import numpy as np
from mne.preprocessing import ICA
import mne
from matplotlib import pyplot as plt
import json
import sys, os
import pandas as pd
import joblib
import warnings
warnings.simplefilter("ignore")

# https://github.com/mne-tools/mne-python/issues/2404

config = configparser.ConfigParser()
config.read("config.ini")

_WINDOW = int(config["CHUNK"]["WINDOW"]) 

def _get_ica_map(ica, components=None):
    """Get ICA topomap for components"""
    fast_dot = np.dot
    if components is None:
        components = list(range(ica.n_components_))
    maps = fast_dot(ica.mixing_matrix_[:, components].T,
                    ica.pca_components_[:ica.n_components_])
    return maps

if __name__ == "__main__":

    # path_edf="./edf/1489/1489_alice/edf/A0001489.edf"
    path_edf="./edf/1572/1572_alice/edf/A0001572.edf"
    path_stage="./edf/1572/1572_alice/csv/STAGE.csv"
    
    eeg = EEG(path_edf=path_edf, path_stage=path_stage)

    # name = input("Subject no: ")

   
    _components = []

    N_SAMPLE = int(config["DEFAULT"]["N_SAMPLE_PER_SUBJECT"])
    counter = 0 

    # print(eeg.tasks)
    for i, task in enumerate(eeg.tasks):
        print("Processing task {}".format(i))
        if counter > N_SAMPLE:
            break

        for idx, chunk in enumerate(chunks(task, _WINDOW)):
            print("Processing chunk {}".format(idx))

            if counter > N_SAMPLE:
                break

            _min, _max = np.min(chunk), np.max(chunk)
            
            # Getting sample data
            sample_raw = eeg.raw.copy().crop(_min, _max, include_tmax=False)

            # Apply baseline correction
            sample_raw_baseline = sample_raw.copy()
            sample_raw_baseline.apply_function(baseline_calc)

            # sample_raw.copy().pick('Fp1').plot()
            # sample_raw_baseline.copy().pick('Fp1').plot()

            # break
            # Apply bandpass filter
            sample_raw_bandpass = sample_raw_baseline.copy()
            sample_raw_bandpass.apply_function(butter_bandpass_filter)
            
            # Apply ICA
            sample_raw_train = sample_raw_bandpass.copy()
            sample_raw_corrected = sample_raw_bandpass.copy()
            
            # ICA Train
            ica = ICA(method="extended-infomax", random_state=1)
            ica.fit(sample_raw_corrected)
            
            # Plot ICA component
            ica.plot_sources(inst=sample_raw_train) 
            ica.plot_components(inst=sample_raw_train)
            # Classifying EOG
            print("Classifying EOG...")

            # Get maps
            maps = _get_ica_map(ica).T
            scalings = np.linalg.norm(maps, axis=0)
            maps /= scalings[None, :]
            X = maps.T
            
            # Load classification model
            clf = joblib.load("./models/eog_classifier.joblib")
            # Predict EOG
            eog_preds = clf.predict(X)
            list_of_eog = np.where(eog_preds == 1)[0]

            # Zeroing out EOG components
            if len(list_of_eog) > 0:
                print("Found EOG in the following components: {}".format(str(list_of_eog).strip('[]')))
                ica.exclude = list_of_eog
                # raw_ = eeg.raw.copy()
                # sample_raw_corrected = sample_raw_bandpass.copy()

                ica.apply(sample_raw_corrected)

                sample_raw.plot(title="RAW")
                sample_raw_corrected.plot(title="RAW_CORRECTED")
            else:
                print("EOG not found in this chunk.")

            to_continue = None
            while to_continue not in ['y', 'n']:
                to_continue = input("Do you want to process another chunks? [y/n]")
                if to_continue == 'n': 
                    sys.exit()