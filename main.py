import configparser
from eeg import EEG
from utils import chunks, baseline_calc, butter_bandpass_filter
import numpy as np
from mne.preprocessing import ICA

config = configparser.ConfigParser()
config.read("config.ini")

_WINDOW = int(config["CHUNK"]["WINDOW"]) 



if __name__ == "__main__":
    
    eeg = EEG(path_edf="./edf/1489/1489_alice/edf/A0001489.edf", \
        path_stage="./edf/1489/1489_alice/csv/STAGE.csv")


    for task in eeg.tasks:
        for chunk in chunks(task, _WINDOW):
            _min, _max = np.min(chunk), np.max(chunk)
            
            # Getting sample data
            sample_raw = eeg.raw.copy().crop(_min, _max, include_tmax=False)

            # Apply baseline correction
            sample_raw_baseline = sample_raw.copy()
            sample_raw_baseline.apply_function(baseline_calc)
            
            # Apply bandpass filter
            sample_raw_bandpass = sample_raw_baseline.copy()
            sample_raw_bandpass.apply_function(butter_bandpass_filter)
            
            # Apply ICA
            sample_raw_train = sample_raw_bandpass.copy()
            sample_raw_corrected = sample_raw_bandpass.copy()
            
            # Train
            ica = ICA(method="extended-infomax", random_state=1)
            ica.fit(sample_raw_train)
            
            # Plot ICA component 
            ica.plot_components(inst=sample_raw_train)
            