import configparser
from eeg_v2 import EEG
from utils import chunks, baseline_calc, butter_bandpass_filter
import numpy as np
from mne.preprocessing import ICA
import mne
from matplotlib import pyplot as plt
import json
import sys, os
import pandas as pd

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
    path_edf="./edf/1578_alice/edf/A0001578.edf"
    path_stage="./edf/1578_alice/csv/STAGE.csv"
    
    eeg = EEG(path_edf=path_edf, path_stage=path_stage)

    name = input("Subject no: ")

   
    _components = []

    N_SAMPLE = int(config["DEFAULT"]["N_SAMPLE_PER_SUBJECT"])
    counter = 0 

    if not os.path.exists("csvs"):
        os.mkdir("csvs")

    for task in eeg.tasks:
        if counter > N_SAMPLE:
            break

        for idx, chunk in enumerate(chunks(task, _WINDOW)):

            if counter > N_SAMPLE:
                break

            _min, _max = np.min(chunk), np.max(chunk)
            
            # Getting sample data
            sample_raw = eeg.raw.copy().crop(_min, _max, include_tmax=False)
            eog_channels = eeg.eog_channels.copy().crop(_min, _max, include_tmax=False)

            # Apply baseline correction
            sample_raw_baseline = sample_raw.copy()
            sample_raw_baseline.apply_function(baseline_calc)
            # eog_channels.apply_function(baseline_calc)
            
            # Apply bandpass filter
            sample_raw_bandpass = sample_raw_baseline.copy()
            sample_raw_bandpass.apply_function(butter_bandpass_filter)
            # eog_channels.apply_function(butter_bandpass_filter)

            # Apply ICA
            sample_raw_train = sample_raw_bandpass.copy()
            sample_raw_corrected = sample_raw_bandpass.copy()

            # Train
            ica = ICA(method="extended-infomax", random_state=1)
            ica.fit(sample_raw_corrected)
            
            # Plot ICA component
            eog_channels.plot(title="EOG_Channels", duration=15, n_channels=10, scalings=dict(eeg=1200e-6))
            ica.plot_sources(inst=sample_raw_train) 
            ica.plot_components(inst=sample_raw_train)

            # eog_min, eog_max = np.min(eog_channels.get_data()[0,:]),np.max(eog_channels.get_data()[0,:])
            # _source_0 = ica.get_sources(inst=sample_raw_train).get_data()[0,:]
            # eeg_min, eeg_max = np.min(_source_0), np.max(_source_0)
            # eeg_min, eeg_max = np.min(ica.get_sources(inst=sample_raw_train)[0,:]),np.max(ica.get_sources(inst=sample_raw_train)[0,:])

            import pdb; pdb.set_trace()
            
            while True:
                try:
                    list_of_eog = input("List of components seperated by space: ")
                    list_of_eog = list(map(int, list_of_eog.split()))
                    break
                except ValueError:
                    print("Try again..")
            
            if list_of_eog:

                ica.exclude = list_of_eog
                # raw_ = eeg.raw.copy()
                # sample_raw_corrected = sample_raw_bandpass.copy()

                ica.apply(sample_raw_corrected)

                sample_raw.plot(title="RAW")
                sample_raw_corrected.plot(title="RAW_CORRECTED")


                to_save = input("Save this trunk? [y/n]: ")
                # print("=================================================================================")

                if to_save == 'y':  
                    maps = _get_ica_map(ica).T
                    scalings = np.linalg.norm(maps, axis=0)
                    maps /= scalings[None, :]
                    
                    # maps[:,0] = 0

                    components = ica.get_sources(inst=sample_raw_train).get_data()
                    components_name = ica.ch_names

                    for idx2, comp_name in enumerate(components_name):
                        tmp = {
                            "name" : "{comp_name}_{idx}".format(comp_name=comp_name, idx=idx),
                            # "component" : list(components[idx2, :]),
                            "map" : list(maps[:, idx2]),
                            "label" : "EOG" if idx2 in list_of_eog else "Non-EOG"
                        }
                        # Save file to csv
                        temp = pd.DataFrame(components[idx2,:].reshape(-1, len(components[idx2,: ])), columns = [f"comp_{i}" for i in range(len(components[idx2,:]))])
                            
                        temp['name'] = "{comp_name}_{idx}".format(comp_name=comp_name, idx=idx)
                        temp['label'] = "EOG" if idx2 in list_of_eog else "Non-EOG"
                        # print(temp)
                        comp_map = list(maps[:, idx2])
                        # print(comp_map)
                        for i in range(len(comp_map)):
                            temp[f"map_component_{i}"] = comp_map[i]
                        temp.to_csv(f'./csvs/{name}_{idx}_{comp_name}.csv', index=False, header=True)
                        # print(maps[:, idx2].shape)
                        # sys.exit()
                        try:
                            _components.append(tmp)
                            counter+=1
                        
                        except Exception as e:
                            break

    j = {
        "name" : name,
        "data" : _components
    }

    with open("data.json", "w") as f:
        json.dump(j, f)



            # import pdb; pdb.set_trace()


            # print(maps[0,:])
            # input()

            # fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(20, 7))

            # info_temp = mne.create_info(eeg.ch_names, 1,
            #                 ch_types='eeg', montage="standard_1005")

            # evoked = mne.EvokedArray(maps, info_temp, 0, comment='', nave=1,
            #              kind='average', verbose=None)

            # evoked.plot_topomap(vmin=-0.6e6, vmax=0.6e6, cmap='jet',
            #                     size=2, contours=None, time_format='', colorbar=False,
            #                     axes=axes, show=False)

            # plt.show()
            # break