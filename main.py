import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from setup import Setup as setup
import preprocessing

new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

def prep_data():
    ############### IMPORT DATA & SIGNAL PROCESSING ###############
    # list of raw data files in local data folder
    subject_ind = '1'
    data_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S'+subject_ind)
    raw_data_list = os.listdir(data_folder_path)

    for file_name in raw_data_list:
        if file_name.endswith('eeg.xdf'):
            preprocessed = False
            for name in raw_data_list:
                if file_name.replace('.xdf','.fif') in name:
                    preprocessed = True
                    print(file_name, 'already has a preprocessed .fif file.')
            if preprocessed == False and not file_name.startswith('training'):
                print(file_name, 'is not preprocessed.')
                # create dict of lsl stream info
                source = ['BrainVision', 'ForeheadE-tattoo']
                n_ch = [4, 6]
                ch_name = [['AF8','Fp2','Fp1','AF7'], ['AF8','Fp2','Fp1','AF7','hEOG','vEOG']]
                ch_type = [['eeg']*4+['eog']*0, ['eeg']*4+['eog']*2]
                daq_source = {'source': source,'n_ch': n_ch,'ch_name': ch_name,'ch_type': ch_type}
                # read .xdf files
                raw_path = os.path.join(data_folder_path, file_name)
                # create setup and raws from different systems
                raw_setup = setup(data_path=raw_path, data_type='xdf',stream_source=daq_source)
                raw_dict = raw_setup.raw_dict

                # preprocess raws
                for key in raw_dict.keys():
                    raw = raw_dict[key]
                    # bandpass filtering
                    filters = preprocessing.Filtering(raw=raw, l_freq=0.01, h_freq=50, picks=['eeg','eog'])
                    raw = filters.external_artifact_rejection(resample=False, notch=False)
                    # filters = preprocessing.Filtering(raw=raw, l_freq=0.01, h_freq=50, picks='eeg')
                    # raw = filters.external_artifact_rejection(resample=False, notch=False)
                    # if 'eog' in ch_type:
                    #     filters = preprocessing.Filtering(raw=raw, l_freq=0.01, h_freq=10, picks='eog')
                    #     raw = filters.external_artifact_rejection(resample=False, notch=False)
                    raw = filters.resample(new_sfreq=50)
                    # interactively annotate bad signal
                    interactive_annot = raw_setup.annotate_interactively(raw=raw)
                    print(interactive_annot)
                    # save preprocessed raw as .fif
                    raw_name = key + '_' + file_name.replace('.xdf','.fif')
                    raw.save(os.path.join(data_folder_path, raw_name), overwrite=True)

def eye_oc():
    pass

def n_back_analysis():
    ############### IMPORT DATA & SIGNAL PROCESSING ###############
    # list of raw data files in local data folder
    subject_ind = '1'
    data_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S'+subject_ind)
    raw_data_list = os.listdir(data_folder_path)

    # get path of EEG montage
    montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/passive electrodes/BrainCap','BrainCap 64 Channel','BC-64.bvef')
    # initilize epoch and event lists
    epochs_list = []
    theta_epochs_list = []
    alpha_epochs_list = []
    beta_epochs_list = []
    events_list = []

    for file_name in raw_data_list:
        if file_name.endswith('eeg.fif'):
            raw_path = os.path.join(data_folder_path, file_name)
            raw = mne.io.read_raw_fif(raw_path)
            raw.load_data()
            custom_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, 'fixation': 10, 'response_alpha': 100, 'response_pos': 101}
            # custom_mapping = {'0-Back': 0, '1-Back': 1, '2-Back': 2, '3-Back': 3, 'fixation': 10, 'character response': 100, 'location response': 101}
            events, event_dict = mne.events_from_annotations(raw, event_id=custom_mapping)
            montage = mne.channels.read_custom_montage(montage_path)
            if file_name.startswith('BrainVision'):
                montage = mne.channels.make_standard_montage('easycap-M1', head_size='auto')
                sphere = None
                # montage.plot(show=False)
            else:
                montage = mne.channels.make_standard_montage('standard_1020', head_size='auto')
                sphere = (0, 0.02, 0, 0.1)
                # montage.plot(show=False, sphere=sphere)
            # plt.show(block=True)
            raw.set_montage(montage)

            # bandpass filtering
            # filters = preprocessing.Filtering(raw=raw, l_freq=1, h_freq=30)
            # raw = filters.external_artifact_rejection(resample=False, notch=False)
            ica = preprocessing.Indepndent_Component_Analysis(raw, n_components=raw.info['nchan']-2, seed=92)
            reconst_raw = ica.perfrom_ICA()

            epochs = mne.Epochs(raw=reconst_raw, events=events, event_id=event_dict,event_repeated='drop', tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
            # n0_back = epochs['0']
            # n1_back = epochs['1']
            # n2_back = epochs['2']
            # n3_back = epochs['3']
            # for epk in [n0_back, n1_back, n2_back, n3_back]:
            #     # global field power and spatial plot
            #     epk.plot_image(picks='eeg',combine='mean')
            #     evk = epk.average()
            #     evk.plot(gfp=True, spatial_colors=True)
            epochs_list.append(epochs)
    # concatenate all epochs from different trials
    all_epochs = mne.concatenate_epochs(epochs_list)
    # epoch analysis
    all_n0_back = all_epochs['0'].average()
    all_n1_back = all_epochs['1'].average()
    all_n2_back = all_epochs['2'].average()
    all_n3_back = all_epochs['3'].average()
    for evk in [all_n0_back, all_n1_back, all_n2_back, all_n3_back]:
        # global field power and spatial plot
        evk.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-4, 4]))
        # spatial plot + topomap
        evk.plot_joint()

if __name__ == '__main__':
    prep_data()
    n_back_analysis()
