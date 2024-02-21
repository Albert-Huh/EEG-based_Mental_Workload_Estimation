import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from setup import Setup as setup
import preprocessing

new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

def main():
    ############### IMPORT DATA & SIGNAL PROCESSING ###############
    # list of raw data files in local data folder
    data_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/motion')
    raw_data_list = os.listdir(data_folder_path)

    # get path of EEG montage
    montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/passive electrodes/BrainCap','BrainCap 64 Channel','BC-64.bvef')
    # initilize epoch and event lists
    bv_epochs_list = []
    et_epochs_list = []
    bv_theta_epochs_list = []
    alpha_epochs_list = []
    bv_beta_epochs_list = []
    et_theta_epochs_list = []
    et_alpha_epochs_list = []
    et_beta_epochs_list = []

    events_list = []

    for file_name in raw_data_list:
        if file_name.endswith('.xdf'):
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
                ch_name = [['AF7', 'Fp1', 'Fp2', 'AF8'], ['AF8','Fp2','Fp1','AF7','hEOG','vEOG']]
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
                    filters = preprocessing.Filtering(raw=raw, l_freq=0.01, h_freq=50, picks='eeg')
                    raw = filters.external_artifact_rejection(resample=False, notch=False)
                    if 'eog' in ch_type:
                        filters = preprocessing.Filtering(raw=raw, l_freq=0.01, h_freq=10, picks='eog')
                        raw = filters.external_artifact_rejection(resample=False, notch=False)
                    # interactively annotate bad signal
                    interactive_annot = raw_setup.annotate_interactively(raw=raw)
                    print(interactive_annot)
                    # save preprocessed raw as .fif
                    raw_name = key + '_' + file_name.replace('.xdf','.fif')
                    raw.save(os.path.join(data_folder_path, raw_name), overwrite=True)

        if file_name.endswith('E-tattoo_eye movements.fif'):
            raw_path = os.path.join(data_folder_path, file_name)
            et_eyemove_raw = mne.io.read_raw_fif(raw_path)
            filters = preprocessing.Filtering(raw=et_eyemove_raw, l_freq=0.01, h_freq=50)
            # et_eyemove_raw = filters.external_artifact_rejection(resample=False, notch=False)
            et_eyemove_raw = filters.resample(new_sfreq=50)
            
            t_start = 17.4
            t_end = 20.4
            et_lookUp = et_eyemove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            et_lookUp._data[4] = et_lookUp._data[4] + 350e-6
            et_lookUp._data[5] = et_lookUp._data[5] + 450e-6
            t_start = 21.0
            t_end = 24.0
            et_lookDown = et_eyemove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            et_lookDown._data[4] = et_lookDown._data[4] + 350e-6
            et_lookDown._data[5] = et_lookDown._data[5] + 450e-6
            t_start = 24.3
            t_end = 27.3
            et_lookLeft = et_eyemove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            et_lookLeft._data[4] = et_lookLeft._data[4] + 350e-6
            et_lookLeft._data[5] = et_lookLeft._data[5] + 450e-6
            t_start = 28.0
            t_end = 31.0
            et_lookRight = et_eyemove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            et_lookRight._data[4] = et_lookRight._data[4] + 350e-6
            et_lookRight._data[5] = et_lookRight._data[5] + 450e-6
            t_start = 105.9
            t_end = 109.9
            et_weakBlink = et_eyemove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            t_start = 110.5
            t_end = 114.5
            et_strongBlink = et_eyemove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            
            custom_mapping = {'up': 1, 'down': 2, 'left': 3, 'right': 4, 'weak': 5, 'strong': 6}
            events, event_dict = mne.events_from_annotations(et_eyemove_raw, event_id=custom_mapping)
            montage = mne.channels.make_standard_montage('standard_1020', head_size='auto')
            sphere = (0, 0.02, 0, 0.1)
            et_eyemove_raw.set_montage(montage)
        if file_name.endswith('E-tattoo_forehead.fif'):
            raw_path = os.path.join(data_folder_path, file_name)
            et_forehead_raw = mne.io.read_raw_fif(raw_path)
            et_forehead_raw.load_data()
            filters = preprocessing.Filtering(raw=et_forehead_raw, l_freq=0.01, h_freq=50)
            # et_forehead_raw = filters.external_artifact_rejection(resample=False, notch=False)
            et_forehead_raw = filters.resample(new_sfreq=50)

            t_start = 250.6
            t_end = 255.6
            et_raiseEyebrow = et_forehead_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            t_start = 255.9
            t_end = 260.9
            et_smile = et_forehead_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            t_start = 261.11
            t_end = 266.11
            et_swallow = et_forehead_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()

        if file_name.endswith('Vision_forehead.fif'):
            raw_path = os.path.join(data_folder_path, file_name)
            bv_forehead_raw = mne.io.read_raw_fif(raw_path)
            bv_forehead_raw.load_data()
            filters = preprocessing.Filtering(raw=bv_forehead_raw, l_freq=0.01, h_freq=50)
            # bv_forehead_raw = filters.external_artifact_rejection(resample=False, notch=False)
            bv_forehead_raw = filters.resample(new_sfreq=50)

            t_start = 252.8
            t_end = 257.8
            bv_raiseEyebrow = bv_forehead_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            t_start = 258.2
            t_end = 263.2
            bv_smile = bv_forehead_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            t_start = 263.3
            t_end = 268.3
            bv_swallow = bv_forehead_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()

        if file_name.endswith('E-tattoo_head movements 2.fif'):
            raw_path = os.path.join(data_folder_path, file_name)
            et_headmove_raw = mne.io.read_raw_fif(raw_path)
            et_headmove_raw.load_data()
            filters = preprocessing.Filtering(raw=et_headmove_raw, l_freq=0.01, h_freq=50)
            # et_headmove_raw = filters.external_artifact_rejection(resample=False, notch=False)
            et_headmove_raw = filters.resample(new_sfreq=50)

            t_start = 215.12
            t_end = 230.12
            et_faceUDLR = et_headmove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            # et_faceUp = et_headmove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            # et_faceDown = et_headmove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            # et_faceLeft = et_headmove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            # et_faceRight = et_headmove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()

        if file_name.endswith('Vision_head movements 2.fif'):
            raw_path = os.path.join(data_folder_path, file_name)
            bv_headmove_raw = mne.io.read_raw_fif(raw_path)
            bv_headmove_raw.load_data()
            filters = preprocessing.Filtering(raw=bv_headmove_raw, l_freq=0.01, h_freq=50)
            # bv_headmove_raw = filters.external_artifact_rejection(resample=False, notch=False)
            bv_headmove_raw = filters.resample(new_sfreq=50)

            t_start = 215.0
            t_end = 230.0
            bv_faceUDLR = bv_headmove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            # bv_faceUp = bv_headmove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            # bv_faceDown = bv_headmove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            # bv_faceLeft = bv_headmove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            # bv_faceRight = bv_headmove_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()

        if file_name.endswith('E-tattoo_walk and run 2.fif'):
            raw_path = os.path.join(data_folder_path, file_name)
            et_walkrun_raw = mne.io.read_raw_fif(raw_path)
            et_walkrun_raw.load_data()
            filters = preprocessing.Filtering(raw=et_walkrun_raw, l_freq=0.01, h_freq=50)
            # et_walkrun_raw = filters.external_artifact_rejection(resample=False, notch=False)
            et_walkrun_raw = filters.resample(new_sfreq=50)

            t_start = 15.5
            t_end = 20.5
            et_walk = et_walkrun_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            t_start = 7.5
            t_end = 12.5
            et_run = et_walkrun_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()

        if file_name.endswith('Vision_walk and run 2.fif'):
            raw_path = os.path.join(data_folder_path, file_name)
            bv_walkrun_raw = mne.io.read_raw_fif(raw_path)
            bv_walkrun_raw.load_data()
            filters = preprocessing.Filtering(raw=bv_walkrun_raw, l_freq=0.01, h_freq=50)
            # bv_walkrun_raw = filters.external_artifact_rejection(resample=False, notch=False)
            bv_walkrun_raw = filters.resample(new_sfreq=50)

            t_start = 15.5
            t_end = 20.5
            bv_walk = bv_walkrun_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()
            t_start = 7.5
            t_end = 12.5
            bv_run = bv_walkrun_raw.copy().crop(tmin=t_start, tmax=t_end).load_data()


    # fig = et_eyemove_raw.pick('eog').plot(block = False)
    # fig = et_forehead_raw.pick('eeg').plot(block = False)
    # fig = bv_forehead_raw.plot(block = False)
    # fig = et_headmove_raw.pick('eeg').plot(block = False)
    # fig = bv_headmove_raw.plot(block = False)
    # fig = et_walkrun_raw.pick('eeg').plot(block = False)
    # fig = bv_walkrun_raw.plot(block = True)
    
    eye_movement_list = [et_lookUp,et_lookDown,et_lookLeft,et_lookRight,et_weakBlink,et_strongBlink]
    et_raw_eye_movement = mne.concatenate_raws(eye_movement_list)
    et_motion_artifact_list = [et_faceUDLR,et_raiseEyebrow,et_smile,et_swallow,et_walk,et_run]
    et_raw_motion_artifact = mne.concatenate_raws(et_motion_artifact_list).filter(l_freq=0.5, h_freq=None, method='fir')
    bv_motion_artifact_list = [bv_faceUDLR,bv_raiseEyebrow,bv_smile,bv_swallow,bv_walk,bv_run]
    bv_raw_motion_artifact = mne.concatenate_raws(bv_motion_artifact_list).filter(l_freq=0.5, h_freq=None, method='fir')

    scaling = dict(eeg=100e-6, eog=1500e-6)
    fig1 = et_raw_eye_movement.pick('eog').plot(duration=200, scalings=scaling, block = False)
    fig1.set_size_inches(10, 4)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300 
    mpl.rcParams['font.sans-serif'] = "Myriad Pro"
    mpl.rcParams['font.family'] = "sans-serif"
    filename = "eog_raw.svg"
    plt.savefig(filename, format='svg')
    fig2 = et_raw_motion_artifact.pick('eeg').plot(duration=200, scalings=scaling, block = False)
    fig2.set_size_inches(10, 4)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300 
    mpl.rcParams['font.sans-serif'] = "Myriad Pro"
    mpl.rcParams['font.family'] = "sans-serif"
    filename = "et_motion_artifact_raw.svg"
    plt.savefig(filename, format='svg')
    fig3 = bv_raw_motion_artifact.plot(duration=200, scalings=scaling, block = False)
    fig3.set_size_inches(10, 4)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300 
    mpl.rcParams['font.sans-serif'] = "Myriad Pro"
    mpl.rcParams['font.family'] = "sans-serif"
    filename = "bv_motion_artifact_raw.svg"
    plt.savefig(filename, format='svg')
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(et_raw_motion_artifact.times, et_raw_motion_artifact._data[1], label='E-tattoo', color="lightskyblue")
    ax[0].legend()
    ax[0].set_xlabel('Time (s)')
    ax[1].plot(bv_raw_motion_artifact.times, bv_raw_motion_artifact._data[2], label='Brain Vision', color="skyblue")
    ax[1].legend()
    ax[1].set_xlabel('Time (s)')

    fig, ax = plt.subplots(nrows=1)
    ax.vlines(x=[3.5, 7.5,11, 15, 20, 25, 30, 35], ymin=-5, ymax=5, colors='black', ls='--', lw=0.8)
    ax.plot(bv_raw_motion_artifact.times, bv_raw_motion_artifact._data[2]*1e3, label='Brain Vision', color="gray", lw=1)
    ax.plot(et_raw_motion_artifact.times, et_raw_motion_artifact._data[1]*1e3, label='E-tattoo', color="paleturquoise", lw=1)
    ax.legend()
    ax.set_ylim([-2, 2])
    ax.set_xlim([0, 40])
    ax.set_xlabel('Time (s)')
    fig.set_size_inches(10, 2)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300 
    mpl.rcParams['font.sans-serif'] = "Myriad Pro"
    mpl.rcParams['font.family'] = "sans-serif"
    filename = "motion_artifact_raw.svg"
    plt.savefig(filename, format='svg')
    plt.show()

    debug
            # custom_mapping = {'up': 1, 'down': 2, 'left': 3, 'right': 4, 'weak': 5, 'strong': 6}
            # events, event_dict = mne.events_from_annotations(raw, event_id=custom_mapping)

            # montage = mne.channels.make_standard_montage('standard_1020', head_size='auto')
            # sphere = (0, 0.02, 0, 0.1)
            # raw.set_montage(montage)

            # # bandpass filtering
            # filters = preprocessing.Filtering(raw=raw, l_freq=1, h_freq=30)
            # raw = filters.external_artifact_rejection(resample=False, notch=False)

            # ica = preprocessing.Indepndent_Component_Analysis(raw, n_components=raw.info['nchan']-2)
            # ica.perfrom_ICA()

            # epochs = mne.Epochs(raw=raw, events=events, event_id=event_dict, tmin=-2, tmax=5.0, preload=True, picks=['eeg', 'eog'])
            # # bv_epoch_eye_closed = epochs['ec']
            # # bv_epoch_eye_open = epochs['eo']
            # scaling = dict(mag=1e-12, grad=4e-11, eeg=50e-6, eog=150e-6, ecg=5e-4,emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,resp=1, chpi=1e-4, whitened=1e2)
            # fig = epochs[2, 3].plot(events=events, event_id=event_dict,picks=['eeg', 'eog'], scalings=scaling, block=False)
            # fig = raw.plot(block = True)
            # fig.set_size_inches(10, 4)
            # plt.rcParams['figure.dpi'] = 300
            # plt.rcParams['savefig.dpi'] = 300 
            # mpl.rcParams['font.sans-serif'] = "Arial"
            # mpl.rcParams['font.family'] = "sans-serif"
            # filename = "eceo_raw.svg"
            # plt.savefig(filename, format='svg')

if __name__ == '__main__':
    main()
    
    