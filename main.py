import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from setup import Setup as setup
import preprocessing

def main():
    ############### IMPORT DATA & SIGNAL PROCESSING ###############
    # list of raw data files in local data folder
    subject_ind = '1'
    data_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S'+subject_ind)
    raw_data_list = os.listdir(data_folder_path)

    # get path of EEG montage
    montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/passive electrodes/BrainCap','BrainCap 64 Channel','BC-64.bvef')
    # initilize epoch and event lists
    bv_epochs_list = []
    et_epochs_list = []
    bv_theta_epochs_list = []
    bv_alpha_epochs_list = []
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
                n_ch = [18, 6]
                ch_name = [['Fz', 'F3', 'F7', 'C3', 'T7', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P8', 'P4', 'T8', 'C4', 'Cz', 'F8', 'F4'], ['AF8','Fp2','Fp1','AF7','hEOG','vEOG']]
                ch_type = [['eeg']*18+['eog']*0, ['eeg']*4+['eog']*2]
                daq_source = {'source': source,'n_ch': [18, 6],'ch_name': ch_name,'ch_type': ch_type}
                # read .xdf files
                raw_path = os.path.join(data_folder_path, file_name)
                # create setup and raws from different systems
                raw_setup = setup(data_path=raw_path, data_type='xdf',stream_source=daq_source)
                raw_dict = raw_setup.raw_dict

                # preprocess raws
                for key in raw_dict.keys():
                    raw = raw_dict[key]
                    # bandpass filtering
                    filters = preprocessing.Filtering(raw=raw, l_freq=1, h_freq=30)
                    raw = filters.external_artifact_rejection(resample=False, notch=False)
                    # interactively annotate bad signal
                    interactive_annot = raw_setup.annotate_interactively(raw=raw)
                    print(interactive_annot)
                    # save preprocessed raw as .fif
                    raw_name = key + '_' + file_name.replace('.xdf','.fif')
                    raw.save(os.path.join(data_folder_path, raw_name), overwrite=True)
        if file_name.endswith('ECEO.fif'):
            raw_path = os.path.join(data_folder_path, file_name)
            raw = mne.io.read_raw_fif(raw_path)

            # montage = mne.channels.read_custom_montage(montage_path)
            if file_name.startswith('BrainVision'):
                montage = mne.channels.make_standard_montage('easycap-M1', head_size='auto')
                sphere = None
                montage.plot(show=False)
            else:
                montage = mne.channels.make_standard_montage('standard_1020', head_size='auto')
                sphere = (0, 0.02, 0, 0.1)
                montage.plot(show=False, sphere=sphere)
            plt.show(block=True)
            raw.set_montage(montage)

            # bandpass filtering
            filters = preprocessing.Filtering(raw=raw, l_freq=1, h_freq=30)
            raw = filters.external_artifact_rejection(resample=False, notch=False)

            raw.load_data()
            custom_mapping = {'ec': 1, 'eo': 2}
            events, event_dict = mne.events_from_annotations(raw, event_id=custom_mapping)

            # ica = preprocessing.Indepndent_Component_Analysis(raw, n_components=raw.info['nchan']-2)
            # eog = ica.create_physiological_evoked()
            # ica.perfrom_ICA()

            epochs = mne.Epochs(raw=raw, events=events, event_id=event_dict, tmin=-10.0, tmax=10.0, preload=True, picks='eeg')
            bv_epoch_eye_closed = epochs['ec']
            bv_epoch_eye_open = epochs['eo']

            #### Compute Time-Freq
            freqs = np.logspace(*np.log10([1, 30]), num=160)
            n_cycles = freqs / 2.  # different number of cycle per frequency
            # cnorm = matplotlib.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=3)
            # cmap=cnorm
            # baseline=(-5.0, -1.0)

            # BV Closed
            power, itc = mne.time_frequency.tfr_morlet(bv_epoch_eye_closed, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
            power.plot(baseline=(-5.0, -0.1), combine='mean', mode='logratio', title='Closing Epoch Average Power')
            # print(power.data)
            # print(np.nanmax(power.data))
            fig, axis = plt.subplots(1, 2, figsize=(7, 4))
            cnorm = matplotlib.colors.TwoSlopeNorm(vmin=-0.275, vcenter=0, vmax=0.275)
            power.plot_topomap(ch_type='eeg', tmin=-5, tmax=-0.5, fmin=8, fmax=13, baseline=(-5.0, -0.1), mode='logratio', axes=axis[0],sphere=sphere, show=False)
            power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=5, fmin=8, fmax=13, baseline=(-5.0, -0.1), mode='logratio', axes=axis[1], sphere=sphere, show=False)
            mne.viz.tight_layout()
            plt.show(block=True)

            # BV Open
            power, itc = mne.time_frequency.tfr_morlet(bv_epoch_eye_open, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
            power.plot(baseline=(0.1, 5), combine='mean', mode='logratio', title='Opening Epoch Average Power')
            # power.plot([1], baseline=(-5.0, 0), mode='logratio', title=power.ch_names[1])
            fig, axis = plt.subplots(1, 2, figsize=(7, 4))
            cnorm = matplotlib.colors.TwoSlopeNorm(vmin=-0.225, vcenter=-0.1, vmax=0.025)
            power.plot_topomap(ch_type='eeg', tmin=-5, tmax=-0.5, fmin=8, fmax=13, baseline=(0.1, 5), mode='logratio', axes=axis[0], sphere=sphere,show=False)
            power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=5, fmin=8, fmax=13, baseline=(0.1, 5), mode='logratio', axes=axis[1], sphere=sphere,show=False)
            mne.viz.tight_layout()
            plt.show(block=True)



if __name__ == '__main__':
    main()
    
    