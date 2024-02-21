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
        
        if file_name.endswith('tattoo_ECEO.fif'):
            raw_path = os.path.join(data_folder_path, file_name)
            raw = mne.io.read_raw_fif(raw_path)
            raw.load_data()
            custom_mapping = {'ec': 1, 'eo': 2}
            events, event_dict = mne.events_from_annotations(raw, event_id=custom_mapping)
            # montage = mne.channels.read_custom_montage(montage_path)
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
            filters = preprocessing.Filtering(raw=raw, l_freq=1, h_freq=30)
            raw = filters.external_artifact_rejection(resample=False, notch=False)
            # ica = preprocessing.Indepndent_Component_Analysis(raw, n_components=raw.info['nchan']-2)
            # ica.perfrom_ICA()

            epochs = mne.Epochs(raw=raw, events=events, event_id=event_dict, tmin=-5.0, tmax=5.0, preload=True, picks=['eeg', 'eog'])
            bv_epoch_eye_closed = epochs['ec']
            bv_epoch_eye_open = epochs['eo']
            scaling = dict(mag=1e-12, grad=4e-11, eeg=50e-6, eog=150e-6, ecg=5e-4,emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,resp=1, chpi=1e-4, whitened=1e2)
            fig = epochs[2, 3].plot(events=events, event_id=event_dict,picks=['eeg', 'eog'], scalings=scaling, block=False)
            fig.set_size_inches(10, 4)
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300 
            mpl.rcParams['font.sans-serif'] = "Arial"
            mpl.rcParams['font.family'] = "sans-serif"
            filename = "eceo_raw.svg"
            plt.savefig(filename, format='svg')
            # plt.show(block=True)

            # fig = bv_epoch_eye_closed.plot(events=events, event_id=event_dict,picks=['eeg', 'eog'], scalings=scaling, block=False)
            # fig.set_size_inches(10, 4)
            # plt.rcParams['figure.dpi'] = 300
            # plt.rcParams['savefig.dpi'] = 300 
            # mpl.rcParams['font.sans-serif'] = "Arial"
            # mpl.rcParams['font.family'] = "sans-serif"
            # filename = "ec_raw.svg"
            # plt.savefig(filename, format='svg')
            # plt.show(block=True)
            # fig = bv_epoch_eye_open.plot(events=events, event_id=event_dict,picks=['eeg', 'eog'], scalings=scaling, block=False)
            # fig.set_size_inches(10, 4)
            # plt.rcParams['figure.dpi'] = 300
            # plt.rcParams['savefig.dpi'] = 300 
            # mpl.rcParams['font.sans-serif'] = "Arial"
            # mpl.rcParams['font.family'] = "sans-serif"
            # filename = "eo_raw.svg"
            # plt.savefig(filename, format='svg')
            # plt.show(block=True)

            bv_alpha = preprocessing.Filtering(raw, 6, 13)
            bv_alpha_raw = bv_alpha.bandpass(picks='eeg')
            alpha_epochs = mne.Epochs(bv_alpha_raw, events=events, event_id=event_dict, tmin=-5.0, tmax=5.0, preload=True, picks=['eeg', 'eog'])
            scaling = dict(mag=1e-12, grad=4e-11, eeg=25e-6, eog=150e-6, ecg=5e-4,emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,resp=1, chpi=1e-4, whitened=1e2)
            fig = alpha_epochs[2,3].plot(events=events, event_id=event_dict,picks=['eeg', 'eog'], scalings=scaling, block=False)
            fig.set_size_inches(10, 4)
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300 
            mpl.rcParams['font.sans-serif'] = "Arial"
            mpl.rcParams['font.family'] = "sans-serif"
            filename = "eceo_alpha.svg"
            plt.savefig(filename, format='svg')
            # plt.show(block=True)
            
            # fig = alpha_epochs['ec'].plot(events=events, event_id=event_dict,picks=['eeg', 'eog'], scalings=scaling, block=False)
            # # fig = alpha_epochs['ec'].average().plot(show=False)
            # fig.set_size_inches(10, 4)
            # plt.rcParams['figure.dpi'] = 300
            # plt.rcParams['savefig.dpi'] = 300 
            # mpl.rcParams['font.sans-serif'] = "Arial"
            # mpl.rcParams['font.family'] = "sans-serif"
            # filename = "ec_alpha.svg"
            # plt.savefig(filename, format='svg')
            # plt.show(block=True)
            # fig = alpha_epochs['eo'].plot(events=events, event_id=event_dict,picks=['eeg', 'eog'], scalings=scaling, block=False)
            # # fig = alpha_epochs['eo'].average().plot(show=False)
            # fig.set_size_inches(10, 4)
            # plt.rcParams['figure.dpi'] = 300
            # plt.rcParams['savefig.dpi'] = 300 
            # mpl.rcParams['font.sans-serif'] = "Arial"
            # mpl.rcParams['font.family'] = "sans-serif"
            # filename = "eo_alpha.svg"
            # plt.savefig(filename, format='svg')
            # plt.show(block=True)

            #### Compute Time-Freq
            freqs = np.logspace(*np.log10([1, 30]), num=160)
            n_cycles = freqs / 2.  # different number of cycle per frequency
            # cnorm = matplotlib.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=3)
            # cmap=cnorm
            # baseline=(-5.0, -1.0)

            # BV Closed
            power, itc = mne.time_frequency.tfr_morlet(bv_epoch_eye_closed, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
            ''' 
            #TODO figure for time-freq plot of eye oc
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.
            '''
            figs = power.plot(baseline=(-5.0, -0.1), combine='mean', mode='logratio', title='Closing Epoch Average Power')
            fig = figs[0]
            fig.set_size_inches(8, 4)
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300 
            mpl.rcParams['font.sans-serif'] = "Arial"
            mpl.rcParams['font.family'] = "sans-serif"
            filename = "ec_power.png"
            # plt.show(block=True)

            # print(power.data)
            # print(np.nanmax(power.data))
            fig, axis = plt.subplots(1, 2, figsize=(8, 4))
            cnorm = mpl.colors.TwoSlopeNorm(vmin=-0.275, vcenter=0, vmax=0.275)
            power.plot_topomap(ch_type='eeg', tmin=-5, tmax=-0.5, fmin=8, fmax=13, baseline=(-5.0, -0.1), mode='logratio', axes=axis[0],sphere=sphere, show=False)
            power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=5, fmin=8, fmax=13, baseline=(-5.0, -0.1), mode='logratio', axes=axis[1], sphere=sphere, show=False)
            mne.viz.tight_layout()
            # plt.show(block=True)

            # BV Open
            power, itc = mne.time_frequency.tfr_morlet(bv_epoch_eye_open, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
            figs = power.plot(baseline=(0.1, 5), combine='mean', mode='logratio', title='Opening Epoch Average Power')
            fig = figs[0]
            fig.set_size_inches(8, 4)
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300 
            mpl.rcParams['font.sans-serif'] = "Arial"
            mpl.rcParams['font.family'] = "sans-serif"
            filename = "eo_power.png"
            # power.plot([1], baseline=(-5.0, 0), mode='logratio', title=power.ch_names[1])
            fig, axis = plt.subplots(1, 2, figsize=(8, 4))
            cnorm = mpl.colors.TwoSlopeNorm(vmin=-0.225, vcenter=-0.1, vmax=0.025)
            power.plot_topomap(ch_type='eeg', tmin=-5, tmax=-0.5, fmin=8, fmax=13, baseline=(0.1, 5), mode='logratio', axes=axis[0], sphere=sphere,show=False)
            power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=5, fmin=8, fmax=13, baseline=(0.1, 5), mode='logratio', axes=axis[1], sphere=sphere,show=False)
            mne.viz.tight_layout()
            plt.show(block=True)
'''
        if file_name.endswith('_eeg.fif'):
            raw_path = os.path.join(data_folder_path, file_name)
            raw = mne.io.read_raw_fif(raw_path)
            raw.load_data()
            # raw.plot(scalings=dict(eeg=20e-6), duration=5, block=True)
            custom_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, 'fixation': 10, 'response_alpha': 100, 'response_pos': 101}
            events, event_dict = mne.events_from_annotations(raw, event_id=custom_mapping)
            # montage = mne.channels.read_custom_montage(montage_path)
            if file_name.startswith('BrainVision'):
                montage = mne.channels.make_standard_montage('easycap-M1', head_size='auto')
                sphere = None
            else:
                montage = mne.channels.make_standard_montage('standard_1020', head_size='auto')
                sphere = (0, 0.02, 0, 0.1)
            plt.show(block=True)
            raw.set_montage(montage)

            # bandpass filtering
            filters = preprocessing.Filtering(raw=raw, l_freq=1, h_freq=30)
            raw = filters.external_artifact_rejection(resample=False, notch=False)

            # ica = preprocessing.Indepndent_Component_Analysis(raw, n_components=raw.info['nchan']-2)
            # eog = ica.create_physiological_evoked(eog_ch='vEOG')
            # ica.perfrom_ICA()

            epochs = mne.Epochs(raw=raw, events=events, event_id=event_dict,event_repeated='drop', tmin=-0.5, tmax=2.0, preload=True, picks='eeg')
            bv_0back = epochs['0']
            bv_1back = epochs['1']
            bv_2back = epochs['2']
            bv_3back = epochs['3']
            for epk in [bv_0back, bv_1back, bv_2back, bv_3back]:
                # global field power and spatial plot
                epk.plot_image(picks='eeg',combine='mean')
                evk = epk.average()
                evk.plot(gfp=True, spatial_colors=True)
'''

if __name__ == '__main__':
    main()
    
    