import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from setup import Setup as setup
import preprocessing

new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
mpl.rcParams['font.sans-serif'] = "Myriad Pro"
mpl.rcParams['font.family'] = "sans-serif"
plt.rcParams.update({'font.size': 22})

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
        '''
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
        '''
        if file_name.endswith('ECEO_raw.fif'):
            raw_path = os.path.join(data_folder_path, file_name)
            raw = mne.io.read_raw_fif(raw_path)
            raw.load_data()

            # montage = mne.channels.read_custom_montage(montage_path)
            if file_name.startswith('BrainVision'):
                montage = mne.channels.make_standard_montage('easycap-M1', head_size='auto')
                sphere = None
                # montage.plot(show=False)
                dev = 'BV_'
            else:
                montage = mne.channels.make_standard_montage('standard_1020', head_size='auto')
                sphere = (0, 0.02, 0, 0.1)
                # montage.plot(show=False, sphere=sphere)
                dev = 'ET_'
            # plt.show(block=True)
            raw.set_montage(montage)

            ############### Signal Processing ###############
            # bandpass filtering
            filters = preprocessing.Filtering(raw=raw, l_freq=1, h_freq=30, picks='eeg')
            raw_clean = filters.external_artifact_rejection(resample=False, notch=False)
            # ica = preprocessing.Indepndent_Component_Analysis(raw, n_components=raw.info['nchan']-2, seed=60)
            # raw = ica.perfrom_ICA()
            # mne EOGRegression requires eeg refference
            # raw_car = raw.copy().set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg')
            # model_plain = mne.preprocessing.EOGRegression(picks='eeg', picks_artifact='eog').fit(raw_car)
            # # fig = model_plain.plot(vlim=(None, None))  # regression coefficients as topomap
            # # plt.show()
            # raw_clean = model_plain.apply(raw_car)
            if file_name.startswith('ForeheadE-tattoo'):
                filters = preprocessing.Filtering(raw=raw_clean, l_freq=1, h_freq=10, picks='eog')
                raw_clean = filters.external_artifact_rejection(resample=False, notch=False)
            raw_clean.annotations.onset[0:2] = np.array([3.1,30.0])
            # raw_clean.annotations.onset[2:4] = np.array([60.7,85.52]) # S5
            # raw_clean = raw.copy()
            # resmaple raw
            new_sfreq = 100
            raw_clean = raw_clean.resample(sfreq=new_sfreq)
            raw_clean.crop(0,50.0) #s5
            # raw_clean.crop(50.7,100.7) #s5
            if file_name.startswith('ForeheadE-tattoo'):
                scaling = dict(eeg=20e-6, eog=250e-6)
            else:
                scaling = dict(eeg=35e-6, eog=250e-6)
            fig = raw_clean.pick('eeg').plot(duration= 50.0, show_scrollbars=False, scalings=scaling, block=False, title='raw_clean')
            fig.set_size_inches(10, 4)
            mpl.rcParams['font.sans-serif'] = "Myriad Pro"
            mpl.rcParams['font.family'] = "sans-serif"
            filename = dev+"eceo_raw.svg"
            plt.show(block=False)
            plt.savefig(filename, format='svg')
            raw_clean.load_data()

            custom_mapping = {'ec': 1, 'eo': 2}
            events, event_dict = mne.events_from_annotations(raw_clean, event_id=custom_mapping)
            # reject = dict(eeg=500e-6, eog=1000e-6)  
            epochs = mne.Epochs(raw=raw_clean, events=events, event_id=event_dict, tmin=-3.1, tmax=46.9, preload=True, picks='eeg')
            # epochs = mne.Epochs(raw=raw_clean, events=events, event_id=event_dict, tmin=-10.0, tmax=40.0, preload=True, picks='eeg')

            # fig = epochs.plot(events=events, event_id=event_dict,picks=['eeg', 'eog'], scalings=scaling, block=False)
            # fig.set_size_inches(10, 4)
            # plt.rcParams['figure.dpi'] = 300
            # plt.rcParams['savefig.dpi'] = 300 
            # mpl.rcParams['font.sans-serif'] = "Myriad Pro"
            # mpl.rcParams['font.family'] = "sans-serif"
            # filename = "eceo_raw.svg"
            # plt.savefig(filename, format='svg')
            # plt.show(block=True)
            freqs = np.arange(5.0, 20.1, 0.1)
            n_cycles = freqs / 2.0
            timefreq_method = 'morlet'
            if timefreq_method == "multitaper":
                power = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False, time_bandwidth=4.0)
            elif timefreq_method == "morlet":
                power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False)
            elif timefreq_method == "stockwell":
                power = mne.time_frequency.tfr_stockwell(epochs, fmin=freqs[0], fmax=freqs[-1], width=1, return_itc=False)
                # power_ec = mne.time_frequency.tfr_stockwell(epochs["closed"], fmin=freqs[0], fmax=freqs[-1], width=1, return_itc=False)
                # power_eo = mne.time_frequency.tfr_stockwell(epochs["open"], fmin=freqs[0], fmax=freqs[-1], width=1, return_itc=False)
            else:
                raise ValueError("Invalid timefreq method")

            if timefreq_method == "multitaper" or timefreq_method == "morlet":
                timefreq_per_epoch = [power.data[epoc, 0:4, :, :] * (10**12) for epoc in range(power.data.shape[0])]
                timefreq = np.concatenate(timefreq_per_epoch, axis=2)
                # for ch in range(timefreq.shape[0]):
                #     plt.figure()
                #     sns.heatmap(timefreq[ch, :, :], cmap="crest", robust=True)
                #     plt.yticks(range(timefreq.shape[1]), freqs)
                # plt.show(block=False)

                timefreq_averaged_ch = np.mean(timefreq, axis=0)
                timefreq_averaged_ch = np.flip(timefreq_averaged_ch, 0)
                baseline_power = np.mean(timefreq_averaged_ch[:,4000:], axis=1)
                # baseline_power = np.mean(timefreq_averaged_ch[:,4000:])
                timefreq_db_averaged_ch = 10*np.log10(timefreq_averaged_ch/baseline_power[:,None])
                plt.figure(figsize=(10,4))
                ax = sns.heatmap(timefreq_averaged_ch, cmap='crest', robust=True, cbar_kws={'label': 'PSD ($\mathrm{\u03bcV}^2/\mathrm{Hz}$)'})
                plt.yticks(np.linspace(start=0, stop=timefreq_averaged_ch.shape[0], num=7), np.flip(np.round(np.linspace(start=power.freqs[0], stop=power.freqs[-1], num=7), decimals=1), 0))
                xtick_locs = np.linspace(start=0, stop=int(np.floor(timefreq_averaged_ch.shape[1])), num=6)
                xtick_labels = np.round(np.linspace(start=0, stop=50, num=6))
                plt.xticks(xtick_locs, xtick_labels.astype(int), rotation='horizontal')
                plt.ylabel("Frequency (Hz)")
                plt.xlabel("Time (s)")
                plt.rcParams['figure.dpi'] = 300
                plt.rcParams['savefig.dpi'] = 300 
                mpl.rcParams['font.sans-serif'] = "Myriad Pro"
                mpl.rcParams['font.family'] = "sans-serif"
                # use matplotlib.colorbar.Colorbar object
                cbar = ax.collections[0].colorbar
                # here set the labelsize by 20
                cbar.ax.tick_params(labelsize=18)
                filename = dev+"tfr_raw.png"
                plt.savefig(filename, format='png')

                plt.figure(figsize=(10,4))
                ax = sns.heatmap(timefreq_db_averaged_ch, cmap=sns.color_palette("coolwarm", as_cmap=True), robust=True, cbar_kws={'label': 'Power (dB)'})
                plt.yticks(np.linspace(start=0, stop=timefreq_db_averaged_ch.shape[0], num=7), np.flip(np.round(np.linspace(start=power.freqs[0], stop=power.freqs[-1], num=7), decimals=1), 0))
                xtick_locs = np.linspace(start=0, stop=int(np.floor(timefreq_db_averaged_ch.shape[1])), num=6)
                xtick_labels = np.round(np.linspace(start=0, stop=50, num=6))
                plt.xticks(xtick_locs, xtick_labels.astype(int), rotation='horizontal')
                plt.ylabel("Frequency (Hz)")
                plt.xlabel("Time (s)")
                plt.rcParams['figure.dpi'] = 300
                plt.rcParams['savefig.dpi'] = 300 
                mpl.rcParams['font.sans-serif'] = "Myriad Pro"
                mpl.rcParams['font.family'] = "sans-serif"
                # use matplotlib.colorbar.Colorbar object
                cbar = ax.collections[0].colorbar
                # here set the labelsize by 20
                cbar.ax.tick_params(labelsize=18)
                filename = dev+"tfr_db_raw.png"
                plt.savefig(filename, format='png')
                plt.show(block=False)

            elif timefreq_method == "stockwell":  # stockwell doesn't offer TFR per epoch but only averaged TFR 
                timefreq_per_epoch = [power.data[0:4, :, :] * (10**12)]
                timefreq = np.concatenate(timefreq_per_epoch, axis=2)
                # for ch in range(timefreq.shape[0]):
                #     plt.figure()
                #     sns.heatmap(timefreq[ch, :, :], cmap="crest", robust=True, cbar_kws={'label': 'Power spectral density ($\mathrm{\u03bcV}^2/\mathrm{Hz}$)'})
                #     # plt.yticks(range(timefreq.shape[1]), power_ec.freqs)
                #     plt.yticks(np.linspace(start=0, stop=timefreq.shape[1], num=10), np.round(np.linspace(start=power_ec.freqs[0], stop=power_ec.freqs[timefreq.shape[1]-1], num=10)))
                #     ec_xtick_locs = np.linspace(start=0, stop=int(np.floor(timefreq.shape[2]/2)), num=11)
                #     ec_xtick_labels = np.round(np.linspace(start=0, stop=tmax, num=11), decimals=1)
                #     eo_xtick_locs = np.linspace(start=int(np.ceil(timefreq.shape[2]/2)), stop=timefreq.shape[2], num=11)
                #     eo_xtick_labels = np.round(np.linspace(start=0, stop=tmax, num=11), decimals=1)
                #     plt.xticks(np.concatenate((ec_xtick_locs, eo_xtick_locs)), np.concatenate((ec_xtick_labels, eo_xtick_labels)))
                # plt.show(block=False)

                timefreq_averaged_ch = np.mean(timefreq, axis=0)
                timefreq_averaged_ch = np.flip(timefreq_averaged_ch, 0)
                plt.figure(figsize=(10,4))
                ax = sns.heatmap(timefreq_averaged_ch, cmap="crest", robust=True, cbar_kws={'label': 'PSD ($\mathrm{\u03bcV}^2/\mathrm{Hz}$)'})
                plt.yticks(np.linspace(start=0, stop=timefreq_averaged_ch.shape[0], num=7), np.flip(np.round(np.linspace(start=power.freqs[0], stop=power.freqs[-1], num=7), decimals=1), 0))
                xtick_locs = np.linspace(start=0, stop=int(np.floor(timefreq_averaged_ch.shape[1])), num=6)
                xtick_labels = np.round(np.linspace(start=0, stop=50, num=6), decimals=1)
                plt.xticks(xtick_locs, xtick_labels, rotation='horizontal')
                plt.ylabel("Frequency (Hz)")
                plt.xlabel("Time (s)")
                plt.rcParams['figure.dpi'] = 300
                plt.rcParams['savefig.dpi'] = 300 
                mpl.rcParams['font.sans-serif'] = "Myriad Pro"
                mpl.rcParams['font.family'] = "sans-serif"
                # use matplotlib.colorbar.Colorbar object
                cbar = ax.collections[0].colorbar
                # here set the labelsize by 20
                cbar.ax.tick_params(labelsize=18)
                filename = dev+"tfr_raw.png"
                plt.show(block=False)
                # plt.savefig("eceo.png", dpi=300, bbox_inches="tight")
            ''''''
            # fig = bv_epoch_eye_closed.plot(events=events, event_id=event_dict,picks=['eeg', 'eog'], scalings=scaling, block=False)
            # fig.set_size_inches(10, 4)
            # plt.rcParams['figure.dpi'] = 300
            # plt.rcParams['savefig.dpi'] = 300 
            # mpl.rcParams['font.sans-serif'] = "Myriad Pro"
            # mpl.rcParams['font.family'] = "sans-serif"
            # filename = "ec_raw.svg"
            # plt.savefig(filename, format='svg')
            # plt.show(block=True)
            # fig = bv_epoch_eye_open.plot(events=events, event_id=event_dict,picks=['eeg', 'eog'], scalings=scaling, block=False)
            # fig.set_size_inches(10, 4)
            # plt.rcParams['figure.dpi'] = 300
            # plt.rcParams['savefig.dpi'] = 300 
            # mpl.rcParams['font.sans-serif'] = "Myriad Pro"
            # mpl.rcParams['font.family'] = "sans-serif"
            # filename = "eo_raw.svg"
            # plt.savefig(filename, format='svg')
            # plt.show(block=True)

            alpha_filter = preprocessing.Filtering(raw_clean, 8, 12)
            raw_alpha = alpha_filter.bandpass(picks='eeg')
            fig = raw_alpha.plot(duration= 50.0, show_scrollbars=False, scalings=scaling, block=False, title='alpha_raw')
            # alpha_epochs = mne.Epochs(raw_alpha, events=events, event_id=event_dict, tmin=-5.0, tmax=5.0, preload=True, picks=['eeg', 'eog'])
            # fig = alpha_epochs.plot(events=events, event_id=event_dict,picks=['eeg', 'eog'], scalings=scaling, block=False)
            fig.set_size_inches(10, 4)
            mpl.rcParams['font.sans-serif'] = "Myriad Pro"
            mpl.rcParams['font.family'] = "sans-serif"
            filename = dev+"eceo_alpha.svg"
            plt.savefig(filename, format='svg')
            plt.show(block=True)

            # fig = alpha_epochs['ec'].plot(events=events, event_id=event_dict,picks=['eeg', 'eog'], scalings=scaling, block=False)
            # # fig = alpha_epochs['ec'].average().plot(show=False)
            # fig.set_size_inches(10, 4)
            # plt.rcParams['figure.dpi'] = 300
            # plt.rcParams['savefig.dpi'] = 300 
            # mpl.rcParams['font.sans-serif'] = "Myriad Pro"
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
            ''' 
            #### Compute Time-Freq
            freqs = np.logspace(*np.log10([1, 30]), num=160)
            n_cycles = freqs / 2.  # different number of cycle per frequency
            # cnorm = matplotlib.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=3)
            # cmap=cnorm
            # baseline=(-5.0, -1.0)

            # BV Closed
            power, itc = mne.time_frequency.tfr_morlet(bv_epoch_eye_closed, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
            
            #TODO figure for time-freq plot of eye oc
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.
            
            figs = power.plot(baseline=(-5.0, -0.1), combine='mean', mode='logratio', title='Closing Epoch Average Power')
            fig = figs[0]
            fig.set_size_inches(8, 4)
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300 
            mpl.rcParams['font.sans-serif'] = "Myriad Pro"
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
            mpl.rcParams['font.sans-serif'] = "Myriad Pro"
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

if __name__ == '__main__':
    main()