import os
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
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
                    filters = preprocessing.Filtering(raw=raw, l_freq=1, h_freq=50, picks=['eeg','eog'])
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
            ica = preprocessing.Indepndent_Component_Analysis(raw, n_components=raw.info['nchan']-2, seed=97)
            reconst_raw = ica.perfrom_ICA()

            tmin, tmax = -0.2, 1.8
            epochs = mne.Epochs(raw=reconst_raw, events=events, event_id=event_dict,event_repeated='drop', tmin=tmin-0.3, tmax=tmax+0.3, preload=True, picks='eeg', baseline=None)
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
    all_epochs = all_epochs['0', '1', '2', '3']
    # epoch analysis
    freqs = np.arange(1, 25)  # frequencies from 1-25Hz
    vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
    baseline = (-0.2, 0)  # baseline interval (in s)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

    kwargs = dict(
        n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
    )  # for cluster test

    tfr = mne.time_frequency.tfr_multitaper(
        epochs,
        freqs=freqs,
        n_cycles=freqs,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=2,
    )
    tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
    event_ids = {'0': 0, '1': 1, '2': 2, '3': 3}

    for event in event_ids:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(
            1, 5, figsize=(16, 4), gridspec_kw={"width_ratios": [10, 10, 10, 10, 1]}
        )
        for ch, ax in enumerate(axes[:-1]):  # for each channel
            # positive clusters
            _, c1, p1, _ = mne.stats.permutation_cluster_1samp_test(tfr_ev.data[:, ch], tail=1, **kwargs)
            # negative clusters
            _, c2, p2, _ = mne.stats.permutation_cluster_1samp_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

            # note that we keep clusters with p <= 0.05 from the combined clusters
            # of two independent tests; in this example, we do not correct for
            # these two comparisons
            c = np.stack(c1 + c2, axis=2)  # combined clusters
            p = np.concatenate((p1, p2))  # combined p-values
            mask = c[..., p <= 0.05].any(axis=-1)

            # plot TFR (ERDS map with masking)
            tfr_ev.average().plot(
                [ch],
                cmap="RdBu",
                cnorm=cnorm,
                axes=ax,
                colorbar=False,
                show=False,
                mask=mask,
                mask_style="mask",
            )

            ax.set_title(epochs.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")
        fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
        fig.suptitle(f"ERDS ({event}-Back)")
        plt.show()

    df = tfr.to_data_frame(time_format=None)
    df.head()

    df = tfr.to_data_frame(time_format=None, long_format=True)

    # Map to frequency bands:
    freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 35, "gamma": 140}
    df["band"] = pd.cut(
        df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:]
    )

    # Filter to retain only relevant frequency bands:
    freq_bands_of_interest = ["delta", "theta", "alpha", "beta"]
    df = df[df.band.isin(freq_bands_of_interest)]
    df["band"] = df["band"].cat.remove_unused_categories()

    # Order channels for plotting:
    df["channel"] = df["channel"].cat.reorder_categories(("AF8", "Fp2", "Fp1", "AF7"), ordered=True)

    g = sns.FacetGrid(df, row="band", col="channel", margin_titles=True)
    g.map(sns.lineplot, "time", "value", "condition", n_boot=10)
    axline_kw = dict(color="black", linestyle="dashed", linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)
    g.set(ylim=(-1.5, 1.5))
    g.set_axis_labels("Time (s)", "ERDS")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(ncol=2, loc="lower center")
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)

    df_mean = (
        df.query("time > 1")
        .groupby(["condition", "epoch", "band", "channel"], observed=False)[["value"]]
        .mean()
        .reset_index()
    )

    g = sns.FacetGrid(
        df_mean, col="condition", col_order=["0", "1", "2", "3"], margin_titles=True
    )
    g = g.map(
        sns.violinplot,
        "channel",
        "value",
        "band",
        cut=0,
        palette="deep",
        order=["AF8", "Fp2", "Fp1", "AF7"],
        hue_order=freq_bands_of_interest,
        linewidth=0.5,
    ).add_legend(ncol=4, loc="lower center")

    g.map(plt.axhline, **axline_kw)
    g.set_axis_labels("", "ERDS")
    g.set_titles(col_template="{col_name}-Back", row_template="{row_name}")
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

    plt.show()

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
