import os
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from setup import Setup as setup
from setup import N_back_report as nback
import preprocessing

# graphic render params
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

############### IMPORT & CONVERT DATA ###############
def prep_data():
    # get list of raw data file names in local data folder
    subject_idx = '1'
    data_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S'+subject_idx)
    raw_data_list = os.listdir(data_folder_path)

    for file_name in raw_data_list:
        if file_name.endswith('.xdf'):
            preprocessed = False
            for name in raw_data_list:
                if file_name.replace('.xdf','_raw.fif') in name:
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
                    # filters = preprocessing.Filtering(raw=raw, l_freq=1, h_freq=50, picks=['eeg','eog'])
                    # raw = filters.external_artifact_rejection(resample=False, notch=False)
                    ''' comment out ' to process raw
                    filters = preprocessing.Filtering(raw=raw, l_freq=0.01, h_freq=50, picks='eeg')
                    raw = filters.external_artifact_rejection(resample=False, notch=False)
                    if 'eog' in ch_type:
                        filters = preprocessing.Filtering(raw=raw, l_freq=0.01, h_freq=10, picks='eog')
                        raw = filters.external_artifact_rejection(resample=False, notch=False)

                    resmaple raw
                    raw = filters.resample(new_sfreq=50)
                    '''
                    
                    # interactively annotate bad signal
                    interactive_annot = raw_setup.annotate_interactively(raw=raw)
                    print(interactive_annot)

                    if file_name.endswith('P002_ses-S003_task-Default_run-003_eeg.xdf'):
                        t_start = float(input('t_start: '))
                        t_end = float(input('t_end: '))
                        raw = raw.crop(tmin=t_start, tmax=t_end)
                    elif file_name.endswith('P005_ses-S001_task-Default_run-003_eeg.xdf'):
                        raw.plot(block=True)
                        continue
                        t_start = float(input('t_start: '))
                        t_end = float(input('t_end: '))
                        raw = raw.crop(tmin=t_start, tmax=t_end)
                    # save preprocessed raw as .fif
                    raw_name = key + '_' + file_name.replace('.xdf','_raw.fif')
                    raw.save(os.path.join(data_folder_path, raw_name), overwrite=True)

def eye_oc():
    pass

############### SIGNAL PROCESSING & N-BACK ANALYSIS ###############
def n_back_analysis():
    # list of raw data files in local data folder
    subject_idx = '1'
    data_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S'+subject_idx)
    raw_data_list = os.listdir(data_folder_path)

    # get path of EEG montage
    montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/passive electrodes/BrainCap','BrainCap 64 Channel','BC-64.bvef')
    # initilize epoch and event lists
    run_list = []
    epochs_list = []
    events_list = []
    report_list = []
    reaction_time_list = []

    ############### Import Data ###############
    for file_name in raw_data_list:
        if file_name.endswith('eeg_raw.fif'):
            run_idx = int(file_name.split("run-")[1].split("_eeg_raw.fif")[0])
            run_list.append(run_idx)
            raw_path = os.path.join(data_folder_path, file_name)
            raw = mne.io.read_raw_fif(raw_path)
            raw.load_data()
            # Show imported raw data
            # raw.plot(block=True)
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
            
            ############### Signal Processing ###############
            # bandpass filtering
            filters = preprocessing.Filtering(raw=raw, l_freq=3, h_freq=50)
            raw = filters.external_artifact_rejection(resample=False, notch=False)
            # raw.plot(block=False, title='raw')

            '''ica = preprocessing.Indepndent_Component_Analysis(raw, n_components=raw.info['nchan']-2, seed=90)
            reconst_raw = ica.perfrom_ICA()
            reconst_raw.plot(block=True)
            debug'''

            # perform regression to correct EOG artifacts
            '''# EOGRegression without refference EEG
            raw_eeg = raw[:4, :][0]
            raw_eog = raw[4:, :][0]
            b = np.linalg.inv(raw_eog @ raw_eog.T) @ raw_eog @ raw_eeg.T
            # b = np.linalg.solve(raw_eog @ raw_eog.T, raw_eog @ raw_eeg.T)
            print(b.shape)
            print(b.T)
            eeg_corrected = (raw_eeg.T - raw_eog.T @ b).T
            raw_clean1 = raw.copy()
            raw_clean1._data[:4, :] = eeg_corrected
            raw_clean1.plot(block=False, title='raw_clean1')
            '''

            # mne EOGRegression requires eeg refference
            raw_car = raw.copy().set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg')
            model_plain = mne.preprocessing.EOGRegression(picks="eeg", picks_artifact="eog").fit(raw_car)
            print(model_plain.coef_.shape)
            print(model_plain.coef_)
            # fig = model_plain.plot(vlim=(None, None))  # regression coefficients as topomap
            # plt.show()

            raw_clean = model_plain.apply(raw_car)

            # Show the corrected data
            # raw_clean.plot(block=True, title='raw_clean')
            
            # resmaple raw
            new_sfreq = 200
            raw_clean = raw_clean.resample(sfreq=new_sfreq)
            # raw_clean = raw.resample(sfreq=new_sfreq)
            raw_clean.load_data()


            ############### Create Event & Import N-bsck Report Data ###############
            # remove overlapping events and get reaction time
            remv_idx = []
            reaction_time = []
            annot = raw_clean.annotations
            for i in range(len(annot.onset)-1):
                del_t = annot.onset[i+1] - annot.onset[i]
                # get overlapping events idx
                if del_t < 1/new_sfreq:
                    remv_idx.append(i)
                else:
                    # get reaction time
                    if annot.description[i].isdigit() and annot.description[i+1].startswith('response_'):
                        # remove too slow responses
                        '''
                        The fastest (simple) reaction time to a stimulus 
                        is about 100 milliseconds, and the time it takes 
                        for a sensory stimulus to become conscious is 
                        typically a few hundred milliseconds. 
                        This makes sense in the environment in which 
                        human beings evolved.
                        '''
                        if 2.0> del_t > 0.1:
                            reaction_time.append([del_t, annot.description[i], annot.description[i+1]])
            # remove overlapping events
            raw_clean.annotations.delete(remv_idx)
            reaction_time_list.append(reaction_time)

            custom_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, 'fixation': 10, 'response_alpha': 100, 'response_pos': 101}
            events, event_dict = mne.events_from_annotations(raw_clean, event_id=custom_mapping)
            events = mne.pick_events(events, exclude=[10,100,101])
            event_dict = {key: event_dict[key] for key in event_dict if key not in ['fixation', 'response_alpha','response_pos']}
            
            tmin, tmax = -0.2, 1.6 # tmin, tmax = -0.1, 1.5 (band-power)
            reject = dict(eeg=80e-6)      # unit: V (EEG channels)
            epochs = mne.Epochs(raw=raw_clean, events=events, event_id=event_dict, event_repeated='drop', tmin=tmin, tmax=tmax, preload=True, reject=reject, picks='eeg', baseline=None)
            baseline_tmin, baseline_tmax = -0.1, 0
            baseline = (baseline_tmin, baseline_tmax)
            epochs.apply_baseline(baseline)
            
            epochs_list.append(epochs)
            events_list.append(events)
        
        # 
        if file_name.endswith('.txt'):
            report_path = os.path.join(data_folder_path, file_name)
            lines = nback.read_report_txt(report_path=report_path)
            key_list = nback.get_nback_key()
            report = nback.get_nback_report_data(lines, key_list)
            report_list.append(report)
    del raw, raw_clean, epochs  # free up memory

    # fatigue and sleepniess questionnarie
    '''
    Initial survey, after run surveys
    '''
    survey1 = {"Lack of Energy": [2,2,2,2],
        "Physical Exertion": [2,1,1,1],
        "Physical Discomfort": [1,2,2,1],
        "Lack of Motivation": [1,2,2,2],
        "Sleepiness": [2,3,3,3]} # run 0 1 2 3
    survey2 = {"Lack of Energy": [2,2,3,3],
        "Physical Exertion": [2,1,1,1],
        "Physical Discomfort": [1,1,1,1],
        "Lack of Motivation": [1,2,3,4],
        "Sleepiness": [3,2,2,3]} # run 0 1 2 4
    survey3 = {"Lack of Energy": [2,1,1,1],
        "Physical Exertion": [0,0,0,0],
        "Physical Discomfort": [1,1,1,2],
        "Lack of Motivation": [1,0,2,1],
        "Sleepiness": [2,2,2,1]} # run 0 1 2 3
    survey5 = {"Lack of Energy": [5,2,4,2],
        "Physical Exertion": [3,3,6,6],
        "Physical Discomfort": [4,3,6,6],
        "Lack of Motivation": [2,1,7,3],
        "Sleepiness": [4,1,3,2]} # run 0 1 2 4

    ############### FEATURE ANALYSIS ###############
    reaction_time_run = []
    for run in reaction_time_list:
        temp = [[0],[0],[0],[0]]
        for i in range(len(run)):
            n = int(run[i][1])
            temp[n].append(run[i][0])
        temp = [sum(x)/len(x) for x in temp]
        reaction_time_run.append(temp)
    grand_avg_reaction_time = np.array(reaction_time_run).mean(axis=0)
    
    # df = pd.DataFrame.from_dict(report)
    # df.index.name = 'block'
    
    
    run_ids = []
    frames = []
    for run_id, r in enumerate(report_list, 1):
        run_ids.append(run_id)
        temp_df = pd.DataFrame(r)
        temp_df[['Hit','Miss', 'False Alarm']] = pd.DataFrame(temp_df.criterion.tolist(), index= temp_df.index)
        temp_df = pd.concat([temp_df.drop(['nasa_tlx', 'criterion'], axis=1), temp_df['nasa_tlx'].apply(pd.Series)], axis=1)
        frames.append(temp_df)
    df = pd.concat(frames, keys=run_ids)
    df.index.names = ['run','block']
    print(df)
    
    debug


#     for run in report_list:
#         tlx = df = pd.DataFrame(run[")
#         keys = ["Mental Demand", "Physical Demand", "Temporal Demand", "Performance", "Effort", "Frustration"]
#         for block in range(len(tlx)):
            

    # concatenate all epochs from different trials
    all_epochs = mne.concatenate_epochs(epochs_list)
    # all_epochs = all_epochs['0', '1', '2', '3']

    # visualize epoch
    all_n0_back = all_epochs['0']
    all_n1_back = all_epochs['1']
    all_n2_back = all_epochs['2']
    all_n3_back = all_epochs['3']
    epcs = [all_n0_back, all_n1_back, all_n2_back, all_n3_back]
    for epc in epcs:
        epc.plot_image(picks="eeg", combine="std")

    # evoked analysis
    evokeds_list = [all_n0_back.average(), all_n1_back.average(), all_n2_back.average(), all_n3_back.average()]
    conds = ("0", "1", "2", "3")
    evks = dict(zip(conds, evokeds_list))
    def custom_func(x):
        return x.max(axis=1)

    for combine in ("mean", "median", "gfp", custom_func):
        mne.viz.plot_compare_evokeds(evks, picks="eeg", combine=combine)

    for evk in evokeds_list:
        evk.plot(picks="eeg")
    '''
    freqs = np.arange(3, 50)  # frequencies from 3-50Hz
    vmin, vmax = -1, 1  # set min and max ERDS values in plot
    baseline = (-0.1, 0)  # baseline interval (in s)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

    kwargs = dict(
        n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
    )  # for cluster test

    tfr = mne.time_frequency.tfr_multitaper(
        all_epochs,
        freqs=freqs,
        n_cycles=freqs,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=2,
    )
    tfr.crop(tmin, tmax).apply_baseline(baseline, mode="mean")
    event_ids = {'0': 0, '1': 1, '2': 2, '3': 3}

    for event in event_ids:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(
            1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]}
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

            ax.set_title(all_epochs.ch_names[ch], fontsize=10)
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
    freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 30}
    df["band"] = pd.cut(
        df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:]
    )

    # Filter to retain only relevant frequency bands:
    freq_bands_of_interest = ["theta", "alpha", "beta"]
    df = df[df.band.isin(freq_bands_of_interest)]
    df["band"] = df["band"].cat.remove_unused_categories()

    # Order channels for plotting:
    df["channel"] = df["channel"].cat.reorder_categories(("AF8", "Fp2", "Fp1", "AF7"), ordered=True)

    g = sns.FacetGrid(df, row="band", col="channel", margin_titles=True)
    g.map(sns.lineplot, "time", "value", "condition", n_boot=10)
    axline_kw = dict(color="black", linestyle="dashed", linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)
    g.set(ylim=(-1.5, 2.5))
    g.set_axis_labels("Time (s)", "ERDS")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(ncol=2, loc="lower center")
    g.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)

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
    g.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

    plt.show()
    '''


if __name__ == '__main__':
    # prep_data()
    # eye_oc()
    n_back_analysis()