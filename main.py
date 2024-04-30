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
import open_tat

# graphic render params
new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none'}
mpl.rcParams.update(new_rc_params)

############### IMPORT & CONVERT DATA ###############
def prep_data():
    # get list of raw data file names in local data folder
    subject_idx = '4'
    data_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S'+subject_idx)
    raw_data_list = os.listdir(data_folder_path)

    for file_name in raw_data_list:
        if file_name.endswith('eeg_raw.fif'):
            preprocessed = True
            run_idx = int(file_name.split('run-')[1].split('_eeg_raw.fif')[0])
            raw_path = os.path.join(data_folder_path, file_name)
            raw = mne.io.read_raw_fif(raw_path)
            # Show imported raw data
            raw.plot(block=True,title='R'+str(run_idx))
        if file_name.endswith('eeg.xdf'):
            preprocessed = False
            for name in raw_data_list:
                if file_name.replace('.xdf','_raw.fif') in name:
                    preprocessed = True
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
    subject_idx = '4'
    data_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S'+subject_idx)
    raw_data_list = os.listdir(data_folder_path)

    # get path of EEG montage
    montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/passive electrodes/BrainCap','BrainCap 64 Channel','BC-64.bvef')
    # initilize epoch and event lists
    run_list = []
    epochs_list = []
    events_list = []
    report_list = []
    criterion_list = []
    reaction_time_list = []

    ############### Import Data ###############
    for file_name in raw_data_list:
        if file_name.endswith('.txt'):
            report_path = os.path.join(data_folder_path, file_name)
            n_back_report = nback(report_path=report_path)
            lines = n_back_report.read_report_txt(report_path=report_path)
            key_list = n_back_report.get_nback_key(full = False)
            report = n_back_report.get_nback_report_data(lines, key_list, full = False)
            criterion = n_back_report.create_criterion_list()
            report_list.append(report)
            criterion_list.append(criterion)
    for file_name in raw_data_list:        
        if file_name.endswith('eeg_raw.fif'):
            run_idx = int(file_name.split('run-')[1].split('_eeg_raw.fif')[0])
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
            filters = preprocessing.Filtering(raw=raw, l_freq=2, h_freq=50)
            raw = filters.external_artifact_rejection(resample=False, notch=False)
            # raw.plot(block=False, title='raw')

            '''ica = preprocessing.Indepndent_Component_Analysis(raw, n_components=raw.info['nchan']-2, seed=90)
            reconst_raw = ica.perfrom_ICA()
            reconst_raw.plot(block=True)
            debug'''

            '''
            # perform regression to correct EOG artifacts
            # EOGRegression without refference EEG
            raw_eeg = raw[:4, :][0]
            raw_eeg = raw_eeg - np.mean(raw_eeg, axis =0, keepdims=True)
            raw_eog = raw[4:, :][0]
            raw_eog = raw_eog - np.mean(raw_eog, axis =0, keepdims=True)
            b = np.linalg.inv(raw_eog @ raw_eog.T) @ raw_eog @ raw_eeg.T
            # b = np.linalg.solve(raw_eog @ raw_eog.T, raw_eog @ raw_eeg.T)
            print(b.shape)
            print(b.T)
            eeg_corrected = (raw_eeg.T - raw_eog.T @ b).T
            raw_clean = raw.copy()
            raw_clean._data[:4, :] = eeg_corrected

            '''
            if subject_idx == '3':
                raw_car = raw.copy().set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg')
                ica = preprocessing.Indepndent_Component_Analysis(raw_car, n_components=raw_car.info['nchan']-2, seed=90)
                raw_clean = ica.perfrom_ICA()
            else:
                # mne EOGRegression requires eeg refference
                raw_car = raw.copy().set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg')
                model_plain = mne.preprocessing.EOGRegression(picks='eeg', picks_artifact='eog').fit(raw_car)
                # fig = model_plain.plot(vlim=(None, None))  # regression coefficients as topomap
                # plt.show()

                raw_clean = model_plain.apply(raw_car)


            # Show the corrected data
            # raw.plot(block=False, title='raw')
            raw_clean.plot(block=True, title='raw_clean')
            
            # resmaple raw
            new_sfreq = 200
            raw_clean = raw_clean.resample(sfreq=new_sfreq)
            # raw_clean = raw.resample(sfreq=new_sfreq)
            raw_clean.load_data()


            ############### Create Event & Import N-bsck Report Data ###############
            # remove overlapping events
            remv_idx = []
            annot = raw_clean.annotations
            for i in range(len(annot.onset)-1):
                del_t = annot.onset[i+1] - annot.onset[i]
                # get overlapping events idx
                if del_t < 1/new_sfreq:
                    remv_idx.append(i)
            # remove overlapping events
            raw_clean.annotations.delete(remv_idx)

            custom_mapping = {'0': 0, '1': 1, '2': 2, '3': 3,
                              'fixation': 40, 'response_alpha': 100, 'response_pos': 101}
            events, event_dict = mne.events_from_annotations(raw_clean, event_id=custom_mapping)
            # events = mne.pick_events(events, exclude=[40,100,101])
            for i, re in enumerate(report_list):
                if re['run'] == run_idx:
                    k = 0
                    for j in range(len(events)):
                        if events[j,2] not in [40, 100, 101]:
                            events[j,2] += criterion_list[i][k]
                            k += 1
                    # events[:,2] += criterion_list[i]
            new_event_dict = {'0/correct_rejection': 0, '0/hit': 10, '0/miss': 20, '0/false_alarm': 30,
                              '1/correct_rejection': 1, '1/hit': 11, '1/miss': 21, '1/false_alarm': 31,
                              '2/correct_rejection': 2, '2/hit': 12, '2/miss': 22, '2/false_alarm': 32,
                              '3/correct_rejection': 3, '3/hit': 13, '3/miss': 23, '3/false_alarm': 33,
                              'fixation': 40, 'response_alpha': 100, 'response_pos': 101}
                        
            # get reaction time
            rt_event_dict = {key: new_event_dict[key] for key 
                             in new_event_dict if key 
                             not in ['0/correct_rejection','0/miss','1/correct_rejection','1/miss',
                              '2/correct_rejection','2/miss','3/correct_rejection','3/miss','fixation']}
            rt_event_dict = {y: x for x, y in rt_event_dict.items()}
            new_annot = mne.annotations_from_events(events=events, sfreq=new_sfreq, event_desc=rt_event_dict)
            reaction_time = []
            for i in range(len(new_annot.onset)-1):
                # get reaction time
                del_t = new_annot.onset[i+1] - new_annot.onset[i]
                if new_annot.description[i][0].isdigit() and new_annot.description[i+1].startswith('response_'):
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
                        reaction_time.append([del_t, new_annot.description[i], new_annot.description[i+1]])
                elif new_annot.description[i][0].isdigit() and new_annot.description[i+1][0].isdigit():
                    reaction_time.append([1.8, new_annot.description[i], 'time_out'])
            reaction_time_list.append(reaction_time)
            # critetion_event_dict = {'hit': +10, 'miss': +20, 'false_alarm': +30}
            tmin, tmax = -0.2, 1.6 # tmin, tmax = -0.1, 1.5 (band-power)
            reject = dict(eeg=80e-6)      # unit: V (EEG channels)
            epochs = mne.Epochs(raw=raw_clean, events=events, event_id=new_event_dict, event_repeated='drop', tmin=tmin, tmax=tmax, preload=True, reject=reject, picks='eeg', baseline=None, on_missing='ignore')
            baseline_tmin, baseline_tmax = -0.1, 0
            baseline = (baseline_tmin, baseline_tmax)
            epochs.apply_baseline(baseline)
            
            epochs_list.append(epochs)
            events_list.append(events)
    del raw, raw_clean, epochs  # free up memory
    # fatigue and sleepniess questionnarie
    '''
    Initial survey, after run surveys
    '''
    survey1 = {'Lack of Energy': [2,2,2,2],
        'Physical Exertion': [2,1,1,1],
        'Physical Discomfort': [1,2,2,1],
        'Lack of Motivation': [1,2,2,2],
        'Sleepiness': [2,3,3,3]} # run 0 1 2 3
    survey2 = {'Lack of Energy': [2,2,3,3],
        'Physical Exertion': [2,1,1,1],
        'Physical Discomfort': [1,1,1,1],
        'Lack of Motivation': [1,2,3,4],
        'Sleepiness': [3,2,2,3]} # run 0 1 2 4
    survey3 = {'Lack of Energy': [2,1,1,1],
        'Physical Exertion': [0,0,0,0],
        'Physical Discomfort': [1,1,1,2],
        'Lack of Motivation': [1,0,2,1],
        'Sleepiness': [2,2,2,1]} # run 0 1 2 3
    survey4 = {'Lack of Energy': [0,2,4,4],
        'Physical Exertion': [1,3,4,4],
        'Physical Discomfort': [1,1,1,1],
        'Lack of Motivation': [0,0,2,1],
        'Sleepiness': [0,2,2,3]} # run 0 1 2 4
    survey5 = {'Lack of Energy': [5,2,4,2],
        'Physical Exertion': [3,3,6,6],
        'Physical Discomfort': [4,3,6,6],
        'Lack of Motivation': [2,1,7,3],
        'Sleepiness': [4,1,3,2]} # run 0 1 2 4
    survey6 = {'Lack of Energy': [4,7,8,9],
        'Physical Exertion': [5,5,4,6],
        'Physical Discomfort': [3,7,8,9],
        'Lack of Motivation': [5,7,8,9],
        'Sleepiness': [2,2,6,0]} # run 0 1 2 4

    ############### FEATURE ANALYSIS ###############
    run_ids = []
    frames = []
    for run_id, r in enumerate(report_list, 1):
        run_ids.append(run_id)
        temp_df = pd.DataFrame(r)
        temp_df[['Hit','Miss', 'False Alarm']] = pd.DataFrame(temp_df.criterion.tolist(), index= temp_df.index)
        temp_df = pd.concat([temp_df.drop(['nasa_tlx', 'criterion'], axis=1), temp_df['nasa_tlx'].apply(pd.Series)], axis=1)
        temp_df.rename(columns={'run':'Run_id','nback_sequence':'N'}, inplace=True)
        temp_df[['N']] = temp_df[['N']].astype(str)
        temp_df['Total TLX'] = temp_df['Mental Demand'] + temp_df['Physical Demand'] + temp_df['Temporal Demand'] + temp_df['Performance'] + temp_df['Effort'] + temp_df['Frustration']
        frames.append(temp_df)
    df = pd.concat(frames, keys=run_ids)
    df.index.names = ['Run','Trial']
    df = df.reset_index()
    print(df)

    df2 = df.drop(['Hit', 'Miss', 'False Alarm', 'Total TLX'], axis=1)
    df2 = df2.melt(['Run','Trial', 'Run_id', 'N'])
    df2.rename(columns={'variable':'Questionnaire', 'value': 'Scale'}, inplace=True)
    print(df2)

    # p = sns.catplot(data=df2, x='Questionnaire', y='Scale', hue='N', row='Run',
    #                 kind='bar', height=5, aspect=2.0,hue_order=['0','1','2','3'])
    # p.despine(offset=5, trim=True)
    # plt.show()

    # p = sns.catplot(data=df2, x='Questionnaire', y='Scale', hue='N',
    #                 kind='bar', height=5, aspect=4.0,hue_order=['0','1','2','3'])
    # p.despine(offset=5, trim=True)
    # plt.show()

    rt_list = []
    for rt in reaction_time_list:
        rt_run = np.zeros((4,3))
        temp_hit = [[0],[0],[0],[0]]
        temp_fa = [[0],[0],[0],[0]]
        for i in range(len(rt)):
            cond = rt[i][1]
            n = int(cond[0])
            if cond.endswith('hit'):
                temp_hit[n].append(rt[i][0])
            elif cond.endswith('false_alarm'):
                temp_fa[n].append(rt[i][0])
        for i in range(4):
            h = temp_hit[i]
            f = temp_fa[i]
            if len(h) == 1:
                rt_run[i,0] = np.nan
            else:
                rt_run[i,0] = np.sum(h)/(len(h)-1)
            if len(f) == 1:
                rt_run[i,1] = np.nan
            else:
                rt_run[i,1] = np.sum(f)/(len(f)-1)
            total = temp_hit[i] + temp_fa[i]
            if len(total) == 2:
                rt_run[i,2] = np.nan
            else:
                rt_run[i,2] = np.sum(total)/(len(total)-2)
        rt_list.append(rt_run)
    rt_list = [pd.DataFrame(x) for x in rt_list]
    df4 = pd.concat(rt_list, keys=run_ids)
    df4.index.names = ['Run','N']
    df4 = df4.reset_index()
    df4.rename(columns={0:'RT:Hit', 1:'RT:False Alarm', 2:'RT'}, inplace=True)


    df3 = df.drop(['Trial','Mental Demand', 'Physical Demand', 'Temporal Demand', 'Performance', 'Effort', 'Frustration', 'Total TLX'], axis=1)
    df3['TN'] = 20 - df3['Hit'] - df3['Miss'] - df3['False Alarm']
    df3 = df3.melt(['Run','N', 'Run_id'])
    df3.rename(columns={'variable':'Criterion', 'value': 'Count'}, inplace=True)
    df3 = pd.pivot_table(df3, values='Count', index=['Run','N'], columns='Criterion', aggfunc='sum')
    df3['Detection Rate'] = df3['Hit']/(df3['Hit'] + df3['Miss'])
    df3['False Alarm Rate'] = df3['False Alarm']/(df3['False Alarm'] + df3['TN'])
    df3 = df3.reset_index()
    df3 = df3.rename_axis(None, axis=1)
    cols_to_use = df4.columns.difference(df3.columns)
    df3 = pd.merge(df3, df4[cols_to_use], left_index=True, right_index=True, how='outer')
    '''
    p = sns.catplot(
        data=df3, x='N', y='Detection Rate', col='Run',
        kind='bar', height=5, aspect=0.8, order=['0','1','2','3'])
    p.despine(offset=5, trim=True)
    plt.show()

    p = sns.catplot(
        data=df3, x='N', y='Detection Rate',
        kind='bar', height=5, aspect=1.0, order=['0','1','2','3'])
    p.despine(offset=5, trim=True)
    plt.show()

    p = sns.catplot(
        data=df3, x='N', y='False Alarm Rate', col='Run',
        kind='bar', height=5, aspect=0.8, order=['0','1','2','3'])
    p.despine(offset=5, trim=True)
    plt.show()

    p = sns.catplot(
        data=df3, x='N', y='False Alarm Rate',
        kind='bar', height=5, aspect=1.0, order=['0','1','2','3'])
    p.despine(offset=5, trim=True)
    plt.show()
    '''

    # concatenate all epochs from different runs
    all_epochs = mne.concatenate_epochs(epochs_list)
    # all_epochs = all_epochs['0', '1', '2', '3']

    # visualize epoch
    all_n0_back = all_epochs['0']
    all_n1_back = all_epochs['1']
    all_n2_back = all_epochs['2']
    all_n3_back = all_epochs['3']
    all_hit = all_epochs['hit']
    all_miss = all_epochs['miss']
    all_fa = all_epochs['false_alarm']
    all_cr = all_epochs['correct_rejection']

    epcs = [all_n0_back, all_n1_back, all_n2_back, all_n3_back, all_hit, all_miss, all_fa, all_cr]
    for epc in epcs:
        epc.plot_image(picks='eeg', combine='std')

    # evoked analysis
    evokeds_list = [all_n0_back.average(), 
                    all_n1_back.average(), 
                    all_n2_back.average(), 
                    all_n3_back.average(), 
                    all_hit.average(), 
                    all_miss.average(), 
                    all_fa.average()]
    evokeds_list = [all_hit.average(), 
                    all_miss.average(), 
                    all_fa.average(),
                    all_cr.average()]
    conds = ('hit', 'correct_rejection')
    evks = dict(zip(conds, evokeds_list))
    conds = ('0/correct_rejection', '0/hit', '0/miss', '0/false_alarm',
            '1/correct_rejection', '1/hit', '1/miss', '1/false_alarm',
            '2/correct_rejection', '2/hit', '2/miss', '2/false_alarm',
            '3/correct_rejection', '3/hit', '3/miss', '3/false_alarm')
    
    evks = dict(zip(conds, evokeds_list))

    mne.viz.plot_compare_evokeds(
        evks,time_unit='ms',
        colors=dict(hit=0, correct_rejection=1))
    # for evk in evokeds_list:
    #     evk.plot(picks='eeg')
    mne.viz.plot_compare_evokeds(
        all_hit.average(),time_unit='ms',
        colors=dict(hit=0, correct_rejection=1))
    
    
    def custom_func(x):
        return x.max(axis=1)
    for combine in ('mean', 'median', 'gfp', custom_func):
        mne.viz.plot_compare_evokeds(evks, picks='eeg', combine=combine)


    '''
    freqs = np.arange(3, 50)  # frequencies from 3-50Hz
    vmin, vmax = -1, 1  # set min and max ERDS values in plot
    baseline = (-0.1, 0)  # baseline interval (in s)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

    kwargs = dict(
        n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type='mask'
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
    tfr.crop(tmin, tmax).apply_baseline(baseline, mode='mean')
    event_ids = {'0': 0, '1': 1, '2': 2, '3': 3}

    for event in event_ids:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(
            1, 4, figsize=(12, 4), gridspec_kw={'width_ratios': [10, 10, 10, 1]}
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
                cmap='RdBu',
                cnorm=cnorm,
                axes=ax,
                colorbar=False,
                show=False,
                mask=mask,
                mask_style='mask',
            )

            ax.set_title(all_epochs.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color='black', linestyle=':')  # event
            if ch != 0:
                ax.set_ylabel('')
                ax.set_yticklabels('')
        fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale('linear')
        fig.suptitle(f'ERDS ({event}-Back)')
        plt.show()

    df = tfr.to_data_frame(time_format=None)
    df.head()

    df = tfr.to_data_frame(time_format=None, long_format=True)

    # Map to frequency bands:
    freq_bounds = {'_': 0, 'delta': 3, 'theta': 7, 'alpha': 13, 'beta': 30}
    df['band'] = pd.cut(
        df['freq'], list(freq_bounds.values()), labels=list(freq_bounds)[1:]
    )

    # Filter to retain only relevant frequency bands:
    freq_bands_of_interest = ['theta', 'alpha', 'beta']
    df = df[df.band.isin(freq_bands_of_interest)]
    df['band'] = df['band'].cat.remove_unused_categories()

    # Order channels for plotting:
    df['channel'] = df['channel'].cat.reorder_categories(('AF8', 'Fp2', 'Fp1', 'AF7'), ordered=True)

    g = sns.FacetGrid(df, row='band', col='channel', margin_titles=True)
    g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
    axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)
    g.set(ylim=(-1.5, 2.5))
    g.set_axis_labels('Time (s)', 'ERDS')
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    g.add_legend(ncol=2, loc='lower center')
    g.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)

    df_mean = (
        df.query('time > 1')
        .groupby(['condition', 'epoch', 'band', 'channel'], observed=False)[['value']]
        .mean()
        .reset_index()
    )

    g = sns.FacetGrid(
        df_mean, col='condition', col_order=['0', '1', '2', '3'], margin_titles=True
    )
    g = g.map(
        sns.violinplot,
        'channel',
        'value',
        'band',
        cut=0,
        palette='deep',
        order=['AF8', 'Fp2', 'Fp1', 'AF7'],
        hue_order=freq_bands_of_interest,
        linewidth=0.5,
    ).add_legend(ncol=4, loc='lower center')

    g.map(plt.axhline, **axline_kw)
    g.set_axis_labels('', 'ERDS')
    g.set_titles(col_template='{col_name}-Back', row_template='{row_name}')
    g.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

    plt.show()
    '''


if __name__ == '__main__':
    prep_data()
    # eye_oc()
    # n_back_analysis()