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
new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none'}
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
    target_epoch_list = []
    events_list = []
    report_list = []
    criterion_list = []
    reaction_time_list = []

    ############### Import Data ###############
    for file_name in raw_data_list:
        if file_name.endswith('.txt'):
            report_path = os.path.join(data_folder_path, file_name)
            # n_back_report = nback(report_path=report_path)
            lines = nback.read_report_txt(report_path=report_path)
            key_list = nback.get_nback_key(full = False)
            report = nback.get_nback_report_data(lines, key_list, full = False)
            # criterion = nback.create_criterion_list()
            report_list.append(report)
            # criterion_list.append(criterion)
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
            model_plain = mne.preprocessing.EOGRegression(picks='eeg', picks_artifact='eog').fit(raw_car)
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
            if run_idx <3:
                idx = run_idx-1
            else:
                idx = 2
            # critetion_events = events[:,2] = criterion_list[idx]
            # critetion_events = mne.pick_events(critetion_events, exclude=[0,1,2,3])
            # critetion_event_dict = {'hit': 100, 'miss': 200, 'false_alarm': 300}
            tmin, tmax = -0.2, 1.6 # tmin, tmax = -0.1, 1.5 (band-power)
            reject = dict(eeg=80e-6)      # unit: V (EEG channels)
            epochs = mne.Epochs(raw=raw_clean, events=events, event_id=event_dict, event_repeated='drop', tmin=tmin, tmax=tmax, preload=True, reject=reject, picks='eeg', baseline=None)
            # target_epochs = mne.Epochs(raw=raw_clean, events=critetion_events, event_id=critetion_event_dict, event_repeated='drop', tmin=tmin, tmax=tmax, preload=True, reject=reject, picks='eeg', baseline=None)
            baseline_tmin, baseline_tmax = -0.1, 0
            baseline = (baseline_tmin, baseline_tmax)
            epochs.apply_baseline(baseline)
            # target_epochs.apply_baseline(baseline)
            
            epochs_list.append(epochs)
            events_list.append(events)
            # target_epoch_list.append(target_epochs)
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
    survey5 = {'Lack of Energy': [5,2,4,2],
        'Physical Exertion': [3,3,6,6],
        'Physical Discomfort': [4,3,6,6],
        'Lack of Motivation': [2,1,7,3],
        'Sleepiness': [4,1,3,2]} # run 0 1 2 4

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
        temp_df.rename(columns={'nback_sequence':'N'}, inplace=True)
        temp_df[['N']] = temp_df[['N']].astype(str)
        temp_df['Total TLX'] = temp_df['Mental Demand'] + temp_df['Physical Demand'] + temp_df['Temporal Demand'] + temp_df['Performance'] + temp_df['Effort'] + temp_df['Frustration']
        frames.append(temp_df)
    df = pd.concat(frames, keys=run_ids)
    
    #### EEG BAND POWERS ####

    # TODO: 
    # Find the grand average (try not to do it across trials... 0-16)
    # If sill weird, then try to figure out with raw data (with and without EOG correction)
    
    # Frequency band
    # Theta: 4 - 8 Hz
    # Alpha: 8 - 13 Hz
    # Beta: 13 - 30 Hz
            
    theta_low_bound = 4
    theta_high_bound = 8
    alpha_low_bound = 8
    alpha_high_bound = 13
    beta_low_bound = 13
    beta_high_bound = 30
    
    # Keeping track of information based on the runs 
    theta_runs_list = []
    alpha_runs_list = []
    beta_runs_list = []
    
    band_powers = []
    band_power_by_trial = []

    epoch_num = 0
    for epoch in epochs_list:   # Loops through each run
        theta_list = {'run': epoch_num, 'psd': [], 'freq': []}
        alpha_list = {'run': epoch_num, 'psd': [], 'freq': []}
        beta_list = {'run': epoch_num, 'psd': [], 'freq': []}
        
        for n in range(4):  # Loops through each n-back trial
            data = epoch['{}'.format(n)].get_data(copy=True)
            psd_extraction = mne.time_frequency.psd_array_multitaper(x=data, sfreq=200)
            psds = psd_extraction[0] * 1e12     # Put the units on a micro volt ^ 2 scale
            freqs = psd_extraction[1]
            
            # Compute manually if the psd is incorrect
            # print(psds)
            
            t = a = b = 0   # Averages per n per trial
            
            # Integrate with Riemann sums and linear interpolation
            for i in range(len(freqs)-1):  # Why is the number of frequencies larger than the number of psds? (Are the rest zeroes?) 
                if (i == (len(psds)-1)): # For safety purposes
                    break
                
                theta = 0
                alpha = 0
                beta = 0
                
                # In each psd, it returns the recorded signal from each channel
                # TODO: Is it okay to just average all of the channels?
                b1 = np.average(psds[i], axis=None, weights=None, returned=False)
                b2 = np.average(psds[i+1], axis=None, weights=None, returned=False)
                h = freqs[i+1]-freqs[i]
                area = 0.5*(b1+b2)/h   # Where h is dx, the distance between the previous and next sample
                
                # TODO: Figure out more practical implementation since your dx is not consistent between samples...
                if ((freqs[i] >= theta_low_bound) & (freqs[i] < theta_high_bound)):
                    t += area
                    theta = area
                    theta_list['psd'].append(psds[i])
                    theta_list['freq'].append(freqs[i])
                elif ((freqs[i] >= alpha_low_bound) & (freqs[i] < alpha_high_bound)):
                    a += area
                    alpha = area
                    alpha_list['psd'].append(psds[i])
                    alpha_list['freq'].append(freqs[i])
                elif ((freqs[i] >= beta_low_bound) & (freqs[i] < beta_high_bound)):
                    b += area
                    beta = area
                    beta_list['psd'].append(psds[i])
                    beta_list['freq'].append(freqs[i])
                else:
                    continue
                
                # Band powers per run
                run = {'run': epoch_num, 'n': n, 'theta': theta, 'alpha': alpha, 'beta': beta}
                band_powers.append(run)
                
            trial = {'run': epoch_num, 'n': n, 'theta': t, 'alpha': a, 'beta': b}
            band_power_by_trial.append(trial)
            
        
        # PSD and corresponding frequency (simulate and check against the epoch.compute_psd())
        theta_runs_list.append(theta_list)
        alpha_runs_list.append(alpha_list)
        beta_runs_list.append(beta_list)
        
        epoch_num += 1
    
    
    #### Testing ####
    
    # Create a data frame that can be used to interface for sns plotting
    # band_power_df = pd.DataFrame(band_powers)s
    # print(band_power_df)
    
    # This is by the dx, or each 'slice' of the area under the curve for PSD
    # ax = band_power_df.plot.bar(x='n', y='alpha')   # Somewhat working, kinda ugly
    # plt.show()s
    
    # TODO: MOST USED CODE
    # This is the ideal one, group together features that seem to have similar characteristics
    band_power_by_trial_df = pd.DataFrame(band_power_by_trial)
    
    # Everything combined onto one figure
    # band_power_by_trial_df.plot.bar(x='n', y=['theta', 'alpha', 'beta'])
    
    # Each of these are the summation of alpha, theta, and beta for each category by n
    # Separated by the runs (figure wise)
    # band_power_by_trial_df.groupby('run').plot.bar(x='n', y=['theta', 'alpha', 'beta'])
    
    # Grand average
    grand_average = band_power_by_trial_df.groupby('n').sum()
    # grand_average.drop(['run'])   # Remove runs from here later
    print(grand_average)
    grand_average.plot.bar(y=['theta', 'alpha', 'beta'])   
    plt.show()
    
    
    #### SOMETHING IS WEIRD SO TRY TO FIGURE OUT WHY WITH RAW DATA ####
    
    
    #### THESE DONT WORK!! Is it the yscale that is the issue? ####
    # p = sns.catplot(
    #     data=band_power_df, x='n', y='theta', col='run',
    #     kind='bar', height=5, aspect=0.6, order=['0','1','2','3'])
    # p.despine(offset=5, trim=True)
    # plt.autoscale()
    # plt.show()
    
    # p = sns.catplot(
    #     data=band_power_df, x='n', y='alpha', col='run',
    #     kind='bar', height=5, aspect=0.6, order=['0','1','2','3'])
    # p.despine(offset=5, trim=True)
    # plt.autoscale()
    # plt.show()
    
    # p = sns.catplot(
    #     data=band_power_df, x='n', y='beta', col='run',
    #     kind='bar', height=5, aspect=0.6, order=['0','1','2','3'])
    # p.despine(offset=5, trim=True)
    # plt.autoscale()
    # plt.show()

    # Complete visualization
    # psd = epoch['{}'.format(epoch_num)].compute_psd()
    # p = psd.plot(exclude="bads", amplitude=False)
    # plt.show()
    
    df.index.names = ['Run','Trial']
    df = df.reset_index()
    print(df)
    
    debug

    p = sns.catplot(
        data=df, x='N', y='Total TLX', col='Run',
        kind='bar', height=5, aspect=0.6,order=['0','1','2','3'])
    p.despine(offset=5, trim=True)
    plt.show()

    p = sns.catplot(
        data=df, x='Trial', y='Total TLX', col='Run',
        kind='point', height=5, aspect=0.8,)
    p.despine(offset=5, trim=True)
    plt.show()

    df2 = df.drop(['Hit', 'Miss', 'False Alarm', 'Total TLX'], axis=1)
    df2 = df2.melt(['Run','Trial', 'N'])
    df2.rename(columns={'variable':'Questionnaire', 'value': 'Scale'}, inplace=True)
    print(df2)

    p = sns.catplot(data=df2, x='Questionnaire', y='Scale', hue='N', row='Run',
                    kind='bar', height=5, aspect=2.0,hue_order=['0','1','2','3'])
    p.despine(offset=5, trim=True)
    plt.show()

    p = sns.catplot(data=df2, x='Questionnaire', y='Scale', hue='N',
                    kind='bar', height=5, aspect=4.0,hue_order=['0','1','2','3'])
    p.despine(offset=5, trim=True)
    plt.show()

    df3 = df.drop(['Trial','Mental Demand', 'Physical Demand', 'Temporal Demand', 'Performance', 'Effort', 'Frustration', 'Total TLX'], axis=1)
    df3['TN'] = 20 - df3['Hit'] - df3['Miss'] - df3['False Alarm']
    df3 = df3.melt(['Run','N'])
    df3.rename(columns={'variable':'Criterion', 'value': 'Count'}, inplace=True)
    df3 = pd.pivot_table(df3, values='Count', index=['Run','N'], columns='Criterion', aggfunc='sum')
    df3['Detection Rate'] = df3['Hit']/(df3['Hit'] + df3['Miss'])
    df3['False Alarm Rate'] = df3['False Alarm']/(df3['False Alarm'] + df3['TN'])
    df3 = df3.reset_index()
    df3 = df3.rename_axis(None, axis=1)
    print(df3)

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
    
    # concatenate all epochs from different trials
    all_epochs = mne.concatenate_epochs(epochs_list)
    all_target_epochs = mne.concatenate_epochs(target_epochs_list)
    # all_epochs = all_epochs['0', '1', '2', '3']

    # visualize epoch
    all_n0_back = all_epochs['0']
    all_n1_back = all_epochs['1']
    all_n2_back = all_epochs['2']
    all_n3_back = all_epochs['3']
    all_hit = all_target_epochs['hit']
    all_miss = all_target_epochs['miss']
    all_fa = all_target_epochs['false_alarm']

    epcs = [all_n0_back, all_n1_back, all_n2_back, all_n3_back]
    # for epc in epcs:
    #     epc.plot_image(picks='eeg', combine='std')

    # evoked analysis
    evokeds_list = [all_n0_back.average(), all_n1_back.average(), all_n2_back.average(), all_n3_back.average()]
    target_evokeds_list = [all_hit.average(), all_miss.average(), all_fa.average()]
    conds = ('0', '1', '2', '3')
    target_conds = ('hit', 'miss', 'false_alarm')
    evks = dict(zip(conds, evokeds_list))
    target_evks = dict(zip(conds, evokeds_list))  
    def custom_func(x):
        return x.max(axis=1)

    for combine in ('mean', 'median', 'gfp', custom_func):
        mne.viz.plot_compare_evokeds(evks, picks='eeg', combine=combine)
        mne.viz.plot_compare_evokeds(target_evks, picks='eeg', combine=combine)

    # for evk in evokeds_list:
    #     evk.plot(picks='eeg')
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
    # prep_data()
    # eye_oc()
    n_back_analysis()
