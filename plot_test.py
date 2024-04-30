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


# list of raw data files in local data folder
subject_ids = ['1','2','3','4','5','6']
df_list = []
rt_df_list = []
all_subject_epoch_list = []
# get path of EEG montage
montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/passive electrodes/BrainCap','BrainCap 64 Channel','BC-64.bvef')

for s_id in subject_ids:

    data_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S'+s_id)
    raw_data_list = os.listdir(data_folder_path)
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
            filters = preprocessing.Filtering(raw=raw, l_freq=1, h_freq=50)
            raw = filters.external_artifact_rejection(resample=False, notch=False)
            # raw.plot(block=False, title='raw')

            if s_id == '3':
                raw_car = raw.copy().set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg')
                ica = preprocessing.Indepndent_Component_Analysis(raw_car, n_components=raw_car.info['nchan']-2, seed=90)
                raw_clean = ica.perfrom_ICA(exclude=[2,3])
            else:
                # mne EOGRegression requires eeg refference
                raw_car = raw.copy().set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg')
                model_plain = mne.preprocessing.EOGRegression(picks='eeg', picks_artifact='eog').fit(raw_car)
                # fig = model_plain.plot(vlim=(None, None))  # regression coefficients as topomap
                # plt.show()
                raw_clean = model_plain.apply(raw_car)


            # Show the corrected data
            # raw.plot(block=False, title='raw')
            # raw_clean.plot(block=True, title='raw_clean')
            
            # resmaple raw
            new_sfreq = 240
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
            tmin, tmax = -0.2, 1.2 # tmin, tmax = -0.1, 1.5 (band-power)
            reject = dict(eeg=60e-6)      # unit: V (EEG channels)
            epochs = mne.Epochs(raw=raw_clean, events=events, event_id=new_event_dict, event_repeated='drop', tmin=tmin, tmax=tmax, preload=True, reject=reject, picks='eeg', baseline=None, on_missing='ignore')
            baseline_tmin, baseline_tmax = -0.15, 0
            baseline = (baseline_tmin, baseline_tmax)
            epochs.apply_baseline(baseline)
            
            epochs_list.append(epochs)
            events_list.append(events)
    del raw, raw_clean, epochs  # free up memory
    # concatenate all epochs from different runs
    all_run_epochs = mne.concatenate_epochs(epochs_list)
    all_subject_epoch_list.append(all_run_epochs)

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
    df_list.append(df)

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
    rt_df = pd.concat(rt_list, keys=run_ids)
    rt_df.index.names = ['Run','N']
    rt_df = rt_df.reset_index()
    rt_df.rename(columns={0:'RT:Hit', 1:'RT:False Alarm', 2:'RT'}, inplace=True)
    rt_df_list.append(rt_df)
    
############### FEATURE ANALYSIS ###############
# NASA-TLX and N-back Performance Features
df = pd.concat(df_list, keys=subject_ids)
df.index.names = ['Subject','Index']
df = df.reset_index()
df = df.drop('Index', axis=1)

rt_df = pd.concat(rt_df_list, keys=subject_ids)
rt_df.index.names = ['Subject','Index']
rt_df = rt_df.reset_index()
rt_df = rt_df.drop('Index', axis=1)

df2 = df.drop(['Run_id', 'Hit', 'Miss', 'False Alarm', 'Total TLX'], axis=1)
df2 = df2.melt(['Subject','Run','Trial', 'N'])
df2.rename(columns={'variable':'Questionnaire', 'value': 'Scale'}, inplace=True)

df3 = df.drop(['Trial','Run_id','Mental Demand', 'Physical Demand', 'Temporal Demand', 'Performance', 'Effort', 'Frustration', 'Total TLX'], axis=1)
df3['Correct Rejection'] = 20 - df3['Hit'] - df3['Miss'] - df3['False Alarm']
df3 = df3.melt(['Subject','Run','N'])
df3.rename(columns={'variable':'Criterion', 'value': 'Count'}, inplace=True)
df3 = pd.pivot_table(df3, values='Count', index=['Subject','Run','N'], columns='Criterion', aggfunc='sum')
df3['Detection Rate'] = df3['Hit']/(df3['Hit'] + df3['Miss'])
df3['False Alarm Rate'] = df3['False Alarm']/(df3['False Alarm'] + df3['Correct Rejection'])
df3 = df3.reset_index()
df3 = df3.rename_axis(None, axis=1)
cols_to_use = rt_df.columns.difference(df3.columns)
df3 = pd.merge(df3, rt_df[cols_to_use], left_index=True, right_index=True, how='outer')

df_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data')
df_path = os.path.join(df_folder_path, file_name)
pd.to_pickle(df, ".\data\.pkl") 
# setting font sizeto 30
plt.rcParams.update({'font.size': 16})
'''
p = sns.catplot(data=df2, x='Questionnaire', y='Scale', hue='N',
                kind='bar', height=5, aspect=2.0,hue_order=['0','1','2','3'])
p.despine(offset=5, trim=True)
p.set_xticklabels(fontsize=12)
plt.show()

p = sns.catplot(
    data=df3, x='N', y='Detection Rate',
    kind='bar', height=5, aspect=1.0, order=['0','1','2','3'])
p.despine(offset=5, trim=True)
plt.show()

p = sns.catplot(
    data=df3, x='N', y='False Alarm Rate',
    kind='bar', height=5, aspect=1.0, order=['0','1','2','3'])
p.despine(offset=5, trim=True)
plt.show()

p = sns.catplot(
    data=df3, x='N', y='RT:Hit',
    kind='bar', height=5, aspect=1.0, order=['0','1','2','3'])
p.despine(offset=5, trim=True)
plt.show()
'''

# EEG Features
# concatenate all epochs from different subjects
all_epochs = mne.concatenate_epochs(epochs_list)

# visualize epoch
all_n0_back = all_epochs['0']
all_n1_back = all_epochs['1']
all_n2_back = all_epochs['2']
all_n3_back = all_epochs['3']
all_hit = all_epochs['hit']
all_miss = all_epochs['miss']
all_fa = all_epochs['false_alarm']
all_cr = all_epochs['correct_rejection']
'''
epcs = [all_n0_back, all_n1_back, all_n2_back, all_n3_back, all_hit, all_miss, all_fa, all_cr]
titles = ['0-Back (EEG: gfp)',
          '1-Back (EEG: gfp)',
          '2-Back (EEG: gfp)',
          '3-Back (EEG: gfp)',
          'Hit (EEG: gfp)',
          'Miss (EEG: gfp)',
          'False Alarm (EEG: gfp)',
          'Correct Rejection (EEG: gfp)']
for i, epc in enumerate(epcs):
    epc.plot_image(picks='eeg', combine='gfp', title=titles[i])
'''

'''
# evoked analysis
conds = ('hit', 'false_alarm', 'miss', 'correct_rejection')
evokeds_list = [all_hit.average(), all_fa.average(), all_miss.average(), all_cr.average()]
evks = dict(zip(conds, evokeds_list))
mne.viz.plot_compare_evokeds(evks,picks='eeg',
    time_unit='ms',title='Detection Criterion Comparison')

conds = ('0', '1', '2', '3')
evokeds_list = [all_n0_back.average(), all_n1_back.average(), all_n2_back.average(), all_n3_back.average()]
evks = dict(zip(conds, evokeds_list))
mne.viz.plot_compare_evokeds(evks,picks='eeg',
    time_unit='ms',title='N Comparison')

# conds = [('0/hit','0/false_alarm','0/miss'),
#          ('1/hit','1/false_alarm','1/miss'),
#          ('2/hit','2/false_alarm','2/miss'),
#          ('3/hit','3/false_alarm','3/miss')]
# for i, cond in enumerate(conds):
#     evokeds_list = [all_epochs[c].average() for c in cond]
#     evks = dict(zip(conds, evokeds_list))
#     mne.viz.plot_compare_evokeds(evks,picks='eeg',
#         time_unit='ms',title=str(i)+'-Back')
'''
psd_info_df = pd.DataFrame()
for n in range(4):  # Go through each n-back trial
    data = all_epochs['{}'.format(n)].get_data(copy=True)    # There should be 4 
    psd_extraction = mne.time_frequency.psd_array_multitaper(x=data, sfreq=new_sfreq, fmin=3, fmax=30)
    psds = psd_extraction[0] * 1e12     # Put the units on a micro volt ^ 2 scale
    freqs = psd_extraction[1]
    
    theta_lb, theta_ub = 4, (np.where(freqs <= 8))[0][-1]
    alpha_lb, alpha_ub = (theta_ub + 1), (np.where(freqs <= 12))[0][-1]
    beta_lb, beta_ub = (alpha_ub + 1), (np.where(freqs <= 30))[0][-1]
    
    # PSDs construction:
    # np_array with shape: (4, 4, 1046) (3D array)
    # What I am guessing it is structure by:
    # 1st layer: 4 epochs relating to that one trial
    # 2nd layer: 4 channels sampling for the same epoch
    # 3rd layer: 1046 sampled values (time series data >> but I'm guessing by frequency since psd applied)
    
    # Find the average across all 4 channels
    avg_psd = np.mean(psds, axis=1)
    
    for e in range(len(avg_psd)):   # This will separate each epoch
        theta = (avg_psd[e])[theta_lb:theta_ub]
        alpha = (avg_psd[e])[alpha_lb:alpha_ub]
        beta = (avg_psd[e])[beta_lb:beta_ub]
        theta_freq =  freqs[theta_lb:theta_ub]
        alpha_freq = freqs[alpha_lb:alpha_ub]
        beta_freq = freqs[beta_lb:beta_ub]
        theta_bp = np.trapz(y=theta, x=theta_freq)
        alpha_bp = np.trapz(y=alpha, x=alpha_freq)
        beta_bp = np.trapz(y=beta, x=beta_freq)
        info = dict({
            'n' : n,
            'theta' : theta,
            'alpha' : alpha,
            'beta' : beta,
            'theta freq' : theta_freq,
            'alpha freq' : alpha_freq,
            'beta freq' : beta_freq,
            'theta band power' :  theta_bp,
            'alpha band power' : alpha_bp,
            'beta band power' : beta_bp,
            '(theta+beta)/alpha' :  (theta_bp+beta_bp)/alpha_bp,
            'theta/(alpha+beta)' : theta_bp/(alpha_bp+beta_bp),
            'beta/(theta+alpha)' : beta_bp/(theta_bp+alpha_bp)
        })
        
        psd_info_df = pd.concat([psd_info_df, pd.DataFrame([info])], ignore_index=True)
fig, axes = plt.subplots(2, 3, sharex='all',
                             gridspec_kw=dict(left=0.2, right=0.8, bottom=0.1, top=0.9),
                             figsize=(5,4))
sns.barplot(data=psd_info_df, x='n', y='theta band power', label='Theta Band Power', ax=axes[0,0])
sns.barplot(data=psd_info_df, x='n', y='alpha band power', label='Alpha Band Power', ax=axes[0,1])
sns.barplot(data=psd_info_df, x='n', y='beta band power', label='Beta Band Power', ax=axes[0,2])
sns.barplot(data=psd_info_df, x='n', y='(theta+beta)/alpha', label='(theta+beta)/alpha', ax=axes[1,0])
sns.barplot(data=psd_info_df, x='n', y='theta/(alpha+beta)', label='theta/(alpha+beta)', ax=axes[1,1])
sns.barplot(data=psd_info_df, x='n', y='beta/(theta+alpha)', label='beta/(theta+alpha)', ax=axes[1,2])

# Combine the plots into one figure
fig.suptitle('Band Powers vs. N')
fig.subplots_adjust(wspace=0.5)
sns.despine(fig, offset=5, trim=True)
plt.show()

conds = ['hit', 'false_alarm', 'miss', 'correct_rejection']
psd_info_df = pd.DataFrame()
for n in conds:  # Go through each n-back trial
    data = all_epochs[n].get_data(copy=True)    # There should be 4 
    psd_extraction = mne.time_frequency.psd_array_multitaper(x=data, sfreq=200, fmin=3, fmax=30)
    psds = psd_extraction[0] * 1e12     # Put the units on a micro volt ^ 2 scale
    freqs = psd_extraction[1]
    
    theta_lb, theta_ub = 4, (np.where(freqs <= 8))[0][-1]
    alpha_lb, alpha_ub = (theta_ub + 1), (np.where(freqs <= 13))[0][-1]
    beta_lb, beta_ub = (alpha_ub + 1), (np.where(freqs <= 30))[0][-1]
    
    # PSDs construction:
    # np_array with shape: (4, 4, 1046) (3D array)
    # What I am guessing it is structure by:
    # 1st layer: 4 epochs relating to that one trial
    # 2nd layer: 4 channels sampling for the same epoch
    # 3rd layer: 1046 sampled values (time series data >> but I'm guessing by frequency since psd applied)
    
    # Find the average across all 4 channels
    avg_psd = np.mean(psds, axis=1)
    
    for e in range(len(avg_psd)):   # This will separate each epoch
        theta = (avg_psd[e])[theta_lb:theta_ub]
        alpha = (avg_psd[e])[alpha_lb:alpha_ub]
        beta = (avg_psd[e])[beta_lb:beta_ub]

        theta_freq =  freqs[theta_lb:theta_ub]
        alpha_freq = freqs[alpha_lb:alpha_ub]
        beta_freq = freqs[beta_lb:beta_ub]
        
        info = dict({
            'criterion' : n,
            'theta' : theta,
            'alpha' : alpha,
            'beta' : beta,
            'theta freq' : theta_freq,
            'alpha freq' : alpha_freq,
            'beta freq' : beta_freq,
            'theta band power' :  np.trapz(y=theta, x=theta_freq),
            'alpha band power' : np.trapz(y=alpha, x=alpha_freq),
            'beta band power' : np.trapz(y=beta, x=beta_freq)
        })
        
        psd_info_df = pd.concat([psd_info_df, pd.DataFrame([info])], ignore_index=True)
fig, axes = plt.subplots(1, 3, sharex='all',
                             gridspec_kw=dict(left=0.2, right=0.8, bottom=0.1, top=0.9),
                             figsize=(5,4))
sns.barplot(data=psd_info_df, x='criterion', y='theta band power', label='Theta Band Power', ax=axes[0])
sns.barplot(data=psd_info_df, x='criterion', y='alpha band power', label='Alpha Band Power', ax=axes[1])
sns.barplot(data=psd_info_df, x='criterion', y='beta band power', label='Beta Band Power', ax=axes[2])

# Combine the plots into one figure
fig.suptitle('Band Powers vs. Criterion')
fig.subplots_adjust(wspace=0.5)
sns.despine(fig, offset=5, trim=True)
plt.show()
debug