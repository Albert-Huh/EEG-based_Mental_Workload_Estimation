import os
import mne
import numpy as np
import pandas as pd
from setup import N_back_report as nback

subjects = ['S1','S2','S3','S4','S5','S6']
df_list = []
rt_df_list = []
all_subject_epoch_list = []
# get path of EEG montage
montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/passive electrodes/BrainCap','BrainCap 64 Channel','BC-64.bvef')

for s in subjects:
    data_folder_path = os.path.join(os.getcwd(), 'data/HS_Data', s)
    raw_data_list = os.listdir(data_folder_path)
    # initilize epoch and event lists
    run_list = []
    epochs_list = []
    events_list = []
    report_list = []
    criterion_list = []
    reaction_time_list = []

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
        if file_name.endswith('_.fif'):
            run_idx = int(file_name.split('run-')[1].split('_.fif')[0])
            raw_path = os.path.join(data_folder_path, file_name)
            raw = mne.io.read_raw_fif(raw_path)
            raw.load_data()
            sfreq = raw.info['sfreq']
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

            # remove overlapping events
            remv_idx = []
            annot = raw.annotations
            for i in range(len(annot.onset)-1):
                del_t = annot.onset[i+1] - annot.onset[i]
                # get overlapping events idx
                if del_t < 1/sfreq:
                    remv_idx.append(i)
            # remove overlapping events
            raw.annotations.delete(remv_idx)

            custom_mapping = {'0': 0, '1': 1, '2': 2, '3': 3,
                              'fixation': 40, 'response_alpha': 100, 'response_pos': 101}
            events, event_dict = mne.events_from_annotations(raw, event_id=custom_mapping)
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
            new_annot = mne.annotations_from_events(events=events, sfreq=sfreq, event_desc=rt_event_dict)
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
    run_ids = []
    frames = []
    for run_id, r in enumerate(report_list, 1):
        run_ids.append(run_id)
        temp_df = pd.DataFrame(r)
        temp_df[['Hit','Miss', 'False Alarm']] = pd.DataFrame(temp_df.criterion.tolist(), index= temp_df.index)
        temp_df = pd.concat([temp_df.drop(['nasa_tlx', 'criterion'], axis=1), temp_df['nasa_tlx'].apply(pd.Series)], axis=1)
        temp_df.rename(columns={'run':'Run_id','nback_sequence':'N'}, inplace=True)
        temp_df[['N']] = temp_df[['N']].astype(str)
        temp_df[['Performance']] = 100-temp_df[['Performance']]
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
    rt_df[['N']].astype(str)
    rt_df.rename(columns={0:'RT:Hit', 1:'RT:False Alarm', 2:'RT'}, inplace=True)
    rt_df_list.append(rt_df)

subject_ids = ['1','2','3','4','5','6']
# NASA-TLX and N-back Performance Features
df = pd.concat(df_list, keys=subject_ids)
df.index.names = ['Subject','Index']
df = df.reset_index()
df = df.drop('Index', axis=1)
df.to_pickle("./tlx_criterion_by_trial.pkl")  
rt_df = pd.concat(rt_df_list, keys=subject_ids)
rt_df.index.names = ['Subject','Index']
rt_df = rt_df.reset_index()
rt_df = rt_df.drop('Index', axis=1)
rt_df.to_pickle("./rt_by_run.pkl") 