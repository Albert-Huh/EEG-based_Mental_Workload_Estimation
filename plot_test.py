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
subject_ids = ['1','2','3','5']
df_list = []
for s_id in subject_ids:

    data_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S'+s_id)
    raw_data_list = os.listdir(data_folder_path)
    # initilize epoch and event lists
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
    df.index.names = ['Run','Trial']
    df = df.reset_index()
    df_list.append(df)
    

df = pd.concat(df_list, keys=subject_ids)
df.index.names = ['Subject','Index']
df = df.reset_index()
df = df.drop('Index', axis=1)
print(df)

# p = sns.catplot(
#     data=df, x='N', y='Total TLX', col='Run',
#     kind='bar', height=5, aspect=0.6,order=['0','1','2','3'])
# p.despine(offset=5, trim=True)
# plt.show()

# p = sns.catplot(
#     data=df, x='Trial', y='Total TLX', col='Run',
#     kind='point', height=5, aspect=0.8,)
# p.despine(offset=5, trim=True)
# plt.show()

df2 = df.drop(['Hit', 'Miss', 'False Alarm', 'Total TLX'], axis=1)
df2 = df2.melt(['Subject','Run','Trial', 'N'])
df2.rename(columns={'variable':'Questionnaire', 'value': 'Scale'}, inplace=True)
print(df2)

# p = sns.catplot(data=df2, x='Questionnaire', y='Scale', hue='N', row='Run',
#                 kind='bar', height=5, aspect=2.0,hue_order=['0','1','2','3'])
# p.despine(offset=5, trim=True)
# plt.show()

# setting font sizeto 30
plt.rcParams.update({'font.size': 16})

p = sns.catplot(data=df2, x='Questionnaire', y='Scale', hue='N',
                kind='bar', height=5, aspect=2.0,hue_order=['0','1','2','3'])
p.despine(offset=5, trim=True)
p.set_xticklabels(fontsize=12)
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

# p = sns.catplot(
#     data=df3, x='N', y='Detection Rate', col='Run',
#     kind='bar', height=5, aspect=0.8, order=['0','1','2','3'])
# p.despine(offset=5, trim=True)
# plt.show()

p = sns.catplot(
    data=df3, x='N', y='Detection Rate',
    kind='bar', height=5, aspect=1.0, order=['0','1','2','3'])
p.despine(offset=5, trim=True)
plt.show()

# p = sns.catplot(
#     data=df3, x='N', y='False Alarm Rate', col='Run',
#     kind='bar', height=5, aspect=0.8, order=['0','1','2','3'])
# p.despine(offset=5, trim=True)
# plt.show()

p = sns.catplot(
    data=df3, x='N', y='False Alarm Rate',
    kind='bar', height=5, aspect=1.0, order=['0','1','2','3'])
p.despine(offset=5, trim=True)
plt.show()
