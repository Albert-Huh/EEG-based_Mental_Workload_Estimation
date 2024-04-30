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

    subjects = ['S1','S2','S3','S4','S5','S6']
    subjects = ['S6']
    for s in subjects:
        data_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data', s)
        raw_data_list = os.listdir(data_folder_path)
        for file_name in raw_data_list:
            if file_name.endswith('eeg.xdf'):
                preprocessed = False
                for name in raw_data_list:
                    if file_name.replace('eeg.xdf','_.fif') in name:
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
                    raw = open_tat.xdf2mne_2(raw_path, plot=False)

                    # preprocess raws
                    # interactively annotate bad signal
                    
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
                    raw_name = file_name.replace('eeg.xdf','.fif')
                    raw.plot(block=True)
                    raw.save(os.path.join(data_folder_path, raw_name), overwrite=True)

if __name__ == '__main__':
    prep_data()