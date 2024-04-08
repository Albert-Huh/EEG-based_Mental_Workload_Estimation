import os
import pyxdf
import mne
import numpy as np
import re
import matplotlib.pyplot as plt

def _xdf_tiebreak(data_path, streams):
        names = []
        for stream in streams:
            names.append(stream['info']['name'][0])
        print('Resolving streams...', end='\n')

        winning_streams = []
        unique_names = np.unique(names)
        for name in unique_names:
            candidate_ids = pyxdf.match_streaminfos(pyxdf.resolve_streams(data_path), [{'name': name}])
            candidate_streams = [stream for stream in streams if stream['info']['stream_id'] in candidate_ids and len(stream['time_series'])>1]
            stream_len = [len(stream['time_series']) for stream in candidate_streams]
            stamp_min = [min(stream['time_stamps']) for stream in candidate_streams]
            winner_idx = sorted(range(len(stamp_min)), key=lambda k: stamp_min[k])
            winners = [candidate_streams[i] for i in winner_idx]
            # max_stream_len = max(stream_len)
            # winner_idx = stream_len.index(max_stream_len)
            # winning_streams.append(candidate_streams[winner_idx])
            for winner in winners:
                winning_streams.append(winner)
        return winning_streams

# get list of raw data file names in local data folder
subject_idx = '6'
data_folder_path = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S'+subject_idx)
raw_data_list = os.listdir(data_folder_path)

for file_name in raw_data_list:
    if file_name.endswith('eeg.xdf'):
        preprocessed = False
        for name in raw_data_list:
            if file_name.replace('.xdf','_raw.fif') in name:
                preprocessed = True
                print(file_name, 'already has a preprocessed .fif file.')
        if preprocessed == False and not file_name.startswith('training'):
            print(file_name, 'is not preprocessed.')
            # load .xdf file
            raw_path = os.path.join(data_folder_path, file_name)
            streams, header = pyxdf.load_xdf(raw_path, dejitter_timestamps=True, handle_clock_resets=False,
                                                jitter_break_threshold_seconds=0.04, 
                                                jitter_break_threshold_samples=50)
            #fix bad xdf streams
            streams = _xdf_tiebreak(raw_path, streams)
            streams = streams[0:14]
            fig, ax = plt.subplots(nrows=len(streams))
            fig.suptitle('Raw XDF')
            for i, stream in enumerate(streams):
                ax[i].plot(stream['time_stamps'],stream['time_series'].T[0,:],label=i)
                ax[i].set_xlabel('Time (s)')
                ax[i].set_ylabel('Signal')
            plt.show()
            debug 

            # detect trigger/STIM stream id
            list_stim_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(raw_path), [{'type': 'Markers'}])
            list_stim_id = list_stim_id + pyxdf.match_streaminfos(pyxdf.resolve_streams(raw_path), [{'type': 'stim'}])

            # detect the EEG stream id
            list_eeg_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(raw_path), [{'type': 'EEG'}])

            # define STIM and EEG streams
            stim_stream = None
            eeg_stream = []
            for stream in streams:
                stream_id = stream['info']['stream_id']
                if stream['info']['stream_id'] in list_stim_id:
                    if len(stream['time_stamps']) != 0:
                        stim_stream = stream
                elif stream['info']['stream_id'] in list_eeg_id: # and np.any(stream['time_stamps'])
                    eeg_stream.append(stream)
                    
            if stim_stream == None:
                print('STIM stream not found')
            assert eeg_stream is not [], 'EEG stream not found'

            #get first and last timestamps
            first_samp = min(stream['time_stamps'][0] for stream in eeg_stream)
            last_samp = max(stream['time_stamps'][-1] for stream in eeg_stream)
            for stream in eeg_stream:
            # find first timestamp
                first_samp = max(stream['time_stamps'][0], first_samp) if abs(stream['time_stamps'][0]-first_samp) < 2 else min(stream['time_stamps'][0], first_samp)
                last_samp = min(stream['time_stamps'][-1], last_samp) if abs(stream['time_stamps'][-1]-last_samp) < 2 else max(stream['time_stamps'][-1], last_samp)
                print(stream['info']['name'])
                print(stream['time_stamps'][0:3],stream['time_stamps'][-3:])
            print('first time stamp is {}'.format(first_samp))
            print('last time stamp is {}'.format(last_samp))

            # timestamps correction
            # last_samp -= first_samp
            if stim_stream != None:
                stim_stream['time_stamps'] -= first_samp

            # truncate EEG streams between first and last timestamps
            strat_ind = 0
            end_ind = 0
            for stream in eeg_stream:
                nominal_srate = float(stream['info']['nominal_srate'][0])
                for i in range(len(stream['time_stamps'])):
                    if abs(stream['time_stamps'][i]-first_samp) <= 1/nominal_srate/2:
                        strat_ind = i
                    if abs(stream['time_stamps'][i]-last_samp) <= 1/nominal_srate/2:
                        end_ind = i
                stream['time_stamps'] = stream['time_stamps'][strat_ind:end_ind+1] - first_samp
                stream['time_series'] = stream['time_series'][strat_ind:end_ind+1,:]

            # seperate streams from different EEG systems
            raw_dict = {}

            # create Forehead E-tattoo raw
            source = ['ForeheadE-tattoo']
            n_ch = [6]
            ch_name = [['AF8','Fp2','Fp1','AF7','hEOG','vEOG']]
            ch_type = [['eeg']*4+['eog']*2]
            daq_source = {'source': source,'n_ch': n_ch,'ch_name': ch_name,'ch_type': ch_type}
            source_ind = daq_source['source'].index('ForeheadE-tattoo')
            n_ch = daq_source['n_ch'][source_ind]
            et_stream = [0] * n_ch
            for stream in eeg_stream:
                stream_id = stream['info']['stream_id'] 
                if 'Pulse' in stream['info']['name'][0]:        
                    if 'EEG' in stream['info']['name'][0]:
                        i = re.search(r'\d', stream['info']['name'][0])
                        eeg_num = int(stream['info']['name'][0][i.start()])
                        et_stream[eeg_num-1] = stream
                    if 'hEOG' in stream['info']['name'][0]:
                        et_stream[4] = stream
                    if 'vEOG' in stream['info']['name'][0]:
                        et_stream[5] = stream
            
            # create et_data
            et_stream_len = et_stream[0]['time_series'].T.shape[1]
            for stream in et_stream: # find min sample size
                et_stream_len = min(et_stream_len,stream['time_series'].T.shape[1])
            VREF = 2.5
            PGA_gain = 24
            lsb = (2* VREF) / (PGA_gain * 2 ** 24)
            et_scale = lsb
            et_data = np.ndarray(shape=(n_ch,et_stream_len), dtype=float)
            srate = np.ndarray(shape=(n_ch,1), dtype=float)
            for i in range(n_ch):
                et_data[i] = et_stream[i]['time_series'].T[:,:et_stream_len] * et_scale
                srate[i] = et_stream[i]['info']['effective_srate']
            # create et_info
            ch_name = daq_source['ch_name'][source_ind] # get ForeheadE-tattoo stream channel name
            ch_type = daq_source['ch_type'][source_ind] # get ForeheadE-tattoo stream channel type
            sfreq = min(srate)
            # sfreq = stream['info']['effective_srate'] # get sampling frequnecy
            # sfreq = float(et_stream[0]['info']['nominal_srate'][0]) # get sampling frequnecy
            et_info = mne.create_info(ch_name, sfreq, ch_type) # create mne info

            # create raw
            et_raw = mne.io.RawArray(et_data, et_info)
            raw_dict['ForeheadE-tattoo'] = et_raw

            # create annotation and set to raws
            for stream in eeg_stream:
                print(stream['info']['name'])
                print(stream['time_stamps'].shape)
                if stream['time_stamps'].shape[0]>0:
                    print(stream['time_stamps'][0])
            if stim_stream != None:
                onset = stim_stream['time_stamps']
                description = np.array([item for sub in stim_stream['time_series'] for item in sub])
                ind_remove = []
                for i in range(len(description)):
                    if description[i] == '':
                        ind_remove.append(i)
                onset = np.delete(onset, ind_remove)
                duration = np.zeros(onset.shape)
                description = np.delete(description, ind_remove)
                for key in raw_dict.keys():
                    print(key)
                    self.set_annotation(raw_dict[key], onset=onset, duration=duration, description=description)
                    raw_dict[key] # dict of mne.io.Raw
            # return raw_dict # dict of mne.io.Raw