import os
import pyxdf
import mne
import numpy as np
import re

class Setup:
    def __init__(self,data_path: str = None, data_type: str = None, stream_source: dict = None):
        assert data_path is not None, ' data_path is not defined'
        assert data_type is not None, ' data_type is not defined'
        if not isinstance(data_path, str):
            raise TypeError
        if not isinstance(data_type, str):
            raise TypeError

        self.data_path = data_path
        self.stream_source = stream_source
        if data_type == 'binary':
            self.raw = mne.io.read_raw_fif(data_path)
        elif data_type == 'vhdr':
            self.raw = mne.io.read_raw_brainvision(data_path)
        elif data_type -- 'edf':
            self.raw = mne.io.read_raw_edf(data_path)
        elif data_type == 'xdf':
            # check if stream_source is claimed (dict {'source': list of str,'n_ch': list of int,'ch_name': list of list of str,'ch_type': list of list of str})
            assert stream_source is not None, 'stream_source of xdf file is not defined'
            if not isinstance(stream_source, dict):
                raise TypeError
            self.raw_list = self.read_xdf()
        else:
            print('data_type is not supported')
    
    def read_xdf(self):
        # load .xdf file
        streams, header = pyxdf.load_xdf(self.data_path)

        # detect trigger/STIM stream id
        list_stim_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(self.data_path), [{'type': 'Markers'}])

        # detect the EEG stream id
        list_eeg_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(self.data_path), [{'type': 'EEG'}])

        # define STIM and EEG streams and get first and last timestamps
        first_samp = 0.0
        last_samp = 1000000000
        stim_stream = None
        eeg_stream = []
        for stream in streams:
            stream_id = stream['info']['stream_id']
            if stream['info']['stream_id'] in list_stim_id:
                stim_stream = stream
            elif stream['info']['stream_id'] in list_eeg_id:
                eeg_stream.append(stream)
                # find first timestamp
                if stream['time_stamps'][0] > first_samp:
                    first_samp = stream['time_stamps'][0]
                if stream['time_stamps'][-1] < last_samp:
                    last_samp = stream['time_stamps'][-1]
                assert stim_stream is not None, 'STIM stream not found'
                assert stim_stream is not [], 'EEG stream not found'
                print('first time stamp is {}'.format(first_samp))
                print('last time stamp is {}'.format(first_samp))

        # timestamps correction
        last_samp -= first_samp
        stim_stream['time_stamps'] -= first_samp
        for stream in eeg_stream:
            stream['time_stamps'] -= first_samp

        # truncate EEG streams between first and last timestamps
        strat_ind = 0
        end_ind = 0
        for stream in eeg_stream:
            for i in range(len(stream['time_stamps'])):
                if stream['time_stamps'][i] <= 0.0:
                    strat_ind = i+1
                if stream['time_stamps'][i] <= last_samp:
                    end_ind = i
            stream['time_stamps'] = stream['time_stamps'][strat_ind:end_ind+1]
            stream['time_series'] = stream['time_series'][strat_ind:end_ind+1,:]

        # seperate streams from different EEG systems
        raw_list = []

        # create BrainVision raw
        if 'BrainVision' in self.stream_source['source']:
            source_ind = self.stream_source['source'].index('LiveAmp')
            n_ch = self.stream_source['n_ch'][source_ind]
            for stream in eeg_stream:
                stream_id = stream['info']['stream_id'] 
                if 'LiveAmp' in stream['info']['name'][0]:
                    print('Found LiveAmp eeg stream {}'.format(stream_id))

                    # create bv_data
                    stream['time_series'] = stream['time_series'][:,:n_ch]
                    scale = 1e-6
                    bv_data = stream['time_series'].T * scale # get BrainVision stream data

                    # create bv_info
                    ch_name = self.stream_source['ch_name'][source_ind] # get BrainVision stream channel name
                    ch_type = self.stream_source['ch_type'][source_ind] # get BrainVision stream channel type
                    sfreq = float(stream['info']['nominal_srate'][0]) # get sampling frequnecy
                    bv_info = mne.create_info(ch_name, sfreq, ch_type) # create mne info

                    #create raw
                    bv_raw = mne.io.RawArray(bv_data, bv_info)
                    raw_list.append(bv_raw)
        else:
            print('BrainVision EEG stream not found')

        # create Forehead E-tattoo raw
        if 'ForeheadE-tattoo' in self.stream_source['source']:
            source_ind = self.stream_source['source'].index('LiveAmp')
            n_ch = self.stream_source['n_ch'][source_ind]
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
            scale = 1e-8
            et_data = np.ndarray(shape=(n_ch,et_stream_len), dtype=float)
            for i in range(n_ch):
                et_data[i] = et_stream[i]['time_series'].T[:,:et_stream_len] * scale

            # create et_info
            ch_name = self.stream_source['ch_name'][source_ind] # get BrainVision stream channel name
            ch_type = self.stream_source['ch_type'][source_ind] # get BrainVision stream channel type
            sfreq = float(et_stream[0]['info']['nominal_srate'][0]) # get sampling frequnecy
            et_info = mne.create_info(ch_name, sfreq, ch_type) # create mne info

            # create raw
            et_raw = mne.io.RawArray(et_data, et_info)
            raw_list.append(et_raw)
        else:
            print('Forehead E-tattoo stream not found')
        assert raw_list is not [], 'source is not supported'
        return raw_list