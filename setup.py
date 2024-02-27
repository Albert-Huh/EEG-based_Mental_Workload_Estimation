import pyxdf
import mne
import numpy as np
import re
import matplotlib.pyplot as plt

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
        elif data_type == 'edf':
            self.raw = mne.io.read_raw_edf(data_path)
        elif data_type == 'xdf':
            # check if stream_source is claimed (dict {'source': list of str,'n_ch': list of int,'ch_name': list of list of str,'ch_type': list of list of str})
            assert stream_source is not None, 'stream_source of xdf file is not defined'
            if not isinstance(stream_source, dict):
                raise TypeError
            self.raw_dict = self.read_xdf()
        else:
            print('data_type is not supported')
    
    def read_xdf(self):
        # load .xdf file
        streams, header = pyxdf.load_xdf(self.data_path, dejitter_timestamps=True, handle_clock_resets=False,
                                         jitter_break_threshold_seconds=0.04, 
                                         jitter_break_threshold_samples=50)
        #fix bad xdf streams
        streams = self._xdf_tiebreak(streams)
        streams = self._drop_bad_stream(streams)

        # detect trigger/STIM stream id
        list_stim_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(self.data_path), [{'type': 'Markers'}])
        list_stim_id = list_stim_id + pyxdf.match_streaminfos(pyxdf.resolve_streams(self.data_path), [{'type': 'stim'}])

        # detect the EEG stream id
        list_eeg_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(self.data_path), [{'type': 'EEG'}])

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
        
        # create BrainVision raw
        if 'BrainVision' in self.stream_source['source']:
            source_ind = self.stream_source['source'].index('BrainVision')
            n_ch = self.stream_source['n_ch'][source_ind]
            for stream in eeg_stream:
                stream_id = stream['info']['stream_id'] 
                if 'LiveAmp' in stream['info']['name'][0]:
                    print('Found LiveAmp eeg stream {}'.format(stream_id))

                    # create bv_data
                    stream['time_series'] = stream['time_series'][:,:n_ch]
                    bv_scale = 1e-6
                    bv_data = stream['time_series'].T * bv_scale # get BrainVision stream data

                    # create bv_info
                    ch_name = self.stream_source['ch_name'][source_ind] # get BrainVision stream channel name
                    ch_type = self.stream_source['ch_type'][source_ind] # get BrainVision stream channel type
                    sfreq = stream['info']['effective_srate'] # get sampling frequnecy
                    # sfreq = float(stream['info']['nominal_srate'][0]) # get sampling frequnecy
                    bv_info = mne.create_info(ch_name, sfreq, ch_type) # create mne info

                    #create raw
                    bv_raw = mne.io.RawArray(bv_data, bv_info)
                    raw_dict['BrainVision'] = bv_raw
                else:
                    print('BrainVision EEG stream is not not found')
        else:
            print('BrainVision EEG stream not found')

        # create Forehead E-tattoo raw
        if 'ForeheadE-tattoo' in self.stream_source['source']:
            source_ind = self.stream_source['source'].index('ForeheadE-tattoo')
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
            ch_name = self.stream_source['ch_name'][source_ind] # get ForeheadE-tattoo stream channel name
            ch_type = self.stream_source['ch_type'][source_ind] # get ForeheadE-tattoo stream channel type
            sfreq = min(srate)
            # sfreq = stream['info']['effective_srate'] # get sampling frequnecy
            # sfreq = float(et_stream[0]['info']['nominal_srate'][0]) # get sampling frequnecy
            et_info = mne.create_info(ch_name, sfreq, ch_type) # create mne info

            # create raw
            et_raw = mne.io.RawArray(et_data, et_info)
            raw_dict['ForeheadE-tattoo'] = et_raw
        else:
            print('Forehead E-tattoo stream not found')

        # create Combined raw
        if 'Combined' in self.stream_source['source']:
            if 'BrainVision' in raw_dict.keys() and 'ForeheadE-tattoo' in raw_dict.keys():
                source_ind = self.stream_source['source'].index('Combined')
                combined_stream_len = eeg_stream[0]['time_series'].T.shape[1]   
                for stream in eeg_stream: # find min sample size
                    combined_stream_len = min(combined_stream_len,stream['time_series'].T.shape[1])

                bv_ch_num = 4
                et_ch_num = 6
                cumbined_ch_num = self.stream_source['n_ch'][source_ind]
                cumbined_ch_name = self.stream_source['ch_name'][source_ind]
                cumbined_ch_type = self.stream_source['ch_type'][source_ind]
                cumbined_data = np.ndarray(shape=(cumbined_ch_num,combined_stream_len), dtype=float)
                for i in range(len(eeg_stream)):
                    if i == 0:
                        cumbined_data[:bv_ch_num] = eeg_stream[i]['time_series'].T[:,:combined_stream_len] * bv_scale
                    else:
                        cumbined_data[i+bv_ch_num-1] = eeg_stream[i]['time_series'].T[:,:combined_stream_len] * et_scale
                sfreq = bv_info['sfreq'] # get sampling frequnecy
                combined_info = mne.create_info(cumbined_ch_name, sfreq, cumbined_ch_type)
                combined_raw = mne.io.RawArray(cumbined_data, combined_info)
                raw_dict['Combined'] = combined_raw
        assert raw_dict is not {}, 'source is not supported'
        
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
                raw_dict[key]
        return raw_dict # dict of mne.io.Raw

    def _xdf_tiebreak(self, streams):
        names = []
        fig, ax = plt.subplots(nrows=2)
        fig.suptitle('time_stamps')
        for stream in streams:
            names.append(stream['info']['name'][0])
            
            '''ax[0].plot(range(len(stream['time_stamps'])),stream['time_stamps'], label=stream['info']['name'][0])
            ax[1].plot(range(len(np.diff(stream['time_stamps']))),np.diff(stream['time_stamps']), label=stream['info']['name'][0])
            print(stream['time_stamps'][0:3],stream['time_stamps'][268:272],stream['time_stamps'][-3:])'''

            # S2 timestamps debugging
            '''
            fig, ax = plt.subplots(nrows=1)
            fig.suptitle('time_stamps '+stream['info']['name'][0])
            ax.plot(range(len(stream['time_stamps'])),stream['time_stamps'], color='black')
            ax.set_xlabel('Index')
            ax.set_ylabel('Stamp')
            plt.show()
            print(stream['time_stamps'][0:3],stream['time_stamps'][269:271],stream['time_stamps'][-3:])
            debug
            '''
        '''ax[0].plot(range(len(streams[4]['time_stamps'])),streams[4]['time_stamps'], label=streams[4]['info']['name'][0])
        ax[1].plot(range(len(np.diff(streams[4]['time_stamps']))),np.diff(streams[4]['time_stamps']), label=streams[4]['info']['name'][0])
        ax[0].set_xlabel('Index')
        ax[0].set_ylabel('Stamp')
        ax[1].set_xlabel('Index')
        ax[1].set_ylabel('Stamp Diff')
        ax[0].legend()
        ax[1].legend()
        plt.show()
        debug'''
        print('Resolving streams...', end='\n')

        winning_streams = []
        unique_names = np.unique(names)
        for name in unique_names:
            candidate_ids = pyxdf.match_streaminfos(pyxdf.resolve_streams(self.data_path), [{'name': name}])
            candidate_streams = [stream for stream in streams if stream['info']['stream_id'] in candidate_ids]
            stream_len = [len(stream['time_series']) for stream in candidate_streams]
            max_stream_len = max(stream_len)
            winner_idx = stream_len.index(max_stream_len)
            winning_streams.append(candidate_streams[winner_idx])
        return winning_streams
    
    def _drop_bad_stream(self, streams):
        # drop empty stream
        for stream in streams:
            if np.any(stream['time_stamps']) == False:
                streams.remove(stream)
        eeg_stream_ids = pyxdf.match_streaminfos(pyxdf.resolve_streams(self.data_path), [{'type': 'EEG'}])
        eeg_streams = [stream for stream in streams if stream['info']['stream_id'] in eeg_stream_ids and np.any(stream['time_stamps'])]
        first_samp = min(stream['time_stamps'][0] for stream in eeg_streams)
        last_samp = max(stream['time_stamps'][-1] for stream in eeg_streams)
        for stream in eeg_streams:
            # drop stream with siginificant packet lost
            if np.abs(stream['info']['effective_srate']-float(stream['info']['nominal_srate'][0])) > 10:
                streams.remove(stream)
                continue
            # drop disconnected stream
            if np.abs(stream['time_stamps'][0]-first_samp) > 10 or np.abs(stream['time_stamps'][-1]-last_samp) > 10:
                streams.remove(stream)
        return streams

    def set_annotation(self, raw: mne.io.Raw, onset: np.ndarray, duration: np.ndarray, description: np.ndarray):
        my_annot = mne.Annotations(onset=onset, duration=duration, description=description)
        raw.set_annotations(my_annot)

    def get_annotation_info(self, raw: mne.io.Raw):
        onset = raw.annotations.onset
        duration = raw.annotations.duration
        description = raw.annotations.description
        return onset, duration, description # ndarray or float, float, str

    def annotate_interactively(self, raw: mne.io.Raw):
        fig = raw.plot()
        fig.fake_keypress('a')
        plt.show(block=True)
        interactive_annot = raw.annotations
        return interactive_annot # class mne.Annotations
    
    def read_report_txt(self, reprot_path: str):
        file = open(reprot_path, 'r')
        # read .txt file line by line
        content = list(file)
        lines = []
        for i in content:
            lines.append(i.replace('\n',''))
        # close the Report.txt
        file.close()
        return lines # list of str

    def get_nback_key(self):
        # get the list of indication stings to extract data
        key_alpha_solution = 'Alpha Answer '
        key_alpha_user = 'User Input Alpha: '
        key_position_solution = 'Position Answer '
        key_position_user = 'User Input Position: '
        key_nasa_tlx = 'NASA TLX Responses: '
        key_list = [key_alpha_solution, key_position_solution, key_alpha_user, key_position_user, key_nasa_tlx]
        return key_list # list of str

    def get_nback_report_data(self, lines: list, key_list: list):
        # get the list of nback sequence
        nback_sequence = lines[5] # Sequence is on line 6
        sequence = [int(s) for s in list(nback_sequence) if s.isdigit()]
        # create report dict data object
        report = {}
        report['nback_sequence'] = sequence
        report['sol_alphabet'] = []
        report['sol_position'] = []
        report['user_alphabet'] = []
        report['user_position'] = []
        report['nasa_tlx'] = []
        key_ind = ['nback_sequence','sol_alphabet','sol_position','user_alphabet','user_position','nasa_tlx']

        line_idx = 0
        flag = 0
        line_timestamps = []
        timestamps = []
        # loop through the file line by line
        for line in lines:
            # checking string is present in line or not
            str_idx = 0
            for string in key_list:
                if string in line:
                    lst = line.split('[',1)[1].replace(']','').split(', ') # split str to a list of str
                    lst = [s == 'True' for s in lst] # convert list of str to list of booleen
                    report[key_ind[str_idx]].append(lst)
                str_idx += 1
            line_idx += 1
        if flag == 0:
            print('The key string for stim timestamps were not found')
        else:
            print('The key string for stim timestamps were found in line', line_timestamps)
        return report # dict of list
