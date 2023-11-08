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
        elif data_type -- 'edf':
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
        streams, header = pyxdf.load_xdf(self.data_path)

        # detect trigger/STIM stream id
        list_stim_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(self.data_path), [{'type': 'Markers'}])
        list_stim_id = list_stim_id + pyxdf.match_streaminfos(pyxdf.resolve_streams(self.data_path), [{'type': 'stim'}])

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
                if len(stream['time_series']) != 0:
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
        raw_dict = {}

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
                    raw_dict['BrainVision'] = bv_raw
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
            raw_dict['ForeheadE-tattoo'] = et_raw
        else:
            print('Forehead E-tattoo stream not found')
        assert raw_dict is not {}, 'source is not supported'

        # create annotation and set to raws
        onset = stim_stream['time_stamps']
        description = np.array([item for sub in stim_stream['time_series'] for item in sub])
        ind_remove = []
        for i in range(len(description)):
            if description[i] == '':
                ind_remove.append(i)
        onset = np.delete(onset, ind_remove)
        duration = np.zeros(onset.shape)
        description = np.delete(description, ind_remove)
        for key in raw_dict.keyskeys():
            self.set_annotation(raw_dict[key], onset=onset, duration=duration, description=description)
            raw_dict[key]
        return raw_dict # dict of mne.io.Raw

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
