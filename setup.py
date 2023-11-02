import os
import pyxdf
import mne

class Setup:
    def __init__(self,data_path: str, data_type: str):
        if not isinstance(data_path, str):
            raise TypeError
        if not isinstance(data_type, str):
            raise TypeError

        self.data_path = data_path
        if data_type == 'binary':
            self.raw = mne.io.read_raw_fif(data_path)
        elif data_type == 'brainvision':
            self.raw = mne.io.read_raw_brainvision(data_path)
        elif data_type == 'xdf':

            '''
            Debuging
            '''

            stream_type = ['Markes','EEG'] # sls stream types
            daq_system = ['LiveAmp', 'Pulse'] # daq system names
            streams, header = pyxdf.load_xdf(data_path)
            for i in range(len(streams)):
                if streams[i]['info']['type'][0] == 'Markers':
                    markers = streams[i]['time_series']
                    marker_stamps = streams[i]['time_stamps'].T






        pass