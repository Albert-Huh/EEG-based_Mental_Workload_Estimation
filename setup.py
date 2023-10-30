import os
import pyxdf
import mne

class Setup:
    def __init__(self,data_path: string, data_type: string):
        if not isistance(data_path, string):
            raise TypeError
        elif if not isistance(data_type, string):
            raise TypeError

        self.data_path = data_path
        if data_type == 'binary':
            self.raw = mne.io.read_raw_fif(data_path)
        elif data_type == 'brainvision':
            self.raw = mne.io.read_raw_brainvision(data_path)
        elif data_type == 'xdf':
            
            
        pass