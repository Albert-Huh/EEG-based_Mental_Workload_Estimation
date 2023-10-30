import os
import pyxdf
import mne


def __init__(self, data_path=None, data_type=None):
    self.data_path = data_path
    if data_type == 'Binary':
        self.raw = mne.io.read_raw_fif(data_path)
    pass