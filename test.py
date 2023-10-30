import pyxdf
import os
import mne
import numpy as np

fname = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S1/ECEO.xdf')
streams, header = pyxdf.load_xdf(fname)
# data = streams[1]["time_series"].T
# # assert data.shape[0] == 5  # four raw EEG plus one stim channel
# # data[:4:2] -= data[1:4:2]  # subtract (rereference) to get two bipolar EEG
# # data = data[::2]  # subselect
# # data[:2] *= 1e-6 / 50 / 2  # uV -> V and preamp gain
# sfreq = float(streams[1]["info"]["nominal_srate"][0])
# info = mne.create_info(36, sfreq)
# raw = mne.io.RawArray(data, info)
# raw.plot(scalings=dict(eeg=100e-6), duration=1, start=14)
a = streams[4]["time_series"].T[:,:49840]
print(a.shape)
scale = 1e-8
tattoo_data = np.concatenate((streams[4]["time_series"].T[:,:49840],streams[3]["time_series"].T[:,:49840],streams[8]["time_series"].T,streams[5]["time_series"].T[:,:49840],streams[6]["time_series"].T,streams[7]["time_series"].T), axis=0) * scale
sfreq = float(streams[4]["info"]["nominal_srate"][0])
tattoo_info = mne.create_info(['AF8','Fp2','Fp1','AF7','hEOG','vEOG'], sfreq, ['eeg','eeg','eeg','eeg','eog','eog']) # ['AF8','Fp2','Fp1','AF7','hEOG','vEOG']
raw = mne.io.RawArray(tattoo_data, tattoo_info)
raw.plot(scalings=dict(eeg=100e-6))

mainloop()