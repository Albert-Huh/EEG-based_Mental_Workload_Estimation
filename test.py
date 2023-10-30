import pyxdf
import os
import mne
import numpy as np

fname = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S1/ECEO.xdf')
streams, header = pyxdf.load_xdf(fname)

bv_ch_num = 18
bv_ch_name = ['Fz', 'F3', 'F7', 'C3', 'T7', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P8', 'P4', 'T8', 'C4', 'Cz', 'F8', 'F4']
bv_ch_type = ['eeg'] * bv_ch_num

bv_streams = streams[1] # get BrainVision stream
bv_stream_len = bv_streams["time_series"].T.shape[1]
bv_scale = 1e-6
bv_data = bv_streams["time_series"].T[:18,:] * bv_scale # get BrainVision stream data
bv_sfreq = float(bv_streams["info"]["nominal_srate"][0]) # get sampling frequnecy
bv_info = mne.create_info(bv_ch_name, bv_sfreq, bv_ch_type) # create mne info
bv_raw = mne.io.RawArray(bv_data, bv_info)
bv_raw.plot(scalings=dict(eeg=100e-6), duration=5)


et_ch_num = 6
et_ch_name = ['AF8','Fp2','Fp1','AF7','hEOG','vEOG']
et_ch_type = ['eeg','eeg','eeg','eeg','eog','eog']

et_streams = streams[3:]
et_stream_len = et_streams[0]["time_series"].T.shape[1]
for stream in et_streams:
    et_stream_len = min(et_stream_len,stream["time_series"].T.shape[1])

et_scale = 1e-8
et_data = np.ndarray(shape=(et_ch_num,et_stream_len), dtype=float)
for i in range(len(et_streams)):
    et_data[i] = et_streams[i]["time_series"].T[:,:et_stream_len] * et_scale
et_sfreq = float(et_streams[0]["info"]["nominal_srate"][0])

et_info = mne.create_info(et_ch_num, et_sfreq, et_ch_type) # ['AF8','Fp2','Fp1','AF7','hEOG','vEOG']
et_raw = mne.io.RawArray(et_data, et_info)
et_raw.plot(scalings=dict(eeg=100e-6), duration=5)

mainloop()