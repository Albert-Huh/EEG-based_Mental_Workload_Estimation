import pyxdf
import os
import mne
import numpy as np

fname = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/S1/ECEO.xdf')
streams, header = pyxdf.load_xdf(fname)

# detect trigger/STIM stream id
list_stim_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(fname), [{'type': 'Markers'}])

# detect the EEG stream id
list_eeg_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(fname), [{'type': 'EEG'}])

# define STIM and EEG streams; get first and last timestamps
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
            strat_ind = i
        if stream['time_stamps'][i] <= last_samp:
            end_ind = i
    print(strat_ind, end_ind)
    stream['time_stamps'] = stream['time_stamps'][strat_ind:end_ind+1]
    stream['time_series'] = stream['time_series'][strat_ind:end_ind+1,:]

# seperate streams from different EEG systems
bv_stream = None
et_stream = []
liveamp_eeg_id = []
pulse_eeg_id = []       
for stream in eeg_stream:
    stream_id = stream['info']['stream_id']
    if 'LiveAmp' in stream['info']['name'][0]:
        liveamp_eeg_id.append(stream['info']['stream_id'])
        bv_stream = stream
        print('Found LiveAmp eeg stream {}'.format(stream_id))
    elif 'Pulse' in stream['info']['name'][0]:
        pulse_eeg_id.append(stream['info']['stream_id'])
        et_stream.append(stream)
assert bv_stream is not None, 'BrainVision EEG stream not found'
assert et_stream is not [], 'Pulse EEG stream not found'

# example raw
bv_ch_num = 18
bv_ch_name = ['Fz', 'F3', 'F7', 'C3', 'T7', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P8', 'P4', 'T8', 'C4', 'Cz', 'F8', 'F4']
bv_ch_type = ['eeg'] * bv_ch_num
bv_stream_len = bv_stream['time_series'].T.shape[1]
bv_scale = 1e-6
bv_data = bv_stream['time_series'].T[:18,:] * bv_scale # get BrainVision stream data
bv_sfreq = float(bv_stream['info']['nominal_srate'][0]) # get sampling frequnecy
bv_info = mne.create_info(bv_ch_name, bv_sfreq, bv_ch_type) # create mne info
bv_raw = mne.io.RawArray(bv_data, bv_info)

# generate and add timestamped annotations to RawArray
onsets = stim_stream["time_stamps"]
descriptions = [item for sub in stim_stream["time_series"] for item in sub]
bv_raw.annotations.append(onsets, [0] * len(onsets), descriptions)

bv_raw.plot(scalings=dict(eeg=100e-6), duration=5)

debug
for i in range(len(streams)):
    print(i)
    print(streams[i]['info']['name'][0])
    print(streams[i]['info']['type'][0])

bv_ch_num = 18
bv_ch_name = ['Fz', 'F3', 'F7', 'C3', 'T7', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P8', 'P4', 'T8', 'C4', 'Cz', 'F8', 'F4']
bv_ch_type = ['eeg'] * bv_ch_num

bv_stream = streams[1] # get BrainVision stream
bv_stream_len = bv_stream['time_series'].T.shape[1]
bv_scale = 1e-6
bv_data = bv_stream['time_series'].T[:18,:] * bv_scale # get BrainVision stream data
bv_sfreq = float(bv_stream['info']['nominal_srate'][0]) # get sampling frequnecy
bv_info = mne.create_info(bv_ch_name, bv_sfreq, bv_ch_type) # create mne info
bv_raw = mne.io.RawArray(bv_data, bv_info)
bv_raw.plot(scalings=dict(eeg=100e-6), duration=5)

et_ch_num = 6
et_ch_name = ['AF8','Fp2','Fp1','AF7','hEOG','vEOG']
et_ch_type = ['eeg','eeg','eeg','eeg','eog','eog']

et_stream = streams[3:]
et_stream_len = et_stream[0]['time_series'].T.shape[1]
for stream in et_stream:
    et_stream_len = min(et_stream_len,stream['time_series'].T.shape[1])

et_scale = 1e-8
et_data = np.ndarray(shape=(et_ch_num,et_stream_len), dtype=float)
for i in range(len(et_stream)):
    et_data[i] = et_stream[i]['time_series'].T[:,:et_stream_len] * et_scale
et_sfreq = float(et_stream[0]['info']['nominal_srate'][0])

et_info = mne.create_info(et_ch_num, et_sfreq, et_ch_type) # ['AF8','Fp2','Fp1','AF7','hEOG','vEOG']
et_raw = mne.io.RawArray(et_data, et_info)
et_raw.plot(scalings=dict(eeg=100e-6), duration=5)

mainloop()