import pyxdf
import os
import mne
import numpy as np
import re
import preprocessing

# load .xdf file
fname = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/motion/eye movements.xdf')
streams, header = pyxdf.load_xdf(fname)

# detect trigger/STIM stream id
list_stim_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(fname), [{'type': 'Markers'}])
list_stim_id = list_stim_id + pyxdf.match_streaminfos(pyxdf.resolve_streams(fname), [{'type': 'stim'}])

# detect the EEG stream id
list_eeg_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(fname), [{'type': 'EEG'}])

# define STIM and EEG streams and get first and last timestamps
first_samp = 0.0
last_samp = 1000000000
stim_stream = None
eeg_stream = []
for stream in streams:
    stream_id = stream['info']['stream_id']
    if stream['info']['stream_id'] in list_stim_id and np.any(stream['time_stamps']):
        stim_stream = stream
    elif stream['info']['stream_id'] in list_eeg_id and np.any(stream['time_stamps']):
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
    print([stream['time_stamps'][0],stream['time_stamps'][-1]])

# seperate streams from different EEG systems
bv_ch_num = 4 # BrainVision system
bv_stream = None
liveamp_eeg_id = []

et_ch_num = 6 # E-tattoo system
et_stream = [0] * et_ch_num
pulse_eeg_id = [0] * et_ch_num    

for stream in eeg_stream:
    stream_id = stream['info']['stream_id'] 
    if 'LiveAmp' in stream['info']['name'][0]:
        liveamp_eeg_id.append(stream['info']['stream_id'])
        stream['time_series'] = stream['time_series'][:,:bv_ch_num]
        bv_stream = stream
        print('Found LiveAmp eeg stream {}'.format(stream_id))
    elif 'Pulse' in stream['info']['name'][0]:        
        if 'EEG' in stream['info']['name'][0]:
            i = re.search(r'\d', stream['info']['name'][0])
            eeg_num = int(stream['info']['name'][0][i.start()])
            pulse_eeg_id[eeg_num-1] = stream['info']['stream_id']
            et_stream[eeg_num-1] = stream
        if 'hEOG' in stream['info']['name'][0]:
            pulse_eeg_id[4] = stream['info']['stream_id']
            et_stream[4] = stream
        if 'vEOG' in stream['info']['name'][0]:
            pulse_eeg_id[5] = stream['info']['stream_id']
            et_stream[5] = stream
assert bv_stream is not None, 'BrainVision EEG stream not found'
assert et_stream is not [], 'E-tattoo EEG stream not found'

# manually create mne raw and info
bv_ch_name = ['BV-AF7','BV-Fp1','BV-Fp2','BV-AF8']
bv_ch_type = ['eeg'] * bv_ch_num
bv_stream_len = bv_stream['time_series'].T.shape[1]
bv_scale = 1e-6
bv_data = bv_stream['time_series'].T * bv_scale # get BrainVision stream data
bv_sfreq = float(bv_stream['info']['effective_srate']) # get sampling frequnecy
bv_info = mne.create_info(bv_ch_name, bv_sfreq, bv_ch_type) # create mne info
bv_raw = mne.io.RawArray(bv_data, bv_info)
# filter
filters = preprocessing.Filtering(raw=bv_raw, l_freq=1, h_freq=30)
bv_raw = filters.external_artifact_rejection(resample=False, notch=False)
bv_raw = filters.resample(new_sfreq=150)
# bv_raw.plot(scalings=dict(eeg=25e-6), duration=5, block=False)
bv_data = bv_raw.get_data()

et_ch_name = ['ET-AF8','ET-Fp2','ET-Fp1','ET-AF7','hEOG','vEOG']
et_ch_type = ['eeg','eeg','eeg','eeg','eog','eog']
et_stream_len = et_stream[0]['time_series'].T.shape[1]
for stream in et_stream: # find min sample size
    et_stream_len = min(et_stream_len,stream['time_series'].T.shape[1])
et_scale = 1e-8
et_data = np.ndarray(shape=(et_ch_num,et_stream_len), dtype=float)
et_sfreq = float(et_stream[0]['info']['effective_srate'])
for i in range(len(et_stream)):
    et_data[i] = et_stream[i]['time_series'].T[:,:et_stream_len] * et_scale
    et_sfreq = min(et_sfreq, float(et_stream[i]['info']['effective_srate']))

et_info = mne.create_info(et_ch_name, et_sfreq, et_ch_type) # ['AF8','Fp2','Fp1','AF7','hEOG','vEOG']
et_raw = mne.io.RawArray(et_data, et_info)
# filter
filters = preprocessing.Filtering(raw=et_raw, l_freq=1, h_freq=30)
et_raw = filters.external_artifact_rejection(resample=False, notch=False)
et_raw = filters.resample(new_sfreq=150)
# et_raw.plot(scalings=dict(eeg=25e-6), duration=5, block=True)
et_data = et_raw.get_data()

print(bv_data.shape)
print(et_data.shape)

# combined raw
combined_stream_len = min(bv_data.shape[1],et_data.shape[1])-10
cumbined_ch_num = bv_ch_num + et_ch_num
cumbined_ch_name = bv_ch_name + et_ch_name
cumbined_ch_type = bv_ch_type + et_ch_type
cumbined_data = np.ndarray(shape=(cumbined_ch_num,combined_stream_len), dtype=float)
cumbined_data[:bv_ch_num] = bv_data[:,:combined_stream_len]
cumbined_data[bv_ch_num:] = et_data[:,10:combined_stream_len+10]
sfreq = bv_raw.info['sfreq'] # get sampling frequnecy
combined_info = mne.create_info(cumbined_ch_name, sfreq, cumbined_ch_type)
combined_raw = mne.io.RawArray(cumbined_data, combined_info)

# combined_stream_len = et_stream[0]['time_series'].T.shape[1]
# for stream in eeg_stream: # find min sample size
#     combined_stream_len = min(combined_stream_len,stream['time_series'].T.shape[1])

# cumbined_ch_num = bv_ch_num + et_ch_num
# cumbined_ch_name = bv_ch_name + et_ch_name
# cumbined_ch_type = bv_ch_type + et_ch_type
# cumbined_data = np.ndarray(shape=(cumbined_ch_num,combined_stream_len), dtype=float)
# for i in range(len(eeg_stream)):
#     if i == 0:
#         cumbined_data[:bv_ch_num] = eeg_stream[i]['time_series'].T[:,:combined_stream_len] * bv_scale
#     else:
#         cumbined_data[i+bv_ch_num-1] = eeg_stream[i]['time_series'].T[:,:combined_stream_len] * et_scale
# sfreq = 250
# combined_info = mne.create_info(cumbined_ch_name, sfreq, cumbined_ch_type) # ['AF8','Fp2','Fp1','AF7','hEOG','vEOG']
# raw = mne.io.RawArray(cumbined_data, combined_info)

# generate and add timestamped annotations to RawArray
onsets = stim_stream['time_stamps']
descriptions = np.array([item for sub in stim_stream['time_series'] for item in sub])
ind_remove = []
for i in range(len(descriptions)):
    if descriptions[i] == '':
        ind_remove.append(i)
onsets = np.delete(onsets, ind_remove)
descriptions = np.delete(descriptions, ind_remove)

combined_raw.annotations.append(onsets, [0] * len(onsets), descriptions)

# # filter
# filters = preprocessing.Filtering(raw=raw, l_freq=1, h_freq=30)
# raw = filters.external_artifact_rejection(resample=False, notch=False)

# plot mne.Raw
# bv_raw.plot(scalings=dict(eeg=20e-6), duration=5, block=False)
# et_raw.plot(scalings=dict(eeg=20e-6), duration=5, block=True)
combined_raw.plot(scalings=dict(eeg=25e-6), duration=5, block=True)

mainloop()


'''
    # specific frequency bands
    FREQ_BANDS = {"theta": [4.0, 8.0],
                "alpha": [8.0, 13.0],
                "beta": [13.0, 30.0]}
    evk_power = []
    for evk in evokeds_list:
        spectrum = evk.compute_psd(method='multitaper', fmin=4, fmax=30, tmin=None, tmax=None, normalization='length')
        psds, freqs = spectrum.get_data(return_freqs=True)
        
        X = []
        # Raw of Evoked type
        if len(psds.shape) == 2:
            for fmin, fmax in FREQ_BANDS.values():
                band_power = psds[:, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
                X.append(band_power)
            grand_avg_band_power = []
            for power in X:
                grand_avg_power = np.mean(power) # average across channels
                grand_avg_band_power.append(grand_avg_power)
        # Epochs type
        elif len(psds.shape) == 3:
            for fmin, fmax in FREQ_BANDS.values():
                band_power = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
                X.append(band_power)
            grand_avg_band_power = []
            for power in X:
                grand_avg_power = np.mean(power, axis=1)#.reshape(1,4)
                grand_avg_band_power.append(grand_avg_power)

        evk_power.append(grand_avg_band_power)
'''