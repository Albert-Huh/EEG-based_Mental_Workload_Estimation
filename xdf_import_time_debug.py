import pyxdf
import os
import mne
import numpy as np
import re
import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
# load .xdf file
fname = os.path.join(os.getcwd(), 'data/UT_Experiment_Data/motion/eye movements.xdf')
streams, header = pyxdf.load_xdf(fname,dejitter_timestamps=True,
                                 jitter_break_threshold_seconds=0.04,
                                  jitter_break_threshold_samples=50)

# detect trigger/STIM stream id
list_stim_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(fname), [{'type': 'Markers'}])
list_stim_id = list_stim_id + pyxdf.match_streaminfos(pyxdf.resolve_streams(fname), [{'type': 'stim'}])

# detect the Pulse packet stream id
list_packet_id = pyxdf.match_streaminfos(pyxdf.resolve_streams(fname), [{'type': 'misc'}])
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
    elif stream['info']['stream_id'] in list_packet_id and np.any(stream['time_stamps']):
        if 'Packet' in stream['info']['name'][0]:
            packet_stream = stream
    elif stream['info']['stream_id'] in list_eeg_id and np.any(stream['time_stamps']):
        eeg_stream.append(stream)
        # find first timestamp
        first_samp = max(stream['time_stamps'][0], first_samp)
        last_samp = min(stream['time_stamps'][-1], last_samp)
assert stim_stream is not None, 'STIM stream not found'
assert stim_stream is not [], 'EEG stream not found'
print('first time stamp is {}'.format(first_samp))
print('last time stamp is {}'.format(first_samp))

## timestamps correction

# last_samp -= first_samp
stim_stream['time_stamps'] -= first_samp
packet_stream['time_stamps'] -= first_samp

# start_time = 0
# end_time = last_samp

# truncate EEG streams between first and last timestamps
strat_ind = 0
end_ind = 0
sfreq = eeg_stream[0]['info']['effective_srate']
for stream in eeg_stream:
    # stream['time_stamps'] -= first_samp # time offset correction
    nominal_srate = float(stream['info']['nominal_srate'][0])
    # assert abs(nominal_srate-sfreq) < 50, 'sampling rates of EEG streams are different'    
    # if nominal_srate < sfreq:
    #     sfreq = nominal_srate
    for i in range(len(stream['time_stamps'])):
        if abs(stream['time_stamps'][i]-first_samp) <= 1/nominal_srate/2:
            strat_ind = i
        if abs(stream['time_stamps'][i]-last_samp) <= 1/nominal_srate/2:
            end_ind = i
    # start_time = stream['time_stamps'][strat_ind]
    # end_time = stream['time_stamps'][end_ind]
    stream['time_stamps'] = stream['time_stamps'][strat_ind:end_ind+1] - first_samp
    stream['time_series'] = stream['time_series'][strat_ind:end_ind+1,:]
    print([stream['time_stamps'][0:5],stream['time_stamps'][-5:]],len(stream['time_stamps']))



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
bv_sfreq = bv_stream['info']['effective_srate'] # get sampling frequnecy
bv_data = butter_highpass_filter(bv_data, 0.1, bv_sfreq, order=5)
bv_info = mne.create_info(bv_ch_name, bv_sfreq, bv_ch_type) # create mne info
bv_raw = mne.io.RawArray(bv_data, bv_info, first_samp= bv_stream['time_stamps'][0])
bv_raw.time = bv_stream['time_stamps']
# bv_raw.plot(scalings=dict(eeg=20e-6), duration=5, block=True)

et_ch_name = ['ET-AF8','ET-Fp2','ET-Fp1','ET-AF7','hEOG','vEOG']
et_ch_type = ['eeg','eeg','eeg','eeg','eog','eog']
et_stream_len = et_stream[0]['time_series'].T.shape[1]
for stream in et_stream: # find min sample size
    et_stream_len = min(et_stream_len,stream['time_series'].T.shape[1])
VREF = 2.5
PGA_gain = 24
lsb = (2* VREF) / (PGA_gain * 2 ** 24)
et_scale = lsb
et_data = np.ndarray(shape=(et_ch_num,et_stream_len), dtype=float)
srate = np.ndarray(shape=(et_ch_num,1), dtype=float)
for i in range(len(et_stream)):
    et_data[i] = et_stream[i]['time_series'].T[:,:et_stream_len] * et_scale
    srate[i] = et_stream[i]['info']['effective_srate']
et_data = butter_highpass_filter(et_data, 0.1, et_stream[1]['info']['effective_srate'], order=5)

et_fp2 = et_stream[1]['time_series'].T * et_scale
et_fp2 = butter_highpass_filter(et_fp2, 0.1, et_stream[1]['info']['effective_srate'], order=5)

fig, ax = plt.subplots(nrows=2)
fig.suptitle('BV vs ET Raw XDF')
ax[0].plot(bv_stream['time_stamps'],bv_stream['time_series'].T[2,:],label='BV', color='lightskyblue')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('EEG')
ax[1].plot(et_stream[1]['time_stamps'],et_stream[1]['time_series'].T[0,:],label='ET', color='lightcoral')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('EEG')

fig, ax = plt.subplots(nrows=2)
fig.suptitle('BV Time Stamps vs Value XDF')
ax[0].plot(bv_stream['time_stamps'],label='Time stamps', color='skyblue')
ax[0].set_xlabel('Index')
ax[0].set_ylabel('Time (s)')
ax[1].plot(bv_stream['time_series'].T[2,:],label='Time series', color='lightskyblue')
ax[1].set_xlabel('Index')
ax[1].set_ylabel('EEG')

fig, ax = plt.subplots(nrows=2)
fig.suptitle('ET Time Stamps vs Value XDF')
ax[0].plot(et_stream[1]['time_stamps'][:-2],label='Time stamps', color='coral')
ax[0].set_xlabel('Index')
ax[0].set_ylabel('Time (s)')
ax[1].plot(et_stream[1]['time_series'].T[0,:-2],label='Time series', color='lightcoral')
ax[1].set_xlabel('Index')
ax[1].set_ylabel('EEG')

fig, ax = plt.subplots(nrows=2)
fig.suptitle('BV vs ET Filtered Raw')
ax[0].plot(bv_stream['time_stamps'],bv_data[2,:],label='BV', color='lightskyblue')
ax[0].set_xlabel('Index')
ax[0].set_ylabel('Time (s)')
ax[1].plot(et_stream[1]['time_stamps'],et_fp2[0,:],label='ET', color='lightcoral')
ax[1].set_xlabel('Index')
ax[1].set_ylabel('EEG')

print('Pulse packet effective_srate: ')
print(packet_stream['info']['effective_srate'])
print('Pulse channel effective_srate: ')
print(srate)
# print(et_stream[0]['time_stamps'][0:300])
missing_packet_ind = []
missing_packet_duration = []
missing_packet = []
for i in range(len(packet_stream['time_stamps'])-1):
    old = packet_stream['time_series'].T[0,i]
    new = packet_stream['time_series'].T[0,i+1]
    lost = new - old
    if lost > 1:
        missing_packet_ind.append(i+1)
        missing_packet_duration.append(lost)
        missing_packet = missing_packet + list(range(i+1,i+int(lost)))
        
print('Missing packet index: ')
print(missing_packet_ind)
print('Missing packet duration: ')
print(missing_packet_duration)
print('Missing packet number: ', len(missing_packet_ind))
missing_packet_timestamp = []
for i in range(len(bv_stream['time_stamps'])-1):
    old = bv_stream['time_stamps'].T[i]
    new = bv_stream['time_stamps'].T[i+1]
    lost = new - old
    if lost > 1.5/250:
        missing_packet_timestamp.append(lost)
print('Missing packet time: ')
print(missing_packet_timestamp)
print('First and last packet: ')
print(packet_stream['time_series'].T[0,0],packet_stream['time_series'].T[0,-1])
print('Missing packet num: ')
print(len(missing_packet))

# Plotting the time series of given dataframe
fig, ax = plt.subplots(nrows=2)
fig.suptitle('Packet Time vs. Packet ID')
ax[0].plot(range(len(packet_stream['time_stamps']))[2690:2700], packet_stream['time_stamps'][2690:2700],label='Packet Ind vs Packet time', color='blue')
ax[0].set_xlabel('Index')
ax[0].set_ylabel('Time (s)')
ax[1].plot(range(len(packet_stream['time_stamps']))[2690:2700], packet_stream['time_series'][2690:2700],label='Packet Ind vs Packet ID', color='red')
ax[1].set_xlabel('Index')
ax[1].set_ylabel('ID')
plt.show()

et_sfreq = et_stream[0]['info']['effective_srate']
# et_sfreq = float(et_stream[0]['info']['nominal_srate'][0])
print(et_sfreq)
# et_sfreq = 250

et_info = mne.create_info(et_ch_name, et_sfreq, et_ch_type) # ['AF8','Fp2','Fp1','AF7','hEOG','vEOG']
et_raw = mne.io.RawArray(et_data, et_info)

original_time_stamps = np.copy(et_stream[1]['time_stamps'][:et_stream_len])
original_time_series = np.copy(et_data[1,:])
corrected_time_stamps = np.copy(et_stream[1]['time_stamps'][:et_stream_len])
corrected_time_series = np.copy(et_data[1,:])
fig, ax = plt.subplots(nrows=2)
fig.suptitle('XDF vs. MNE')
ax[0].plot(original_time_stamps,original_time_series,label='original', color='blue')
ax[0].legend()
ax[0].set_xlabel('Time (s)')
ax[1].plot(et_raw.times, et_raw._data[1], label='MNE', color="red")
ax[1].legend()
ax[1].set_xlabel('Time (s)')

''' manual correction (don't need anymore)
for i in range(len(et_stream)):
    corrected_time_stamps = np.ndarray(shape=bv_stream['time_stamps'].shape)
    corrected_time_series = np.ndarray(shape=(len(bv_stream['time_stamps']),1))
    
    corrected_time_stamps[0] = et_stream[i]['time_stamps'][0]
    packet_ind = 1
    element = 0
    for j in range(len(corrected_time_stamps)):
        lost_element = 0
        if lost_element == 0 and packet_ind<len(packet_stream['time_series']):
            old_packet = packet_stream['time_series'].T[0,packet_ind-1]
            new_packet = packet_stream['time_series'].T[0,packet_ind]
            diff_packet = new_packet - old_packet
            lost_element = (diff_packet-1)*10
            packet_ind += 1
        if diff_packet < 1.5 and element<len(et_stream[i]['time_stamps']):
            corrected_time_stamps[j] = et_stream[i]['time_stamps'][element]
            corrected_time_series[j] = et_stream[i]['time_series'][element]
            element += 1
        elif diff_packet >= 1.5 and lost_element>0:
            corrected_time_stamps[j] = bv_stream['time_stamps'][j]
            # corrected_time_stamps[j] = et_stream[i]['time_series'][j-1]+1/250
            lost_element -= 1
            corrected_time_series[j] = np.nan
    corrected_time_stamps = np.squeeze(corrected_time_stamps)
    corrected_time_series = np.squeeze(corrected_time_series)
    nans = np.isnan(corrected_time_series)
    corrected_time_series[nans]= np.interp(corrected_time_stamps[nans], corrected_time_stamps[~nans], corrected_time_series[~nans])
    # x = np.linspace(0, 2*np.pi, 10)
    # y = np.sin(x)
    # xvals = np.linspace(0, 2*np.pi, 50)
    # yinterp = np.interp(xvals, x, y)
    et_data[i] = corrected_time_series.T * et_scale
et_data = butter_highpass_filter(et_data, 1, bv_sfreq, order=5)
    
fig, ax = plt.subplots(nrows=1)
fig.suptitle('BV vs Corrected ET')
ax.plot(bv_stream['time_stamps'],bv_data[0,:],label='BV', color='blue')
ax.plot(bv_stream['time_stamps'], et_data[0,:], label='Corrected ET', color="red")
ax.legend()
ax.set_xlabel('Time (s)')
plt.show()


# Time_stamp correction with packet id
first_stamp = corrected_time_stamps[0]
for i in range(len(corrected_time_stamps)):
    corrected_time_stamps[i] = first_stamp + i*1/250
for i in range(len(missing_packet_ind)):
    packet_stream['time_stamps'][missing_packet_ind[i]:] += (missing_packet_duration[i]-1)*1/25
    corrected_time_stamps[missing_packet_ind[i]*10:] += (missing_packet_duration[i]-1)*1/25
print('Original time stamps: ')
print(original_time_stamps[-5:])
print('Corrected time stamps: ')
print(corrected_time_stamps[-5:])
fig5 = plt.figure('Figure 5')
plt.plot(range(len(packet_stream['time_stamps'])), packet_stream['time_stamps']) 
# Giving title to the chart using plt.title
plt.title('Packet Index vs. Corrected Packet Time stamps')

# Providing x and y label to the chart
plt.xlabel('Index')
plt.ylabel('Time Stamp')

fig6 = plt.figure('Figure 6')
plt.plot(range(len(corrected_time_stamps)), corrected_time_stamps) 
# Giving title to the chart using plt.title
plt.title('LSL Stream Index vs. Corrected LSL Stream Time stamps')

# Providing x and y label to the chart
plt.xlabel('Index')
plt.ylabel('Time Stamp')

# Providing x and y label to the chart
plt.xlabel('Index')
plt.ylabel('Time Stamp')

fig7 = plt.figure('Figure 7')
plt.plot(range(len(bv_stream['time_stamps'])), bv_stream['time_stamps']) 
# Giving title to the chart using plt.title
plt.title('BV Stream Index vs. BV Stream Time stamps')

# Providing x and y label to the chart
plt.xlabel('Index')
plt.ylabel('Time Stamp')

fig, ax = plt.subplots(nrows=2)
fig.suptitle('LSL vs. corrected')
ax[0].plot(original_time_stamps,original_time_series,label='original', color='blue')
ax[0].legend()
ax[0].set_xlabel('Time (s)')
ax[1].plot(corrected_time_stamps,corrected_time_series,label='corrected', color="red")
ax[1].legend()
ax[1].set_xlabel('Time (s)')

fig, ax = plt.subplots(nrows=4)
fig.suptitle('LSL vs Corrected LSL vs MNE')
ax[0].plot(original_time_stamps,original_time_series,label='original', color='blue')
ax[0].plot(bv_stream['time_stamps'], bv_data[0,:], label='BV', color="red")
ax[0].legend()
ax[0].set_xlabel('Time (s)')
ax[1].plot(et_raw.times, et_raw._data[0], label='MNE', color="red")
ax[1].legend()
ax[1].set_xlabel('Time (s)')
ax[2].plot(bv_stream['time_stamps'], bv_data[0,:], label='BV', color="red")
ax[2].legend()
ax[2].set_xlabel('Time (s)')
ax[3].plot(corrected_time_stamps,corrected_time_series,label='corrected', color='blue')
ax[3].legend()
ax[3].set_xlabel('Time (s)')
plt.show()
'''
fig, ax = plt.subplots(nrows=4)
fig.suptitle('XDF vs MNE')
ax[0].plot(bv_stream['time_stamps'], bv_data[0,:], label='BV-XDF', color="lightskyblue")
ax[0].legend()
ax[0].set_xlabel('Time (s)')
ax[1].plot(bv_raw.times, bv_raw._data[0], label='BV-MNE', color="skyblue")
ax[1].legend()
ax[1].set_xlabel('Time (s)')
ax[2].plot(et_stream[1]['time_stamps'],et_fp2[0,:], label='ET-XDF', color="lightcoral")
ax[2].legend()
ax[2].set_xlabel('Time (s)')
ax[3].plot(et_raw.times, et_raw._data[1],label='ET-MNE', color='coral')
ax[3].legend()
ax[3].set_xlabel('Time (s)')
plt.show()

debug


# combined raw

raw = bv_raw
raw.add_channels(et_raw)
'''
combined_stream_len = eeg_stream[0]['time_series'].T.shape[1]   
for stream in eeg_stream: # find min sample size
    combined_stream_len = min(combined_stream_len,stream['time_series'].T.shape[1])

cumbined_ch_num = bv_ch_num + et_ch_num
cumbined_ch_name = bv_ch_name + et_ch_name
cumbined_ch_type = bv_ch_type + et_ch_type
cumbined_data = np.ndarray(shape=(cumbined_ch_num,combined_stream_len), dtype=float)
for i in range(len(eeg_stream)):
    if i == 0:
        cumbined_data[:bv_ch_num] = eeg_stream[i]['time_series'].T[:,:combined_stream_len] * bv_scale
    else:
        cumbined_data[i+bv_ch_num-1] = eeg_stream[i]['time_series'].T[:,:combined_stream_len] * et_scale
sfreq = 250
combined_info = mne.create_info(cumbined_ch_name, sfreq, cumbined_ch_type) # ['AF8','Fp2','Fp1','AF7','hEOG','vEOG']
raw = mne.io.RawArray(cumbined_data, combined_info)
'''

# generate and add timestamped annotations to RawArray
onsets = stim_stream['time_stamps']
descriptions = np.array([item for sub in stim_stream['time_series'] for item in sub])
ind_remove = []
for i in range(len(descriptions)):
    if descriptions[i] == '':
        ind_remove.append(i)
onsets = np.delete(onsets, ind_remove)
descriptions = np.delete(descriptions, ind_remove)

# bv_raw.annotations.append(onsets, [0] * len(onsets), descriptions)
# et_raw.annotations.append(onsets, [0] * len(onsets), descriptions)
raw.annotations.append(onsets, [0] * len(onsets), descriptions)

# filter
filters = preprocessing.Filtering(raw=raw, l_freq=1, h_freq=30)
raw = filters.external_artifact_rejection(resample=False, notch=False)

# plot mne.Raw
# bv_raw.plot(scalings=dict(eeg=20e-6), duration=5, block=False)
# et_raw.plot(scalings=dict(eeg=20e-6), duration=5, block=True)
raw.plot(scalings=dict(eeg=25e-6), duration=5, block=True)

mainloop()