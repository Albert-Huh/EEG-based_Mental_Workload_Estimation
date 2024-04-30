import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import mne
import copy
import seaborn as sns
import pyxdf
import scipy

# TODO: honestly this file should be renamed to something that doesnt mention tat since it covers xdf as well 

PACKETS_PER_SECOND = 25
ch_names = ["AF8", "Fp2", "Fp1", "AF7", "EOGh", "EOGv"]

def _xdf_tiebreak(streams, stream_names, expected_stream_count):
    names = []
    for s in streams:
        names.append(s['info']['name'][0])

    print('Resolving streams...', end='')
    winning_stream_idxs = []
    winning_stream_names = []
    unique_strings, unique_indices = np.unique(stream_names, return_index=True)

    idxs = [[ii for ii, name in enumerate(names) if unique_string in name] for unique_string in unique_strings]

    for cft_idxs in idxs:
        temps = []
        for idx in cft_idxs:
            temp = len(streams[idx]['time_series'])
            temps.append(temp) 

        winner = cft_idxs[temps.index(max(temps))]
        winning_stream_idxs.append(winner)
        winning_stream_names.append(names[winner])
        
    assert(len(winning_stream_names) % expected_stream_count == 0)
    print("OK")

    winning_streams = [streams[i] for i in winning_stream_idxs]
    return winning_streams, winning_stream_names, winning_stream_idxs 


def xdf2mne(xdf_file, plot=False, bad_chs=[], *, report_loss=False, report_time=False):
    """Load in EEG E-TATTOO .xdf file. Only the first 2 outputs are needed for data processing with MNE. 
    Parameters:
        xdf_file (str): string for file location 
        plot (bool): whether to plot the data, should be False for batch processing 
        bad_chs (list): zero-indexed list of bad channel indices, should be known from previous plotting or experiments. only important for plotting 
        report_loss (bool): whether to return the packet loss rate.

    Returns:
        raw (mne.io.RawArray): mne.io.RawArray object
        events (np.array): events in the format that mne expects
        chdata_array (np.array): np array of shape (6, samples) containing EEG data
        user_markers (list): list of events, each with a label and time in seconds, [] if no events are found
        unique_marker_labels (list): list of unique marker labels, might be helpful for identifying what kind of experiment was run, [] if no events are found
   """

    pyxdf.load_xdf(xdf_file, dejitter_timestamps=True)
    streams, fileheader = pyxdf.load_xdf(xdf_file)
    names = []
    for s in streams:
        names.append(s['info']['name'][0])
    etat_stream_names = [name for name in names if 'Pulse' in name]
    marker_stream_names = [name for name in names if 'Phil' in name]
    brainvision_stream_names = [name for name in names if 'LiveAmp' in name]
    nback_stream_names = [name for name in names if 'n_back' in name]
    
    EXPECTED_ETAT_STREAMS = 7
    EXPECTED_MARKER_STREAMS = 1
    EXPECTED_BRAINVISION_STREAMS = 1
    EXPECTED_NBACK_STREAMS = 1

    etat_streams, etat_stream_names, etat_stream_idxs = _xdf_tiebreak(streams, etat_stream_names, EXPECTED_ETAT_STREAMS)
    marker_streams, marker_stream_names, marker_stream_idxs = _xdf_tiebreak(streams, marker_stream_names, EXPECTED_MARKER_STREAMS)
    brainvision_streams, brainvision_stream_names, brainvision_stream_idxs = _xdf_tiebreak(streams, brainvision_stream_names, EXPECTED_BRAINVISION_STREAMS)
    nback_streams, nback_stream_names, nback_stream_idxs = _xdf_tiebreak(streams, nback_stream_names, EXPECTED_NBACK_STREAMS)
    # print(etat_stream_names)
    print(nback_streams[0]['time_series'])
    debug
    desired_order = ["EEG1", "EEG2", "EEG3", "EEG4", "hEOG", "vEOG", "Packet"]
    etat_streams_final = []
    for i, _ in enumerate(desired_order):
        for j, name in enumerate(etat_stream_names):
            if desired_order[i] in name:
                etat_streams_final.append(etat_streams[j])
                break

    #get first and last timestamps
    first_samp = min(stream['time_stamps'][0] for stream in etat_streams)
    last_samp = max(stream['time_stamps'][-1] for stream in etat_streams)
    for stream in etat_streams:
    # find first timestamp
        first_samp = max(stream['time_stamps'][0], first_samp) if abs(stream['time_stamps'][0]-first_samp) < 2 else min(stream['time_stamps'][0], first_samp)
        last_samp = min(stream['time_stamps'][-1], last_samp) if abs(stream['time_stamps'][-1]-last_samp) < 2 else max(stream['time_stamps'][-1], last_samp)
        print(stream['info']['name'])
        print(stream['time_stamps'][0:3],stream['time_stamps'][-3:])
    print('first time stamp is {}'.format(first_samp))
    print('last time stamp is {}'.format(last_samp))

    # timestamps correction
    # last_samp -= first_samp
    nback_streams[0]['time_stamps'] -= first_samp
    
    total_traversed = etat_streams_final[-1]["time_series"][-1] - etat_streams_final[-1]["time_series"][0]
    loss = 0
    y_pkt = [] 
    for pkt in range(len(etat_streams_final[-1]["time_series"])-1):
        y_pkt.append(etat_streams_final[-1]["time_series"][pkt])
        loss = loss + etat_streams_final[-1]["time_series"][pkt+1] - etat_streams_final[-1]["time_series"][pkt] - 1
        if etat_streams_final[-1]["time_series"][pkt+1] - etat_streams_final[-1]["time_series"][pkt] - 1 >= 1:
            aloalo = etat_streams_final[-1]["time_series"][pkt+1] - etat_streams_final[-1]["time_series"][pkt] - 1
            while aloalo > 0:
                y_pkt.append(etat_streams_final[-1]["time_series"][pkt])
                aloalo = aloalo - 1
    y_pkt.append(etat_streams_final[-1]["time_series"][-2])
    y_pkt.append(etat_streams_final[-1]["time_series"][-1])
    loss_rate = loss/total_traversed
    loss_rate = loss_rate.item()

    if report_loss:
        t_pkt = etat_streams_final[-1]["time_stamps"] - etat_streams_final[-1]["time_series"][0]
        return t_pkt, y_pkt - etat_streams_final[-1]["time_series"][0], loss_rate

    chdata_list = []   
    for i in range(len(ch_names)):
        chdata_list.append(etat_streams_final[i]['time_series'])

    time_list = []
    for i in range(len(ch_names)):
        time_list.append(etat_streams_final[i]['time_stamps'])

    lowest_samples = np.inf
    for ch in chdata_list:
        samples = ch.shape[0]
        if samples < lowest_samples:
            lowest_samples = samples

    for c, ch in enumerate(chdata_list): 
        chdata_list[c] = ch[:lowest_samples]
        time_list[c] = time_list[c][:lowest_samples]
    chdata_array = np.squeeze(np.array(chdata_list))
    time_array = np.squeeze(np.array(time_list))

    VREF = 2.5
    PGA_gain = 24
    lsb = (2* VREF) / (PGA_gain * 2 ** 24)
    chdata_array = chdata_array * lsb

    effective_fs_list = [] 
    for i in range(len(ch_names)):
        effective_fs_list.append(etat_streams_final[i]['info']['effective_srate'])
    print(f"Effective sampling rates: {effective_fs_list}")

    info = mne.create_info(ch_names=ch_names, ch_types=["eeg","eeg","eeg","eeg","eog","eog"], sfreq=max(effective_fs_list))
    
    raw = mne.io.RawArray(chdata_array, info)
    
    onset = nback_streams[0]['time_stamps']
    description = np.array([item for sub in nback_streams[0]['time_series'] for item in sub])
    ind_remove = []
    for i in range(len(description)):
        if description[i] == '':
            ind_remove.append(i)
    onset = np.delete(onset, ind_remove)
    duration = np.zeros(onset.shape)
    description = np.delete(description, ind_remove)
    my_annot = mne.Annotations(onset=onset, duration=duration, description=description)
    raw.set_annotations(my_annot)
    raw.plot(block=True, title='raw')

    return raw
