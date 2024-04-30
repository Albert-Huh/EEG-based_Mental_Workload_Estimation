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
from mpl_toolkits.axes_grid1 import ImageGrid, inset_locator, make_axes_locatable
import scipy

# TODO: honestly this file should be renamed to something that doesnt mention tat since it covers xdf as well 

PACKETS_PER_SECOND = 25
ch_names = ["AF8", "Fp2", "Fp1", "AF7", "EOGh", "EOGv"]

def mne2eceo(raw, events, user_markers, unique_marker_labels, closed="ec", open="eo", tmax=20, timefreq_method="stockwell", plot=True):
    event_dict = dict(closed=unique_marker_labels.index(closed), open=unique_marker_labels.index(open))
    epochs = mne.Epochs(
        raw, events, event_id=event_dict, tmin=0.5, tmax=tmax, baseline=(None,None), preload=True, reject=None, proj=False  # baseline (None,None)
    )
    unique_marker_labels = [i for i in unique_marker_labels if i] # removes empty markers 

    # epochs, _ = mne.set_eeg_reference(epochs, 'average')
    # model = mne.preprocessing.EOGRegression(picks="eeg", picks_artifact="eog").fit(epochs)
    # epochs = model.apply(epochs)

    assert(user_markers != [])
    # epochs.plot(scalings='auto', show=True, block=False)
    try:
        spectrum = epochs.compute_psd(method='welch', fmin=2, fmax=70).plot(picks="eeg", exclude="bads")
    except ValueError:
        spectrum =  epochs.compute_psd(method='welch', fmin=2, fmax=raw.info["lowpass"]).plot(picks="eeg", exclude="bads")
    # spectrum_ec = epochs["closed"].compute_psd(method='welch', fmin=2, fmax=70).plot(picks="eeg", exclude="bads")
    # spectrum_eo = epochs["open"].compute_psd(method='welch', fmin=2, fmax=70).plot(picks="eeg", exclude="bads")
   
    # freqs = np.logspace(*np.log10([4, 62]), num=20)
    freqs = np.arange(5.0, 27.5, 2.5)
    n_cycles = freqs / 2.0
    if timefreq_method == "multitaper":
        power = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False, time_bandwidth=4.0)
    elif timefreq_method == "morlet":
        power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False)
    elif timefreq_method == "stockwell":
        power_ec = mne.time_frequency.tfr_stockwell(epochs["closed"], fmin=freqs[0], fmax=freqs[-1], width=1, return_itc=False)
        power_eo = mne.time_frequency.tfr_stockwell(epochs["open"], fmin=freqs[0], fmax=freqs[-1], width=1, return_itc=False)
    else:
        raise ValueError("Invalid timefreq method")

    if timefreq_method == "multitaper" or timefreq_method == "morlet":
        timefreq_per_epoch = [power.data[epoc, 0:4, :, :] * (10**12) for epoc in range(power.data.shape[0])]
        timefreq = np.concatenate(timefreq_per_epoch, axis=2)
        for ch in range(timefreq.shape[0]):
            plt.figure()
            sns.heatmap(timefreq[ch, :, :], cmap="crest", robust=True)
            plt.yticks(range(timefreq.shape[1]), freqs)
        plt.show(block=False)

        timefreq_averaged_ch = np.mean(timefreq, axis=0)
        plt.figure()
        sns.heatmap(timefreq_averaged_ch, cmap="crest", robust=True)
        plt.yticks(range(timefreq_averaged_ch.shape[0]), freqs)
             
    elif timefreq_method == "stockwell":  # stockwell doesn't offer TFR per epoch but only averaged TFR 
        view_window = np.where((power_ec.times>=0) & (power_ec.times<=20))
        # timefreq_per_epoch = [power_ec.data[0:4, :, :] * (10**12), power_eo.data[0:4, :, :] * (10**12)]
        timefreq_per_epoch = [np.squeeze(power_ec.data[0:4, :, view_window]) * (10**12), np.squeeze(power_eo.data[0:4, :, view_window]) * (10**12)]
        if plot: 
            timefreq = np.concatenate(timefreq_per_epoch, axis=2)
            for ch in range(timefreq.shape[0]):
                plt.figure()
                sns.heatmap(timefreq[ch, :, :], cmap="crest", robust=True, cbar_kws={'label': 'Power spectral density ($\mathrm{\u03bcV}^2/\mathrm{Hz}$)'})
                # plt.yticks(range(timefreq.shape[1]), power_ec.freqs)
                plt.yticks(np.linspace(start=0, stop=timefreq.shape[1], num=10), np.round(np.linspace(start=power_ec.freqs[0], stop=power_ec.freqs[timefreq.shape[1]-1], num=10)))
                ec_xtick_locs = np.linspace(start=0, stop=int(np.floor(timefreq.shape[2]/2)), num=11)
                ec_xtick_labels = np.round(np.linspace(start=0, stop=20, num=11), decimals=1)
                eo_xtick_locs = np.linspace(start=int(np.ceil(timefreq.shape[2]/2)), stop=timefreq.shape[2], num=11)
                eo_xtick_labels = np.round(np.linspace(start=0, stop=20, num=11), decimals=1)
                plt.xticks(np.concatenate((ec_xtick_locs, eo_xtick_locs)), np.concatenate((ec_xtick_labels, eo_xtick_labels)))
            

            timefreq_averaged_ch = np.mean(timefreq, axis=0)
            plt.figure(figsize=(10,4))
            sns.heatmap(timefreq_averaged_ch, cmap="crest", robust=True, cbar_kws={'label': 'Power spectral density ($\mathrm{\u03bcV}^2/\mathrm{Hz}$)'})
            plt.yticks(np.linspace(start=0, stop=timefreq.shape[1], num=10), np.round(np.linspace(start=power_ec.freqs[0], stop=power_ec.freqs[timefreq.shape[1]-1], num=10)))
            ec_xtick_locs = np.linspace(start=0, stop=int(np.floor(timefreq.shape[2]/2)), num=11)
            ec_xtick_labels = np.round(np.linspace(start=0, stop=20, num=11), decimals=1)
            eo_xtick_locs = np.linspace(start=int(np.ceil(timefreq.shape[2]/2)), stop=timefreq.shape[2], num=11)
            eo_xtick_labels = np.round(np.linspace(start=0, stop=20, num=11), decimals=1)
            plt.xticks(np.concatenate((ec_xtick_locs, eo_xtick_locs)), np.concatenate((ec_xtick_labels, eo_xtick_labels)))
            plt.ylabel("Frequency (Hz)")
            plt.xlabel("Time (s)")
            # plt.savefig("eceo.png", dpi=300, bbox_inches="tight")
            plt.show(block=False)
    
    if plot:
        plt.show(block=True)

    output_freqs = power_ec.freqs
    output_times = power_ec.times 
    # assume ec and eo are same size    

    return epochs, spectrum, timefreq_per_epoch, output_freqs, output_times


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

def xdf2mne_2(xdf_file, plot=False, bad_chs=[], *, report_loss=False, report_time=False):
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

    streams, fileheader = pyxdf.load_xdf(xdf_file,dejitter_timestamps=True, handle_clock_resets=False,
                                         jitter_break_threshold_seconds=0.04,)
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

    desired_order = ["EEG1", "EEG2", "EEG3", "EEG4", "hEOG", "vEOG", "Packet"]
    etat_streams_final = []
    for i, _ in enumerate(desired_order):
        for j, name in enumerate(etat_stream_names):
            if desired_order[i] in name:
                etat_streams_final.append(etat_streams[j])
                break

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

    '''
    HH: I added some lines below
    '''
    # drop packet stream
    etat_streams_final = etat_streams_final[:-1]

    # fix flipped streams (some streams may have decending time_stamps)
    for stream in etat_streams_final:
        p = stream['time_stamps'].argsort()
        stream['time_stamps'] = stream['time_stamps'][p]
        stream['time_series'] = stream['time_series'][p]

    # get first and last timestamps
    first_samp = min(stream['time_stamps'][0] for stream in etat_streams_final)
    last_samp = max(stream['time_stamps'][-1] for stream in etat_streams_final)
    for stream in etat_streams_final:
        first_samp = max(stream['time_stamps'][0], first_samp) if abs(stream['time_stamps'][0]-first_samp) < 2 else min(stream['time_stamps'][0], first_samp)
        last_samp = min(stream['time_stamps'][-1], last_samp) if abs(stream['time_stamps'][-1]-last_samp) < 2 else max(stream['time_stamps'][-1], last_samp)
        print(stream['info']['name'])
        print(stream['time_stamps'][0:3],stream['time_stamps'][-3:])
    print('first time stamp is {}'.format(first_samp))
    print('last time stamp is {}'.format(last_samp))

    # timestamps correction
    if nback_streams != None:
        nback_streams[0]['time_stamps'] -= first_samp
    
    # truncate EEG streams between first and last timestamps
    strat_ind = 0
    end_ind = 0
    for stream in etat_streams_final:
        nominal_srate = float(stream['info']['nominal_srate'][0])
        for i in range(len(stream['time_stamps'])):
            if abs(stream['time_stamps'][i]-first_samp) <= 1/nominal_srate/2:
                strat_ind = i
            if abs(stream['time_stamps'][i]-last_samp) <= 1/nominal_srate/2:
                end_ind = i
        stream['time_stamps'] = stream['time_stamps'][strat_ind:end_ind+1] - first_samp
        stream['time_series'] = stream['time_series'][strat_ind:end_ind+1,:]

    # create et_data
    et_stream_len = etat_streams_final[0]['time_series'].T.shape[1]
    for stream in etat_streams_final: # find min sample size
        et_stream_len = min(et_stream_len,stream['time_series'].T.shape[1])
    VREF = 2.5
    PGA_gain = 24
    lsb = (2* VREF) / (PGA_gain * 2 ** 24)
    n_ch = len(ch_names)
    chdata_array = np.ndarray(shape=(n_ch,et_stream_len), dtype=float)
    for i in range(n_ch):
        chdata_array[i] = etat_streams_final[i]['time_series'].T[:,:et_stream_len] * lsb
    time_array = etat_streams_final[0]['time_stamps']
    
    # create et_info
    effective_fs_list = [] 
    for i in range(len(ch_names)):
        effective_fs_list.append(etat_streams_final[i]['info']['effective_srate'])
    print(f"Effective sampling rates: {effective_fs_list}")

    # fs = 250 
    fs = min(effective_fs_list, key=lambda x:abs(x-250)); assert(fs > 0)

    info = mne.create_info(
        ch_names=ch_names, ch_types=['eeg']*4+['eog']*2, sfreq=fs #sfreq=np.mean(effective_fs_list)
    )
    raw = mne.io.RawArray(chdata_array, info)
    if bad_chs != []:
        raw.info["bads"] = [ch_names[i] for i in bad_chs]

    # create annotation (STIM Markers) and set to raws
    if nback_streams != None:
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

    # create MNE event
    custom_mapping = {'0': 0, '1': 1, '2': 2, '3': 3,
                              'fixation': 40, 'response_alpha': 100, 'response_pos': 101}
    events, event_dict = mne.events_from_annotations(raw, event_id=custom_mapping)

    if plot:
        raw = raw.filter(l_freq = 0.5, h_freq=70)
        raw.plot(scalings='auto', show=True, block=False, remove_dc=True, events=events)
        plt.show(block=True) 

    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")

    if report_time == True:
        return raw, events, chdata_array, user_markers, unique_marker_labels, time_array
    else:
        return raw
    
def ar_psd(X, fs):
    """Just for convenience, can be modified later"""
    nfft = X.shape[0]
    f, Pxx_den = scipy.signal.welch(X, fs = fs,\
                        # window = np.ones(nfft),\
                        window = "hamming",\
                        # nperseg = nfft,\
                        scaling="density",\
                        nperseg = 1024, average='median',
                        detrend="linear")
    # f, Pxx_den = scipy.signal.welch(X, fs = fs,\
    #                     window = "hamming",\
    #                     scaling="density",\
    #                     nperseg = np.floor(X.shape[1]/8), 
    #                     noverlap = None,
    #                     nfft = 4096)
    return f, Pxx_den

# Example usage 
if __name__ == "__main__":
    raw = mne.io.read_raw_fif(r"C:\Users\mcvai\forehead-eeg-device\data\sub-P001\ses-S003\eeg\BrainVision_ECEO.fif", preload=True)

    # 1: Just load in the data and plot it
    raw, events, chdata_array, user_markers, unique_marker_labels = xdf2mne(r"C:\Users\mcvai\forehead-eeg-device\data\sub-P001\ses-S003\eeg\ECEO.xdf", plot=True)
    epochs, _, _ = mne2eceo(raw, events, user_markers, unique_marker_labels, closed="ec", open="eo", tmax=20)
    input("hi")
    raw, events, chdata_array, user_markers, unique_marker_labels = xdf2mne(r"C:\Users\mcvai\forehead-eeg-device\data\sub-P001\ses-S003\eeg\sub-P001_ses-S003_task-Default_run-001_eeg.xdf", plot=True)
    raw, events, chdata_array, user_markers, unique_marker_labels = xdf2mne(r"C:\Users\mcvai\forehead-eeg-device\data\sub-P001\ses-S003\eeg\sub-P001_ses-S003_task-Default_run-002_eeg.xdf", plot=True)
    raw, events, chdata_array, user_markers, unique_marker_labels = xdf2mne(r"C:\Users\mcvai\forehead-eeg-device\data\sub-P001\ses-S003\eeg\sub-P001_ses-S003_task-Default_run-003_eeg.xdf", plot=True)

    # 2. EC EO comparison 
    raw, events, chdata_array, user_markers, unique_marker_labels = xdf2mne(r"C:\Users\mcvai\forehead-eeg-device\data\sub-P002\ses-S002\eeg\ECEO.xdf", plot=False)
    epochs = mne2eceo(raw, events, user_markers, unique_marker_labels, closed="ec", open="eo", tmax=20)
    # ec_spec = epochs["closed"].compute_psd(method='multitaper', fmin=0.5, fmax=70, tmin=None, tmax=None, picks=None, proj=False)
    # eo_spec = epochs["open"].compute_psd(method='multitaper', fmin=0.5, fmax=70, tmin=None, tmax=None, picks=None, proj=False)  # works well but ar_psd looks a little bit better? 

    ec = epochs["closed"]._data  # (epochs, channels, samples)
    eo = epochs["open"]._data
    
    ec_xf, ec_y = ar_psd(ec, fs=250)
    eo_xf, eo_y = ar_psd(eo, fs=250)

    # sns.set_theme(style="darkgrid")
    # sns.set_theme(style="whitegrid")
    f, axes = plt.subplots(2,2, constrained_layout=True, figsize=(10, 10))
    for ch, ax in enumerate(axes.reshape(-1)):
        y = ec_y[:, ch, :]
        y_average = np.mean(y, axis=0)  # average across epochs
        y_ci = 1.96 * np.std(y, axis=0)/np.sqrt(len(ec_xf))
        y_err = np.array([y_average - y_ci, y_average + y_ci])
        sns.lineplot(x=ec_xf, y=y_average, color="tab:orange", label="eyes closed", ax=ax)
        ax.fill_between(ec_xf, y_err[0, :], y_err[1, :], color="tab:orange", alpha=0.25)
        
        y = eo_y[:, ch, :]
        y_average = np.mean(y, axis=0)  # average across epochs
        y_ci = 1.96 * np.std(y, axis=0)/np.sqrt(len(ec_xf))
        y_err = np.array([y_average - y_ci, y_average + y_ci])
        sns.lineplot(x=eo_xf, y=y_average, color="tab:blue", label="eyes open", ax=ax)  # err_style="band", err_kws = {'y1': y_err[0, :], 'y2': y_err[1, :]}, didnt work for some reason
        ax.fill_between(eo_xf, y_err[0, :], y_err[1, :], color="tab:blue", alpha=0.25)

        ax.set_xlim([0.5, 70])
        ax.set_title(ch_names[ch])
        ax.set_ylabel('Power spectral density ($\mathrm{\u03bcV}^2/\mathrm{Hz}$)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_yscale('log')

    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    for ch, ax in enumerate(axes.reshape(-1)):
        axinset = inset_locator.inset_axes(ax, width="30%", height="30%", loc=2)
        raw.copy().pick(ch_names[ch]).plot_sensors(title="", axes=axinset)
    plt.show(block=True)

    path = r"C:\Users\mcvai\forehead-eeg-device\circuit-characterization\rld behavior\CL 125k gel"
    filename = r"Wed Feb 07 111517 CST 2024 cl 125k gel eceo"
    raw, events, chdata_array, user_markers, unique_marker_labels = tat2mne(path, filename, plot=False, bad_chs=[])
    epochs = mne2eceo(raw, events, user_markers, unique_marker_labels, closed="ec", open="eo")