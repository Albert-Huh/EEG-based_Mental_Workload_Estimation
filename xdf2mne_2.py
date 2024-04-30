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