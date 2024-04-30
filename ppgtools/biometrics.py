from scipy import stats, interpolate, signal
import matplotlib.pyplot as plt
import numpy as np
import bisect
from ppgtools import biosignal, sigpeaks, sigplot, sigpro
import copy

col_width = 16

def average_metrics(time_and_metrics, average_type = 'mean', moving_filt = None, tfs = 1):
    time_and_metrics_interp = []
    N = 5
    
    if moving_filt == None:
        N = 0
    
    min_time = time_and_metrics[0][0][int(N/2)]
    max_time = time_and_metrics[0][0][-int(N/2)-1]

    for i in range(1, len(time_and_metrics)):
        if N >= len(time_and_metrics[i][0]):
            raise ValueError(f"N ({N}) is larger than the length of metric {i}.")

        min_time = np.max([time_and_metrics[i][0][int(N/2)], min_time])
        max_time = np.min([time_and_metrics[i][0][-int(N/2)-1], max_time])

    print(f"{min_time}, {max_time}")
    t = np.linspace(min_time, max_time, int((max_time-min_time) * tfs))
    #plt.figure()
    for i in range(0, len(time_and_metrics)):
        cur_timestamps = copy.deepcopy(time_and_metrics[i][0])
        cur_metrics = copy.deepcopy(time_and_metrics[i][1])
        
        #Preprocess data
        if moving_filt == None:
            pass
        elif moving_filt == 'mean':
            cur_timestamps = time_and_metrics[i][0][int(N/2):int(-N/2)]
            cur_metrics = sigpro.moving_average(time_and_metrics[i][1], N)
        elif moving_filt == 'median':
            cur_timestamps = time_and_metrics[i][0][int(N/2):int(-N/2)]
            cur_metrics = signal.medfilt(time_and_metrics[i][1], N)
        else:
            raise ValueError("Unknown moving filter type.")
        
        #Interpolate the time and metrics to a constant time rate.
        interp = interpolate.interp1d(cur_timestamps, cur_metrics)
            
        time_and_metrics_interp.append(interp(t))
        
        #plt.plot(t, interp(t))
    
    time_and_metrics_interp = np.array(time_and_metrics_interp)
    
    if average_type == 'mean':
        time_and_metrics_avg = np.mean(time_and_metrics_interp, axis = 0)
    elif average_type == 'median':
        time_and_metrics_avg = np.median(time_and_metrics_interp, axis = 0)
        
    return t, time_and_metrics_avg

def calc_respiratory_rate(sig, tfs = 1000, rr_alg = 'lowpass', peak_alg = 'adaptive'):
    '''
    Function to extract the respiratory rate from a PPG signal.

    Parameters
    ----------
    sig : BioSignal
        Signal to extract RR from.
    rr_alg : {'lowpass', 'envelope'}, optional
        Type of algorithm to extract RR. Lowpass looks at the low frequency.
        Envelope looks at the envelop of the signal. The default is 'lowpass'.
    peak_alg : {'adaptive', 'dist'}, optional
        Type of peak detection algorithm to determine the respiration peaks.
        The default is 'adaptive'.

    Raises
    ------
    ValueError
        An error will be raised if an unknown algorithms is provided.

    Returns
    -------
    peaks : array_like
        Timestamps (indices) of the breaths.
    bpm : array_like
        Extracted RR in breaths per minute.

    '''
    ref_sig = copy.deepcopy(sig)
            
    if rr_alg == 'lowpass':
        #Filter the signal to a lower band
        ref_sig.filter_signal("bandpass", [0.1, 0.3])
    elif rr_alg == 'envelope':
        #Currently, I just use the dist to get the peaks for the envelope
        ref_sig.filter_signal("bandpass", [0.3, 8])
        peaks = ref_sig.find_peaks_distance(0.7)
        interp = interpolate.interp1d(peaks, ref_sig.data[peaks])
        ref_sig.data = interp(np.linspace(peaks[0], peaks[-1], len(peaks)))
    else:
        raise ValueError("Unknown respiratory rate algorithm.")
        
        
    if peak_alg == 'adaptive':
        peaks = ref_sig.find_mins_and_peaks_adaptive_threshold(0, fc = [0.1, 0.3], plot = False)[0]
    elif peak_alg == 'dist':
        peaks = ref_sig.find_peaks_distance(0.08)
    else:
        raise ValueError("Unknown peak detection algorithm")

    ibi = np.diff(peaks) / ref_sig.fs
    bpm = 60 / ibi
    peaks = peaks[0:-1]
        
    return peaks, bpm

def calc_heart_rate(inp, tfs = 1000, peak_alg = 'adaptive'):
    '''
    Function to calculate the heartrate from a PPG signal.

    Parameters
    ----------
    inp : BioSignal
        Signal to calculate HR from.
    tfs : int, optional
        Target sample rate. The default is 1000 Hz.
    peak_alg : {'adaptive', 'pan-tomp', 'dist'}, optional
        What algorithm to use for the peak detection. The default is an adaptive
        threshold.

    Returns
    -------
    peaks : array_like
        Indices of where the HR was calculated from.
    bpm : array_like
        Interbeat intervals in BPM.

    '''
    sig = copy.deepcopy(inp)
    
    #Resample to a higher rate to get better temporal resolution
    sig.resample(tfs)
    
    #Normalize the amplitude
    sig.data = sig.data / max(sig.data)
    
    #Peak enhancement
    #sig.data = sig.data
    
    #Find the peaks
    if peak_alg == 'adaptive':
        peaks = sig.find_mins_and_peaks_adaptive_threshold(1.5, fc = [1, 3], plot = True)[0]
    elif peak_alg == 'pan-tomp':
        peaks = sig.find_peaks_pan_tompkins()
    elif peak_alg == 'dist':
        peaks = sig.find_peaks_distance(0.7)
    else:
        raise ValueError("Unknown algorithm")
                
    ibi = np.diff(peaks) / sig.fs
    bpm = 60 / ibi
    peaks = peaks[0:-1]
    
    return peaks, bpm

def calc_spo2(red, ir, peak_alg = 'adaptive'):
    '''
    Function to calculate the SpO2 from red and IR channels of signal.

    Parameters
    ----------
    red : BioSignal
        BioSignal containg the red PPG channel.
    ir : BioSignal
        BioSignal containg the ir PPG channel.
    peak_alg : {'adaptive', 'dist'}, optional
        What algorithm to use for the peak detection. The default is an adaptive
        threshold.
        
    Returns
    -------
    spo2 : array_like
        Time series of the calculated SpO2. The first dimension contains
        timestamps. The second dimension contains the SpO2 data. The third
        dimension contains the raw R ratios.

    '''
    print(f"Red signal: \"{red.name}\"")
    print(f"IR signal: \"{ir.name}\"")
    #First, find the peaks and minimums. We need a filtered version of the signals.
    
    
    #Find the peaks
    if peak_alg == 'adaptive':
        red_mins_peaks = red.find_mins_and_peaks_adaptive_threshold(0.9, fc = [0.5, 4])
        ir_mins_peaks = ir.find_mins_and_peaks_adaptive_threshold(0.9, fc = [0.5, 4])  
        
        ir.plot(peaks = np.concatenate((ir_mins_peaks[0], ir_mins_peaks[1])))
        red.plot(peaks = np.concatenate((red_mins_peaks[0], red_mins_peaks[1])))
    elif peak_alg == 'dist':
        red_filt = copy.deepcopy(red)
        ir_filt = copy.deepcopy(ir)
        
        red_filt.filter_signal("bandpass", [1.5, 8])
        ir_filt.filter_signal("bandpass", [1.5, 8])
    
        red_peaks = red_filt.find_peaks_distance(0.5)
        ir_peaks = ir_filt.find_peaks_distance(0.5)
        red_mins = sigpeaks.find_feet_local_min(red_filt.data, red_peaks)
        ir_mins = sigpeaks.find_feet_local_min(ir_filt.data, ir_peaks)
        red_mins_peaks = [red_mins, red_peaks]
        ir_mins_peaks = [ir_mins, ir_peaks]
        
        ir_filt.plot(peaks = np.concatenate((ir_peaks, ir_mins)))
        red_filt.plot(peaks = np.concatenate((ir_peaks, ir_mins)))
        
    else:
        raise ValueError("Unknown algorithm")
    
    
    #For each channel, find the peak-to-peak amplitude.
    red_pp = red.data[red_mins_peaks[1]] - red.data[red_mins_peaks[0]]
    ir_pp = ir.data[ir_mins_peaks[1]] - ir.data[ir_mins_peaks[0]]
    
    #Divide each peak-to-peak amplitude by the DC signal amplitude. Here, the
    #DC amplitude is the average between the peak and min.
    red_dc = (red.data[red_mins_peaks[1]] + red.data[red_mins_peaks[0]]) / 2
    ir_dc = (ir.data[ir_mins_peaks[1]] + ir.data[ir_mins_peaks[0]]) / 2
    
    #Align the peaks and data
    red_indices, ir_indices = sigpeaks.align_peaks(red_mins_peaks[0], ir_mins_peaks[0])
    
    '''
    plt.figure()
    sigplot.plot_biosignals([red_filt, ir_filt], points_of_interest = [np.array(red_mins_peaks[1])[red_indices], np.array(ir_mins_peaks[1])[ir_indices]])
    '''
    
    #Calculate R
    R = (red_pp[red_indices] / red_dc[red_indices]) / (ir_pp[ir_indices] / ir_dc[ir_indices])
    
    #Using the MAX86141 calibration curve
    #https://www.maximintegrated.com/en/design/technical-documents/app-notes/6/6845.html
    spo2 = -16.666666 * R * R + 8.333333 * R + 100
    
    return np.array(ir_mins_peaks[0])[ir_indices], spo2, R
    
'''////////////////// Blood Pressure High Level Functions///////////'''
#Outputs PWV^2 * delta blood volume
def pivot_btb(t_amps, amps, t_pwv, pwv):
    out = []
    
    for i in range (0, len(t_amps)):
        upper_time = bisect.bisect_left(t_pwv, t_amps[i])
        
        if upper_time >= len(t_pwv):
            upper_time = len(t_pwv) - 1
        
        if upper_time < len(pwv):
            #Linearly interpolate to find the exact PWV value at the pulse time
            a = (pwv[upper_time] - pwv[upper_time - 1]) / (t_pwv[upper_time] - t_pwv[upper_time - 1])
            new_pwv = pwv[upper_time - 1] + a * (t_amps[i] - t_pwv[upper_time-1])
            out.append(new_pwv**2 * amps[i]**1.5)
        
    return out

def compare_metrics(time_and_metrics, align_to = 0):
    #Select a "gold standard" that the other metrics should be aligned to.
    standard = time_and_metrics[align_to]
    
    if len(standard) == 0:
        return None, None
    
    aligned_metrics = np.zeros((len(time_and_metrics), len(standard[1])))
    print(aligned_metrics.shape)
    
    #Iterate for every timestamp in the gold standard
    #for t in time_and_metrics[align_to][0]:
    for i in range (len(time_and_metrics[align_to][0])):
        t = time_and_metrics[align_to][0][i]
        #print(f"t: {t}")
        #Iterate for every other metric
        #for metric in time_and_metrics:        
        for j in range(len(time_and_metrics)):
            metric = time_and_metrics[j]
            if len(metric[0] != 0):
                #Find nearest timestamp in that metric
                t2 = np.asarray(metric[0])
                idx = (np.abs(t2 - t)).argmin()
                aligned_metrics[j][i] = metric[1][idx]
                #print(f"\t{j}: {metric[1][idx]}")
        
    aligned_t = time_and_metrics[align_to][0]   
    return aligned_t, aligned_metrics
    


# =============================================================================
# Function to compare data obtained from two different sources
# I.e., blood pressure from finapres and e-tattoo   
# =============================================================================
def compare_data(t_fina, dat_fina, t_ppg, dat_ppg, fina_threshold, fina_offset = 0):
    print("Comparing Finapres and PPG data...")
    aligned_dat_ppg = []
    aligned_dat_fina = []
    aligned_t = []
    
    for i in range (0, len(t_fina)):
        if(dat_fina[i] > fina_threshold and i > 10):
            aligned_dat_fina.append(dat_fina[i])
            aligned_t.append(t_fina[i])
            upper_time = bisect.bisect_left(t_ppg, t_fina[i])
            
            if upper_time >= len(t_ppg):
                upper_time = upper_time - len(t_ppg)
            
            #Check which PPG BP is closer in time
            if t_ppg[upper_time] - t_fina[i] > t_fina[i] - t_ppg[upper_time-1]:
                aligned_dat_ppg.append(dat_ppg[upper_time - 1])
            else:
                aligned_dat_ppg.append(dat_ppg[upper_time])

    print(len(aligned_dat_fina))
    print(len(aligned_dat_ppg))
    print(len(aligned_t))
    
    if fina_offset > 0:
        aligned_dat_fina = aligned_dat_fina[fina_offset:-1]
        aligned_dat_ppg = aligned_dat_ppg[0:len(aligned_dat_ppg)-fina_offset-1]
        aligned_t = aligned_t[0:len(aligned_t)-fina_offset-1]
    elif fina_offset < 0:
        aligned_dat_fina = aligned_dat_fina[0:len(aligned_dat_fina)+fina_offset-1]
        aligned_dat_ppg = aligned_dat_ppg[-fina_offset:-1]
        aligned_t = aligned_t[0:len(aligned_t)+fina_offset-1]    
    
    r = stats.pearsonr(aligned_dat_fina, aligned_dat_ppg)      
    print("Pearson's r: " + str(r))
    m, b = np.polyfit(aligned_dat_fina, aligned_dat_ppg, 1)
    print("Line of best fit: y = " + str(m) + "x + " + str(b))

    aligned_dat_ppg = aligned_dat_ppg / m - b/m
            
    fig = plt.figure(figsize=(12.8,4.8))
    ax1 = fig.add_axes([0.2, 0.2, 0.3, 0.7])
    ax2 = fig.add_axes([0.6, 0.2, 0.3, 0.7], sharey = ax1)
    
    ax1.plot(aligned_dat_fina, aligned_dat_ppg, "ko")
    #plt.plot(aligned_dat_fina, m*np.array(aligned_dat_fina) + b)
    ax1.plot(aligned_dat_fina, aligned_dat_fina)
    ax1.set_xlabel("Finapres BP (mm Hg)")
    ax1.set_ylabel("E-Tattoo (mm Hg)")
    
    ax2.scatter(aligned_t, aligned_dat_fina, label = "Finapres")
    ax2.scatter(aligned_t, aligned_dat_ppg, label = "E-Tattoo")
    ax2.legend()
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Pressure (mm Hg)")
    plt.show()      
