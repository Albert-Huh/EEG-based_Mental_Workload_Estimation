import matplotlib.pyplot as plt
import numpy as np
from ppgtools.sigpro import calc_FFT, filter_signal, moving_average

col_width = 16
 
'''/////////////////Plotting and Display////////////////////////'''

# =============================================================================
# Used to plot 3 channels of PPG signals
# =============================================================================
def plot_waveforms(distal_sig, proximal_sig, lf_sig = [], title = "", fs = 500, lfs = 100):
    fig = plt.figure()
    
    height = 0.4
    y_offset = 0.1
    if len(lf_sig) > 0:
        height = 0.3
        y_offset = 0.35
    
    ax1 = fig.add_axes([0.1, y_offset + height, 0.8, height])
    ax2 = fig.add_axes([0.1, y_offset, 0.8, height], sharex = ax1)
    
    t = np.linspace(0, len(distal_sig) / fs, len(distal_sig))
    ax1.set_title("Distal Waveform", position=(0.025, 0.05), horizontalalignment='left')
    ax1.set_yticks([])
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (s)")
    ax1.plot(t, distal_sig)

    t = np.linspace(0, len(proximal_sig) / fs, len(proximal_sig))
    ax2.set_title("Proximal Waveform",  position=(0.025, 0.05), horizontalalignment='left')
    ax2.set_yticks([])
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("Time (s)")
    ax2.plot(t, proximal_sig)

    if len(lf_sig) > 0:
        ax3 = fig.add_axes([0.1, 0.05, 0.8, height], sharex = ax1)
        ax3.set_title("ECG Signal", position=(0.025, 0.05), horizontalalignment='left')
        ax3.set_yticks([])
        ax3.set_ylabel("Amplitude")
        ax3.set_xlabel("Time (s)")
        t = np.linspace(0, len(distal_sig) / fs, len(lf_sig))
        ax3.plot(t, lf_sig)
        
    fig.suptitle(title)
    plt.show()

# =============================================================================
# Used to plot PTT, PP, and PWV
# =============================================================================
def plot_extracted_data(t_ptt, ptt, t_pp, pp, title = "", fs = 500, lfs = 100, ma = 1, fc_lpf = 0):
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])#[0.1, y_offset + height, 0.8, height])
    ax1.set_title("PTT", position=(0.025, 0.05), horizontalalignment='left')
    ax1.set_ylabel("PTT (ms)")
    ax1.set_xlabel("Time (s)")
    ax1.plot(t_ptt, ptt, label = 'Raw')
    
    # Do filtering of PTT
    ptt_fs = 1 / (t_ptt[1] - t_ptt[0])
    if ma > 1:
        ptt_ma = moving_average(ptt, ma * ptt_fs)
        print(ma * ptt_fs)
        ax1.plot(t_ptt, ptt_ma, label='Moving Average')
    if fc_lpf > 0:
        ptt_lpf = filter_signal(ptt, "lowpass", [fc_lpf], ptt_fs)
        ax1.plot(t_ptt, ptt_lpf, label='Low Pass Filter')
    ax1.legend()

        
# =============================================================================
# Plots the FFT of the signal    
# =============================================================================
def plot_FFT(sig, fs, positive_only = False, freq_range = []):
    freq, signal_fx = calc_FFT(sig, fs, positive_only)
        
    plt.figure()
    plt.title("FFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    if len(freq_range) == 2:
        plt.xlim(freq_range)
    
    plt.plot(freq, abs(signal_fx) **2)
    plt.show()
    