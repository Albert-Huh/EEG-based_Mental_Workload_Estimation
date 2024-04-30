'''
BioSignal Class
'''
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import copy

from ppgtools import sigpro, sigpeaks

class BioSignal:   
    def __init__(self, index, name, bytes_per_point, fs, bit_resolution, signed, little_endian):
        self.index = index
        self.name = name
        self.bytes_per_point = bytes_per_point
        self.fs = fs
        self.bit_resolution = bit_resolution
        self.signed = signed
        self.little_endian = little_endian
        self.data = np.zeros(0)
        self.units = "a.u."
        
    def getEndian(self):
        if self.little_endian:
            return 'little'
        return 'big'

    def pad_disconnect_events(self, markers, method = "hold"):
        '''
        Function to pad the data when the tattoo disconnected

        Parameters
        ----------
        markers : array_like
            Event markers containing disconnect and reconnect events.
        method : {zero, hold}, optional
            What the padded data should be. Zero fills with zeros, hold fills with last valid data.
        Returns
        -------
        markers_updated : array_like
            Event markers with updated timestamps.

        '''
        valid = {'zero', 'hold'}
        if method not in valid:
            raise ValueError("Method must be one of %r."%valid)
        
        disconnect_time_data   = 0  #Timestamp of the disconnect, in the data
        last_disconnect        = 0  #Timestamp of the disconnect, in unix (phone)
        total_downtime         = 0  #Total amount of downtime, in seconds
        
        markers_updated = copy.deepcopy(markers)
        
        for e in markers_updated:
            if "reconnected" in e.label:
                #Formatted in hr:mn:sc
                reconnect_time = e.label[-20:-9]
                days = int(reconnect_time[0:2]) - int(last_disconnect[0:2])
                hours = int(reconnect_time[3:5]) - int(last_disconnect[3:5])
                mins = int(reconnect_time[6:8]) - int(last_disconnect[6:8])
                secs = int(reconnect_time[9:]) - int(last_disconnect[9:])
                
                downtime_s = 86400 * days + hours * 3600 + mins * 60 + secs
                total_downtime += downtime_s
                
                #Pad the signal.
                pad = np.zeros(int(downtime_s * self.fs))
                pad -= 1000
                                
                if method == 'hold':
                    last_valid = self.data[int(disconnect_time_data * self.fs)]
                    pad += last_valid
                
                self.data = np.insert(self.data, int(disconnect_time_data * self.fs), pad)
                
                
            elif "disconnected" in e.label:
                last_disconnect = e.label[-20:-9]
                disconnect_time_data = e.t + total_downtime
                
            e.t += total_downtime
        
        return markers_updated
    
    
    def pad_disconnect_events_new(self, markers, method = "hold"):
        '''
        Function to pad the data when the tattoo disconnected

        Parameters
        ----------
        markers : array_like
            Event markers containing disconnect and reconnect events.
        method : {zero, hold}, optional
            What the padded data should be. Zero fills with zeros, hold fills with last valid data.
        Returns
        -------
        markers_updated : array_like
            Event markers with updated timestamps.

        '''
        valid = {'zero', 'hold'}
        if method not in valid:
            raise ValueError("Method must be one of %r."%valid)
        
        disconnect_time_data   = 0  #Timestamp of the disconnect, in the data
        last_disconnect        = 0  #Timestamp of the disconnect, in unix (phone)
        total_downtime         = 0  #Total amount of downtime, in seconds
        
        markers_updated = copy.deepcopy(markers)
        
        for e in markers_updated:
            if "Packet interval" in e.label:
                #Formatted in hr:mn:sc
                reconnect_time = e.label[17:]

                downtime_ms = float(reconnect_time)
                total_downtime += downtime_ms/1000
                
                #Pad the signal.
                pad = np.zeros(int((downtime_ms * self.fs) /1000))
                # pad -= 1000
                                
                if method == 'hold':
                    last_valid = self.data[int(disconnect_time_data * self.fs)]
                    pad += last_valid
                
                self.data = np.insert(self.data, int(disconnect_time_data * self.fs), pad)
                
                
            elif "disconnected" in e.label:
                last_disconnect = e.label[-20:-9]
                disconnect_time_data = e.t + total_downtime
                
            e.t += total_downtime
        
        return markers_updated
    
    
    def resample(self, tfs, aa = False, change_self = True):
        '''
        Resample the data to a new sampling rate using Fourier method.
        Recommend tfs being an integer factor of fs.

        Parameters
        ----------
        tfs : int
            Target sample rate.
        aa : bool, optional
            Set to true if an anti-aliasing filter should be used before resampling. The default is False.
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        output = sigpro.resample(self.data, self.fs, tfs, aa)
        
        if change_self:
            self.data = output
            self.fs = tfs
        
        return output

    def downsample(self, tfs, aa = False, change_self = True):
        '''
        Downsamples the data to a new sampling rate by decimation.
        Recomend the target sampling rate to be an integer factor of the original.

        Parameters
        ----------
        tfs : int
            Target sample rate in Hz.
        aa : bool, optional
            Set to true if an anti-aliasing filter should be used before downsampling. The default is False.
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        
        output = sigpro.downsample(self.data, self.fs, tfs, aa)
        
        if change_self:
            self.data = output
            self.fs = tfs
            
        return output
        
    
    def upsample(self, num_padding, change_self = True):
        '''
        Helper function that will upsample the data to a higher sampling rate.

        Parameters
        ----------
        num_padding : int
            How many zeros to put between each data point.
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
    
        output = sigpro.upsample(self.data, self.fs, num_padding)
        
        if change_self:
            self.data = output
            self.fs = self.fs * (1 + num_padding)
            
        return output        
    

    def remove_mean(self, change_self = True):
        '''
        Removes the mean from the data i.e., make the DC offset 0.
        
        Parameters
        ----------
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        output = sigpro.remove_mean(self.data)
        
        if change_self:
            self.data = output
        
        return output
    
    def filter_signal(self, filter_pass, fc, order = 4, filter_design = "butter", zero_phase = True, change_self = True):  
        '''
        Filters the signal

        Parameters
        ----------
        filter_pass : {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
            Filter type
        fc : array_like
            Critical freqencies.
        order : int, optional
            The order of the filter. The default is 4.
        filter_design : {'butter', 'bessel', 'chebyii', 'fir'}, optional
            How the filter coefficients are calculated. The default is "butter".
        zero_phase : bool, optional
            If true, uses filtfilt. Otherwise, uses lfilter. The default is True.
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.
            
        Raises
        ------
        ValueError
            If a filter_design is not recognized, it will throw this error.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        output = sigpro.filter_signal(self.data, self.fs, filter_pass, fc, order, filter_design, zero_phase)
                    
        if change_self:
            self.data = output
        
        return output    

    def calc_fft(self, positive_only = False, plot = False):
        '''
        Calculates and returns the FFT of the data.

        Parameters
        ----------
        positive_only : bool, optional
            If true, only the positive frequencies of the FFT will be returned. The default is False.
        plt : bool, optional
            If true, the FTT will be plotted. The default is False.
        
        Returns
        -------
        freq : array_like
            The DST sample frequencies.
        signal_fx : array_like
            The FFT complex amplitude values.

        '''
        
        return sigpro.calc_fft(self.data, self.fs, positive_only, plot)
        

    def trim_signal(self, lower_bound = 0, upper_bound = 0, change_self = True):
        '''
        Truncates the signal.

        Parameters
        ----------
        lower_bound : float, optional
            Where to start the signal, in seconds. The default is 0.
        upper_bound : float, optional
            Where to cut off the signal (original data), in seconds. The default is 0.
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.
        
        Returns
        -------
        output : array_like
            The data after the change.

        '''
        output = sigpro.trim_signal(self.data, self.fs, lower_bound, upper_bound)
        
        if change_self:
            self.data = output
        
        return output
        
    def plot(self, markers = [], peaks = [], **kwargs):
        plt.figure()
        t = np.linspace(0, len(self.data) / self.fs, len(self.data))
        plt.plot(t, self.data, **kwargs)
        
        if len(peaks) > 0:
            plt.plot(t[peaks], self.data[peaks], 'r.')
        
        plt.xlabel('Time [s]')
        plt.ylabel(self.units)
        
        
        y_point = 35#min(self.data)
        for e in markers:
            plt.annotate(e.label, [e.t, y_point], rotation = 'vertical', size = 12)
            plt.axvline(e.t, c = 'red')        
        
        plt.title(self.name)
        
    def plot_stft(self, win_len = 1):
        x = self.data
        nperseg = win_len * self.fs
        
        #plt.figure()
        f, t, Zxx = signal.stft(x, self.fs, nperseg=nperseg)
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.ylim([0, 100])
        plt.show()
        
    def convert_Si7050(self):
        self.data = self.data * 175.72 / 65536 - 46.85
        self.units = 'Temperature [C]'
        
    def convert_TMP117(self):
        self.data = self.data * 7.8125 / 1000
        self.units = 'Temperature [C]'

    def shift(self, delay_amount, change_self = True):
        '''
        Function to shift the signal in time.

        Parameters
        ----------
        delay_amount : float
            The amount to delay the signal, in seconds. A positive amount will move up the signal, and a negative amount will truncate the end of the signal.
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        
        output = sigpro.shift(self.data, self.fs, delay_amount)
        
        if change_self:
            self.data = output
            
        return output
    
    def invert(self, change_self = True):
        '''
        Function to flip the waveform

        Parameters
        ----------
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        
        output = sigpro.invert(self.data)
        
        if change_self:
            self.data = output
            
        return output
    
    def find_mins_and_peaks_adaptive_threshold(self, window_size, fc = None, plot = False):
        '''
        Function to determine the location of peaks using an adaptive threshold.
        The adaptive threshold is obtained from the moving average of the original signal.
        
        NOTE: This takes the window size in seconds, as opposed to samples.
    
        Parameters
        ----------
        inp : array_like
            Signal to find peaks in.
        window_size : int
            Moving average size in seconds.
        fc : float, optional
        Cutoff frequencies for interesection calculation. The default is no filtering.
    
        Returns
        -------
        mins : array_like
            Indices of the estimated peaks.
        peaks : array_like
            Indices of the estimated peaks.
        plot : bool, optional
            Whether to plot the adaptive threshold or not. The default is false.
        '''
        
        return sigpeaks.find_mins_and_peaks_adaptive_threshold(self.data, self.fs, int(window_size * self.fs), fc, plot)
    
    def find_peaks_pan_tompkins(self):
        '''
        Function to determine the location of peaks using the Pan-Tompkins method.

        Returns
        -------
        peaks : array_like
            Indices of the estimated peaks.

        '''
        peaks = sigpeaks.find_peaks_pan_tompkins(self.data, self.fs)
        
        return peaks
    
    def find_peaks_distance(self, distance):
        '''
        Simple peak detection using scipy.signal.find_peaks.
        The minimum distance between peaks must be supplied.
    
        Parameters
        ----------
        inp : array_like
            Signal to find peaks.
        distance : float
            Time in seconds that each peak must be seperated by.
    
        Returns
        -------
        peaks : array_like
            Index locations of the peaks.
    
        '''
        peaks = sigpeaks.find_peaks_distance(self.data, distance = int(self.fs * distance))
        
        return peaks
    
    def get_peak_frequency(self, fc = [], padlen = 0):
        '''
        Function to get the peak frequency of the signal.

        Parameters
        ----------
        fc : array_like, optional
            If filtering is desired, use this as the cutoff frequencies. 
            Both the lower cutoff and higher cutoff need to be used.
            The default is [].
        padlen : int, optional
            How many zeros to pad the FFT by.

        Returns
        -------
        Peak frequency in Hz.

        '''
        return sigpeaks.get_peak_frequency(self.data, self.fs, fc, padlen)
        
    def find_feet_intersecting_tangents(self, peaks = []):
        if len(peaks) == 0:
            peaks = self.find_peaks_adaptive_threshold(0.5)
        return sigpeaks.find_feet_intersecting_tangents(self.data, peaks)
    
    def find_feet_sdm(self, peaks = []):
        if len(peaks) == 0:
            peaks = self.find_peaks_adaptive_threshold(0.5)
        return sigpeaks.find_feet_sdm(self.data, peaks)
    
    def sync(self, markers):
        last_sync_time = 0
        last_sync_time_data = 0
        sync_data = np.array([])
        
        for m in markers:
            if "Sync time: " in m.label:
                sync_time_data = m.t
                sync_time = float(''.join(c for c in m.label if c.isdigit()))/1000   
                sync_interval = sync_time - last_sync_time
                expected_data_length = sync_time * self.fs
                points_to_add = int(expected_data_length - len(sync_data))
                
                
                start = int(last_sync_time_data * self.fs)
                end = int(sync_time_data * self.fs)
                sublist = self.data[start:end]
                #print(str(m.t) + " " + str(sync_time))
                print(str(points_to_add) + " " + str(len(sublist)) + " " + str(start) + " " + str(end))
                print(len(self.data))
                sublist = signal.resample(sublist, points_to_add)
            
                
                sync_data = np.hstack((sync_data, sublist))
                
                m.t = sync_time
                last_sync_time = sync_time
                last_sync_time_data = sync_time_data
            
        self.data = sync_data