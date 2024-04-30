'''
BioSignal Class
'''
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

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
        
    def getEndian(self):
        if self.little_endian:
            return 'little'
        return 'big'

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
        output : TYPE
            DESCRIPTION.

        '''
        if(aa):
            self.filter_signal('lowpass', [tfs/2], self.fs)
            
        output = signal.resample(self.data, int(self.fs / tfs) * len(self.data))
        
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
        new_data : array_like
            The data after the change.

        '''
        
        #Filter the data to half the target sampling rate
        if(aa):
            self.filter_signal('lowpass', [tfs/2], self.fs)
        
        
        modulus = int(self.fs / tfs)
        output = []
        for i in range (0, len(self.data)):
            if(i % modulus == 0):
                output.append(self.data[i])
                #output.append(4)
        
        output.append(self.data[len(self.data)-1])
        output = np.asarray(output)
        
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
        new_data : array_like
            The data after the change.

        '''
    
        output = []
        
        for i in range (0, len(self.data)):
            output.append(self.data[i])
            for j in range(0, num_padding):
                output.append(0)
        
        output = np.asarray(output)
        
        if change_self:
            self.data = output
            output = self.filter_signal('lowpass', [self.fs/2], self.fs, change_self = change_self)
            self.fs = self.fs * (1 + num_padding)
        else:
            #This is so bad
            temp = self.data
            self.data = output
            output = self.filter_signal('lowpass', [self.fs/2], self.fs, change_self = change_self)
            self.data = temp
            
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
        new_data : array_like
            The data after the change.

        '''
        mean = np.mean(self.data)
        
        if change_self:
            self.data -= mean
            return self.data
        else:
            return self.data - mean
    
    
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
        new_data : array_like
            The data after the change.

        '''
        w = [float(i) / (float(self.fs) / 2) for i in fc] # Normalize the frequency
        
        if filter_design == 'butter': 
            b, a = signal.butter(order, w, filter_pass)
        elif filter_design == 'bessel':
            b, a = signal.bessel(order, w, filter_pass)
        elif filter_design == 'chebyii':
            b, a = signal.cheby2(order, 6, w, filter_pass)
        elif filter_design == 'fir':
            b = signal.filtwin(order, w, pass_zero = False)
            a = 1
        else:
            print(str(filter_design) + " not recognized")
            raise ValueError
        
        output = np.zeros(len(self.data))
        if zero_phase:
            output = signal.filtfilt(b, a, self.data)
        else:
            output = signal.lfilter(b, a, self.data)
                    
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
        
        #I should switch to scipy.fft
        signal_fx = np.fft.fft(self.data)
        freq = np.fft.fftfreq(len(signal_fx), d = 1/self.fs)
                
        #Truncate to positive frequencies only
        if(positive_only):
            n = int(len(self.data) / 2)
            signal_fx = signal_fx[1:n]
            freq = freq[1:n]
            
        if(plot):
            plt.figure()
            plt.title("FFT of " + self.name)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.plot(freq, abs(signal_fx) **2)
            plt.show()
        
        return freq, signal_fx
        

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
        new_data : array_like
            The data after the change.

        '''
        lower_bound = int(lower_bound * self.fs)
        upper_bound = int(upper_bound * self.fs)        
        
        if upper_bound <= lower_bound or upper_bound >= len(self.data):
            upper_bound = len(signal)
        
        if change_self: 
            self.data = self.data[lower_bound : upper_bound]
            return self.data
        else:
            return self.data[lower_bound : upper_bound]
        
    def plot(self):
        plt.figure()
        t = np.linspace(0, len(self.data) / self.fs, len(self.data))
        plt.plot(t, self.data)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [a.u.]')
        plt.yticks([])
        plt.title(self.name)
        
    def convert_Si7050(self):
        self.data = self.data * 175.72 / 65536 - 46.85

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
        new_data : array_like
            The data after the change.

        '''
        
        delay_in_samples = int(delay_amount * self.fs)
        out = np.zeros(len(self.data) - abs(delay_in_samples))
        
        if delay_in_samples >= 0:
            out = self.data[delay_in_samples:]
        else:
            out = self.data[0:delay_in_samples]
        
        if(change_self):
            self.data = out
        
        return out
    
    def invert(self, change_self = True):
        '''
        Function to flip the waveform

        Parameters
        ----------
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        new_data : array_like
            The data after the change.

        '''
        
        out = -self.data
        
        if(change_self):
            self.data = out
            
        return out