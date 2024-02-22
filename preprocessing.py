import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
matplotlib.use('TkAgg')

class Filtering:
    def __init__(self, raw, l_freq, h_freq, nf_freqs=(60,120), picks='eeg'):
        # filter parameters
        self.raw = raw.copy()
        self.sfreq = raw.info['sfreq']
        self.picks = picks
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.nf_freqs = nf_freqs

    def highpass(self, cutoff_freq=None, filt_type='fir', iir_params=None,
        show_filt_params = False, verbose='warning'):
        
        if cutoff_freq == None:
            cutoff_freq = self.l_freq
        picks=self.picks
        filt_raw = self.raw.filter(
            l_freq=cutoff_freq, h_freq=None, picks=picks, method=filt_type,
            iir_params=iir_params, verbose=verbose)
        if show_filt_params == True:
            # check the filter parameter
            filter_params = mne.filter.create_filter(
                data=self.raw.get_data(), sfreq=self.sfreq, l_freq=cutoff_freq, 
                h_freq=None, method=filt_type, iir_params=iir_params)
            mne.viz.plot_filter(h=filter_params, sfreq=self.sfreq, flim=(0.01, self.sfreq/2))
        return filt_raw

    def lowpass(self, cutoff_freq=None, filt_type='fir', iir_params=None, 
        show_filt_params = False, verbose='warning'):
        
        if cutoff_freq == None:
            cutoff_freq = self.h_freq
        picks=self.picks
        filt_raw = self.raw.filter(
            l_freq=None, h_freq=cutoff_freq, picks=picks, method=filt_type,
            iir_params=iir_params, verbose=verbose)
        if show_filt_params == True:
            # check the filter parameter
            filter_params = mne.filter.create_filter(
                data=self.raw.get_data(), sfreq=self.sfreq, l_freq=cutoff_freq, 
                h_freq=None, method=filt_type, iir_params=iir_params)
            mne.viz.plot_filter(h=filter_params, sfreq=self.sfreq, flim=(0.01, self.sfreq/2))
        return filt_raw

    def notch(self, target_freqs=None, notch_wdith=None, filt_type='fir', 
        iir_params=None, verbose='warning'):
        if target_freqs == None:
            target_freqs = self.nf_freqs
        picks=self.picks
        filt_raw = self.raw.notch_filter(
            freqs=target_freqs, picks=picks,
            method=filt_type, iir_params=iir_params, verbose=verbose)
        return filt_raw

    def resample(self,new_sfreq=None, window='boxcar', events=None, verbose='warning'):
        
        if new_sfreq == None:
            new_sfreq = 4*self.h_freqeq
        if new_sfreq > self.sfreq:
            print('Error: New sampling frequency is greather than previous sampling freqeuncy. (Recommended new_f_s = 4 x high_f_c)')
            return
        filt_raw = self.raw.resample(sfreq=new_sfreq,
            npad='auto', window=window, events=events, verbose=verbose)
        return filt_raw

    def bandpass(self, cutoff_freq=None, filt_type='fir', iir_params=None,
        show_filt_params = False, verbose='warning'):

        picks=self.picks
        if cutoff_freq == None:
            cutoff_freq = [self.l_freq, self.h_freq]
        filt_raw = self.raw.filter(l_freq=cutoff_freq[0], h_freq=cutoff_freq[1], picks=picks, method=filt_type, 
                                   iir_params=iir_params, verbose=verbose)
        if show_filt_params == True:
            # check the filter parameter
            filter_params = mne.filter.create_filter(
                data=self.raw.get_data(), sfreq=self.sfreq, l_freq=cutoff_freq[0], 
                h_freq=cutoff_freq[1], method=filt_type, iir_params=iir_params)
            mne.viz.plot_filter(h=filter_params, sfreq=self.sfreq, flim=(0.01, self.sfreq/2))
        return filt_raw

    def external_artifact_rejection(self, resample=False, notch=False):
        self.raw.load_data()
        if resample:
            self.raw = self.resample()
        if notch:
            self.raw = self.notch()
        filt_raw = self.bandpass()
        return filt_raw
    
class Indepndent_Component_Analysis:
    def __init__(self,raw, n_components=None,seed=random.randint(1,100), max_iter='auto', 
        find_eog_peaks=True,find_ecg_peaks=False):
        
        self.raw = raw
        self.n_components = raw.info['nchan']-1 if n_components==None else n_components
        self.random_state = seed
        self.max_iter = max_iter
        self.find_eog_peaks = find_eog_peaks
        self.find_ecg_peaks = find_ecg_peaks    
        self.ica = mne.preprocessing.ICA(n_components=n_components, random_state=seed, 
            max_iter= max_iter)
        self.eog_evoked = None
        self.ecg_evoked = None

    def setup_ICA(self):

        self.ica.fit(self.raw) # whitening and n-comps ICA performed
        
    def visualize_ICA_components(self):
        
        self.ica.plot_components()
        plt.show()
        self.ica.plot_sources(self.raw)
        # col = self.n_components if self.n_components < 4 else 4
        # row = 1 if self.n_components < 4 else self.n_components // 4 +1
        # fig, axis = plt.subplots(row, col)
        ica_picks = list(np.arange(0,self.n_components,1))
        self.ica.plot_properties(self.raw, picks=ica_picks)

    def exclude_ica(self, list_exclude=[]):

        self.ica.exclude = list_exclude
    
    def create_physiological_evoked(self, baseline=(-0.5, -0.2), eog_ch='EOG',verbose='warning'):

        if self.find_eog_peaks == True:
            eog_evoked = mne.preprocessing.create_eog_epochs(self.raw, ch_name=eog_ch, picks='eeg', 
            baseline=baseline, verbose=verbose).average()
            self.eog_evoked = eog_evoked
        if self.find_ecg_peaks == True:
            ecg_evoked = mne.preprocessing.create_ecg_epochs(self.raw, picks='eeg', 
            baseline=baseline, verbose=verbose).average()
            self.ecg_evoked = ecg_evoked
        return eog_evoked if 'eog_evoked' in locals() else None, ecg_evoked if 'ecg_evoked' in locals() else None

    def visualize_physiological_evoked(self, evoked_epoch):

        evoked_epoch.plot_image(combine='mean')
        evoked_epoch.average().plot_joint()
        evoked_epoch.average().plot_image()

    def find_physiological_artifacts(self, eog_treshold='auto', ecg_treshold='auto', 
        reject_by_annotation=True, measure='correlation', plot_fig=True, verbose='warning'):

        if self.find_eog_peaks == True:
            for file_name in self.raw.ch_names:
                eog_ch = None
                if file_name.endswith('EOG'):
                    eog_ch = file_name
                    if eog_ch.startswith('h'): # replace 'hEOG' with 'vEOG' or 'EOG'
                        eog_ch = file_name
                else:
                    eog_ch = None
            eog_indices, eog_scores = self.ica.find_bads_eog(self.raw, ch_name=eog_ch, 
                threshold=eog_treshold, start=None, stop=None, l_freq=1, h_freq=10, 
                reject_by_annotation=reject_by_annotation, measure=measure, verbose=verbose)
            print(eog_indices, eog_scores)
            if plot_fig == True:
                # barplot of ICA component "EOG match" scores
                self.ica.plot_scores(eog_scores)
                # plot diagnostics
                if len(eog_indices) != 0:
                    self.ica.plot_properties(self.raw, picks=eog_indices)
                # plot ICs applied to raw data, with EOG matches highlighted
                self.ica.plot_sources(self.raw)
                plt.show()
                # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
                self.ica.plot_sources(self.eog_evoked)
                plt.show()
        else:
            print('Warning: EOG indices were not found.')
        if self.find_ecg_peaks == True:
            ecg_indices, ecg_scores = self.ica.find_bads_ecg(self.raw, ch_name=None, 
            threshold=ecg_treshold, start=None, stop=None, l_freq=8, h_freq=16, method='ctps', 
            reject_by_annotation=True, measure=measure, verbose=verbose)
            if plot_fig == True:
                # barplot of ICA component "ECG match" scores
                self.ica.plot_scores(ecg_scores)
                # plot diagnostics
                self.ica.plot_properties(self.raw, picks=ecg_indices)
                # plot ICs applied to raw data, with ECG matches highlighted
                self.ica.plot_sources(self.raw)
                plt.show()
        else:
            print('Warning: ECG indices were not found.')
        return eog_indices if 'eog_indices' in locals() else [], eog_scores if 'eog_scores' in locals() else [], ecg_indices if 'ecg_indices' in locals() else [], ecg_scores if 'ecg_scores' in locals() else []

    def perfrom_ICA(self):
        eog_indices, ecg_indices = [], []
        self.setup_ICA()
        # self.visualize_ICA_components() # Comment while debugging
        eog_indices, _, ecg_indices, _ = self.find_physiological_artifacts(
            eog_treshold='auto', ecg_treshold='auto', reject_by_annotation=True,
            measure='correlation', plot_fig=False, verbose='warning')
        print(eog_indices + ecg_indices)
        self.visualize_ICA_components()
        self.exclude_ica(eog_indices + ecg_indices)
        exclude = list(input('Components to exclude: ').split(','))
        self.ica.exclude = [int(x) for x in exclude]
        print(self.ica.exclude)
        self.ica.plot_overlay(self.raw, exclude=self.ica.exclude, picks='eeg')
        reconst_raw = self.raw.copy()
        self.ica.apply(reconst_raw)
        return reconst_raw
