'''
Functions for preprocessing the EEG data, including filtering, epoching, and calculating ERD/ERS.
'''

import scipy
from scipy.signal import iirnotch, filtfilt
from scipy.signal import butter
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import mne


def bandpass_filter(data, fs, lowcut, highcut, order):
    '''
    Applies a bandpass filter to the data using a Butterworth filter design.
    Parameters:
    - data: 2D array of shape (samples, channels)
    - fs: Sampling frequency in Hz
    - lowcut: Low cutoff frequency in Hz
    - highcut: High cutoff frequency in Hz
    - order: Order of the filter
    Returns:
    - Data filtered with the specified filter.
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq #the filter works in a normalized frequency range (0 to 1), where 1 corresponds to the Nyquist frequency
    high = highcut / nyq

    b,a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)



def notch_filter(data, fs, notch_freq, quality_factor=30):
    '''Applies a notch filter to the data.
    Arguments:
    - data: 2D array of shape (samples, channels)
    - fs: Sampling frequency in Hz
    - notch_freq: Frequency to remove in Hz
    - quality_factor: Quality factor of the notch filter
    Returns:
    - Data filtered with the specified notch filter.
    '''
    b, a = iirnotch(notch_freq, quality_factor, fs)
    return filtfilt(b, a, data, axis=0)



def trim_data(data, fs, start_time):
    '''Aims at trimming the data in order to neglect initial noise (if present)
    Arguments:
    - data: 2D array of shape (samples, channels)
    - fs: Sampling frequency in Hz
    - start_time: Time to start trimming from in seconds
    Returns:
    - Trimmed data.
    '''
    start_sample = int(start_time * fs)
    return data[start_sample:, :]



def get_trigger_onsets(trig, fs):
    '''Extracts the onsets of triggers from the trigger signal (detects the transitions to 1 or -1).
    Arguments:
    - trig: 2D array of shape (samples, channels)
    - fs: Sampling frequency in Hz
    Returns:
    - Array of onset times in seconds.
    '''
    trig_1d = trig.flatten()
    onsets = np.where((trig_1d[1:] != 0) & (trig_1d[:-1] == 0))[0] + 1
    
    onset_times = onsets / fs
    return onset_times