'''
Document with all the functions related to data processing, such as epoching and feature extraction, required for the 
calculation of ERD/ERS and the subsequent analysis of the data.
'''


import scipy
from scipy.signal import iirnotch, filtfilt
from scipy.signal import butter
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import mne
from scipy.signal import welch

from preprocessing_functions import get_trigger_onsets


def epoch_data(data, trig, fs, time_after, period_of_interest):
    '''Epochs the data based on trigger onsets, separating left and right hand movement intentions.
    Arguments:
    - data: 2D array of shape (samples, channels)
    - trig: 2D array of shape (1, samples)
    - fs: Sampling frequency in Hz
    - time_after: Time after the trigger onset to include in the epoch in seconds
    - period_of_interest: Duration of the epoch in seconds (time after the trigger onset to consider for analysis)
    Returns:
    - Arrays of epochs for right and left hand movements.
    '''
    samples_after = int(time_after * fs)
    samples_interest = int(period_of_interest * fs)

    trigger_events_time = get_trigger_onsets(trig, fs)

    right_hand_epochs = []
    left_hand_epochs = []

    trig_flat = trig.flatten()

    for onset_time in trigger_events_time:
        idx_onset = int(round(onset_time * fs))

        window = trig_flat[max(0, idx_onset-2):idx_onset+3]
        label = int(np.sign(np.sum(window)))

        start_idx = idx_onset + samples_after
        end_idx = start_idx + samples_interest

        if end_idx <= data.shape[0]:
            epoch = data[start_idx:end_idx, :]

            if label == -1:
                right_hand_epochs.append(epoch)
            elif label == 1:
                left_hand_epochs.append(epoch)

    return np.array(right_hand_epochs), np.array(left_hand_epochs)



def get_band_power(epochs, fs, low=8, high=30):
    """
    Gets the band power for each epoch and channel, then averages across epochs to get a single value per channel.
    Arguments:
    epochs: (trials, samples, channels)
    returns: (channels,)
    """

    band_power_all = []

    for epoch in epochs:
        ch_power = []

        for ch in range(epoch.shape[1]):

            f, Pxx = welch(epoch[:, ch], fs=fs, nperseg=fs*2)

            idx = (f >= low) & (f <= high)
            band_power = np.trapz(Pxx[idx], f[idx])

            ch_power.append(band_power)

        band_power_all.append(ch_power)

    return np.mean(band_power_all, axis=0)



def calculate_erd_ers(active_epochs, ref_epochs, fs, low, high):
    '''Calculates the ERD/ERS values for the given active and reference epochs.
    Arguments:
    - active_epochs: 3D array of shape (n_active_epochs, n_samples, n_channels) containing the epochs of interest (e.g., left or right hand movement)
    - ref_epochs: 3D array of shape (n_ref_epochs, n_samples, n_channels) containing the reference epochs (e.g., baseline or rest periods)
    fs: sampling frequency in Hz
    low: lower cutoff frequency that we want to present
    high: upper cutoff frequency that we want to present'''
    
    A = get_band_power(active_epochs, fs, low, high)
    R = get_band_power(ref_epochs, fs, low, high)

    return 10 * np.log10(A / R)