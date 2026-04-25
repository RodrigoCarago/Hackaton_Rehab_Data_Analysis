from pathlib import Path
import numpy as np
import scipy.io as sio
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

def adaptive_filter(raw, low=1.0, high=30.0):
    nyquist = raw.info['sfreq'] / 2
    h_freq = min(high, nyquist - 1)

    if h_freq <= low:
        raise ValueError(
            f"Invalid band: High cutoff ({h_freq}) <= ({low})."
            f" Nyquist = {nyquist}. Try using lower 'low' or higher sampling rate."
            )

    raw.notch_filter(freqs=[50, 60], picks='eeg', verbose=False)
    raw.filter(l_freq=low, h_freq=h_freq, picks='eeg')

    ica = ICA(n_components=0.95, random_state=42, max_iter=800)
    ica.fit(raw)
    fig = raw.plot(scalings='auto', show=False)
    fig.set_size_inches(15, 10)
    

    plt.show()
    ica.apply(raw)

    raw_cleaned = raw.copy()
    return raw_cleaned

#Run standalone
if __name__ == "__main__":
    import mne

    # Load your raw file from mat
    mat_path = Path("stroke-rehab\P2_pre_training.mat")
    mat = sio.loadmat(mat_path)
    eeg = mat["y"].astype(np.float64).T  # (channels, samples)
    trig = mat["trig"].flatten().astype(int)
    fs = int(mat["fs"][0, 0])
    CHANNEL_NAMES = ['FC3','FCz','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','Pz']
    info = mne.create_info(CHANNEL_NAMES, fs, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore")
    raw = mne.io.RawArray(eeg, info)

    # Run your cleaning pipeline
    cleaned = adaptive_filter(raw)

    # Save cleaned data back to mat
    sio.savemat(str(mat_path), {'y': cleaned.get_data().T, 'trig': trig, 'fs': fs})

    print("Preprocessing complete.")

# 1) Load.mat
mat_path = Path("stroke-rehab\P2_pre_training.mat")
mat = sio.loadmat(mat_path)
p1 = {
    "eeg": mat["y"].astype(np.float64),         # (samples, channels)
    "trig": mat["trig"].flatten().astype(int),  # señal de trigger nivelada
    "fs": int(mat["fs"][0, 0]),
}
print("p1 keys:", p1.keys())
print("eeg shape:", p1["eeg"].shape, "| trig shape:", p1["trig"].shape, "| fs:", p1["fs"])

# 2) Epochs 2 to 6 according to triggers
fs = p1["fs"]
t_start, t_end = 2.0, 6.0
n_start, n_end = int(t_start * fs), int(t_end * fs)
diff_trig = np.diff(p1["trig"], prepend=0)
onset_left = np.where(diff_trig == 1)[0]    # left
onset_right = np.where(diff_trig == -1)[0]  # right
epochs, labels = [], []
for onset in onset_left:
    s, e = onset + n_start, onset + n_end
    if s >= 0 and e <= len(p1["eeg"]):
        epochs.append(p1["eeg"][s:e, :].T)  # (channels, time)
        labels.append(1)  # left -> 1 para MNE event_id
for onset in onset_right:
    s, e = onset + n_start, onset + n_end
    if s >= 0 and e <= len(p1["eeg"]):
        epochs.append(p1["eeg"][s:e, :].T)
        labels.append(2)  # right -> 2
X = np.array(epochs)         # (n_epochs, n_channels, n_times)
y = np.array(labels, int)    # (n_epochs,)
print("X shape:", X.shape, "| y shape:", y.shape)

# 3) MNE EpochsArray
CHANNEL_NAMES = ['FC3','FCz','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','Pz']
info = mne.create_info(CHANNEL_NAMES, fs, ch_types="eeg")
info.set_montage("standard_1020", on_missing="ignore")
events = np.column_stack([
    np.arange(len(y), dtype=int),       # sample index del evento (artificial, válido para EpochsArray)
    np.zeros(len(y), dtype=int),
    y
])
p1_pre_mne = mne.EpochsArray(
    X, info, events=events,
    event_id={"left": 1, "right": 2},
    tmin=t_start,
    baseline=None,
    verbose=False,
)
print(p1_pre_mne)

# Visualization
p1_pre_mne.plot(scalings="auto")
plt.show() # pyright: ignore[reportUndefinedVariable]