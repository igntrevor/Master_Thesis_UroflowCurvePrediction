import os
import numpy as np
import pandas as pd
import librosa
import scipy.fft

def extract_fft_features(audio_file, fmin, fmax, nbins):
    y, sr = librosa.load(audio_file, sr=None)

    fft_vals = np.abs(scipy.fft.rfft(y))
    freqs = scipy.fft.rfftfreq(len(y), d=1/sr)

    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    fft_vals = fft_vals[mask]

    bins = np.linspace(freqs.min(), freqs.max(), nbins + 1)
    features = []

    for i in range(nbins):
        idx = (freqs >= bins[i]) & (freqs < bins[i + 1])
        features.append(fft_vals[idx].sum())

    return np.array(features, dtype=np.float32)


def load_segmented_dataset(base_dir, fmin, fmax, nbins):
    X, y = [], []

    for patient in sorted(os.listdir(base_dir)):
        patient_dir = os.path.join(base_dir, patient)
        if not os.path.isdir(patient_dir):
            continue

        labels = pd.read_csv(os.path.join(patient_dir, "labels.csv"))

        for _, row in labels.iterrows():
            audio_file = os.path.join(patient_dir, row["audio_file"])
            features = extract_fft_features(audio_file, fmin, fmax, nbins)

            X.append(features)
            y.append(row["flow"])

    return np.array(X), np.array(y)

