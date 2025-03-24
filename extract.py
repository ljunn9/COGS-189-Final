import numpy as np
from scipy.signal import welch
from sklearn.cross_decomposition import CCA

def extract_fft(trial, fs=250):
    features = []
    for ch in trial:
        freqs, psd = welch(ch, fs=fs, nperseg=128)
        features.extend(psd[:20])
    return features

def generate_reference_signals(freqs, t):
    return [np.array([np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t)]).T for f in freqs]


