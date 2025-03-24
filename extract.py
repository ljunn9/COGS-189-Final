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

def run_cca(trial, ref_signals):
    cca = CCA(n_components=1)
    trial_data = trial.T
    corrs = []
    for ref in ref_signals:
        X_c, Y_c = cca.fit_transform(trial_data, ref)
        corr = np.corrcoef(X_c.T[0], Y_c.T[0])[0, 1]
        corrs.append(corr)
    return np.argmax(corrs)
