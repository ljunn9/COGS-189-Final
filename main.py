import numpy as np
import os
from google.colab import files
from src.1_preprocessing import preprocess_trial, load_labels
from src.2_feature_extraction import extract_fft, run_cca, generate_reference_signals
from src.3_model_training import train_and_evaluate

uploaded = files.upload()

eeg_data = np.load("eeg_data_all.npy")
fft_features = np.load("fft_features_all.npy")
labels = np.load("labels_all.npy")

ref_freqs = [10, 12, 15]
fs = 250
duration = 1.4
t = np.linspace(0, duration, int(fs * duration))
ref_signals = generate_reference_signals(ref_freqs, t)

cca_preds = np.array([run_cca(trial, ref_signals) for trial in eeg_data]).reshape(-1, 1)

X_combined = np.hstack((fft_features, cca_preds))

train_and_evaluate(X_combined, labels, save_path="models")
