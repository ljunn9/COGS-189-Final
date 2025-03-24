import numpy as np
import os
import glob
from google.colab import drive
import zipfile
import os


drive.mount('/content/drive')

data_dir = "/content/drive/MyDrive/SSVEP Data"
unzip_to = "/content/eeg_data"

for file in os.listdir(data_dir):
    if file.endswith(".zip"):
        with zipfile.ZipFile(os.path.join(data_dir, file), 'r') as zip_ref:
            zip_ref.extractall(unzip_to)

print("Unzipped EEG data to:", unzip_to)


for zipname in os.listdir(data_dir):
    if zipname.endswith(".zip"):
        path = os.path.join(data_dir, zipname)
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(unzip_to)

for root, dirs, files in os.walk("/content/eeg_data"):
    for file in files:
        print(os.path.join(root, file))


base_path = "/content/eeg_data"

eeg_trial_files = glob.glob(os.path.join(base_path, "ses-*", "eeg-trials_2-per-class_run-*.npy"))
print(f"Found {len(eeg_trial_files)} EEG trial files.")

eeg_trials = [np.load(f) for f in eeg_trial_files]
eeg_data = np.concatenate(eeg_trials, axis=0) 
print("Combined EEG shape:", eeg_data.shape)
