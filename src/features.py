import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

# ─────────────────── Constants ───────────────────
TARGET_SR = 16000
CLIP_DURATION = 6.0
N_MELS = 128
FMAX = 8000


# ─────────────────── PyTorch Dataset ───────────────────

class ElephantAcousticDataset(Dataset):
    """
    PyTorch Dataset that loads audio files from a CSV index,
    converts them to Mel-spectrograms, and returns (feature, label) pairs.
    
    Supports 4-class classification: Roar, Rumble, Trumpet, Non_Elephant.
    """
    def __init__(self, csv_file, split='train', feature_type='mel', target_sr=TARGET_SR, duration=CLIP_DURATION):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split]
        if 'corrupted' in self.data.columns:
            self.data = self.data[~self.data['corrupted']]
        self.data = self.data.reset_index(drop=True)
        self.feature_type = feature_type
        self.target_sr = target_sr
        self.duration = duration
        
        # 4-class Classification
        self.label_map = {
            'Roar': 0,
            'Rumble': 1,
            'Trumpet': 2,
            'Non_Elephant': 3,
        }
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filepath = row['filepath']
        label_str = row['label']
        label = self.label_map.get(label_str, 0)
        
        # Load and normalise audio
        y = load_and_pad_audio(filepath, sr=self.target_sr, duration=self.duration)
            
        # Extract features
        if self.feature_type == 'mel':
            S = librosa.feature.melspectrogram(y=y, sr=self.target_sr, n_mels=N_MELS, fmax=FMAX)
            S_db = librosa.power_to_db(S, ref=np.max)
            # Add channel dimension: (1, n_mels, time_steps)
            feature = torch.FloatTensor(S_db).unsqueeze(0)
        elif self.feature_type == 'mfcc':
            mfcc = librosa.feature.mfcc(y=y, sr=self.target_sr, n_mfcc=40)
            feature = torch.FloatTensor(mfcc).unsqueeze(0)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
            
        return feature, torch.tensor(label, dtype=torch.long)


# ─────────────────── Standalone Feature Functions ───────────────────

def load_and_pad_audio(filepath, sr=TARGET_SR, duration=CLIP_DURATION):
    """
    Load an audio file, resample to target_sr, and pad/trim to exact duration.
    Returns a 1D numpy array of audio samples.
    """
    y, _ = librosa.load(filepath, sr=sr, duration=duration)
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    elif len(y) > target_length:
        y = y[:target_length]
    return y


def extract_mel_spectrogram(filepath, sr=TARGET_SR, duration=CLIP_DURATION):
    """
    Load an audio file and return its Mel-spectrogram as a PyTorch tensor.
    
    Returns:
        torch.FloatTensor of shape (1, 128, time_steps) — ready for CNN input.
    """
    y = load_and_pad_audio(filepath, sr=sr, duration=duration)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)
    return torch.FloatTensor(S_db).unsqueeze(0)


def extract_acoustic_characteristics(filepath):
    """
    Analyse an audio file and extract its acoustic characteristics.
    
    Returns:
        dict with keys:
            - 'duration': float (seconds)
            - 'dominant_freq': float (Hz)
            - 'rms_energy': float (average loudness)
            - 'zero_crossing_rate': float (average ZCR)
    """
    y, sr = librosa.load(filepath, sr=TARGET_SR)
    
    # Duration
    duration = round(librosa.get_duration(y=y, sr=sr), 2)
    
    # Dominant frequency (frequency with highest energy)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    magnitude_sum = np.sum(S, axis=1)
    dominant_freq_idx = np.argmax(magnitude_sum)
    dominant_freq = round(float(freqs[dominant_freq_idx]), 2)
    
    # RMS energy (average loudness)
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = round(float(np.mean(rms)), 4)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    avg_zcr = round(float(np.mean(zcr)), 4)
    
    return {
        'duration': duration,
        'dominant_freq': dominant_freq,
        'rms_energy': avg_rms,
        'zero_crossing_rate': avg_zcr,
    }


# ─────────────────── Visualization ───────────────────

def generate_sample_plots(csv_file, output_path):
    """Generate side-by-side waveform + Mel-spectrogram plots for a Rumble and Non_Elephant sample."""
    print("Generating sample plots...")
    df = pd.read_csv(csv_file)
    if 'corrupted' in df.columns:
        df = df[~df['corrupted']]
    
    # Get one elephant and one non-elephant sample
    ele = df[df['label'] == 'Rumble'].iloc[0]
    non = df[df['label'] == 'Non_Elephant'].iloc[0]
    samples = [ele, non]
    
    plt.figure(figsize=(15, 10))
    for i, row in enumerate(samples):
        y = load_and_pad_audio(row['filepath'])
        
        # Waveform
        plt.subplot(2, 2, i + 1)
        librosa.display.waveshow(y, sr=TARGET_SR)
        plt.title(f"Waveform: {row['label']}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        # Mel-spectrogram
        plt.subplot(2, 2, i + 3)
        S = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS, fmax=FMAX)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=TARGET_SR, x_axis='time', y_axis='mel', fmax=FMAX)
        plt.colorbar(img, format='%+2.0f dB')
        plt.title(f"Mel-spectrogram: {row['label']}")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Sample plots saved to {output_path}")

if __name__ == '__main__':
    csv_path = r"c:\Users\kamal\Downloads\MACONFLIC-main\elephant_vocalization_detection\data\dataset_index.csv"
    output_path = r"c:\Users\kamal\Downloads\MACONFLIC-main\elephant_vocalization_detection\data\sample_features.png"
    generate_sample_plots(csv_path, output_path)
