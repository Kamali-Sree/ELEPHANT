import numpy as np
from scipy.signal import butter, lfilter
import math

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Clip normal_cutoff to be strictly between 0 and 1
    normal_cutoff = np.clip(normal_cutoff, 0.001, 0.999)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def simulate_distance(audio_signal, sample_rate, distance_meters):
    """
    Simulates how an audio signal sounds from a certain distance away.
    1. Attenuates volume (Inverse Square Law proxy).
    2. Applies high-frequency air absorption (low-pass filter).
    
    Reference distance is 10 meters (audio_signal remains unchanged).
    """
    # Prevent divide by zero or extreme close-up amplification
    distance = max(10.0, float(distance_meters))
    
    # 1. Volume Attenuation
    # Acoustic pressure drops by 1/r. 
    # At 10m, multiplier is 1.0. At 100m, multiplier is 0.1
    amplitude_multiplier = 10.0 / distance
    audio_attenuated = audio_signal * amplitude_multiplier
    
    # 2. High-frequency Air absorption (Low-pass Filter)
    # Higher frequencies are absorbed by air faster over distance.
    # At 10m, cutoff is high (e.g. 8000 Hz)
    # At 1000m, cutoff drops significantly (e.g. 400 Hz)
    base_cutoff = 8000
    decay_rate = -math.log(400/8000) / 1000 
    cutoff_freq = base_cutoff * math.exp(-decay_rate * (distance - 10))
    
    # Ensure cutoff doesn't exceed Nyquist
    cutoff_freq = min(cutoff_freq, sample_rate / 2 - 10)
    cutoff_freq = max(cutoff_freq, 50.0) # Bottom out at 50Hz
    
    audio_simulated = apply_lowpass(audio_attenuated, cutoff_freq, sample_rate)
    
    return audio_simulated
