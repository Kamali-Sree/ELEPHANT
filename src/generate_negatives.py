import os
import numpy as np
import wave
import struct

def generate_noise(duration_sec, sample_rate, noise_type='white'):
    num_samples = int(duration_sec * sample_rate)
    
    if noise_type == 'white':
        noise = np.random.normal(0, 1, num_samples)
    elif noise_type == 'brown':
        white = np.random.normal(0, 1, num_samples)
        noise = np.cumsum(white)
        noise = noise / np.max(np.abs(noise))
    elif noise_type == 'pink':
        white = np.random.normal(0, 1, num_samples)
        noise = np.convolve(white, np.ones(10)/10, mode='same')
    elif noise_type == 'rain':
        # Rain is essentially pink noise with some amplitude modulation (droplets)
        white = np.random.normal(0, 1, num_samples)
        pink = np.convolve(white, np.ones(10)/10, mode='same')
        drops = np.random.normal(0, 1, num_samples) * (np.random.rand(num_samples) > 0.99)
        noise = pink * 0.5 + drops * 0.5
    elif noise_type == 'wind':
        # Wind is brown noise with low frequency oscillation (LFO) for gusts
        white = np.random.normal(0, 1, num_samples)
        brown = np.cumsum(white)
        brown = brown / np.max(np.abs(brown))
        t = np.linspace(0, duration_sec, num_samples)
        lfo = np.sin(2 * np.pi * 0.2 * t) * 0.5 + 0.5  # 0.2 Hz gust
        noise = brown * lfo
    elif noise_type == 'insects':
        # High frequency chirps
        t = np.linspace(0, duration_sec, num_samples)
        carrier = np.sin(2 * np.pi * 4000 * t)  # 4kHz carrier
        modulator = np.sin(2 * np.pi * 15 * t)  # 15Hz chirp rate
        modulator = np.clip(modulator, 0, 1)
        noise = carrier * modulator + np.random.normal(0, 0.1, num_samples)
    
    # Normalize to -1.0 to 1.0
    noise = noise / np.max(np.abs(noise))
    # Apply some random gain between 0.1 and 0.5
    gain = np.random.uniform(0.1, 0.5)
    noise = noise * gain
    return noise

def save_wav(filepath, audio_data, sample_rate):
    # Convert float32 to int16
    audio_data_int = np.int16(audio_data * 32767)
    with wave.open(filepath, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio_data_int.tobytes())

def create_non_elephant_class(base_dir, splits_counts):
    print("Generating Synthetic 'Non-Elephant' samples for the pipeline...")
    for split, count in splits_counts.items():
        out_dir = os.path.join(base_dir, split, 'Non_Elephant')
        os.makedirs(out_dir, exist_ok=True)
        for i in range(count):
            noise_type = np.random.choice(['white', 'brown', 'pink', 'rain', 'wind', 'insects'])
            audio = generate_noise(6.0, 44100, noise_type)
            filepath = os.path.join(out_dir, f'Non_Elephant_{i+1:03d}.wav')
            save_wav(filepath, audio, 44100)
        print(f"Generated {count} samples in {out_dir}")

if __name__ == '__main__':
    base_dir = r"c:\Users\kamal\Downloads\MACONFLIC-main\elephant_vocalization_detection\data\Audio-Classification-for-Elephant-Sounds\data"
    # Matches roughly the average number of samples per elephant class
    splits_counts = {'train': 77, 'validate': 18, 'test': 9}
    create_non_elephant_class(base_dir, splits_counts)
