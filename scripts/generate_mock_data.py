import numpy as np
import soundfile as sf
import os
from pathlib import Path

# Parameters
SAMPLE_RATE = 16000
DURATION = 6.0
SAMPLES = int(SAMPLE_RATE * DURATION)

def generate_noise(volume=0.05):
    """Generate ambient forest noise (pink/brown noise approximation)"""
    noise = np.random.normal(0, volume, SAMPLES)
    # Simple lowpass filter to make it sound muffled like a forest
    noise = np.convolve(noise, np.ones(10)/10, mode='same')
    return noise

def generate_rumble():
    """Generate a 15-30 Hz infrasonic rumble with harmonics"""
    t = np.linspace(0, DURATION, SAMPLES, endpoint=False)
    # Fundamental frequency varying slightly (frequency modulation)
    f0 = 20.0 + 5.0 * np.sin(2 * np.pi * 0.5 * t) 
    
    wave = 0.5 * np.sin(2 * np.pi * f0 * t) # Fundamental
    wave += 0.25 * np.sin(2 * np.pi * (f0 * 2) * t) # 1st Harmonic
    wave += 0.1 * np.sin(2 * np.pi * (f0 * 3) * t) # 2nd Harmonic
    
    # Add ambient noise
    return np.clip(wave + generate_noise(0.02), -1.0, 1.0)

def generate_trumpet():
    """Generate a 400-800 Hz high pitched burst"""
    t = np.linspace(0, DURATION, SAMPLES, endpoint=False)
    f0 = 600.0 + 200.0 * np.sin(2 * np.pi * 2.0 * t)
    
    # Envelope to make it sound like a burst (fade in, hold, fade out)
    envelope = np.ones(SAMPLES)
    envelope[:4000] = np.linspace(0, 1, 4000)
    envelope[-16000:] = np.linspace(1, 0, 16000)
    
    wave = 0.8 * np.sin(2 * np.pi * f0 * t) * envelope
    return np.clip(wave + generate_noise(0.01), -1.0, 1.0)

def generate_roar():
    """Generate broadband low/mid-frequency noise (chaotic)"""
    t = np.linspace(0, DURATION, SAMPLES, endpoint=False)
    # Chaotic frequency modulation
    wave = 0.6 * np.sin(2 * np.pi * (100.0 + 50.0 * np.random.randn(SAMPLES)) * t)
    
    envelope = np.ones(SAMPLES)
    envelope[:8000] = np.linspace(0, 1, 8000)
    envelope[-8000:] = np.linspace(1, 0, 8000)
    
    return np.clip((wave * envelope) + generate_noise(0.05), -1.0, 1.0)

def main():
    print("Generating Synthetic Elephant Audio Dataset...")
    
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / 'data' / 'raw'
    
    labels = ['Roar', 'Rumble', 'Trumpet', 'Non_Elephant']
    for label in labels:
        (raw_dir / label).mkdir(parents=True, exist_ok=True)
        
    num_samples_per_class = 10
    
    for i in range(num_samples_per_class):
        # 1. Rumble
        sf.write(raw_dir / 'Rumble' / f'synth_rumble_{i}.wav', generate_rumble(), SAMPLE_RATE)
        
        # 2. Trumpet
        sf.write(raw_dir / 'Trumpet' / f'synth_trumpet_{i}.wav', generate_trumpet(), SAMPLE_RATE)
        
        # 3. Roar
        sf.write(raw_dir / 'Roar' / f'synth_roar_{i}.wav', generate_roar(), SAMPLE_RATE)
        
        # 4. Non_Elephant
        sf.write(raw_dir / 'Non_Elephant' / f'synth_noise_{i}.wav', generate_noise(0.1), SAMPLE_RATE)
        
        print(f"Generated sample batch {i+1}/{num_samples_per_class}")
        
    print("\nSuccessfully generated 40 synthetic audio files!")

if __name__ == '__main__':
    main()
