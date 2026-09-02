import os
import torch
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.features import TARGET_SR, CLIP_DURATION, N_MELS, FMAX, load_and_pad_audio
from src.model import ElephantCNN, NUM_CLASSES, CLASS_NAMES
from src.distance_simulation import simulate_distance

def analyze_distance_impact():
    print("Starting Distance Analysis Simulation...")
    
    # 1. Load the trained model
    model_path = os.path.join('models', 'baseline_cnn.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train it first.")
        return

    model = ElephantCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    # 2. Load the test dataset (only original files to avoid augmentation artifacts)
    csv_path = os.path.join('data', 'scaled_dataset_index.csv')
    df = pd.read_csv(csv_path)
    test_df = df[(df['split'] == 'test') & (df['source'] == 'original')]
    
    distances = [10, 50, 100, 200, 300, 500, 1000]
    
    # We will track average confidence for the CORRECT class at each distance
    # Results dictionary: {class_name: {distance: [conf1, conf2, ...]}}
    results = {c: {d: [] for d in distances} for c in ['Roar', 'Rumble', 'Trumpet']}
    
    # 3. Run Simulation
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Analyzing Test Samples"):
        true_label = row['label']
        # We only care about how well it detects the elephant sounds over distance
        if true_label not in results:
            continue
            
        true_idx = CLASS_NAMES.index(true_label)
        filepath = row['filepath']
        
        # Load clean audio once
        clean_audio = load_and_pad_audio(filepath, sr=TARGET_SR, duration=CLIP_DURATION)
        
        for dist in distances:
            # Simulate distance
            sim_audio = simulate_distance(clean_audio, TARGET_SR, dist)
            
            # Extract features (Mel-Spectrogram)
            S = librosa.feature.melspectrogram(y=sim_audio, sr=TARGET_SR, n_mels=N_MELS, fmax=FMAX)
            S_db = librosa.power_to_db(S, ref=np.max)
            feature = torch.FloatTensor(S_db).unsqueeze(0).unsqueeze(0) # (1, 1, 128, time_steps)
            
            # Predict
            with torch.no_grad():
                output = model(feature)
                probs = torch.nn.functional.softmax(output, dim=1)
                
            # Record the confidence of the TRUE class
            true_class_conf = probs[0][true_idx].item()
            results[true_label][dist].append(true_class_conf)
            
    # 4. Aggregate & Plot Results
    print("\nSimulation Complete. Generating Plot...")
    plt.figure(figsize=(10, 6))
    
    colors = {'Roar': 'red', 'Rumble': 'blue', 'Trumpet': 'green'}
    
    for cls in ['Roar', 'Rumble', 'Trumpet']:
        avg_confs = []
        for d in distances:
            confs = results[cls][d]
            avg_confs.append(np.mean(confs) * 100 if confs else 0)
        
        plt.plot(distances, avg_confs, marker='o', label=cls, color=colors[cls], linewidth=2)
        
    plt.axhline(y=50, color='gray', linestyle='--', label='50% Threshold')
    plt.title("Detection Confidence vs Distance (Simulated)")
    plt.xlabel("Distance (meters)")
    plt.ylabel("Average Confidence (%)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'distance_analysis_curve.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {out_path}")

if __name__ == '__main__':
    analyze_distance_impact()
