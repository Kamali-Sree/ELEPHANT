import os
import pandas as pd
import random
from pathlib import Path

def build_csv():
    print("Building Dataset CSV for MACONFLIC...")
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    raw_audio_dir = data_dir / 'raw'
    
    # Output CSV files
    output_csv = data_dir / 'scaled_dataset_index.csv'
    
    # Ensure directories exist
    if not raw_audio_dir.exists():
        print(f"Creating directories at {raw_audio_dir}...")
        for label in ['Roar', 'Rumble', 'Trumpet', 'Non_Elephant']:
            (raw_audio_dir / label).mkdir(parents=True, exist_ok=True)
        
        print("\nACTION REQUIRED: I have created the folder structure!")
        print(f"Please place your .wav files into their respective folders in:\n {raw_audio_dir}")
        print("Then run this script again.")
        return

    # Gather all files
    data_records = []
    valid_extensions = ['.wav', '.mp3', '.flac']
    
    # Look through the label folders
    for label_folder in raw_audio_dir.iterdir():
        if not label_folder.is_dir():
            continue
            
        label = label_folder.name
        if label not in ['Roar', 'Rumble', 'Trumpet', 'Non_Elephant']:
            print(f"Skipping unknown folder: {label}")
            continue
            
        files = [f for f in label_folder.iterdir() if f.suffix.lower() in valid_extensions]
        
        # Shuffle for random train/validate split
        random.shuffle(files)
        
        # 80% Train, 20% Validate
        split_index = int(len(files) * 0.8)
        
        for i, file_path in enumerate(files):
            split = 'train' if i < split_index else 'validate'
            # Store relative path for portability
            relative_path = os.path.relpath(file_path, project_root)
            
            data_records.append({
                'filepath': relative_path,
                'label': label,
                'split': split,
                'corrupted': False
            })
            
    if not data_records:
        print("\nNo audio files found! Please place .wav files inside data/raw/Rumble, data/raw/Non_Elephant, etc.")
        return

    # Create DataFrame and save
    df = pd.DataFrame(data_records)
    df.to_csv(output_csv, index=False)
    
    print(f"\nSuccessfully created CSV at: {output_csv}")
    print(f"Total files indexed: {len(df)}")
    print(df['label'].value_counts())
    print("\nYou can now run: python src/train_v2.py")

if __name__ == '__main__':
    build_csv()
