import os
import csv
from collections import defaultdict
import glob
import librosa

def analyze_dataset(data_dir, output_csv):
    print(f"Analyzing dataset in {data_dir}...")
    
    splits = ['train', 'validate', 'test']
    
    stats = {
        'total_files': 0,
        'labels': set(),
        'class_distribution': defaultdict(int),
        'split_distribution': defaultdict(int),
        'duration_distribution': defaultdict(list),
        'sampling_rates': set()
    }
    
    dataset_index = []
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        labels = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        for label in labels:
            stats['labels'].add(label)
            label_dir = os.path.join(split_dir, label)
            
            # support .wav files
            audio_files = glob.glob(os.path.join(label_dir, '*.wav'))
            
            for audio_file in audio_files:
                stats['total_files'] += 1
                stats['class_distribution'][label] += 1
                stats['split_distribution'][split] += 1
                
                try:
                    duration = librosa.get_duration(path=audio_file)
                    rate = librosa.get_samplerate(audio_file)
                    
                    stats['duration_distribution'][label].append(duration)
                    stats['sampling_rates'].add(rate)
                    
                    dataset_index.append({
                        'filepath': os.path.abspath(audio_file),
                        'split': split,
                        'label': label,
                        'duration_sec': duration,
                        'sampling_rate': rate,
                        'corrupted': False
                    })
                except Exception as e:
                    print(f"Warning: Could not read {audio_file}: {e}")
                    dataset_index.append({
                        'filepath': os.path.abspath(audio_file),
                        'split': split,
                        'label': label,
                        'duration_sec': 0,
                        'sampling_rate': 0,
                        'corrupted': True
                    })

    # Save clean dataset index
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filepath', 'split', 'label', 'duration_sec', 'sampling_rate', 'corrupted'])
        writer.writeheader()
        writer.writerows(dataset_index)
        
    print("\n--- DATASET STATISTICS ---")
    print(f"Total Audio Files: {stats['total_files']}")
    print(f"Labels found: {stats['labels']}")
    print(f"Sampling Rates found: {stats['sampling_rates']}")
    
    print("\nClass Distribution:")
    for label, count in stats['class_distribution'].items():
        print(f"  - {label}: {count} files")
        
    print("\nSplit Distribution:")
    for split, count in stats['split_distribution'].items():
        print(f"  - {split}: {count} files")
        
    print("\nDuration Statistics (min / max / avg):")
    for label, durations in stats['duration_distribution'].items():
        if durations:
            min_d = min(durations)
            max_d = max(durations)
            avg_d = sum(durations) / len(durations)
            print(f"  - {label}: {min_d:.2f}s / {max_d:.2f}s / {avg_d:.2f}s")
            
    print(f"\nDataset index saved to {output_csv}")

if __name__ == '__main__':
    data_dir = r"c:\Users\kamal\Downloads\MACONFLIC-main\elephant_vocalization_detection\data\Audio-Classification-for-Elephant-Sounds\data"
    output_csv = r"c:\Users\kamal\Downloads\MACONFLIC-main\elephant_vocalization_detection\data\dataset_index.csv"
    analyze_dataset(data_dir, output_csv)
