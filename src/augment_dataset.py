"""
augment_dataset.py
==================
Pools all original elephant audio recordings, re-splits them 70/15/15,
and applies scientifically valid audio augmentation to scale each split
to the target count per class.

Augmentation techniques used:
  - Time Shifting:   Simulates vocalizations not starting at clip onset.
  - Pitch Shifting:  Simulates natural variation between individual elephants.
  - Time Stretching: Simulates variation in call duration.
  - Additive Noise:  Simulates varying environmental noise (wind, rain).
  - Gain Variation:  Simulates different recording distances / mic sensitivity.

Every augmented file is clearly named:
    {OriginalName}_aug{N}_{technique}.wav

No original file's augmentations appear in multiple splits (no data leakage).
"""

import os
import glob
import shutil
import random
import numpy as np
import librosa
import soundfile as sf
from collections import defaultdict

# ─────────────────── Configuration ───────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SRC_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "Audio-Classification-for-Elephant-Sounds", "data"
)
DST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "scaled_dataset"
)

TARGET_SR = 16000
CLIP_DURATION = 6.0          # seconds
TARGET_SAMPLES = int(TARGET_SR * CLIP_DURATION)

TRAIN_TARGET_PER_CLASS = 500
VAL_TARGET_PER_CLASS   = 500
TEST_TARGET_PER_CLASS  = 100  # lighter augmentation for honest evaluation

CLASSES = ["Roar", "Rumble", "Trumpet", "Non_Elephant"]

# ─────────────────── Augmentation Functions ───────────────────

def time_shift(y, sr, shift_fraction=None):
    """Shift audio left or right by a random fraction."""
    if shift_fraction is None:
        shift_fraction = np.random.uniform(-0.3, 0.3)
    shift = int(len(y) * shift_fraction)
    return np.roll(y, shift)


def pitch_shift(y, sr, n_steps=None):
    """Shift pitch by n semitones."""
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def time_stretch(y, sr, rate=None):
    """Stretch or compress time without changing pitch."""
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)
    stretched = librosa.effects.time_stretch(y=y, rate=rate)
    # Pad or trim to original length
    if len(stretched) < TARGET_SAMPLES:
        stretched = np.pad(stretched, (0, TARGET_SAMPLES - len(stretched)))
    else:
        stretched = stretched[:TARGET_SAMPLES]
    return stretched


def add_noise(y, sr, noise_level=None):
    """Add random noise (white, brown, or pink)."""
    if noise_level is None:
        noise_level = np.random.uniform(0.002, 0.015)
    noise_type = random.choice(["white", "brown", "pink"])
    n = len(y)
    if noise_type == "white":
        noise = np.random.normal(0, 1, n)
    elif noise_type == "brown":
        noise = np.cumsum(np.random.normal(0, 1, n))
        noise = noise / (np.max(np.abs(noise)) + 1e-9)
    else:  # pink
        noise = np.convolve(np.random.normal(0, 1, n + 9), np.ones(10) / 10, mode="valid")
        noise = noise[:n]
    noise = noise / (np.max(np.abs(noise)) + 1e-9)
    return y + noise_level * noise


def gain_variation(y, sr, gain_db=None):
    """Vary gain (volume) by a random amount in dB."""
    if gain_db is None:
        gain_db = np.random.uniform(-6, 6)
    return y * (10 ** (gain_db / 20))


AUGMENTATIONS = {
    "timeshift":    time_shift,
    "pitchshift":   pitch_shift,
    "timestretch":  time_stretch,
    "addnoise":     add_noise,
    "gainvar":      gain_variation,
}


def apply_random_augmentation(y, sr):
    """Apply 1–2 random augmentation techniques."""
    n_augs = random.choice([1, 2])
    chosen = random.sample(list(AUGMENTATIONS.keys()), n_augs)
    technique_name = "+".join(chosen)
    for name in chosen:
        y = AUGMENTATIONS[name](y, sr)
    # Clip to prevent clipping
    y = np.clip(y, -1.0, 1.0)
    return y, technique_name


# ─────────────────── Core Pipeline ───────────────────

def load_and_normalise(filepath):
    """Load audio, resample to TARGET_SR, pad/trim to CLIP_DURATION."""
    y, sr = librosa.load(filepath, sr=TARGET_SR, duration=CLIP_DURATION)
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    elif len(y) > TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]
    return y, TARGET_SR


def pool_originals():
    """Collect all original .wav files grouped by class."""
    class_files = defaultdict(list)
    for split in ["train", "validate", "test"]:
        for cls in CLASSES:
            cls_dir = os.path.join(SRC_DATA_DIR, split, cls)
            if os.path.isdir(cls_dir):
                for f in sorted(glob.glob(os.path.join(cls_dir, "*.wav"))):
                    class_files[cls].append(f)
    return class_files


def resplit(class_files, train_ratio=0.70, val_ratio=0.15):
    """Re-split originals into train/validate/test (70/15/15)."""
    splits = {"train": defaultdict(list),
              "validate": defaultdict(list),
              "test": defaultdict(list)}
    for cls, files in class_files.items():
        files = sorted(files)
        random.shuffle(files)
        n = len(files)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        splits["train"][cls]    = files[:n_train]
        splits["validate"][cls] = files[n_train:n_train + n_val]
        splits["test"][cls]     = files[n_train + n_val:]
    return splits


def save_wav(filepath, y, sr):
    """Save audio as 16-bit WAV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sf.write(filepath, y, sr, subtype="PCM_16")


def augment_split(split_name, cls, original_files, target_count):
    """
    Copy originals and generate augmented samples until target_count is met.
    Returns list of dicts for the dataset index.
    """
    records = []
    out_dir = os.path.join(DST_DATA_DIR, split_name, cls)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Copy originals
    for fpath in original_files:
        y, sr = load_and_normalise(fpath)
        basename = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(out_dir, f"{basename}.wav")
        save_wav(out_path, y, sr)
        records.append({
            "filepath": os.path.abspath(out_path),
            "split": split_name,
            "label": cls,
            "duration_sec": CLIP_DURATION,
            "sampling_rate": sr,
            "source": "original",
            "augmentation": "none",
            "original_file": os.path.basename(fpath),
        })

    # 2. Augment to reach target
    n_originals = len(original_files)
    n_needed = target_count - n_originals
    if n_needed <= 0:
        return records

    aug_counter = 0
    while aug_counter < n_needed:
        # Pick a random original
        src_path = random.choice(original_files)
        y, sr = load_and_normalise(src_path)
        y_aug, technique = apply_random_augmentation(y, sr)

        basename = os.path.splitext(os.path.basename(src_path))[0]
        aug_counter += 1
        out_name = f"{basename}_aug{aug_counter:04d}_{technique}.wav"
        out_path = os.path.join(out_dir, out_name)
        save_wav(out_path, y_aug, sr)
        records.append({
            "filepath": os.path.abspath(out_path),
            "split": split_name,
            "label": cls,
            "duration_sec": CLIP_DURATION,
            "sampling_rate": sr,
            "source": "augmented",
            "augmentation": technique,
            "original_file": os.path.basename(src_path),
        })

    return records


def main():
    print("=" * 60)
    print("DATASET SCALING PIPELINE")
    print("=" * 60)

    # Step 1: Pool originals
    print("\n[1/4] Pooling all original recordings...")
    class_files = pool_originals()
    for cls, files in class_files.items():
        print(f"  {cls}: {len(files)} original files")

    # Step 2: Re-split
    print("\n[2/4] Re-splitting originals (70/15/15)...")
    splits = resplit(class_files)
    for split_name, classes in splits.items():
        for cls, files in classes.items():
            print(f"  {split_name}/{cls}: {len(files)} originals")

    # Step 3: Clean output directory
    if os.path.exists(DST_DATA_DIR):
        shutil.rmtree(DST_DATA_DIR)

    # Step 4: Augment each split
    targets = {
        "train":    TRAIN_TARGET_PER_CLASS,
        "validate": VAL_TARGET_PER_CLASS,
        "test":     TEST_TARGET_PER_CLASS,
    }

    all_records = []
    print("\n[3/4] Augmenting...")
    for split_name in ["train", "validate", "test"]:
        target = targets[split_name]
        for cls in CLASSES:
            originals = splits[split_name][cls]
            print(f"  {split_name}/{cls}: {len(originals)} originals -> target {target}...", end=" ")
            records = augment_split(split_name, cls, originals, target)
            all_records.extend(records)
            n_orig = sum(1 for r in records if r["source"] == "original")
            n_aug  = sum(1 for r in records if r["source"] == "augmented")
            print(f"done ({n_orig} original + {n_aug} augmented = {len(records)} total)")

    # Step 5: Save dataset index
    import csv
    index_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "scaled_dataset_index.csv"
    )
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filepath", "split", "label", "duration_sec",
            "sampling_rate", "source", "augmentation", "original_file"
        ])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n[4/4] Dataset index saved to {index_path}")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    from collections import Counter
    split_class_counts = Counter()
    split_source_counts = Counter()
    for r in all_records:
        split_class_counts[(r["split"], r["label"])] += 1
        split_source_counts[(r["split"], r["source"])] += 1

    for split in ["train", "validate", "test"]:
        print(f"\n  {split.upper()}:")
        for cls in CLASSES:
            print(f"    {cls}: {split_class_counts[(split, cls)]}")
        print(f"    --- originals: {split_source_counts[(split, 'original')]}, "
              f"augmented: {split_source_counts[(split, 'augmented')]}")

    total = len(all_records)
    print(f"\n  GRAND TOTAL: {total} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
