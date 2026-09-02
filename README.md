# 🐘 Elephant Vocalization Detection Using AI & ML Techniques

## Current Status & Summary
So far, we have built a complete end-to-end AI pipeline for elephant vocalization detection. The project has successfully evolved into a robust 4-class CNN model (capable of classifying Roar, Rumble, Trumpet, and Non-Elephant sounds). We achieved this by building out a full data pipeline that scales the dataset to 4,400 samples using audio augmentation. All steps from automated data downloading, synthetic noise generation, and Mel-spectrogram feature extraction to PyTorch model training, evaluation, and a fully interactive Streamlit web demo are now successfully implemented and operational.

**Recent Updates:**
- **Noise Robustness:** Upgraded the negative class generation to use dynamic physics-based DSP, synthesizing highly realistic jungle environmental sounds (rain, wind gusts, insect chirps) instead of static white noise.
- **Distance Analysis:** Implemented an acoustic physics simulator (`distance_simulation.py`) that applies the Inverse Square Law and high-frequency air absorption to test model detection accuracy over distances up to 1,000 meters.

## 1. Project Overview & Problem Statement

**Problem Statement:**  
*"To develop an AI-based system for detecting elephant vocalizations from acoustic recordings and analysing their acoustic characteristics."*

**The Goal:**  
Build a **Working AI Baseline Model** that can accurately classify **Elephant Vocalization Types** (Rumble, Roar, Trumpet) and distinguish them from **Non-Elephant Background Noise** — a 4-class classification problem.

**Why this approach?**  
This project strictly focuses on *acoustic signal detection* and *characteristic analysis*. It does not attempt to perform multimodal semantic-analysis or claim to "translate" elephant language. By keeping the initial system purely software-focused, we ensure a highly accurate, scientifically defensible core engine that can be extended later to IoT hardware and real-time field deployment.

**Source Dataset:** 
The audio data originates from the **"Elephant Sound Classification Using Raw Audio"** research project (Dewmini et al., *Sensors MDPI, 2025* — [ DOI: 10.3390/s25020352](https://doi.org/10.3390/s25020352)). The original dataset provides curated, pre-labeled 6-second `.wav` clips organised into three elephant caller types (Rumble, Roar, Trumpet) with train/validate/test splits.

## 2. What Has Been Accomplished — Step by Step
Below is a clear, chronological walkthrough of every step completed so far, what each script does, and what it produces.

### ✅ Step 1 — Data Download (`download_data.py`)

**What it does:**  
Automatically downloads the raw elephant audio recordings from the **Dryad African Elephant Acoustic Dataset** — a public repository hosted at `datadryad.org`.

**How it works:**
1. Downloads three zip files from Dryad:
   - `clips.zip` — Short segmented audio clips
   - `labels.zip` — Annotation/label files
   - `audio.zip` — Full-length audio streams
2. Extracts each zip into a corresponding folder inside `data/`.
3. Skips downloads that already exist (resume-safe).
4. Shows a **progress bar** (via `tqdm`) for each download.

**What it produces:**
```
data/
├── clips/       ← extracted audio clips
├── labels/      ← annotation files
└── audio/       ← full audio streams
```

**Command to run:**
```bash
python -m src.download_data
```

---

### ✅ Step 2 — Negative Class Generation (`generate_negatives.py`)

**Why this is needed:**  
The original dataset only contains elephant sounds (Rumble, Roar, Trumpet). For the AI to learn what is **NOT** an elephant, we need a "Non-Elephant" class with realistic background noise.

**What it does:**  
Generates synthetic 6-second `.wav` files containing environmental noise — no elephant sounds at all. The script uses digital signal processing to simulate realistic weather and wildlife instead of relying on downloaded audio files. Each file uses one of several dynamic noise profiles:

| Noise Type | How It's Generated | What It Simulates |
|---|---|---|
| **Rain** | Amplitude-modulated pink noise | A tropical jungle rainstorm |
| **Wind Gusts** | Low-frequency oscillating brown noise | Heavy wind blowing across the microphone |
| **Insects/Crickets** | High-frequency chirps via carrier signals | Ambient jungle nightlife |
| **White/Brown/Pink Noise** | Standard random distributions | Radio static, low thunder, rustling |

**Key parameters:**
- Duration: **6.0 seconds** (matches elephant clips exactly)
- Sample rate: **44,100 Hz**
- Random gain applied: **0.1 – 0.5** (so clips aren't all at the same volume)

**How many samples are generated:**
| Split | Count |
|---|---|
| Train | 77 |
| Validate | 18 |
| Test | 9 |

These counts roughly match the average number of samples in each elephant class, so the dataset starts balanced.

**What it produces:**
```
data/Audio-Classification-for-Elephant-Sounds/data/
├── train/Non_Elephant/      ← 77 synthetic .wav files
├── validate/Non_Elephant/   ← 18 synthetic .wav files
└── test/Non_Elephant/       ← 9 synthetic .wav files
```

**Command to run:**
```bash
python -m src.generate_negatives
```

---

### ✅ Step 3 — Dataset Profiling & Indexing (`prepare_dataset.py`)

**What it does:**  
Scans every `.wav` file across all splits (train/validate/test) and all classes (Roar, Rumble, Trumpet, Non_Elephant), reads their metadata using `librosa`, and builds a master CSV index.

**For each audio file, it extracts:**
- Absolute file path
- Split (train / validate / test)
- Class label (Roar / Rumble / Trumpet / Non_Elephant)
- Duration (in seconds)
- Sampling rate (in Hz)
- Whether the file is corrupted (True/False)

**It also prints a full statistical report:**
- Total number of audio files
- Class distribution (how many files per class)
- Split distribution (how many files per split)
- Duration statistics per class (min / max / average)
- All unique sampling rates found

**What it produces:**
```
data/dataset_index.csv    ← Master index of ALL original audio files with metadata
```

**Command to run:**
```bash
python -m src.prepare_dataset
```

---

### ✅ Step 4 — Dataset Scaling via Audio Augmentation (`augment_dataset.py`)

**Why this is needed:**  
The original dataset has only ~100 recordings per class. This is far too small for a CNN to generalise effectively — it would simply memorise the training data (overfit). We need thousands of samples per class.

**What it does — full pipeline (4 stages):**

#### Stage 1: Pool All Originals
Collects every `.wav` file from all three splits (train/validate/test) and all four classes into a single pool. This ensures no bias from the original split.

#### Stage 2: Clean Re-Split (70 / 15 / 15)
Re-distributes all original files into new train/validate/test splits using a **70/15/15 ratio**. This is critical: **no original file's augmented copies can appear in more than one split** (prevents data leakage).

#### Stage 3: Augment to Target Count
For each split and each class, copies the original files, then generates augmented variants until the target count is reached:

| Split | Target per Class | Total (4 classes) |
|---|---|---|
| **Train** | 500 | 2,000 |
| **Validate** | 500 | 2,000 |
| **Test** | 100 | 400 |
| | | **Grand Total: 4,400** |

Each augmented sample applies **1–2 random techniques** from the following table:

| Technique | Function | What It Simulates | Parameters |
|---|---|---|---|
| **Time Shifting** | `np.roll(y, shift)` | Vocalization not starting at clip onset | ±30% shift |
| **Pitch Shifting** | `librosa.effects.pitch_shift()` | Natural variation between individual elephants | ±2 semitones |
| **Time Stretching** | `librosa.effects.time_stretch()` | Variation in call duration | 0.85× – 1.15× speed |
| **Additive Noise** | White/Brown/Pink noise injection | Varying environmental conditions (wind, rain) | 0.002 – 0.015 level |
| **Gain Variation** | `y × 10^(dB/20)` | Different recording distances / mic sensitivity | ±6 dB |

**File naming convention:**  
Every augmented file is traceable back to its source:
```
{OriginalFileName}_aug{0001}_{pitchshift+addnoise}.wav
```

#### Stage 4: Save Dataset Index
Writes a comprehensive CSV with one row per file:

| Column | Description |
|---|---|
| `filepath` | Absolute path to the .wav file |
| `split` | train / validate / test |
| `label` | Roar / Rumble / Trumpet / Non_Elephant |
| `duration_sec` | Always 6.0 seconds |
| `sampling_rate` | 16,000 Hz (standardised) |
| `source` | "original" or "augmented" |
| `augmentation` | "none" or technique name(s) used |
| `original_file` | Basename of the source recording |

**What it produces:**
```
data/scaled_dataset/
├── train/
│   ├── Roar/           ← 500 .wav files
│   ├── Rumble/         ← 500 .wav files
│   ├── Trumpet/        ← 500 .wav files
│   └── Non_Elephant/   ← 500 .wav files
├── validate/
│   ├── Roar/           ← 500 .wav files
│   ├── Rumble/         ← 500 .wav files
│   ├── Trumpet/        ← 500 .wav files
│   └── Non_Elephant/   ← 500 .wav files
└── test/
    ├── Roar/           ← 100 .wav files
    ├── Rumble/         ← 100 .wav files
    ├── Trumpet/        ← 100 .wav files
    └── Non_Elephant/   ← 100 .wav files

data/scaled_dataset_index.csv   ← Master index of all 4,400 augmented files
```

**Command to run:**
```bash
python -m src.augment_dataset
```

---

### ✅ Step 5 — Feature Extraction & PyTorch Data Loading (`features.py`)

**What it does:**  
Converts raw audio waveforms into numerical representations (features) that a neural network can understand.

**The core feature: Mel-Spectrogram**

A Mel-Spectrogram is a 2D image representation of audio:
- **X-axis** = Time (seconds)
- **Y-axis** = Frequency (Hz), spaced on the Mel scale (mimics human hearing)
- **Color intensity** = Amplitude/Loudness (in decibels)

**Processing pipeline for each audio file:**
1. Load audio at **16,000 Hz** sample rate (resampled if different)
2. Pad or trim to exactly **6.0 seconds** (96,000 samples)
3. Compute **128-band Mel-spectrogram** (fmax = 8,000 Hz)
4. Convert power to decibels (`librosa.power_to_db`)
5. Wrap in a PyTorch `FloatTensor` with shape `(1, 128, time_steps)` — the `1` is the channel dimension (grayscale)

**Alternative feature:** Also supports **MFCC** (40-coefficient Mel-Frequency Cepstral Coefficients) extraction, selectable via `feature_type='mfcc'`.

**4-Class Label Mapping:**
| Class | Label Index |
|---|---|
| Roar | 0 |
| Rumble | 1 |
| Trumpet | 2 |
| Non_Elephant | 3 |

**Additional utility:** `generate_sample_plots()` creates a side-by-side visual comparing a Rumble sample and a Non_Elephant sample — both as waveform and Mel-spectrogram — saved as `data/sample_features.png`.

**Command to generate sample plots:**
```bash
python -m src.features
```

---

### ✅ Step 6 — CNN Model Architecture (`model.py`)

**What it is:**  
A lightweight **3-block Convolutional Neural Network** called `ElephantIntentCNN` designed to classify Mel-spectrogram images.

**Architecture diagram:**
```
Input: (Batch, 1, 128, TimeSteps)  ← Single-channel Mel-spectrogram
         │
         ▼
┌─────────────────────────────────┐
│  Conv2D(1 → 16, 3×3, pad=1)    │
│  BatchNorm2D(16)                │
│  ReLU                           │
│  MaxPool2D(2×2)                 │  ← Detects simple patterns (edges, lines)
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Conv2D(16 → 32, 3×3, pad=1)   │
│  BatchNorm2D(32)                │
│  ReLU                           │
│  MaxPool2D(2×2)                 │  ← Detects intermediate patterns (curves, textures)
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Conv2D(32 → 64, 3×3, pad=1)   │
│  BatchNorm2D(64)                │
│  ReLU                           │
│  MaxPool2D(2×2)                 │  ← Detects complex patterns (frequency sweeps)
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  AdaptiveAvgPool2D(4×4)         │  ← Forces output to fixed 4×4 regardless of input size
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Flatten → Dense(1024 → 128)    │
│  ReLU                           │
│  Dropout(0.5)                   │  ← Randomly drops 50% of neurons to prevent overfitting
│  Dense(128 → 4)                 │  ← Output: 4 class scores
└─────────────────────────────────┘
```

**Why these design choices:**
| Decision | Reason |
|---|---|
| `AdaptiveAvgPool2d(4,4)` | Handles variable-length audio — always produces the same size tensor for the dense layers |
| `BatchNorm` after every conv | Stabilises training by normalising layer outputs; allows higher learning rates |
| `Dropout(0.5)` | Prevents overfitting, especially important when training on augmented data |
| 3 conv blocks (16→32→64) | Progressively deeper feature hierarchy without being too heavy for the dataset size |

---

### ✅ Step 7 — Model Training (`train.py`)

**What it does:**  
Trains the CNN model on the scaled augmented dataset and saves the best weights.

**Training configuration:**
| Parameter | Value |
|---|---|
| Model | `SimpleCNN` (same architecture as ElephantIntentCNN, 4-class output) |
| Optimizer | **Adam** (learning rate = 0.001) |
| Loss Function | **CrossEntropyLoss** (standard for multi-class classification) |
| Epochs | **15** |
| Batch Size | **32** |
| Device | Auto-selects **CUDA GPU** if available, otherwise **CPU** |
| Dataset | `scaled_dataset_index.csv` (4,400 samples) |

**What happens each epoch:**
1. **Training phase:** Feed all training batches through the model, compute loss, backpropagate, update weights.
2. **Validation phase:** Feed all validation batches through the model (no weight updates), compute loss and accuracy.
3. Print both train and val metrics for that epoch.

**What it produces:**
```
models/
├── baseline_cnn.pth        ← Saved model weights (the "learned brain")
└── training_curves.png     ← Plot of Loss and Accuracy per epoch (train vs val)
```

**Command to run:**
```bash
python -m src.train
```

---

### ✅ Step 8 — Model Evaluation (`evaluate.py`)

**What it does:**  
Loads the trained model and evaluates it against the **held-out test set** (data the model has never seen during training).

**Metrics generated:**
| Metric | What It Measures |
|---|---|
| **Accuracy** | % of all test clips correctly classified |
| **Precision** (per class) | When the model predicted this class, how often was it right? |
| **Recall** (per class) | Out of all true samples of this class, how many did the model catch? |
| **F1-Score** (per class) | Harmonic mean of Precision and Recall — balances both |
| **Confusion Matrix** | 4×4 grid showing exactly where the model gets confused |

**Baseline results (from the initial binary model on unscaled data):**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Non-Elephant | 0.53 | 1.00 | 0.69 | 9 |
| Elephant | 1.00 | 0.70 | 0.83 | 27 |
| **Overall Accuracy** | | | **0.78** | **36** |

> ⚠️ **Note:** These results are from the initial binary classifier on unscaled data (only 36 test samples). The 4-class model trained on the scaled augmented dataset (4,400 samples, 400 test) is expected to produce significantly better and more reliable results.

**What it produces:**
```
models/
├── confusion_matrix.png     ← Heatmap of predicted vs actual labels
└── evaluation_report.txt    ← Full classification report (text)
```

**Command to run:**
```bash
python -m src.evaluate
```

---

### ✅ Step 9 — Interactive Demo App (`app.py`)

**What it is:**  
A **Streamlit web application** that lets anyone upload an audio file and get an instant AI prediction — no coding required.

**What you can do:**
1. **Upload** any `.wav`, `.mp3`, or `.flac` audio file
2. **Listen** to the audio playback in the browser
3. Click **"Analyze"** to get:
   - 🏷️ **Prediction:** Elephant Vocalization or Non-Elephant
   - 📊 **Confidence Score:** How certain the model is (0–100%)
   - 📐 **Acoustic Characteristics:** Duration (seconds) and Dominant Frequency (Hz)
   - 🎨 **Mel-Spectrogram:** Visual "fingerprint" of the uploaded audio

**Command to run:**
```bash
streamlit run app.py
```

---

### ✅ Step 10 — Distance Analysis Simulation (`analyze_distance.py`)

**What it does:**  
Runs a physics-based simulation to test how far away the AI can accurately detect an elephant. Since physical microphones placed at 1,000m intervals aren't available, we mathematically simulate distance using acoustic principles.

**How the physics engine works (`distance_simulation.py`):**
1. **Inverse Square Law:** Drops the overall volume as distance increases.
2. **High-Frequency Air Absorption:** Applies a dynamic low-pass filter (high-pitched Trumpets fade away faster than low-pitched Rumbles over long distances).

**What it produces:**
```
results/
└── distance_analysis_curve.png     ← Line graph plotting AI Confidence vs Distance (10m to 1000m)
```

**Command to run:**
```bash
python -m src.analyze_distance
```

---

## 3. Complete File Directory

```
elephant_vocalization_detection/
│
├── .gitignore                 # Git exclusion rules (data, models, results excluded)
├── README.md                  # This file — full project documentation
├── requirements.txt           # Python dependencies (pip install -r requirements.txt)
├── app.py                     # Streamlit web demo — upload audio & get predictions
│
├── src/                       # === ALL SOURCE CODE ===
│   ├── download_data.py       # Step 1: Downloads raw audio from Dryad
│   ├── generate_negatives.py  # Step 2: Creates synthetic Non_Elephant noise samples
│   ├── prepare_dataset.py     # Step 3: Profiles all audio → dataset_index.csv
│   ├── augment_dataset.py     # Step 4: Scales to 4,400 files via augmentation
│   ├── features.py            # Step 5: Audio → Mel-spectrogram tensors + PyTorch Dataset
│   ├── dataset.py             # Legacy PyTorch Dataset class (binary, unused now)
│   ├── model.py               # Step 6: ElephantIntentCNN architecture definition
│   ├── train.py               # Step 7: Training loop — Adam + CrossEntropy
│   ├── evaluate.py            # Step 8: Test evaluation + confusion matrix + report
│   ├── distance_simulation.py # Step 10: Acoustic physics engine for distance attenuation
│   └── analyze_distance.py    # Step 10: Runs distance analysis experiment & plots curve
│
├── data/                      # === ALL AUDIO DATA (git-ignored) ===
│   ├── Audio-Classification-for-Elephant-Sounds/  # Original dataset from GitHub
│   │   └── data/
│   │       ├── train/         # Original train split (Roar, Rumble, Trumpet, Non_Elephant)
│   │       ├── validate/      # Original validation split
│   │       └── test/          # Original test split
│   ├── scaled_dataset/        # Augmented dataset — 4,400 files total
│   │   ├── train/             # 2,000 files (500 × 4 classes)
│   │   ├── validate/          # 2,000 files (500 × 4 classes)
│   │   └── test/              # 400 files (100 × 4 classes)
│   ├── dataset_index.csv      # Index of original files (from Step 3)
│   ├── scaled_dataset_index.csv  # Index of all 4,400 augmented files (from Step 4)
│   └── sample_features.png    # Waveform + spectrogram comparison plot
│
├── models/                    # === TRAINED MODELS & ARTIFACTS (git-ignored) ===
│   ├── baseline_cnn.pth       # Trained 4-class model weights (from Step 7)
│   ├── best_model.pth         # Best binary model (legacy, used by app.py)
│   ├── training_curves.png    # Loss & accuracy plots over epochs
│   ├── confusion_matrix.png   # 4×4 confusion matrix heatmap
│   └── evaluation_report.txt  # Precision/Recall/F1 text report
│
└── results/                   # === EVALUATION OUTPUTS (git-ignored) ===
    └── confusion_matrix.png   # Legacy binary confusion matrix
```

---

## 4. Project Evolution Summary

| Version | Classification Type | Classes | Dataset Size | What Changed |
|---|---|---|---|---|
| **v1 (Initial)** | Binary | Elephant vs Non-Elephant | ~140 samples | Synthetic noise negatives only |
| **v2 (Current)** | **Multi-class (4)** | Roar, Rumble, Trumpet, Non_Elephant | **4,400 samples** | Audio augmentation pipeline, re-split, full scaling |

---

## 5. Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.x | Everything |
| **Deep Learning** | PyTorch | CNN model, training, inference |
| **Audio Processing** | Librosa, SoundFile, NumPy | Loading, resampling, Mel-spectrograms, augmentation |
| **Data Handling** | Pandas, CSV | Dataset indexing and metadata management |
| **Visualization** | Matplotlib, Seaborn | Spectrograms, training curves, confusion matrices |
| **Evaluation** | scikit-learn | Precision, Recall, F1, classification report |
| **Web Interface** | Streamlit | Interactive upload-and-predict demo |
| **Data Download** | Requests, tqdm | Downloading from Dryad with progress bars |

---

## 6. How to Run — Complete Pipeline

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Everything (in order)
```bash
# Step 1: Download raw data from Dryad
python -m src.download_data

# Step 2: Generate synthetic Non_Elephant noise samples
python -m src.generate_negatives

# Step 3: Profile the original dataset → data/dataset_index.csv
python -m src.prepare_dataset

# Step 4: Scale to 4,400 samples via augmentation → data/scaled_dataset/
python -m src.augment_dataset

# Step 5: (Optional) Generate sample waveform/spectrogram plots
python -m src.features

# Step 6: Train the CNN model → models/baseline_cnn.pth
python -m src.train

# Step 7: Evaluate on test set → models/evaluation_report.txt
python -m src.evaluate

# Step 8: Launch the interactive Streamlit demo
streamlit run app.py
```

---

## 7. Future Extensions

| Phase | Description | Status |
|---|---|---|
| **Noise Robustness** | Add real-world weather, traffic, and animal sounds to the negative class to stress-test the model | ✅ Completed (Synthesized Rain/Wind/Insects) |
| **Distance Analysis** | Test detection accuracy at varying microphone-to-elephant distances | ✅ Completed (Acoustic physics simulation) |
| **Infrasound Analysis** | Analyse sub-20Hz rumble frequencies that travel kilometres through the ground | ⏳ Pending |
| **IoT Deployment** | Export the model to edge devices (Raspberry Pi / NVIDIA Jetson) for real-time field monitoring | ⏳ Pending |

---

## 8. References

- **Source Dataset:** Dewmini, H.; Meedeniya, D.; Perera, C. *"Elephant Sound Classification Using Deep Learning Optimization."* Sensors 2025, 25, 352. [https://doi.org/10.3390/s25020352](https://doi.org/10.3390/s25020352)
- **Dryad Data Repository:** [https://datadryad.org](https://datadryad.org) — African Elephant Acoustic Dataset
