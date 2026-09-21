"""
Generate a Grad-CAM visual report with heatmaps for representative
samples of each class. Produces a publication-ready grid.

Usage:
    python -m src.generate_gradcam_report

Output:
    results/gradcam_report.png — Grid showing Grad-CAM overlays per class
    results/gradcam_samples/   — Individual Grad-CAM images per sample
"""

import os
import torch
import pandas as pd
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from src.features import load_and_pad_audio, TARGET_SR, CLIP_DURATION, N_MELS, FMAX
    from src.model_v2 import ElephantCNNv2, NUM_CLASSES, CLASS_NAMES
    from src.gradcam import GradCAM, create_gradcam_overlay
except ImportError:
    from features import load_and_pad_audio, TARGET_SR, CLIP_DURATION, N_MELS, FMAX
    from model_v2 import ElephantCNNv2, NUM_CLASSES, CLASS_NAMES
    from gradcam import GradCAM, create_gradcam_overlay


def generate_report(csv_path=None, model_path=None, num_samples_per_class=1):
    """
    Generate Grad-CAM visual report.

    Args:
        csv_path: Path to the dataset index CSV.
        model_path: Path to the trained model weights.
        num_samples_per_class: Number of samples to visualise per class.
    """
    print("=" * 60)
    print("GRAD-CAM REPORT GENERATOR")
    print("=" * 60)

    # ─── Resolve Paths ───
    if csv_path is None:
        csv_path = os.path.join('data', 'scaled_dataset_index.csv')
        if not os.path.exists(csv_path):
            csv_path = r"c:\Users\kamal\OneDrive\Desktop\PROJECTS\FINAL_YEAR_PROJECT\MACONFLIC-main\elephant_vocalization_detection\data\scaled_dataset_index.csv"

    if model_path is None:
        # Try v2 model first, then fall back to v1
        model_path = os.path.join('models', 'best_model_v2.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join('models', 'baseline_cnn_v2.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join('models', 'best_model.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join('models', 'baseline_cnn.pth')

    print(f"  Dataset: {csv_path}")
    print(f"  Model:   {model_path}")

    # ─── Load Model ───
    print("\n[1/4] Loading model...")
    device = torch.device('cpu')  # Grad-CAM needs gradients, CPU is fine
    model = ElephantCNNv2(num_classes=NUM_CLASSES).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"  ✓ Loaded v2 model from {model_path}")
    except RuntimeError:
        # If v2 model fails (architecture mismatch), try loading v1
        print("  ⚠ v2 model load failed, trying v1 architecture...")
        from model import ElephantCNN as ElephantCNNv1
        model = ElephantCNNv1(num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"  ✓ Loaded v1 model from {model_path}")

    model.eval()

    # Determine the last conv layer for Grad-CAM
    if hasattr(model, 'get_last_conv_layer'):
        target_layer = model.get_last_conv_layer()
    elif hasattr(model, 'conv4'):
        target_layer = model.conv4
    else:
        target_layer = model.conv3
    print(f"  Grad-CAM target: {target_layer}")

    # ─── Load Dataset ───
    print("\n[2/4] Selecting representative samples...")
    df = pd.read_csv(csv_path)
    test_df = df[df['split'] == 'test']

    # Try to use original (non-augmented) samples first for cleaner visualisation
    if 'source' in test_df.columns:
        originals = test_df[test_df['source'] == 'original']
        if len(originals) > 0:
            test_df = originals

    selected_samples = []
    for class_name in CLASS_NAMES:
        class_df = test_df[test_df['label'] == class_name]
        if len(class_df) == 0:
            print(f"  ⚠ No test samples found for class '{class_name}', skipping")
            continue
        samples = class_df.sample(n=min(num_samples_per_class, len(class_df)), random_state=42)
        for _, row in samples.iterrows():
            selected_samples.append(row)

    print(f"  Selected {len(selected_samples)} samples across {len(CLASS_NAMES)} classes")

    # ─── Generate Grad-CAMs ───
    print("\n[3/4] Generating Grad-CAM heatmaps...")
    cam = GradCAM(model, target_layer)

    os.makedirs('results/gradcam_samples', exist_ok=True)
    all_specs = []
    all_heatmaps = []
    all_preds = []
    all_confs = []
    all_true_labels = []

    for i, sample in enumerate(selected_samples):
        filepath = sample['filepath']
        true_label = sample['label']

        # Load audio and extract spectrogram
        y = load_and_pad_audio(filepath, sr=TARGET_SR, duration=CLIP_DURATION)
        S = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS, fmax=FMAX)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Create tensor for model input
        mel_tensor = torch.FloatTensor(S_db).unsqueeze(0)  # (1, 128, T)

        # Generate Grad-CAM
        heatmap, pred_class, confidence = cam.generate(mel_tensor)

        all_specs.append(S_db)
        all_heatmaps.append(heatmap)
        all_preds.append(pred_class)
        all_confs.append(confidence)
        all_true_labels.append(true_label)

        pred_name = CLASS_NAMES[pred_class]
        correct = "✓" if pred_name == true_label else "✗"
        print(f"  [{i+1}/{len(selected_samples)}] True: {true_label:15s} → "
              f"Predicted: {pred_name:15s} ({confidence*100:.1f}%) {correct}")

        # Save individual overlay
        fig = create_gradcam_overlay(
            S_db, heatmap, pred_class, confidence, CLASS_NAMES,
            sr=TARGET_SR, fmax=FMAX,
            save_path=f'results/gradcam_samples/{true_label}_{i}.png'
        )
        plt.close(fig)

    cam.cleanup()

    # ─── Create Summary Grid ───
    print("\n[4/4] Creating summary grid...")
    n = len(selected_samples)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))

    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        true_label = all_true_labels[i]
        pred_name = CLASS_NAMES[all_preds[i]]
        conf_pct = all_confs[i] * 100
        correct = "✓" if pred_name == true_label else "✗"

        # Row 1: Original spectrogram
        librosa.display.specshow(all_specs[i], x_axis='time', y_axis='mel',
                                 sr=TARGET_SR, fmax=FMAX, ax=axes[0, i], cmap='magma')
        axes[0, i].set_title(f'True: {true_label}\nPred: {pred_name} ({conf_pct:.1f}%) {correct}',
                             fontsize=11, fontweight='bold',
                             color='green' if correct == "✓" else 'red')

        # Row 2: Grad-CAM overlay
        librosa.display.specshow(all_specs[i], x_axis='time', y_axis='mel',
                                 sr=TARGET_SR, fmax=FMAX, ax=axes[1, i], cmap='magma')
        extent = axes[1, i].get_xlim() + axes[1, i].get_ylim()
        axes[1, i].imshow(all_heatmaps[i], aspect='auto', origin='lower',
                          cmap='jet', alpha=0.45, extent=extent)
        axes[1, i].set_title('Grad-CAM Overlay', fontsize=10)

    axes[0, 0].set_ylabel('Mel-Spectrogram', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Grad-CAM Overlay', fontsize=12, fontweight='bold')

    plt.suptitle('🐘 Grad-CAM Explainability Report — Elephant Vocalization CNN',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    report_path = 'results/gradcam_report.png'
    fig.savefig(report_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  ✓ Summary grid saved to: {report_path}")
    print(f"  ✓ Individual overlays saved to: results/gradcam_samples/")
    print("=" * 60)
    print("GRAD-CAM REPORT COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    generate_report(num_samples_per_class=1)
