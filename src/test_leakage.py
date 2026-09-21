"""
Data Leakage Test — Evaluate on original (non-augmented) test files only.

If accuracy drops significantly compared to the full augmented test set,
it suggests the model is memorising augmentation artifacts rather than
learning genuine acoustic features.

Usage:
    python -m src.test_leakage

Output:
    results/leakage_test_report.txt
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

try:
    from src.features import ElephantAcousticDataset
    from src.model_v2 import ElephantCNNv2, NUM_CLASSES, CLASS_NAMES
    from src.model import ElephantCNN as ElephantCNNv1
except ImportError:
    from features import ElephantAcousticDataset
    from model_v2 import ElephantCNNv2, NUM_CLASSES, CLASS_NAMES
    from model import ElephantCNN as ElephantCNNv1


class OriginalOnlyDataset(ElephantAcousticDataset):
    """Dataset that loads ONLY original (non-augmented) test samples."""

    def __init__(self, csv_file, feature_type='mel'):
        super().__init__(csv_file, split='test', feature_type=feature_type)

        # Filter to original samples only
        if 'source' in self.data.columns:
            original_count = len(self.data)
            self.data = self.data[self.data['source'] == 'original']
            self.data = self.data.reset_index(drop=True)
            filtered_count = len(self.data)
            print(f"  Filtered: {original_count} total test → {filtered_count} original-only test")
        else:
            print("  ⚠ No 'source' column found — using all test samples")


def evaluate_model(model, data_loader, device):
    """Run evaluation and return predictions + labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def test_leakage(csv_path=None, model_path=None):
    """
    Run the data leakage test.

    Evaluates the model on:
    1. Full augmented test set (400 samples)
    2. Original-only test set (should be ~36 samples)

    And compares the accuracy/F1 between them.
    """
    print("=" * 60)
    print("DATA LEAKAGE TEST")
    print("=" * 60)

    # ─── Resolve Paths ───
    if csv_path is None:
        csv_path = os.path.join('data', 'scaled_dataset_index.csv')
        if not os.path.exists(csv_path):
            csv_path = r"c:\Users\kamal\OneDrive\Desktop\PROJECTS\FINAL_YEAR_PROJECT\MACONFLIC-main\elephant_vocalization_detection\data\scaled_dataset_index.csv"

    if model_path is None:
        for candidate in ['models/best_model_v2.pth', 'models/baseline_cnn_v2.pth',
                          'models/best_model.pth', 'models/baseline_cnn.pth']:
            if os.path.exists(candidate):
                model_path = candidate
                break

    print(f"  Dataset: {csv_path}")
    print(f"  Model:   {model_path}")

    # ─── Load Model ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:  {device}")

    # Try v2 first, then v1
    try:
        model = ElephantCNNv2(num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("  ✓ Loaded v2 model")
    except RuntimeError:
        model = ElephantCNNv1(num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("  ✓ Loaded v1 model (v2 architecture mismatch)")

    # ─── Test 1: Full Augmented Test Set ───
    print("\n[1/2] Evaluating on FULL augmented test set...")
    full_dataset = ElephantAcousticDataset(csv_path, split='test', feature_type='mel')
    full_loader = DataLoader(full_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"  Samples: {len(full_dataset)}")

    full_preds, full_labels = evaluate_model(model, full_loader, device)
    full_acc = accuracy_score(full_labels, full_preds) * 100
    full_report = classification_report(
        full_labels, full_preds,
        target_names=CLASS_NAMES, labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    print(f"  Accuracy: {full_acc:.2f}%")
    print(full_report)

    # ─── Test 2: Original-Only Test Set ───
    print("\n[2/2] Evaluating on ORIGINAL-ONLY test set...")
    orig_dataset = OriginalOnlyDataset(csv_path, feature_type='mel')
    orig_loader = DataLoader(orig_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"  Samples: {len(orig_dataset)}")

    if len(orig_dataset) == 0:
        print("  ⚠ No original test samples found. Cannot perform leakage test.")
        return

    orig_preds, orig_labels = evaluate_model(model, orig_loader, device)
    orig_acc = accuracy_score(orig_labels, orig_preds) * 100
    orig_report = classification_report(
        orig_labels, orig_preds,
        target_names=CLASS_NAMES, labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    print(f"  Accuracy: {orig_acc:.2f}%")
    print(orig_report)

    # ─── Comparison ───
    acc_diff = full_acc - orig_acc
    print("\n" + "=" * 60)
    print("DATA LEAKAGE ANALYSIS")
    print("=" * 60)
    print(f"  Full test set accuracy:      {full_acc:.2f}%  ({len(full_dataset)} samples)")
    print(f"  Original-only accuracy:      {orig_acc:.2f}%  ({len(orig_dataset)} samples)")
    print(f"  Accuracy difference:         {acc_diff:+.2f}%")

    if abs(acc_diff) < 5:
        verdict = "✅ NO significant data leakage detected. The model generalises well."
    elif abs(acc_diff) < 15:
        verdict = "⚠️ MILD data leakage suspected. Accuracy drops moderately on original-only samples."
    else:
        verdict = "🔴 SIGNIFICANT data leakage likely. The model may be memorising augmentation patterns."

    print(f"\n  Verdict: {verdict}")

    # ─── Save Report ───
    os.makedirs('results', exist_ok=True)
    report_path = 'results/leakage_test_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DATA LEAKAGE TEST REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {csv_path}\n\n")
        f.write(f"Full test set ({len(full_dataset)} samples):\n")
        f.write(f"  Accuracy: {full_acc:.2f}%\n")
        f.write(full_report + "\n\n")
        f.write(f"Original-only test set ({len(orig_dataset)} samples):\n")
        f.write(f"  Accuracy: {orig_acc:.2f}%\n")
        f.write(orig_report + "\n\n")
        f.write(f"Accuracy difference: {acc_diff:+.2f}%\n")
        f.write(f"Verdict: {verdict}\n")

    print(f"\n  Report saved to: {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    test_leakage()
