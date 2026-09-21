"""
5-Fold Stratified Cross-Validation for ElephantCNN.

Provides statistically reliable metrics by training 5 separate models
on different data splits and reporting mean ± std for all metrics.

Usage:
    python -m src.cross_validate

Output:
    results/cross_validation_report.txt
    results/cross_validation_summary.json
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from src.features import ElephantAcousticDataset, TARGET_SR, CLIP_DURATION, N_MELS, FMAX
    from src.model_v2 import ElephantCNNv2, NUM_CLASSES, CLASS_NAMES
except ImportError:
    from features import ElephantAcousticDataset, TARGET_SR, CLIP_DURATION, N_MELS, FMAX
    from model_v2 import ElephantCNNv2, NUM_CLASSES, CLASS_NAMES


class FullDataset(ElephantAcousticDataset):
    """
    Modified dataset that loads ALL samples (ignoring the split column)
    so we can perform our own k-fold splitting.
    """
    def __init__(self, csv_file, feature_type='mel'):
        # Read full CSV without filtering by split
        self.data = pd.read_csv(csv_file)
        if 'corrupted' in self.data.columns:
            self.data = self.data[~self.data['corrupted']]

        # Only use ORIGINAL samples (not augmented) to prevent data leakage
        if 'source' in self.data.columns:
            self.data = self.data[self.data['source'] == 'original']

        self.data = self.data.reset_index(drop=True)
        self.feature_type = feature_type
        self.target_sr = TARGET_SR
        self.duration = CLIP_DURATION

        self.label_map = {
            'Roar': 0, 'Rumble': 1, 'Trumpet': 2, 'Non_Elephant': 3,
        }

    def get_labels(self):
        """Return all labels as a numpy array (for stratified splitting)."""
        return np.array([self.label_map.get(l, 0) for l in self.data['label']])


def train_fold(model, train_loader, val_loader, device, epochs=15, lr=0.001):
    """Train a single fold and return validation predictions."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        scheduler.step(val_loss / len(val_loader))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best model for final evaluation
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation on validation set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), best_val_acc


def cross_validate(csv_path=None, n_folds=5, epochs_per_fold=15, batch_size=32):
    """
    Run k-fold stratified cross-validation.

    Args:
        csv_path: Path to the dataset index CSV.
        n_folds: Number of folds (default: 5).
        epochs_per_fold: Epochs to train each fold.
        batch_size: Batch size.
    """
    print("=" * 60)
    print(f"{n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)

    # ─── Resolve Path ───
    if csv_path is None:
        csv_path = os.path.join('data', 'scaled_dataset_index.csv')
        if not os.path.exists(csv_path):
            csv_path = r"c:\Users\kamal\OneDrive\Desktop\PROJECTS\FINAL_YEAR_PROJECT\MACONFLIC-main\elephant_vocalization_detection\data\scaled_dataset_index.csv"

    # ─── Load Dataset ───
    print("\n[1/3] Loading dataset (original samples only)...")
    dataset = FullDataset(csv_path, feature_type='mel')
    labels = dataset.get_labels()
    print(f"  Total original samples: {len(dataset)}")

    class_counts = {CLASS_NAMES[i]: (labels == i).sum() for i in range(NUM_CLASSES)}
    print(f"  Class distribution: {class_counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ─── K-Fold Split ───
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_f1_macros = []
    fold_reports = []
    all_fold_preds = []
    all_fold_labels = []

    print(f"\n[2/3] Training {n_folds} folds ({epochs_per_fold} epochs each)...")
    print("-" * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        print(f"\n  ── Fold {fold+1}/{n_folds} ──")
        print(f"     Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Fresh model for each fold
        model = ElephantCNNv2(num_classes=NUM_CLASSES).to(device)

        preds, true_labels, best_acc = train_fold(
            model, train_loader, val_loader, device,
            epochs=epochs_per_fold, lr=0.001
        )

        accuracy = accuracy_score(true_labels, preds) * 100
        f1_macro = f1_score(true_labels, preds, average='macro') * 100
        report = classification_report(
            true_labels, preds,
            target_names=CLASS_NAMES,
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
            output_dict=True
        )

        fold_accuracies.append(accuracy)
        fold_f1_macros.append(f1_macro)
        fold_reports.append(report)
        all_fold_preds.extend(preds)
        all_fold_labels.extend(true_labels)

        print(f"     Accuracy: {accuracy:.2f}% | Macro F1: {f1_macro:.2f}%")

    print("\n" + "-" * 60)

    # ─── Aggregate Results ───
    print(f"\n[3/3] Aggregating results...")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_macros)
    std_f1 = np.std(fold_f1_macros)

    # Per-class aggregation
    per_class_f1 = {cls: [] for cls in CLASS_NAMES}
    per_class_precision = {cls: [] for cls in CLASS_NAMES}
    per_class_recall = {cls: [] for cls in CLASS_NAMES}

    for report in fold_reports:
        for cls in CLASS_NAMES:
            if cls in report:
                per_class_f1[cls].append(report[cls]['f1-score'] * 100)
                per_class_precision[cls].append(report[cls]['precision'] * 100)
                per_class_recall[cls].append(report[cls]['recall'] * 100)

    # ─── Print Summary ───
    print("\n" + "=" * 60)
    print(f"{n_folds}-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"\n  Overall Accuracy:  {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Macro F1-Score:    {mean_f1:.2f}% ± {std_f1:.2f}%")
    print(f"\n  Per-Class Results (mean ± std):")
    print(f"  {'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print(f"  {'-'*51}")

    for cls in CLASS_NAMES:
        p_mean = np.mean(per_class_precision[cls])
        p_std = np.std(per_class_precision[cls])
        r_mean = np.mean(per_class_recall[cls])
        r_std = np.std(per_class_recall[cls])
        f_mean = np.mean(per_class_f1[cls])
        f_std = np.std(per_class_f1[cls])
        print(f"  {cls:<15} {p_mean:5.1f}±{p_std:4.1f}%  {r_mean:5.1f}±{r_std:4.1f}%  {f_mean:5.1f}±{f_std:4.1f}%")

    print(f"\n  Per-Fold Accuracies: {[f'{a:.2f}%' for a in fold_accuracies]}")

    # ─── Save Results ───
    os.makedirs('results', exist_ok=True)

    # Save text report
    report_path = 'results/cross_validation_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"{n_folds}-FOLD STRATIFIED CROSS-VALIDATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"\nDataset: {csv_path}\n")
        f.write(f"Folds: {n_folds}\n")
        f.write(f"Epochs per fold: {epochs_per_fold}\n")
        f.write(f"Total original samples: {len(dataset)}\n")
        f.write(f"\nOverall Accuracy:  {mean_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write(f"Macro F1-Score:    {mean_f1:.2f}% ± {std_f1:.2f}%\n")
        f.write(f"\nPer-Class Results (mean ± std):\n")
        for cls in CLASS_NAMES:
            f_mean = np.mean(per_class_f1[cls])
            f_std = np.std(per_class_f1[cls])
            f.write(f"  {cls}: F1 = {f_mean:.2f}% ± {f_std:.2f}%\n")
        f.write(f"\nPer-Fold Accuracies: {fold_accuracies}\n")

    # Save JSON summary
    summary_path = 'results/cross_validation_summary.json'
    summary = {
        'n_folds': n_folds,
        'epochs_per_fold': epochs_per_fold,
        'total_samples': len(dataset),
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_f1_macro': mean_f1,
        'std_f1_macro': std_f1,
        'fold_accuracies': fold_accuracies,
        'fold_f1_macros': fold_f1_macros,
        'per_class_f1_mean': {cls: float(np.mean(per_class_f1[cls])) for cls in CLASS_NAMES},
        'per_class_f1_std': {cls: float(np.std(per_class_f1[cls])) for cls in CLASS_NAMES},
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Report saved to: {report_path}")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 60)


if __name__ == '__main__':
    cross_validate(n_folds=5, epochs_per_fold=15, batch_size=32)
