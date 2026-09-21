"""
Enhanced training pipeline for ElephantCNN v2.

Improvements over train.py:
    - ReduceLROnPlateau learning rate scheduler
    - Early stopping (patience=7)
    - Label smoothing (0.1) in CrossEntropyLoss
    - Class-weighted loss to handle Roar underperformance
    - SpecAugment (frequency + time masking) during training
    - Trains for 30-50 epochs (early stopping prevents overfitting)
    - Saves training history as JSON for reproducibility
    - Supports both v1 and v2 model architectures
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

try:
    from src.features import ElephantAcousticDataset
    from src.model_v2 import ElephantCNNv2, NUM_CLASSES, CLASS_NAMES
except ImportError:
    from features import ElephantAcousticDataset
    from model_v2 import ElephantCNNv2, NUM_CLASSES, CLASS_NAMES


# ─── SpecAugment ───────────────────────────────────
class SpecAugment(nn.Module):
    """
    SpecAugment: frequency and time masking for spectrograms.

    Applied during training only. Randomly masks contiguous bands
    of frequency bins and time steps to improve model robustness.

    Reference: Park et al., "SpecAugment: A Simple Data Augmentation
    Method for Automatic Speech Recognition", Interspeech 2019.
    """
    def __init__(self, freq_mask_param=15, time_mask_param=25, num_freq_masks=2, num_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x):
        if not self.training:
            return x

        cloned = x.clone()
        _, _, n_mels, n_time = cloned.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, n_mels - f))
            cloned[:, :, f0:f0 + f, :] = 0

        # Time masking
        for _ in range(self.num_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, n_time - t))
            cloned[:, :, :, t0:t0 + t] = 0

        return cloned


def compute_class_weights(dataset, num_classes):
    """
    Compute inverse-frequency class weights to balance the loss function.
    Classes with fewer correct predictions get higher weight.
    """
    class_counts = np.zeros(num_classes)
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_counts[label.item()] += 1

    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-8)
    # Normalise so weights sum to num_classes
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)


def train_model_v2(data_path, epochs=30, batch_size=32, lr=0.001,
                   patience=7, label_smoothing=0.1, use_class_weights=True,
                   use_spec_augment=True):
    """
    Train the ElephantCNN v2 model with all improvements.

    Args:
        data_path: Path to the scaled_dataset_index.csv
        epochs: Maximum number of epochs (early stopping may stop earlier)
        batch_size: Training batch size
        lr: Initial learning rate
        patience: Early stopping patience (epochs without improvement)
        label_smoothing: Label smoothing factor for CrossEntropyLoss
        use_class_weights: Whether to use class-weighted loss
        use_spec_augment: Whether to apply SpecAugment during training
    """
    print("=" * 60)
    print("ELEPHANT CNN v2 — ENHANCED TRAINING PIPELINE")
    print("=" * 60)
    print(f"Classes: {CLASS_NAMES}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Max epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Initial LR: {lr}")
    print(f"Early stopping patience: {patience}")
    print(f"Label smoothing: {label_smoothing}")
    print(f"Class weights: {use_class_weights}")
    print(f"SpecAugment: {use_spec_augment}")

    # ─── Setup Datasets ───
    print("\n[1/4] Loading datasets...")
    train_dataset = ElephantAcousticDataset(data_path, split='train', feature_type='mel')
    val_dataset = ElephantAcousticDataset(data_path, split='validate', feature_type='mel')
    print(f"  Train samples:      {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ─── Model ───
    model = ElephantCNNv2(num_classes=NUM_CLASSES).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ─── Loss Function ───
    print("\n[2/4] Setting up training components...")
    if use_class_weights:
        print("  Computing class weights...")
        class_weights = compute_class_weights(train_dataset, NUM_CLASSES).to(device)
        print(f"  Class weights: {dict(zip(CLASS_NAMES, class_weights.cpu().numpy().round(3)))}")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ─── Optimizer & Scheduler ───
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    print(f"  Optimizer: Adam (lr={lr}, weight_decay=1e-4)")
    print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")

    # ─── SpecAugment ───
    spec_augment = SpecAugment() if use_spec_augment else None
    if spec_augment:
        spec_augment = spec_augment.to(device)

    # ─── Training Loop ───
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    lr_history = []
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    print(f"\n[3/4] Training for up to {epochs} epochs...")
    print("-" * 80)

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # ─── Training Phase ───
        model.train()
        if spec_augment:
            spec_augment.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply SpecAugment
            if spec_augment:
                inputs = spec_augment(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ─── Validation Phase ───
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # ─── Learning Rate Scheduling ───
        scheduler.step(val_loss)

        # ─── Best Model Tracking ───
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model_v2.pth')
        else:
            epochs_without_improvement += 1

        # ─── Logging ───
        marker = " ★ BEST" if is_best else ""
        print(f"  Epoch {epoch+1:02d}/{epochs} | "
          f"LR: {current_lr:.1e} | "
          f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
          f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%{marker}")

        # ─── Early Stopping ───
        if epochs_without_improvement >= patience:
            print(f"\n  ⚠ Early stopping triggered after {epoch+1} epochs "
                  f"(no improvement for {patience} epochs)")
            break

    print("-" * 80)

    # ─── Save Outputs ───
    print(f"\n[4/4] Saving outputs...")
    os.makedirs('models', exist_ok=True)

    # Save final model (last epoch)
    final_model_path = 'models/baseline_cnn_v2.pth'
    torch.save(model.state_dict(), final_model_path)

    # Save training history as JSON
    history = {
        'class_names': CLASS_NAMES,
        'num_classes': NUM_CLASSES,
        'epochs_trained': len(train_losses),
        'max_epochs': epochs,
        'batch_size': batch_size,
        'initial_lr': lr,
        'label_smoothing': label_smoothing,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'total_params': total_params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'lr_history': lr_history,
    }
    history_path = 'models/training_history_v2.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # ─── Plot Training Curves ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(train_losses, label='Train Loss', color='#e74c3c', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', color='#3498db', linewidth=2)
    axes[0].set_title('Loss per Epoch', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(train_accs, label='Train Acc', color='#e74c3c', linewidth=2)
    axes[1].plot(val_accs, label='Val Acc', color='#3498db', linewidth=2)
    axes[1].axhline(y=best_val_acc, color='green', linestyle='--', alpha=0.5,
                    label=f'Best Val Acc: {best_val_acc:.2f}%')
    axes[1].set_title('Accuracy per Epoch', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(lr_history, color='#9b59b6', linewidth=2, marker='o', markersize=3)
    axes[2].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('ElephantCNN v2 — Training Curves', fontsize=15, fontweight='bold')
    plt.tight_layout()
    curves_path = 'models/training_curves_v2.png'
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Summary ───
    print(f"\n  Final model saved to:    {final_model_path}")
    print(f"  Best model saved to:     models/best_model_v2.pth (Val Acc: {best_val_acc:.2f}%)")
    print(f"  Training curves saved to: {curves_path}")
    print(f"  Training history saved to: {history_path}")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    # Use the scaled dataset in the current project location
    csv_path = os.path.join('data', 'scaled_dataset_index.csv')

    # Fallback to absolute path if relative doesn't exist
    if not os.path.exists(csv_path):
        csv_path = r"c:\Users\kamal\OneDrive\Desktop\PROJECTS\FINAL_YEAR_PROJECT\MACONFLIC-main\elephant_vocalization_detection\data\scaled_dataset_index.csv"

    train_model_v2(
        data_path=csv_path,
        epochs=30,
        batch_size=32,
        lr=0.001,
        patience=7,
        label_smoothing=0.1,
        use_class_weights=True,
        use_spec_augment=True,
    )
