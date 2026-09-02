import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

try:
    from src.features import ElephantAcousticDataset
    from src.model import ElephantCNN, NUM_CLASSES, CLASS_NAMES
except ImportError:
    from features import ElephantAcousticDataset
    from model import ElephantCNN, NUM_CLASSES, CLASS_NAMES


def train_model(data_path, epochs=10, batch_size=16):
    print("=" * 60)
    print("ELEPHANT CNN TRAINING PIPELINE")
    print("=" * 60)
    print(f"Classes: {CLASS_NAMES}")
    print(f"Number of classes: {NUM_CLASSES}")

    # ─── Setup Datasets ───
    print("\n[1/3] Loading datasets...")
    train_dataset = ElephantAcousticDataset(data_path, split='train', feature_type='mel')
    val_dataset = ElephantAcousticDataset(data_path, split='validate', feature_type='mel')
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ─── Model, Loss, Optimizer ───
    model = ElephantCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    # ─── Training Loop ───
    print(f"\n[2/3] Training for {epochs} epochs (batch_size={batch_size})...")
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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

        # --- Validation Phase ---
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

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')

        print(f"  Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
              f"{' (BEST)' if val_acc >= best_val_acc else ''}")

    # ─── Save Outputs ───
    print(f"\n[3/3] Saving outputs...")
    os.makedirs('models', exist_ok=True)

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/training_curves.png')
    plt.close()

    # Save final model (last epoch)
    model_path = 'models/baseline_cnn.pth'
    torch.save(model.state_dict(), model_path)

    print(f"  Final model saved to {model_path}")
    print(f"  Best model saved to models/best_model.pth (Val Acc: {best_val_acc:.2f}%)")
    print(f"  Training curves saved to models/training_curves.png")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    csv_path = r"c:\Users\kamal\Downloads\MACONFLIC-main\elephant_vocalization_detection\data\scaled_dataset_index.csv"
    train_model(csv_path, epochs=15, batch_size=32)
