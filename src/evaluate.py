import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from src.features import ElephantAcousticDataset
    from src.model import ElephantCNN, NUM_CLASSES, CLASS_NAMES
except ImportError:
    from features import ElephantAcousticDataset
    from model import ElephantCNN, NUM_CLASSES, CLASS_NAMES


def evaluate_model(data_path, model_path):
    print("=" * 60)
    print("ELEPHANT CNN EVALUATION PIPELINE")
    print("=" * 60)

    # ─── Load Test Data ───
    print("\n[1/3] Loading test dataset...")
    test_dataset = ElephantAcousticDataset(data_path, split='test', feature_type='mel')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"  Test samples: {len(test_dataset)}")

    # ─── Load Model ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = ElephantCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Model loaded from {model_path}")

    # ─── Run Predictions ───
    all_preds = []
    all_labels = []

    print("\n[2/3] Running predictions...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ─── Generate Report ───
    print("\n[3/3] Generating evaluation report...")
    target_names = CLASS_NAMES  # ['Roar', 'Rumble', 'Trumpet', 'Non_Elephant']
    label_indices = list(range(NUM_CLASSES))

    report = classification_report(
        all_labels, all_preds,
        labels=label_indices,
        target_names=target_names,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("TEST EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Test Samples: {len(test_dataset)}")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=label_indices)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix — Elephant Vocalization Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    os.makedirs('models', exist_ok=True)
    cm_path = 'models/confusion_matrix.png'
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Save text report
    report_path = 'models/evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TEST EVALUATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Test Samples: {len(test_dataset)}\n")
        f.write(f"Classes: {target_names}\n\n")
        f.write(report)
    print(f"Text report saved to {report_path}")
    print("=" * 60)

if __name__ == '__main__':
    csv_path = r"c:\Users\kamal\Downloads\MACONFLIC-main\elephant_vocalization_detection\data\scaled_dataset_index.csv"
    model_path = r"c:\Users\kamal\Downloads\MACONFLIC-main\elephant_vocalization_detection\models\baseline_cnn.pth"
    evaluate_model(csv_path, model_path)
