import torch
import torch.nn as nn
import torch.nn.functional as F

# Class labels used across the entire project
CLASS_NAMES = ['Roar', 'Rumble', 'Trumpet', 'Non_Elephant']
NUM_CLASSES = len(CLASS_NAMES)

class ElephantCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network (CNN) for classifying
    elephant vocalizations from Mel-spectrograms.

    4-class classification: Roar, Rumble, Trumpet, Non_Elephant.

    Architecture:
        3 × [Conv2D → BatchNorm → ReLU → MaxPool]
        → AdaptiveAvgPool(4×4)
        → Dense(1024→128) → Dropout(0.5)
        → Dense(128→num_classes)
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(ElephantCNN, self).__init__()
        # Input shape: (Batch, 1, 128 Mel-bands, Time Steps)

        # 1st Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3rd Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pooling ensures the output to the linear layers is always the same size
        # This handles variable-length audio clips gracefully
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully Connected (Dense) Layers
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Keep backward compatibility — old code that imports ElephantIntentCNN or SimpleCNN
# will still work
ElephantIntentCNN = ElephantCNN
SimpleCNN = ElephantCNN


if __name__ == "__main__":
    model = ElephantCNN(num_classes=NUM_CLASSES)
    print(model)
    print(f"\nClasses: {CLASS_NAMES}")
    print(f"Number of classes: {NUM_CLASSES}")
    dummy_input = torch.randn(2, 1, 128, 188)  # (batch, channels, mel_bands, time_steps)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
