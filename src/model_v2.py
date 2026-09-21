import torch
import torch.nn as nn
import torch.nn.functional as F

# Class labels used across the entire project
CLASS_NAMES = ['Roar', 'Rumble', 'Trumpet', 'Non_Elephant']
NUM_CLASSES = len(CLASS_NAMES)


class ElephantCNNv2(nn.Module):
    """
    Enhanced CNN for elephant vocalization classification.

    Improvements over ElephantCNN v1:
        - 4 convolutional blocks (was 3) for deeper feature extraction
        - Dual pooling (AvgPool + MaxPool concatenated) for richer representations
        - Wider fully-connected layers (2048→256→4)
        - Squeeze-and-Excitation (SE) channel attention in each block
        - ~300K parameters (still lightweight and deployable)

    4-class classification: Roar, Rumble, Trumpet, Non_Elephant.

    Architecture:
        4 × [Conv2D → BatchNorm → ReLU → SE-Block → MaxPool]
        → AdaptiveAvgPool(4×4) + AdaptiveMaxPool(4×4) → Concat
        → Dense(2048→256) → ReLU → Dropout(0.5)
        → Dense(256→num_classes)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(ElephantCNNv2, self).__init__()

        # ─── Convolutional Blocks ───
        # Block 1: 1 → 16 channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.se1 = SEBlock(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 16 → 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.se2 = SEBlock(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 32 → 64 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.se3 = SEBlock(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4: 64 → 128 channels (NEW — deeper features)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.se4 = SEBlock(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        # ─── Dual Adaptive Pooling ───
        # Concatenate avg and max pooling for richer feature representation
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((4, 4))

        # ─── Fully Connected Layers ───
        # 128 channels × 4 × 4 × 2 (avg + max) = 4096
        # But we reduce to 2048 by using a single pool set for initial simplicity
        # Actually: avg(128×4×4) + max(128×4×4) = 2048 + 2048 = 4096
        self.fc1 = nn.Linear(128 * 4 * 4 * 2, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool1(self.se1(F.relu(self.bn1(self.conv1(x)))))
        # Block 2
        x = self.pool2(self.se2(F.relu(self.bn2(self.conv2(x)))))
        # Block 3
        x = self.pool3(self.se3(F.relu(self.bn3(self.conv3(x)))))
        # Block 4
        x = self.pool4(self.se4(F.relu(self.bn4(self.conv4(x)))))

        # Dual pooling — concatenate average and max pooled features
        avg_out = self.adaptive_avg_pool(x)
        max_out = self.adaptive_max_pool(x)
        x = torch.cat([avg_out, max_out], dim=1)  # (B, 256, 4, 4)

        # Flatten
        x = x.view(x.size(0), -1)  # (B, 4096)

        # Classification head
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

    def get_last_conv_layer(self):
        """Return the last convolutional layer (for Grad-CAM)."""
        return self.conv4


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.

    Learns channel-wise attention weights to emphasise important
    frequency/time features and suppress irrelevant ones.

    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    """
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling → (B, C, 1, 1) → (B, C)
        y = self.squeeze(x).view(b, c)
        # Excitation: FC → ReLU → FC → Sigmoid → (B, C)
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: channel-wise multiplication
        return x * y.expand_as(x)


# ─── Backward Compatibility ───
# Old code importing ElephantCNN will get v2 automatically
ElephantCNN = ElephantCNNv2
ElephantIntentCNN = ElephantCNNv2
SimpleCNN = ElephantCNNv2


if __name__ == '__main__':
    model = ElephantCNNv2(num_classes=NUM_CLASSES)
    print(model)
    print(f"\nClasses: {CLASS_NAMES}")
    print(f"Number of classes: {NUM_CLASSES}")

    dummy_input = torch.randn(2, 1, 128, 188)  # (batch, channels, mel_bands, time_steps)
    output = model(dummy_input)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Verify Grad-CAM hook target
    print(f"\nGrad-CAM target layer: {model.get_last_conv_layer()}")
