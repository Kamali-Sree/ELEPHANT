import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights

# Class labels used across the entire project
CLASS_NAMES = ['Roar', 'Rumble', 'Trumpet', 'Non_Elephant']
NUM_CLASSES = len(CLASS_NAMES)

class ElephantCNNv2(nn.Module):
    """
    Enhanced CNN for elephant vocalization classification using Transfer Learning.
    Uses a pre-trained ResNet-18 model from torchvision, modified for 1-channel audio.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(ElephantCNNv2, self).__init__()
        
        # Load pre-trained ResNet-18
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the first convolutional layer to accept 1 channel (grayscale spectrograms) instead of 3 (RGB images)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=original_conv1.bias is not None
        )
        
        # Initialize the new 1-channel weights by averaging the 3-channel weights from the pretrained model
        with torch.no_grad():
            self.resnet.conv1.weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))
            
        # Modify the fully connected layer for our 4 classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

    def get_last_conv_layer(self):
        """Return the last convolutional layer (for Grad-CAM)."""
        return self.resnet.layer4[-1].conv2

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
