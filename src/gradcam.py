"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Elephant CNN.

Generates visual heatmaps showing which regions of a Mel-spectrogram
the CNN focuses on when making a classification decision.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", ICCV 2017.

Usage:
    from src.gradcam import GradCAM
    from src.model_v2 import ElephantCNNv2

    model = ElephantCNNv2()
    model.load_state_dict(torch.load('models/best_model_v2.pth'))
    cam = GradCAM(model, target_layer=model.conv4)

    # Generate heatmap for a single spectrogram
    heatmap = cam.generate(mel_spectrogram_tensor)

    # Overlay on original spectrogram
    overlaid = cam.overlay(mel_spectrogram_tensor, heatmap)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import librosa
import librosa.display


class GradCAM:
    """
    Grad-CAM implementation for CNN-based audio classifiers.

    Hooks into a target convolutional layer to capture activations
    and gradients, then produces a class-discriminative heatmap.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: The trained CNN model (e.g., ElephantCNNv2).
            target_layer: The convolutional layer to visualise
                          (typically the last conv layer).
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        # Storage for hooked values
        self._activations = None
        self._gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook — saves the activation maps."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook — saves the gradients."""
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Mel-spectrogram tensor of shape (1, 1, H, W) or (1, H, W).
                          Will be unsqueezed if needed.
            target_class: The class index to generate the heatmap for.
                          If None, uses the predicted class.

        Returns:
            heatmap: numpy array of shape (H, W) with values in [0, 1].
            predicted_class: int — the model's predicted class index.
            confidence: float — prediction confidence (0-1).
        """
        # Ensure correct shape: (1, 1, H, W)
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

        if target_class is None:
            target_class = predicted_class.item()

        # Backward pass for the target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # Get activations and gradients
        activations = self._activations[0]  # (C, H', W')
        gradients = self._gradients[0]      # (C, H', W')

        # Global Average Pooling of gradients → channel weights
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — we only want positive influence
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input spectrogram dimensions
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = np.array(
            torch.nn.functional.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                size=(input_h, input_w),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
        )

        return cam_resized, predicted_class.item(), confidence.item()

    def cleanup(self):
        """Remove hooks to prevent memory leaks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def create_gradcam_overlay(spectrogram_db, heatmap, predicted_class, confidence,
                           class_names, sr=16000, fmax=8000, save_path=None):
    """
    Create a side-by-side visualization: original spectrogram + Grad-CAM overlay.

    Args:
        spectrogram_db: 2D numpy array of the Mel-spectrogram in dB.
        heatmap: 2D numpy array of the Grad-CAM heatmap (0-1).
        predicted_class: int — predicted class index.
        confidence: float — prediction confidence (0-1).
        class_names: list of class name strings.
        sr: sample rate.
        fmax: maximum frequency for display.
        save_path: optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    class_name = class_names[predicted_class]
    conf_pct = confidence * 100

    # 1. Original Mel-Spectrogram
    img1 = librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel',
                                     sr=sr, fmax=fmax, ax=axes[0], cmap='magma')
    axes[0].set_title('Original Mel-Spectrogram', fontsize=12, fontweight='bold')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    # 2. Grad-CAM Heatmap
    axes[1].imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.9)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Mel Bands')

    # 3. Overlay (Spectrogram + Heatmap)
    img3 = librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel',
                                     sr=sr, fmax=fmax, ax=axes[2], cmap='magma')
    axes[2].imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.4,
                   extent=axes[2].get_xlim() + axes[2].get_ylim())
    axes[2].set_title(f'Overlay — {class_name} ({conf_pct:.1f}%)',
                      fontsize=12, fontweight='bold')

    plt.suptitle(f'Grad-CAM Analysis: Predicted "{class_name}" with {conf_pct:.1f}% confidence',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM overlay saved to {save_path}")

    return fig


def create_gradcam_grid(spectrograms_db, heatmaps, predictions, confidences,
                        class_names, sr=16000, fmax=8000, save_path=None):
    """
    Create a grid showing Grad-CAM overlays for multiple samples (one per class).

    Args:
        spectrograms_db: list of 2D numpy arrays (Mel-spectrograms in dB).
        heatmaps: list of 2D numpy arrays (Grad-CAM heatmaps).
        predictions: list of predicted class indices.
        confidences: list of confidence values.
        class_names: list of class name strings.
        sr: sample rate.
        fmax: max frequency.
        save_path: optional path to save.

    Returns:
        matplotlib Figure object.
    """
    n = len(spectrograms_db)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))

    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        class_name = class_names[predictions[i]]
        conf_pct = confidences[i] * 100

        # Row 1: Original spectrogram
        librosa.display.specshow(spectrograms_db[i], x_axis='time', y_axis='mel',
                                 sr=sr, fmax=fmax, ax=axes[0, i], cmap='magma')
        axes[0, i].set_title(f'{class_name}\n({conf_pct:.1f}%)', fontsize=11, fontweight='bold')

        # Row 2: Grad-CAM overlay
        librosa.display.specshow(spectrograms_db[i], x_axis='time', y_axis='mel',
                                 sr=sr, fmax=fmax, ax=axes[1, i], cmap='magma')
        axes[1, i].imshow(heatmaps[i], aspect='auto', origin='lower', cmap='jet', alpha=0.45,
                          extent=axes[1, i].get_xlim() + axes[1, i].get_ylim())
        axes[1, i].set_title('Grad-CAM Overlay', fontsize=10)

    axes[0, 0].set_ylabel('Mel-Spectrogram', fontsize=11)
    axes[1, 0].set_ylabel('Grad-CAM Overlay', fontsize=11)

    plt.suptitle('Grad-CAM Explainability — Per-Class Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Grad-CAM grid saved to {save_path}")

    return fig
