"""
MACONFLIC — Backend ML Service.

Wraps the trained ElephantCNN model in a clean API for the FastAPI backend.
Handles audio file processing, prediction, and Grad-CAM generation.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import io
import base64
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path so we can import src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import load_and_pad_audio, TARGET_SR, CLIP_DURATION, N_MELS, FMAX
from src.gradcam import GradCAM

# Try v2 model first, fall back to v1
try:
    from src.model_v2 import ElephantCNNv2 as ModelClass, NUM_CLASSES, CLASS_NAMES
    MODEL_VERSION = 'v2'
except ImportError:
    from src.model import ElephantCNN as ModelClass, NUM_CLASSES, CLASS_NAMES
    MODEL_VERSION = 'v1'


class ElephantMLService:
    """
    ML inference service for elephant vocalization classification.

    Provides:
        - predict(audio_path) → classification result
        - predict_with_gradcam(audio_path) → classification + Grad-CAM heatmap
        - get_spectrogram_image(audio_path) → base64-encoded spectrogram PNG
    """

    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(PROJECT_ROOT, 'models')

        self.device = torch.device('cpu')
        self.model = None
        self.model_path = None

        self._load_model(models_dir)

    def _load_model(self, models_dir):
        """Load the best available model."""
        candidates = [
            os.path.join(models_dir, 'best_model_v2.pth'),
            os.path.join(models_dir, 'baseline_cnn_v2.pth'),
            os.path.join(models_dir, 'best_model.pth'),
            os.path.join(models_dir, 'baseline_cnn.pth'),
        ]

        model = ModelClass(num_classes=NUM_CLASSES).to(self.device)

        for path in candidates:
            if os.path.exists(path):
                try:
                    model.load_state_dict(
                        torch.load(path, map_location=self.device, weights_only=True)
                    )
                    model.eval()
                    self.model = model
                    self.model_path = path
                    print(f"[ML Service] Loaded model ({MODEL_VERSION}): {path}")
                    return
                except RuntimeError:
                    continue

        # Try v1 as fallback
        if self.model is None:
            from src.model import ElephantCNN as ModelV1
            model_v1 = ModelV1(num_classes=NUM_CLASSES).to(self.device)
            for path in candidates:
                if os.path.exists(path):
                    try:
                        model_v1.load_state_dict(
                            torch.load(path, map_location=self.device, weights_only=True)
                        )
                        model_v1.eval()
                        self.model = model_v1
                        self.model_path = path
                        print(f"[ML Service] Loaded model (v1 fallback): {path}")
                        return
                    except RuntimeError:
                        continue

        raise FileNotFoundError(
            f"No trained model found in {models_dir}. "
            "Run 'python -m src.train_v2' first."
        )

    def predict(self, audio_path: str) -> dict:
        """
        Classify an audio file.

        Args:
            audio_path: Path to the audio file (.wav, .mp3, .flac).

        Returns:
            dict with keys:
                - predicted_class: str (e.g., 'Roar')
                - confidence: float (0-1)
                - probabilities: dict mapping class names to probabilities
                - acoustic_features: dict with duration, dominant_freq, etc.
        """
        # Load and extract features
        y = load_and_pad_audio(audio_path, sr=TARGET_SR, duration=CLIP_DURATION)
        S = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS, fmax=FMAX)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Create tensor
        mel_tensor = torch.FloatTensor(S_db).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, T)

        # Predict
        with torch.no_grad():
            output = self.model(mel_tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        conf_value = confidence.item()

        # Probabilities per class
        prob_dict = {
            CLASS_NAMES[i]: round(probs[0][i].item(), 4)
            for i in range(NUM_CLASSES)
        }

        # Acoustic features
        acoustic = self._extract_acoustic_features(y, TARGET_SR)

        return {
            'predicted_class': predicted_class,
            'confidence': round(conf_value, 4),
            'probabilities': prob_dict,
            'acoustic_features': acoustic,
        }

    def predict_with_gradcam(self, audio_path: str) -> dict:
        """
        Classify an audio file and generate a Grad-CAM heatmap.

        Returns the same dict as predict(), plus:
            - gradcam_image: base64-encoded PNG of the Grad-CAM overlay
            - spectrogram_image: base64-encoded PNG of the Mel-spectrogram
        """
        result = self.predict(audio_path)

        # Load audio for visualization
        y = load_and_pad_audio(audio_path, sr=TARGET_SR, duration=CLIP_DURATION)
        S = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS, fmax=FMAX)
        S_db = librosa.power_to_db(S, ref=np.max)
        mel_tensor = torch.FloatTensor(S_db).unsqueeze(0)  # (1, 128, T)

        # Generate Grad-CAM
        target_layer = self._get_target_layer()
        cam = GradCAM(self.model, target_layer)

        try:
            heatmap, pred_cls, conf = cam.generate(mel_tensor)

            # Create overlay image
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            # Original spectrogram
            librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                     sr=TARGET_SR, fmax=FMAX, ax=axes[0], cmap='magma')
            axes[0].set_title('Mel-Spectrogram', fontweight='bold')

            # Grad-CAM overlay
            librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                     sr=TARGET_SR, fmax=FMAX, ax=axes[1], cmap='magma')
            extent = axes[1].get_xlim() + axes[1].get_ylim()
            axes[1].imshow(heatmap, aspect='auto', origin='lower',
                          cmap='jet', alpha=0.45, extent=extent)
            axes[1].set_title(f'Grad-CAM: {result["predicted_class"]} ({result["confidence"]*100:.1f}%)',
                             fontweight='bold')

            plt.tight_layout()

            # Convert to base64
            result['gradcam_image'] = self._fig_to_base64(fig)
            plt.close(fig)

        except Exception as e:
            result['gradcam_image'] = None
            result['gradcam_error'] = str(e)
        finally:
            cam.cleanup()

        # Also generate standalone spectrogram
        result['spectrogram_image'] = self._spectrogram_to_base64(S_db)

        return result

    def get_spectrogram_image(self, audio_path: str) -> str:
        """Generate a Mel-spectrogram image as base64 string."""
        y = load_and_pad_audio(audio_path, sr=TARGET_SR, duration=CLIP_DURATION)
        S = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS, fmax=FMAX)
        S_db = librosa.power_to_db(S, ref=np.max)
        return self._spectrogram_to_base64(S_db)

    def _get_target_layer(self):
        """Get the last convolutional layer for Grad-CAM."""
        if hasattr(self.model, 'get_last_conv_layer'):
            return self.model.get_last_conv_layer()
        elif hasattr(self.model, 'conv4'):
            return self.model.conv4
        else:
            return self.model.conv3

    def _extract_acoustic_features(self, y, sr):
        """Extract acoustic characteristics from audio."""
        duration = round(len(y) / sr, 2)

        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        magnitude_sum = np.sum(S, axis=1)
        dominant_freq = round(float(freqs[np.argmax(magnitude_sum)]), 2)

        rms = librosa.feature.rms(y=y)[0]
        avg_rms = round(float(np.mean(rms)), 4)

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        avg_zcr = round(float(np.mean(zcr)), 4)

        return {
            'duration': duration,
            'dominant_freq': dominant_freq,
            'rms_energy': avg_rms,
            'zero_crossing_rate': avg_zcr,
        }

    @staticmethod
    def _fig_to_base64(fig) -> str:
        """Convert a matplotlib figure to a base64-encoded PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    @staticmethod
    def _spectrogram_to_base64(S_db) -> str:
        """Generate a Mel-spectrogram and return as base64 PNG."""
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                       sr=TARGET_SR, fmax=FMAX, ax=ax, cmap='magma')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel-Spectrogram')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
