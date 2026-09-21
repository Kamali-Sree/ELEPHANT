import streamlit as st
import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.features import extract_mel_spectrogram, extract_acoustic_characteristics, \
    load_and_pad_audio, TARGET_SR, CLIP_DURATION, N_MELS, FMAX

# Try v2 model first, fall back to v1
try:
    from src.model_v2 import ElephantCNNv2 as ModelClass, NUM_CLASSES, CLASS_NAMES
except ImportError:
    from src.model import ElephantCNN as ModelClass, NUM_CLASSES, CLASS_NAMES

from src.gradcam import GradCAM, create_gradcam_overlay

# ─── Page Config ───
st.set_page_config(page_title="Elephant Acoustic Analyzer", page_icon="🐘", layout="centered")

st.title("🐘 Elephant Acoustic Analyzer")
st.write("Upload an audio file to classify elephant vocalizations (Roar, Rumble, Trumpet) or detect background noise.")

# ─── Class display config ───
CLASS_ICONS = {
    'Roar': '🦁',
    'Rumble': '🔊',
    'Trumpet': '🎺',
    'Non_Elephant': '🌿',
}

CLASS_DESCRIPTIONS = {
    'Roar': 'Elephant Roar — a loud, aggressive vocalization',
    'Rumble': 'Elephant Rumble — a low-frequency communication call',
    'Trumpet': 'Elephant Trumpet — a high-pitched alarm/excitement call',
    'Non_Elephant': 'Non-Elephant — background noise / no elephant detected',
}


# ─── Load Model ───
@st.cache_resource
def load_model():
    model = ModelClass(num_classes=NUM_CLASSES)

    # Try models in order of preference
    model_candidates = [
        os.path.join('models', 'best_model_v2.pth'),
        os.path.join('models', 'baseline_cnn_v2.pth'),
        os.path.join('models', 'best_model.pth'),
        os.path.join('models', 'baseline_cnn.pth'),
    ]

    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            try:
                model.load_state_dict(torch.load(candidate, map_location='cpu', weights_only=True))
                model_path = candidate
                break
            except RuntimeError:
                continue

    if model_path is None:
        # Try v1 model as last resort
        from src.model import ElephantCNN as ModelV1
        model = ModelV1(num_classes=NUM_CLASSES)
        for candidate in model_candidates:
            if os.path.exists(candidate):
                try:
                    model.load_state_dict(torch.load(candidate, map_location='cpu', weights_only=True))
                    model_path = candidate
                    break
                except RuntimeError:
                    continue

    if model_path:
        model.eval()
        return model, model_path
    else:
        return None, None


model, model_path = load_model()

if model is None:
    st.error("❌ Model not found. Please train the model first by running:\n```\npython -m src.train_v2\n```")
    st.stop()

st.caption(f"Model loaded from: `{model_path}`")

# ─── Sidebar Options ───
with st.sidebar:
    st.header("⚙️ Settings")
    show_gradcam = st.toggle("🔥 Show Grad-CAM Heatmap", value=True,
                              help="Visualise which spectrogram regions the model focuses on")
    show_acoustic = st.toggle("📐 Show Acoustic Characteristics", value=True)
    show_probs = st.toggle("📊 Show Class Probabilities", value=True)

# ─── File Upload ───
uploaded_file = st.file_uploader("Choose Audio File", type=['wav', 'mp3', 'flac'])

if uploaded_file is not None:
    # Save temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(temp_path)

    if st.button("🔍 Analyze", type="primary"):
        with st.spinner("Analyzing audio..."):
            # ─── 1. Prediction ───
            mel_spec = extract_mel_spectrogram(temp_path)
            mel_spec_input = mel_spec.unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output = model(mel_spec_input)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            predicted_class = CLASS_NAMES[predicted_idx.item()]
            conf_score = confidence.item() * 100
            icon = CLASS_ICONS.get(predicted_class, '❓')
            description = CLASS_DESCRIPTIONS.get(predicted_class, '')

            # ─── Display Prediction ───
            st.subheader("🏷️ Prediction")
            if predicted_class == 'Non_Elephant':
                st.info(f"{icon} **{description}** (Confidence: {conf_score:.1f}%)")
            else:
                st.success(f"{icon} **{description}** (Confidence: {conf_score:.1f}%)")

            # ─── Class Probabilities ───
            if show_probs:
                st.subheader("📊 Class Probabilities")
                prob_values = probabilities[0].cpu().numpy()
                for i, cls_name in enumerate(CLASS_NAMES):
                    cls_icon = CLASS_ICONS.get(cls_name, '')
                    st.progress(float(prob_values[i]),
                                text=f"{cls_icon} {cls_name}: {prob_values[i]*100:.1f}%")

            # ─── Acoustic Characteristics ───
            if show_acoustic:
                st.subheader("📐 Acoustic Characteristics")
                chars = extract_acoustic_characteristics(temp_path)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Duration", f"{chars['duration']} sec")
                    st.metric("RMS Energy", f"{chars['rms_energy']:.4f}")
                with col2:
                    st.metric("Dominant Frequency", f"{chars['dominant_freq']} Hz")
                    st.metric("Zero Crossing Rate", f"{chars['zero_crossing_rate']:.4f}")

            # ─── Mel-Spectrogram ───
            st.subheader("🎨 Mel-Spectrogram")
            y, sr = librosa.load(temp_path, sr=TARGET_SR, duration=CLIP_DURATION)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
            S_dB = librosa.power_to_db(S, ref=np.max)

            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel',
                                           sr=sr, fmax=FMAX, ax=ax, cmap='magma')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title(f'Mel-Spectrogram — Predicted: {predicted_class}')
            st.pyplot(fig)
            plt.close(fig)

            # ─── Grad-CAM Heatmap ───
            if show_gradcam:
                st.subheader("🔥 Grad-CAM — Model Attention Heatmap")
                st.caption("Shows which regions of the spectrogram the model focuses on for its prediction.")

                try:
                    # Determine target layer
                    if hasattr(model, 'get_last_conv_layer'):
                        target_layer = model.get_last_conv_layer()
                    elif hasattr(model, 'conv4'):
                        target_layer = model.conv4
                    else:
                        target_layer = model.conv3

                    cam = GradCAM(model, target_layer)
                    heatmap, pred_cls, conf = cam.generate(mel_spec)

                    # Create overlay figure
                    fig_cam = create_gradcam_overlay(
                        S_dB, heatmap, pred_cls, conf, CLASS_NAMES,
                        sr=TARGET_SR, fmax=FMAX
                    )
                    st.pyplot(fig_cam)
                    plt.close(fig_cam)

                    cam.cleanup()

                except Exception as e:
                    st.warning(f"⚠️ Grad-CAM generation failed: {e}")

    # Cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
