import streamlit as st
import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.features import extract_mel_spectrogram, extract_acoustic_characteristics
from src.model import ElephantCNN, NUM_CLASSES, CLASS_NAMES

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
    model = ElephantCNN(num_classes=NUM_CLASSES)
    # Try best_model.pth first, fall back to baseline_cnn.pth
    model_path = os.path.join('models', 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join('models', 'baseline_cnn.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
        return model, model_path
    else:
        return None, None

model, model_path = load_model()

if model is None:
    st.error("❌ Model not found. Please train the model first by running:\n```\npython -m src.train\n```")
    st.stop()

st.caption(f"Model loaded from: `{model_path}`")

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

            # Show all class probabilities
            st.subheader("📊 Class Probabilities")
            prob_values = probabilities[0].cpu().numpy()
            for i, cls_name in enumerate(CLASS_NAMES):
                cls_icon = CLASS_ICONS.get(cls_name, '')
                st.progress(float(prob_values[i]), text=f"{cls_icon} {cls_name}: {prob_values[i]*100:.1f}%")

            # ─── 2. Acoustic Characteristics ───
            st.subheader("📐 Acoustic Characteristics")
            chars = extract_acoustic_characteristics(temp_path)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Duration", f"{chars['duration']} sec")
                st.metric("RMS Energy", f"{chars['rms_energy']:.4f}")
            with col2:
                st.metric("Dominant Frequency", f"{chars['dominant_freq']} Hz")
                st.metric("Zero Crossing Rate", f"{chars['zero_crossing_rate']:.4f}")

            # ─── 3. Spectrogram Visualization ───
            st.subheader("🎨 Mel-Spectrogram")
            y, sr = librosa.load(temp_path, sr=16000, duration=6.0)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)

            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title(f'Mel-Spectrogram — Predicted: {predicted_class}')
            st.pyplot(fig)
            plt.close(fig)

    # Cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
