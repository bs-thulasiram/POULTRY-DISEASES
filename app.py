import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Poultry Disease Identifier",
    page_icon="🐔",
    layout="wide"
)

# --- DARK MODE CSS ---
st.markdown("""
    <style>
    body, .stApp {
        background-color: #121212;
        color: #f0f0f0;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #00e676;
        text-align: center;
        font-size: 44px;
        margin-bottom: 10px;
    }
    .description {
        text-align: center;
        color: #aaa;
        font-size: 18px;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #1e1e1e;
        border: 2px solid #00e676;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .result-box h2 {
        color: #6ee7b7;
        font-size: 28px;
        margin-bottom: 10px;
    }
    .result-box p {
        font-size: 18px;
        color: #ccc;
    }
    .sidebar .sidebar-content {
        background-color: #181818;
    }
    </style>
""", unsafe_allow_html=True)

# --- TITLE & DESCRIPTION ---
st.markdown("<h1>🐣 Poultry Disease Identifier</h1>", unsafe_allow_html=True)
st.markdown("""
    <div class='description'>
        Upload an image of chicken feces to detect <strong>Coccidiosis</strong>, <strong>Salmonella</strong>, <strong>Newcastle Disease</strong>, or confirm if it's <strong>Healthy</strong>.
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("📋 About")
    st.markdown("""
        This tool uses a deep learning model (MobileNetV2) to identify common poultry diseases from chicken fecal images.

        **Detectable Conditions:**
        - 🧫 Coccidiosis  
        - 🦠 Salmonella  
        - 🐤 Newcastle Disease  
        - ✅ Healthy

        💡 Upload a clear, well-lit image for accurate results.
    """)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/mobilenetV2/mobilenetv2.h5", compile=False)

model = load_model()
class_names = {0: 'Coccidiosis', 1: 'Healthy', 2: 'NewCastleDisease', 3: 'Salmonella'}

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload a Chicken Feces Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_file, caption="🖼️ Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🔬 Analyzing the image..."):
            img = Image.open(uploaded_file).resize((128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) / 255.0

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]
            confidence = round(100 * np.max(score), 2)

        # --- DISPLAY RESULT ---
        st.markdown(f"""
        <div class='result-box'>
            <h2>🩺 Prediction: {predicted_class}</h2>
            <p><strong>Confidence Score:</strong> {confidence}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence))
else:
    st.warning("📌 Please upload an image to begin the prediction.")
