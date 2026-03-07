import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ------------------------
# Page configuration
# ------------------------
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="centered"
)

# ------------------------
# Custom CSS for soft dark theme + neon navbar glow
# ------------------------
st.markdown("""
<style>
/* Full page background and text */
.stApp {
    background-color: #1f1f1f;
    color: #f0f0f0;
    font-family: 'Helvetica', sans-serif;
}

/* App title */
h1 {
    color: #39ff14;  /* soft neon green */
    text-shadow: 0 0 4px #39ff14;
    font-weight: normal;
    font-size: 2em;
}

/* Subtitle / description */
h2, h3, .stText, .stMarkdown {
    color: #ffffff;
    font-weight: normal;
}

/* Navigation bar: neon purple → neon blue gradient with glow */
.css-18ni7ap.e8zbici2 {
    background: linear-gradient(90deg, #8a2be2, #00ffff) !important;
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent !important;  /* show gradient */
    font-weight: bold;
    text-shadow:
        0 0 5px #8a2be2,
        0 0 10px #8a2be2,
        0 0 20px #00ffff,
        0 0 30px #00ffff,
        0 0 40px #00ffff; /* neon glow */
}

/* File uploader label */
.css-1kyxreq.edgvbvh3 {
    color: #000000 !important;  /* dark text for 'Browse files' */
    font-weight: bold;
}

/* Uploaded image */
.stImage > img {
    border: 2px solid #39ff14;
    border-radius: 8px;
}

/* Buttons */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #00cfff, #0077ff);
    color: #ffffff;
    font-weight: normal;
    border-radius: 10px;
}

/* Success box (no tumor) */
.stSuccess {
    background: linear-gradient(90deg, #00ff99, #00cc66) !important;
    color: #000000 !important;
    border-radius: 8px;
}

/* Warning box (low confidence) */
.stWarning {
    background: linear-gradient(90deg, #ffcc33, #ff9900) !important;
    color: #000000 !important;
    border-radius: 8px;
}

/* Error box (tumor detected) */
.stError {
    background: linear-gradient(90deg, #ff6666, #ff0033) !important;
    color: #ffffff !important;
    border-radius: 8px;
}

/* Info box (suggestions) */
.stInfo {
    background: linear-gradient(90deg, #ff99ff, #cc66cc) !important;
    color: #000000 !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# App title and description
# ------------------------
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI scan image to detect brain tumors using AI.")

# ------------------------
# Load AI model
# ------------------------
@st.cache_resource
def load_ai_model():
    model = load_model("model.h5")
    return model

model = load_ai_model()

# ------------------------
# Class labels
# ------------------------
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ------------------------
# Image uploader
# ------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Display smaller image
    st.image(uploaded_file, caption="Uploaded MRI Scan", width=300)

    # Preprocess image
    img = load_img(uploaded_file, target_size=(128,128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    label = class_labels[predicted_class]

    st.subheader("Prediction Result")

    # Show results with color-coded advice
    if label == "notumor":
        st.success(f"No Tumor Detected ✅ (Confidence: {confidence*100:.2f}%)")
        st.info("You are clear! No tumor detected. Maintain healthy habits and regular check-ups.")
    else:
        if confidence < 0.6:
            st.warning(f"Tumor Detected: {label} ⚠️ Low confidence ({confidence*100:.2f}%)")
        else:
            st.error(f"Tumor Detected: {label} (Confidence: {confidence*100:.2f}%)")
        
        st.warning("⚠ You should consult a doctor immediately for proper diagnosis and treatment.")
        st.info("💡 Suggestions:\n• Maintain a healthy diet\n• Reduce stress\n• Follow doctor's instructions\n• Avoid self-medication")