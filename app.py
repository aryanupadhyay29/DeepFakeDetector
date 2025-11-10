import streamlit as st
from PIL import Image
from utils import predict_real_or_fake

# --- Page Config ---
st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    body {
        background-color: #0f172a;
        color: white;
    }
    .upload-container {
        border: 2px dashed #4f46e5;
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        background-color: #1e293b;
        transition: all 0.3s ease;
    }
    .upload-container:hover {
        border-color: #6366f1;
        background-color: #111827;
        transform: scale(1.01);
    }
    .uploaded-image {
        border-radius: 12px;
        box-shadow: 4px 4px 12px #0a0f1a, -4px -4px 12px #1e293b;
        margin-top: 20px;
        display: block;
        width: 100%;
        height: 350px;
    }
    .result-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-radius: 20px;
        box-shadow: 8px 8px 20px #0a0f1a, -8px -8px 20px #1e293b;
        padding: 30px;
        margin-top: 20px;
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
    }
    .stProgress > div > div {
        background-color: #4f46e5 !important;
    }
    h1, h2, h3 {
        color: #e0e7ff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Main Layout ---
st.title("🕵️‍♂️ Deepfake Image Detector")
st.markdown("<p style='color:#cbd5e1'>Upload an image to check if it's real or AI-generated.</p>", unsafe_allow_html=True)
st.markdown("---")

# 💡 Adjusted column width: smaller col1, larger col2
col1, col2 = st.columns([0.8, 1.4], gap="large")

with col1:
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag & Drop or Click to Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        st.image(image,  use_container_width=True)

        analyze = st.button("🔍 Analyze Image", use_container_width=True)

        if analyze:
            with st.spinner("Analyzing... Please wait..."):
                try:
                    result, confidence, face_detected = predict_real_or_fake(image)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    result = None
                    confidence = None

with col2:
    if uploaded_file is not None and "result" in locals() and result is not None:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("🧠 Analysis Results")

        if not face_detected:
            st.warning("⚠️ No face detected. The model analyzed the entire image.")

        # Result display
        if result.lower() == "fake":
            st.error(f"### 🟥 Prediction: {result}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            severity = "High" if confidence > 80 else "Medium"
            st.write(f"**Severity Level:** {severity}")
        else:
            st.success(f"### 🟩 Prediction: {result}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            severity = "Low" if confidence < 60 else "Moderate"
            st.write(f"**Severity Level:** {severity}")

        st.progress(confidence / 100)

        with st.expander("ℹ️ How This Works"):
            st.write("""
            The CNN model detects subtle inconsistencies in texture, lighting, and facial structure 
            that help differentiate between authentic and AI-generated images.
            """)
        st.markdown("</div>", unsafe_allow_html=True)
