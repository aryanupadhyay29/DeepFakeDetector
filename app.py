import streamlit as st
from PIL import Image
from utils import predict_real_or_fake


st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🔍",
    layout="centered"
)

def main():
    # App title 
    st.title("🔍 Deepfake Image Detector")
    st.write("""
    Upload an image to check if it's real or a deepfake.
    This application uses a CNN model trained to distinguish between real and fake images.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
       
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)
        
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                # Make prediction
                result, confidence, face_detected = predict_real_or_fake(image)
                
                # Display results
                with col2:
                    st.subheader("Analysis Results")
                    if not face_detected:
                        st.warning("⚠️ No face detected in the image. Results may be less accurate.")
                    
                    st.markdown(f"### Prediction: **{result}**")
                    st.progress(confidence / 100)
                    st.write(f"Confidence: {confidence:.2f}%")
                    
                    if result == "Fake":
                        st.error("This image was detected as potentially manipulated or generated.")
                    else:
                        st.success("This image was detected as likely authentic.")

if __name__ == "__main__":
    main()
