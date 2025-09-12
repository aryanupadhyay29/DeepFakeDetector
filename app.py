import streamlit as st
from PIL import Image
from utils import predict_real_or_fake

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🔍",
    layout="centered"
)

def main():
    """Main function to run the Streamlit app's UI logic."""
    # App title and description
    st.title("🔍 Deepfake Image Detector")
    st.markdown("""
    Upload an image to check if it's real or a deepfake.
    This application uses a Convolutional Neural Network (CNN) to analyze the image.
    """)
    
    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
        
        # Add a button to start the analysis
        if st.button("Analyze Image", use_container_width=True):
            with st.spinner("Analyzing..."):
                # Call the prediction function from utils.py
                try:
                    result, confidence, face_detected = predict_real_or_fake(image)
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    return

                # Display the analysis results
                st.markdown("---")
                st.subheader("Analysis Results")
                
                if not face_detected:
                    st.warning("⚠️ No face was detected in the image. The model analyzed the entire image, and results may be less accurate.")
                
                # Show the prediction with color-coding
                if result == "Fake":
                    st.error(f"### Prediction: **{result}**")
                    st.write(f"Confidence: **{confidence:.2f}%**")
                    st.write("This image was detected as potentially manipulated or AI-generated.")
                else:
                    st.success(f"### Prediction: **{result}**")
                    st.write(f"Confidence: **{confidence:.2f}%**")
                    st.write("This image was detected as likely authentic.")
                
                # Display a progress bar for confidence
                st.progress(confidence / 100)

                # Add an optional section to explain the technology
                with st.expander("How does this work?"):
                    st.write("""
                    The model looks for subtle artifacts and inconsistencies that are often invisible to the human eye,
                    such as a lack of natural imperfections in skin texture, unusual pixel patterns, or distortions in facial features.
                    """)

if __name__ == "__main__":
    main()