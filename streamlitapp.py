import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Ceramic Tile Defect Detection",
    page_icon="🔍",
    layout="centered"
)

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('keras_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the image for prediction"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model's expected input size (224x224 is common for most models)
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def save_prediction_log(image_name, result, confidence):
    """Save prediction results to log (using secrets if database is configured)"""
    try:
        # Check if database secrets are available
        if "DB_USERNAME" in st.secrets and "DB_TOKEN" in st.secrets:
            # You can add database logging here if needed
            # For now, we'll just create a simple log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "image_name": image_name,
                "prediction": result,
                "confidence": confidence,
                "username": st.secrets["DB_USERNAME"]
            }
            # In a real implementation, you would save this to your database
            # using the DB_TOKEN for authentication
            st.session_state.setdefault('prediction_history', []).append(log_entry)
        
        return True
    except Exception as e:
        st.warning(f"Logging not available: {str(e)}")
        return False

def predict_defect(model, image_array):
    """Make prediction on the preprocessed image"""
    try:
        predictions = model.predict(image_array)
        
        # Get the predicted class (0 or 1)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Map to labels
        labels = {0: "Non Defected", 1: "Defected"}
        result = labels[predicted_class]
        
        return result, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def main():
    st.title("🔍 Ceramic Tile Defect Detection")
    st.markdown("Upload an image of a ceramic tile to check for defects")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model could not be loaded. Please check if 'keras_model.h5' exists in the repository.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a ceramic tile"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        with st.spinner('🔄 Analyzing image...'):
            processed_image = preprocess_image(image)
            
            if processed_image is not None:
                result, confidence, raw_predictions = predict_defect(model, processed_image)
                
                if result is not None:
                    # Save prediction log
                    save_prediction_log(uploaded_file.name, result, confidence)
                    
                    with col2:
                        st.subheader("🎯 Prediction Results")
                        
                        # Display result with color coding
                        if result == "Non Defected":
                            st.success(f"**Result:** {result}")
                            st.balloons()
                        else:
                            st.error(f"**Result:** {result}")
                        
                        # Display confidence
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Display detailed probabilities
                        st.subheader("📊 Detailed Probabilities")
                        prob_non_defected = float(raw_predictions[0])
                        prob_defected = float(raw_predictions[1])
                        
                        st.write(f"**Non Defected:** {prob_non_defected:.2%}")
                        st.progress(prob_non_defected)
                        
                        st.write(f"**Defected:** {prob_defected:.2%}")
                        st.progress(prob_defected)
                        
                        # Additional info
                        st.info(
                            f"The model is {confidence:.1%} confident that this tile is **{result.lower()}**"
                        )
    
    # Show prediction history if available
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        with st.expander("📈 Recent Predictions History"):
            for i, log in enumerate(reversed(st.session_state.prediction_history[-5:])):  # Show last 5
                st.write(f"**{i+1}.** {log['image_name']} → {log['prediction']} ({log['confidence']:.1%}) - {log['timestamp'][:19]}")
    
    # Instructions
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Upload an image**: Click on 'Browse files' and select a clear image of a ceramic tile
        2. **Wait for analysis**: The AI model will process your image
        3. **View results**: See if the tile is defected or non-defected with confidence scores
        
        **Supported formats**: PNG, JPG, JPEG
        
        **Tips for best results**:
        - Use clear, well-lit images
        - Ensure the tile fills most of the frame
        - Avoid blurry or dark images
        """)
    
    # Model info
    with st.expander("🤖 About the Model"):
        try:
            username = st.secrets.get("DB_USERNAME", "Anonymous")
        except:
            username = "Anonymous"
            
        st.markdown(f"""
        This ceramic tile defect detection model uses deep learning to classify tiles as:
        - **Class 0**: Non Defected (Good quality tile)
        - **Class 1**: Defected (Tile with defects)
        
        The model analyzes visual patterns in the tile surface to detect cracks, chips, 
        discoloration, or other manufacturing defects.
        
        **Current Configuration:**
        - Model: keras_model.h5
        - Classes: 0 (Non Defected), 1 (Defected)
        - Input Size: 224x224 pixels
        - User: {username}
        """)

    # Debug info (only show if secrets are configured)
    try:
        if st.secrets.get("some_key"):
            with st.expander("🔧 Debug Info"):
                st.write(f"Some Key Value: {st.secrets['some_key']}")
                st.write("Secrets are properly configured!")
    except:
        pass  # Secrets not available, skip debug info

if __name__ == "__main__":
    main()
