import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
from datetime import datetime
import base64
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config with custom theme
st.set_page_config(
    page_title="🔍 AI Ceramic Tile Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    .success-card {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .error-card {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .info-card {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .camera-container {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {'total_processed': 0, 'defected': 0, 'non_defected': 0}

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('keras_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    """Apply image enhancements"""
    try:
        # Brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        
        # Contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        # Sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
        
        return image
    except Exception as e:
        st.error(f"Error enhancing image: {str(e)}")
        return image

def preprocess_image(image, target_size=(224, 224)):
    """Advanced preprocessing with multiple options"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model's expected input size
        image = image.resize(target_size)
        
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

def predict_defect(model, image_array):
    """Make prediction with detailed analysis"""
    try:
        predictions = model.predict(image_array, verbose=0)
        
        # Get the predicted class (0 or 1)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Map to labels
        labels = {0: "Non Defected", 1: "Defected"}
        result = labels[predicted_class]
        
        # Calculate additional metrics
        uncertainty = 1 - confidence
        prob_non_defected = float(predictions[0][0])
        prob_defected = float(predictions[0][1])
        
        return {
            'result': result,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'prob_non_defected': prob_non_defected,
            'prob_defected': prob_defected,
            'raw_predictions': predictions[0]
        }
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def save_prediction_log(image_name, prediction_data):
    """Enhanced logging with detailed data"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "image_name": image_name,
            "prediction": prediction_data['result'],
            "confidence": prediction_data['confidence'],
            "uncertainty": prediction_data['uncertainty'],
            "prob_non_defected": prediction_data['prob_non_defected'],
            "prob_defected": prediction_data['prob_defected'],
            "username": st.secrets.get("DB_USERNAME", "Anonymous")
        }
        
        st.session_state.prediction_history.append(log_entry)
        
        # Update statistics
        st.session_state.processing_stats['total_processed'] += 1
        if prediction_data['result'] == 'Defected':
            st.session_state.processing_stats['defected'] += 1
        else:
            st.session_state.processing_stats['non_defected'] += 1
        
        return True
    except Exception as e:
        st.warning(f"Logging error: {str(e)}")
        return False

def create_prediction_chart(prediction_data):
    """Create interactive prediction visualization"""
    try:
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Prediction Confidence', 'Class Probabilities'),
            specs=[[{"type": "indicator"}, {"type": "bar"}]]
        )
        
        # Confidence gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=prediction_data['confidence'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # Probability bar chart
        fig.add_trace(
            go.Bar(
                x=['Non Defected', 'Defected'],
                y=[prediction_data['prob_non_defected'], prediction_data['prob_defected']],
                marker_color=['green', 'red'],
                text=[f"{prediction_data['prob_non_defected']:.2%}", 
                      f"{prediction_data['prob_defected']:.2%}"],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def create_statistics_dashboard():
    """Create statistics dashboard"""
    stats = st.session_state.processing_stats
    
    if stats['total_processed'] > 0:
        # Create pie chart
        fig = px.pie(
            values=[stats['non_defected'], stats['defected']],
            names=['Non Defected', 'Defected'],
            title='Processing Statistics',
            color_discrete_map={'Non Defected': 'green', 'Defected': 'red'}
        )
        fig.update_layout(height=300)
        return fig
    return None

def camera_input():
    """Enhanced camera input with preview"""
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)
    st.markdown("### 📸 Camera Input")
    
    camera_image = st.camera_input("Take a picture of the ceramic tile", key="camera")
    
    if camera_image:
        st.success("📷 Image captured successfully!")
        return Image.open(camera_image)
    
    st.markdown('</div>', unsafe_allow_html=True)
    return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI Ceramic Tile Inspector</h1>
        <p>Advanced AI-powered defect detection system with real-time analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Control Panel")
        
        # Model status
        model = load_model()
        if model:
            st.success("✅ AI Model Loaded")
        else:
            st.error("❌ Model Loading Failed")
            st.stop()
        
        st.markdown("---")
        
        # Input method selection
        st.subheader("📥 Input Method")
        input_method = st.radio(
            "Choose input method:",
            ["📁 File Upload", "📸 Camera Capture", "🖼️ Sample Images"],
            key="input_method"
        )
        
        st.markdown("---")
        
        # Image enhancement controls
        st.subheader("🎨 Image Enhancement")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
        
        st.markdown("---")
        
        # Processing options
        st.subheader("⚙️ Processing Options")
        auto_enhance = st.checkbox("Auto Enhancement", value=True)
        show_heatmap = st.checkbox("Show Attention Heatmap", value=False)
        batch_mode = st.checkbox("Batch Processing", value=False)
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📊 Session Statistics")
        stats = st.session_state.processing_stats
        st.metric("Total Processed", stats['total_processed'])
        st.metric("Non Defected", stats['non_defected'], delta=f"{stats['non_defected']}")
        st.metric("Defected", stats['defected'], delta=f"{stats['defected']}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Input")
        
        uploaded_image = None
        
        if input_method == "📁 File Upload":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload a clear image of a ceramic tile"
            )
            if uploaded_file:
                uploaded_image = Image.open(uploaded_file)
                image_name = uploaded_file.name
        
        elif input_method == "📸 Camera Capture":
            uploaded_image = camera_input()
            image_name = f"camera_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
    elif input_method == "🖼️ Sample Images":
    # Create a dictionary of sample images with labels
    sample_images = {
        "Sample 1 - Defected": "sample1.jpeg",
        "Sample 2 - Non-Defected": "sample2.jpeg",
        "Sample 3 - Edge Defect": "sample3.jpeg"
    }
    
    sample_option = st.selectbox(
        "Select a sample image:",
        ["Select..."] + list(sample_images.keys())
    )
    
    if sample_option != "Select...":
        try:
            # Load the selected sample image
            image_path = sample_images[sample_option]
            uploaded_image = Image.open(image_path)
            image_name = image_path
            st.success(f"Loaded sample image: {sample_option}")
            
            # Display the sample image
            st.image(uploaded_image, 
                    caption=f"Sample: {sample_option}", 
                    use_column_width=True)
        except Exception as e:
            st.error(f"Error loading sample image: {str(e)}")
        
        # Display uploaded image with enhancements
        if uploaded_image:
            st.subheader("🖼️ Original Image")
            st.image(uploaded_image, caption="Original Image", use_column_width=True)
            
            # Apply enhancements
            if auto_enhance or any([brightness != 1.0, contrast != 1.0, sharpness != 1.0]):
                enhanced_image = enhance_image(uploaded_image, brightness, contrast, sharpness)
                st.subheader("✨ Enhanced Image")
                st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
            else:
                enhanced_image = uploaded_image
    
    with col2:
        st.subheader("🎯 Analysis Results")
        
        if uploaded_image:
            with st.spinner('🔄 AI Analysis in Progress...'):
                # Preprocess image
                processed_image = preprocess_image(enhanced_image)
                
                if processed_image is not None:
                    # Make prediction
                    prediction_data = predict_defect(model, processed_image)
                    
                    if prediction_data:
                        # Save to log
                        save_prediction_log(image_name, prediction_data)
                        
                        # Display results with enhanced UI
                        result = prediction_data['result']
                        confidence = prediction_data['confidence']
                        
                        if result == "Non Defected":
                            st.markdown(f"""
                            <div class="success-card">
                                <h2>✅ {result}</h2>
                                <h3>Confidence: {confidence:.1%}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown(f"""
                            <div class="error-card">
                                <h2>⚠️ {result}</h2>
                                <h3>Confidence: {confidence:.1%}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Interactive chart
                        chart = create_prediction_chart(prediction_data)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        
                        # Detailed metrics
                        st.subheader("📈 Detailed Analysis")
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                "Confidence Score",
                                f"{confidence:.1%}",
                                delta=f"{confidence - 0.5:.1%}"
                            )
                        
                        with metric_col2:
                            st.metric(
                                "Uncertainty",
                                f"{prediction_data['uncertainty']:.1%}",
                                delta=f"{0.5 - prediction_data['uncertainty']:.1%}"
                            )
                        
                        with metric_col3:
                            st.metric(
                                "Risk Level",
                                "High" if prediction_data['uncertainty'] > 0.3 else "Low",
                                delta="Safe" if result == "Non Defected" else "Attention"
                            )
                        
                        # Probability breakdown
                        st.subheader("🔍 Probability Breakdown")
                        prob_col1, prob_col2 = st.columns(2)
                        
                        with prob_col1:
                            st.write("**Non Defected Probability**")
                            st.progress(prediction_data['prob_non_defected'])
                            st.write(f"{prediction_data['prob_non_defected']:.2%}")
                        
                        with prob_col2:
                            st.write("**Defected Probability**")
                            st.progress(prediction_data['prob_defected'])
                            st.write(f"{prediction_data['prob_defected']:.2%}")
                        
                        # AI Insights
                        st.markdown("""
                        <div class="info-card">
                            <h4>🤖 AI Insights</h4>
                            <p>Analysis complete! The AI model has processed your image using advanced deep learning techniques.</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        else:
            st.info("👆 Please upload an image or use camera to start analysis")
    
    # Statistics Dashboard
    if st.session_state.processing_stats['total_processed'] > 0:
        st.markdown("---")
        st.subheader("📊 Session Dashboard")
        
        dashboard_col1, dashboard_col2 = st.columns([1, 1])
        
        with dashboard_col1:
            stats_chart = create_statistics_dashboard()
            if stats_chart:
                st.plotly_chart(stats_chart, use_container_width=True)
        
        with dashboard_col2:
            st.subheader("📈 Recent Analysis History")
            if st.session_state.prediction_history:
                for i, log in enumerate(reversed(st.session_state.prediction_history[-5:])):
                    with st.expander(f"Analysis #{len(st.session_state.prediction_history) - i}"):
                        st.write(f"**Image:** {log['image_name']}")
                        st.write(f"**Result:** {log['prediction']}")
                        st.write(f"**Confidence:** {log['confidence']:.1%}")
                        st.write(f"**Time:** {log['timestamp'][:19]}")
    
    # Advanced Features
    st.markdown("---")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            ### 🚀 Getting Started
            1. **Choose Input Method**: File upload, camera, or samples
            2. **Adjust Settings**: Use sidebar controls for enhancement
            3. **Upload/Capture**: Add your ceramic tile image
            4. **Get Results**: View AI analysis with confidence scores
            
            ### 💡 Tips for Best Results
            - **Lighting**: Use good, even lighting
            - **Focus**: Ensure the image is sharp and clear
            - **Angle**: Capture the tile straight-on
            - **Background**: Use a neutral background
            """)
    
    with feature_col2:
        with st.expander("🤖 About the AI Model"):
            try:
                username = st.secrets.get("DB_USERNAME", "Anonymous")
            except:
                username = "Anonymous"
            
            st.markdown(f"""
            ### 🧠 Model Information
            - **Architecture**: Deep Convolutional Neural Network
            - **Input Size**: 224x224 pixels
            - **Classes**: Binary classification (Defected/Non-Defected)
            - **Training**: Specialized for ceramic tile defects
            
            ### 📊 Performance Metrics
            - **Accuracy**: High precision detection
            - **Speed**: Real-time processing
            - **Reliability**: Industrial-grade performance
            
            **Current User**: {username}
            """)
    
    with feature_col3:
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            ### ⚙️ Processing Pipeline
            1. **Image Preprocessing**: Resize, normalize, enhance
            2. **Feature Extraction**: Deep learning feature maps
            3. **Classification**: Binary defect detection
            4. **Post-processing**: Confidence calculation
            
            ### 🎯 Detection Capabilities
            - **Cracks**: Surface and structural cracks
            - **Chips**: Edge and corner damage
            - **Discoloration**: Color inconsistencies
            - **Surface Defects**: Texture abnormalities
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🔍 AI Ceramic Tile Inspector - Powered by Advanced Deep Learning</p>
        <p>Built with ❤️ using Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
