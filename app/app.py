import streamlit as st
import torch
# import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys

from mtcnn import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import InceptionResnetV1

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- CONFIGURATION ---
DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- UTILS ---
@st.cache_resource
def load_label_map():
    """Reconstruct label map from classes.txt or data_processed directory."""
    # Priority 1: classes.txt (for deployment)
    classes_file = os.path.join(os.path.dirname(__file__), '..', 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return {i: name for i, name in enumerate(classes)}

    # Priority 2: data_processed (for local dev)
    if os.path.exists(DATA_PROCESSED_DIR):
        return None

@st.cache_resource
def load_model(model_path, num_classes):
    """Load the trained model."""
    try:
        # Initialize InceptionResnetV1 for classification
        model = InceptionResnetV1(classify=True, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def get_detector():
    return MTCNN()

def preprocess_image(image, target_size=160):
    """Preprocess image for inference (Resize -> Normalize -> Tensor)."""
    transform = A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    # Convert PIL to Numpy
    image_np = np.array(image)
    augmented = transform(image=image_np)
    return augmented['image'].unsqueeze(0) # Add batch dimension

# --- APP UI ---
st.set_page_config(page_title="Face Recognition Demo", layout="wide", page_icon="👤")

st.title("👤 Face Recognition Demo")
st.markdown("Upload an image to detect faces and recognize the person.")

# Sidebar
st.sidebar.header("Settings")

# 1. Load Label Map
idx_to_class = load_label_map()
if not idx_to_class:
    st.sidebar.error("Could not load label map. Check `data_processed` folder.")
    st.stop()
else:
    st.sidebar.success(f"Loaded {len(idx_to_class)} classes.")

# 2. Load Model
model_path = os.path.join(MODEL_DIR, 'InceptionResnetV1-kfold.pth')
if not os.path.exists(model_path):
    st.sidebar.error(f"Model not found: {model_path}")
    st.stop()

model = load_model(model_path, len(idx_to_class))

# 4. Confidence Threshold
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit & PyTorch")

# Main Area
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display Original Image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    if st.button('Analyze Image', type="primary"):
        with st.spinner('Detecting faces...'):
            detector = get_detector()
            image_np = np.array(image)
            
            # MTCNN expects RGB (which PIL provides, but let's be safe with numpy)
            detections = detector.detect_faces(image_np)
            
            if not detections:
                st.warning("No faces detected.")
            else:
                # Draw boxes on image
                draw_img = image.copy()
                draw = ImageDraw.Draw(draw_img)
                
                results = []
                
                for detection in detections:
                    box = detection['box']
                    confidence = detection['confidence']
                    
                    if confidence < 0.90: # Filter low confidence detections from MTCNN
                        continue
                        
                    x, y, w, h = box
                    # Ensure within bounds
                    x, y = max(0, x), max(0, y)
                    
                    # Crop Face
                    face_crop = image.crop((x, y, x+w, y+h))
                    
                    # Inference
                    input_tensor = preprocess_image(face_crop).to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        conf, pred_idx = torch.max(probs, 1)
                        
                        pred_class = idx_to_class[pred_idx.item()]
                        pred_conf = conf.item()
                    
                    # Store result
                    results.append({
                        'box': box,
                        'class': pred_class,
                        'conf': pred_conf,
                        'crop': face_crop
                    })
                    
                    # Draw Box & Label
                    color = "green" if pred_conf > confidence_threshold else "red"
                    label = f"{pred_class} ({pred_conf:.1%})"
                    
                    draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
                    
                    # Draw text background
                    # text_bbox = draw.textbbox((x, y), label) # Needs newer Pillow
                    # draw.rectangle(text_bbox, fill=color)
                    draw.text((x, y-15), label, fill=color)

                # Show Result Image
                with col2:
                    st.subheader("Detection Results")
                    st.image(draw_img, use_column_width=True)
                
                # Show Individual Crops & Details
                st.markdown("### Detailed Results")
                if not results:
                     st.info("Faces detected but filtered out by MTCNN confidence.")
                
                for res in results:
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.image(res['crop'], width=100, caption="Cropped Face")
                    with c2:
                        st.markdown(f"**Prediction:** `{res['class']}`")
                        st.markdown(f"**Confidence:** `{res['conf']:.2%}`")
                        
                        if res['conf'] > confidence_threshold:
                            st.success("Match Found!")
                        else:
                            st.error("Low Confidence - Unknown or Uncertain")
                        
                        st.progress(res['conf'])
                    st.divider()

