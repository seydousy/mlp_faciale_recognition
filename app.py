# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
import joblib
import os
from werkzeug.utils import secure_filename
from flask.json import JSONEncoder
from flask import send_from_directory

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model architecture
class MLPFaceClassifier(nn.Module):
    def __init__(self, input_dim=100, hidden1=256, hidden2=128, dropout=0.3, n_classes=5):  # Changez à 5
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes)
        )

    def forward(self, x):
        return self.model(x)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Ajout de la route pour servir les fichiers du dossier uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(CustomJSONEncoder, self).default(obj)

app.json_encoder = CustomJSONEncoder
# Load the model and preprocessing tools
try:
    # Load PCA and scaler
    pca = joblib.load('pca_model.pkl')
    scaler = joblib.load('pca_scaler.pkl')
    
    # Load model
    model = MLPFaceClassifier().to(device)
    model.load_state_dict(torch.load('modele_lfw.pth', map_location=device))
    model.eval()
    
    # Class names (replace with your actual class names)
    class_names = ["Colin Powell", "Donald Rumsfeld", 
                   "George W Bush", "Gerhard Schroeder", "Tony Blair"]
    
    model_loaded = True
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_loaded = False
    class_names = []

def preprocess_image(image):
    """Preprocess an image for prediction"""
    # Resize to match the expected input (62x47)
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((47, 62))  # Resize to match LFW dimensions
    
    # Convert to numpy array and flatten
    img_array = np.array(image).flatten()
    
    # Apply the same preprocessing as during training
    img_scaled = scaler.transform([img_array])
    img_pca = pca.transform(img_scaled)
    
    # Convert to tensor
    img_tensor = torch.tensor(img_pca, dtype=torch.float32).to(device)
    
    return img_tensor

def get_prediction(img_tensor):
    """Get model prediction for an image tensor"""
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
    predicted_class = predicted.item()
    confidence = probabilities[0][predicted_class].item() * 100
    
    # Get all probabilities as a list
    all_probs = probabilities[0].cpu().numpy() * 100
    
    return {
        'class': class_names[predicted_class],
        'confidence': confidence,
        'all_probabilities': [{'name': name, 'probability': prob} 
                              for name, prob in zip(class_names, all_probs)]
    }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', model_status=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read and process the image
        img = Image.open(io.BytesIO(file.read()))
        img_tensor = preprocess_image(img)
        
        # Get prediction
        result = get_prediction(img_tensor)
        
        # Save image for display
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(file_path)
        # Return prediction result
        return jsonify({
            'prediction': result,
            'image_path': url_for('uploaded_file', filename=filename)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/webcam_predict', methods=['POST'])
def webcam_predict():
    """Handle webcam image prediction requests"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get the base64 image data
        image_data = request.json.get('image', '')
        if not image_data:
            return jsonify({'error': 'No image data received'}), 400
        
        # Remove the data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode the base64 image
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Process and predict
        img_tensor = preprocess_image(img)
        result = get_prediction(img_tensor)
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)