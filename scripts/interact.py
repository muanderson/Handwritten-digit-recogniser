from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import numpy as np
import io
import matplotlib.pyplot as plt
import os
import time
from model import CNN

# Load model
model = CNN()
model.load_state_dict(torch.load(r'C:\Users\Matthew\Documents\PhD\MNIST\models\best_model_fold_2.pt'))
model.to('cuda:0')
model.eval()

# --- Setup Model for Prediction ---
print("Loading fine-tuned model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model_path = r'C:\Users\Matthew\Documents\PhD\MNIST\models\fine_tuned_best_model_fold_5.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() 
print(f"Model loaded successfully from {model_path} and set to evaluation mode.")

# --- Setup Flask App ---
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    image_file = request.files['file']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # --- Image Preprocessing ---
    image = Image.open(io.BytesIO(image_file.read())).convert('L')
    image_np = np.array(image)

    # Find bounding box to crop the digit
    coords = np.argwhere(image_np > 50) # Find non-black pixels
    if coords.size == 0:
        return jsonify({'prediction': 'No digit found'})

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = image_np[y0:y1, x0:x1]

    # Resize to 20x20 while maintaining aspect ratio, then pad to 28x28
    cropped_img = Image.fromarray(cropped)
    cropped_img.thumbnail((20, 20), Image.LANCZOS)
    
    # Create a new 28x28 black image and paste the digit in the center
    padded_img = Image.new('L', (28, 28), 0)
    paste_x = (28 - cropped_img.width) // 2
    paste_y = (28 - cropped_img.height) // 2
    padded_img.paste(cropped_img, (paste_x, paste_y))

    # --- Prediction Logic ---
    # Normalize the image tensor using MNIST's mean and std
    padded_np = np.array(padded_img).astype(np.float32) / 255.0
    mean, std = 0.1307, 0.3081
    normed = (padded_np - mean) / std

    tensor = torch.tensor(normed).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        
    prediction_result = {
        'prediction': str(pred.item()),
        'confidence': f"{confidence.item():.4f}"
    }

    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(debug=True)