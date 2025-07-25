from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import numpy as np
import io
import matplotlib.pyplot as plt
import os
import time # <-- Import the time module

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 5 * 5, 128)
        self.linearout = nn.Linear(128, 10)

    def forward(self, image):
        x = self.conv1(image)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linearout(x)
        return x

# Load model
model = CNN()
# IMPORTANT: Make sure this path is correct for your system
model.load_state_dict(torch.load(r'C:\Users\Matthew\Documents\PhD\MNIST\models\best_model_fold_2.pt'))
model.to('cuda:0')
model.eval()

app = Flask(__name__, static_folder='static')

# --- NEW: Define the path for your new dataset ---
DATASET_PATH = 'my_drawings'
os.makedirs(DATASET_PATH, exist_ok=True)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # --- MODIFICATION: Get the label from the form data ---
    label = request.form.get('label')
    if label is None:
        return jsonify({'error': 'No label provided'}), 400

    image_file = request.files['file']
    image = Image.open(io.BytesIO(image_file.read())).convert('L')
    image = np.array(image)

    # Threshold to binary: white digit on black background
    image = (image > 50).astype(np.uint8) * 255

    coords = np.argwhere(image)
    if coords.size == 0:
        return jsonify({'prediction': 'No digit found'})

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = image[y0:y1, x0:x1]

    # Resize to 20x20 (retaining anti-aliasing is better for training)
    cropped_img = Image.fromarray(cropped).resize((20, 20), Image.LANCZOS)
    
    # Pad to 28x28 with black background
    padded_img = ImageOps.expand(cropped_img, border=4, fill=0)

    # --- MODIFICATION: Save the image for your dataset ---
    # 1. Create a subdirectory for the label (e.g., 'my_drawings/7')
    label_dir = os.path.join(DATASET_PATH, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # 2. Create a unique filename using a timestamp
    timestamp = int(time.time() * 1000)
    filename = f"{timestamp}.png"
    save_path = os.path.join(label_dir, filename)

    # 3. Save the processed image
    padded_img.save(save_path)


    # --- Prediction logic remains the same ---
    padded = np.array(padded_img).astype(np.float32) / 255.0
    mean, std = 0.1307, 0.3081
    normed = (padded - mean) / std

    tensor = torch.tensor(normed).unsqueeze(0).unsqueeze(0).to('cuda:0')

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()

    # Return both the prediction and the path where the image was saved
    return jsonify({'prediction': str(pred), 'saved_path': save_path})


if __name__ == '__main__':
    app.run(debug=True)