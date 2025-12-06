from flask import Flask, request, jsonify, send_from_directory
# *** NEW: Import CORS ***
from flask_cors import CORS 
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision.models.video import r3d_18

app = Flask(__name__)
# *** NEW: Enable CORS for all routes (CRITICAL for local testing) ***
CORS(app) 
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ LOAD MODEL ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume 'deepfake_model.pth' is in the same directory
model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# Ensure the model file 'deepfake_model.pth' exists
try:
    model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
    model.to(device)
    model.eval()
except FileNotFoundError:
    print("WARNING: Model file 'deepfake_model.pth' not found. Detection will fail.")
    # Set model to evaluation state even if weights failed to load
    model.eval()


# -------------- VIDEO PREPROCESS ---------------
def preprocess_video(video_path, frames_per_clip=8, resize=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Face detection/cropping logic often goes here, but skipping for simplicity
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - 0.5) / 0.5
        frames.append(frame)

    cap.release()
    os.remove(video_path) # Clean up the uploaded file

    if len(frames) == 0:
        raise ValueError("No frames found or video file empty")

    # Select frames
    indices = np.linspace(0, len(frames)-1, frames_per_clip).astype(int)
    frames = [frames[i] for i in indices]

    # Convert to C, T, H, W (Channels, Time, Height, Width)
    frames_np = np.stack(frames).transpose(3,0,1,2) 
    return torch.tensor(frames_np).unsqueeze(0)

@app.route("/")
def home():
    # *** CHANGE: Assuming the frontend HTML is named 'index.html' ***
    # You might need to change this if your file is still "WebsiteDeployment.html"
    return send_from_directory(".", "index.html")

# The endpoint is /detect, as defined here
@app.route("/detect", methods=["POST"])
def detect():
    # Check if a file was uploaded with the correct key 'video'
    if 'video' not in request.files:
        return jsonify({"label": "ERROR", "message": "No video file part in request"}), 400
        
    file = request.files["video"]
    
    if file.filename == '':
        return jsonify({"label": "ERROR", "message": "No selected file"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    try:
        # Preprocess the video
        tensor = preprocess_video(save_path).to(device)

        # Run model inference
        with torch.no_grad():
            output = model(tensor)
            # Get probability scores
            probabilities = torch.softmax(output, dim=1)
            # Prediction: 1=Real, 0=Fake (based on typical dataset labels)
            pred = torch.argmax(probabilities).item() 
            
            # The R3D model output needs to be interpreted:
            # We assume index 1 is 'REAL' and index 0 is 'FAKE' based on the label logic.
            label = "REAL" if pred == 1 else "FAKE"
            
            # Return the result
            return jsonify({
                "label": label, 
                "confidence": probabilities.tolist()
            })

    except ValueError as e:
        return jsonify({"label": "ERROR", "message": str(e)}), 500
    except Exception as e:
        return jsonify({"label": "ERROR", "message": f"Server-side processing failed: {str(e)}"}), 500

# ------------------ RUN ------------------

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)