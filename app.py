from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os

app = Flask(__name__)
CORS(app)

# ---------------- Model Setup ---------------- #
device = torch.device("cpu")  # Change to "cuda" if GPU is available
MODEL_FILE = "resnet50_buffalo_best.pth"
CLASSES_FILE = "classes.txt"

# Load classes
with open(CLASSES_FILE, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
state_dict = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model loaded and ready!")

# ---------------- Login Endpoint ---------------- #
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    print("Received login data:", data)  # DEBUG

    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()
    age = str(data.get("age", "")).strip()

    print(f"Parsed - username: '{username}', password: '{password}', age: '{age}'")  # DEBUG

    # Hardcoded credentials with age check
    if username == "admin" and password == "232" and age == "27":
        return jsonify({"success": True, "message": "Login successful"})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

# ---------------- Prediction Endpoint ---------------- #
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(file.stream).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        return jsonify({
            "breed": classes[pred.item()],
            "confidence": float(conf.item() * 100)
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ---------------- Run App ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}...")
    app.run(host="0.0.0.0", port=port)
