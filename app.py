from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from threading import Lock

# ---------------- FLASK APP ---------------- #
app = Flask(__name__)
CORS(app, origins=["https://b06climb.github.io"])  # Only allow your frontend

DB_FILE = 'users.json'
db_lock = Lock()  # Prevent race conditions

# ---------------- HELPER FUNCTIONS ---------------- #
def load_users():
    if not os.path.exists(DB_FILE):
        return {}
    with db_lock:
        with open(DB_FILE, 'r') as f:
            return json.load(f)

def save_users(users):
    with db_lock:
        with open(DB_FILE, 'w') as f:
            json.dump(users, f, indent=4)

# ---------------- LOGIN / REGISTER ---------------- #
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    name, age, password = data.get('name'), data.get('age'), data.get('password')

    if not name or not password:
        return jsonify({'message': 'Name and password required'}), 400

    users = load_users()

    if name in users:
        if check_password_hash(users[name]['password'], password):
            return jsonify({'message': 'Login successful'}), 200
        else:
            return jsonify({'message': 'Invalid password'}), 401
    else:
        # Register new user
        users[name] = {
            'age': age,
            'password': generate_password_hash(password),
            'data': []
        }
        save_users(users)
        return jsonify({'message': 'New user registered and logged in'}), 201

# ---------------- USER DATA ---------------- #
@app.route('/data', methods=['GET'])
def get_data():
    name = request.args.get('name')
    if not name:
        return jsonify({'message': 'Name required'}), 400

    users = load_users()
    if name in users:
        return jsonify({'data': users[name].get('data', [])}), 200
    else:
        return jsonify({'data': []}), 200

@app.route('/data', methods=['POST'])
def save_data():
    data = request.json
    name, new_record = data.get('name'), data.get('record')
    if not name or new_record is None:
        return jsonify({'message': 'Name and record required'}), 400

    users = load_users()
    if name in users:
        users[name]['data'].append(new_record)
        save_users(users)
        return jsonify({'message': 'Data saved successfully'}), 200
    else:
        return jsonify({'message': 'User not found'}), 404

# ---------------- MODEL ---------------- #
# Load classes
CLASSES_FILE = "classes.txt"
MODEL_FILE = "resnet50_buffalo_best.pth"

if not os.path.exists(CLASSES_FILE) or not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("classes.txt or model file not found. Make sure both are in the repo.")

with open(CLASSES_FILE, "r") as f:
    classes = [line.strip() for line in f.readlines()]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

device = torch.device("cpu")  # Change to "cuda" if GPU available
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
state_dict = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ---------------- PREDICTION ---------------- #
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    try:
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        return jsonify({
            "breed": classes[pred.item()],
            "confidence": float(conf.item())
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ---------------- RUN ---------------- #
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting app on port {port}...")
    app.run(host="0.0.0.0", port=port)

