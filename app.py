from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------- LOGIN / USER DATA ---------------- #
app = Flask(__name__)
CORS(app)
DB_FILE = 'users.json'

def load_users():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(DB_FILE, 'w') as f:
        json.dump(users, f, indent=4)

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    name, age, password = data.get('name'), data.get('age'), data.get('password')
    users = load_users()

    if name in users and users[name]['password'] == password:
        return jsonify({'message': 'Login successful'}), 200
    elif name not in users:
        users[name] = {'age': age, 'password': password, 'data': []}
        save_users(users)
        return jsonify({'message': 'New user registered and logged in'}), 201
    else:
        return jsonify({'message': 'Invalid name or password'}), 401

@app.route('/data', methods=['GET'])
def get_data():
    name = request.args.get('name')
    users = load_users()
    return jsonify(users.get(name, {'message': 'User not found'})), 200

@app.route('/data', methods=['POST'])
def save_data():
    data = request.json
    name, new_record = data.get('name'), data.get('record')
    users = load_users()
    if name in users:
        users[name]['data'].append(new_record)
        save_users(users)
        return jsonify({'message': 'Data saved successfully'}), 200
    else:
        return jsonify({'message': 'User not found'}), 404

# ---------------- MODEL INFERENCE ---------------- #
# Load classes from file
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Preprocessing for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Load model once on CPU only
device = torch.device("cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
state_dict = torch.load("resnet50_buffalo_best.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return jsonify({
        "breed": classes[pred.item()],
        "confidence": float(conf.item())
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)