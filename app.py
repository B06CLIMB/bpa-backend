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
CORS(app)

DB_FILE = 'users.json'
db_lock = Lock()  # Prevent race conditions

# ---------------- USER DATA ---------------- #
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
        # New user registration
        users[name] = {
            'age': age,
            'password': generate_password_hash(password),
            'data': []
        }
        save_users(users)
        return jsonify({'message': 'New user registered and logged in'}), 201

# ---------------- USER DATA ENDPOINTS ---------------- #
@app.route('/data', methods=['GET'])
def get_data():
    name = request.args.get('name')
    if not name:
        return jsonify({'message': 'Name required'}), 400

    users = load_users()
    return jsonify(users.get(name, {'message': 'User not found'})), 200

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

# ---------------- MODEL INFERENCE ---------------- #
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

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
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return jsonify({
        "breed": classes[pred.item()],
        "confidence": float(conf.item())
    })

# ---------------- RUN APP ---------------- #
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

