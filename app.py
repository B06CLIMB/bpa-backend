from flask import Flask, request, jsonify
import json
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# File to act as our simple database
DB_FILE = 'users.json'

# Load user data from the file
def load_users():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, 'r') as f:
        return json.load(f)

# Save user data to the file
def save_users(users):
    with open(DB_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# API Endpoint for Login
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    name = data.get('name')
    age = data.get('age')
    password = data.get('password')

    users = load_users()

    # Check if user exists and password is correct
    if name in users and users[name]['password'] == password:
        return jsonify({'message': 'Login successful'}), 200
    # If user doesn't exist, create a new one
    elif name not in users:
        users[name] = {'age': age, 'password': password, 'data': []}
        save_users(users)
        return jsonify({'message': 'New user registered and logged in'}), 201
    else:
        return jsonify({'message': 'Invalid name or password'}), 401

# API Endpoint to get a user's data
@app.route('/data', methods=['GET'])
def get_data():
    name = request.args.get('name')
    users = load_users()

    if name in users:
        # Return the entire user object, including their age and data
        return jsonify(users[name]), 200
    else:
        return jsonify({'message': 'User not found'}), 404

# API Endpoint to save new data for a user
@app.route('/data', methods=['POST'])
def save_data():
    data = request.json
    name = data.get('name')
    new_record = data.get('record')

    users = load_users()

    if name in users:
        users[name]['data'].append(new_record)
        save_users(users)
        return jsonify({'message': 'Data saved successfully'}), 200
    else:
        return jsonify({'message': 'User not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)