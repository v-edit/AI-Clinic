import os
import json
import hashlib

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

def _load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def _save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def _hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(username: str, password: str):
    users = _load_users()
    if username in users:
        return False, "❌ Username already exists."
    users[username] = _hash_password(password)
    _save_users(users)
    return True, "✅ Account created successfully!"

def login(username: str, password: str):
    users = _load_users()
    if username not in users:
        return False
    return users[username] == _hash_password(password)
