from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, datetime, os, numpy as np
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import jwt
from bson import ObjectId
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
from flask import send_file
from tensorflow.keras.models import load_model
import joblib
# -------------------------------------------------------------
# Load environment
# -------------------------------------------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "voc_project")
SECRET_KEY = os.getenv("SECRET_KEY", "voc_secret_key")
MODEL_PATH = os.getenv("MODEL_PATH", "model/voc_random_forest_model.pkl")

# -------------------------------------------------------------
# Initialize Flask + Mongo
# -------------------------------------------------------------
app = Flask(__name__)
CORS(app)

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
users_collection = db["users"]
sensor_collection = db["sensor_data"]

# -------------------------------------------------------------
# Load trained ML model
# -------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "model/voc_nn_model.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "model/voc_scaler.pkl")
ENCODER_PATH = os.getenv("ENCODER_PATH", "model/label_encoder.pkl")

try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print(f"✅ Neural Network model loaded successfully: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model, scaler, label_encoder = None, None, None


# -------------------------------------------------------------
# Helper: token auth
# -------------------------------------------------------------
def token_required(f):
    def decorator(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
        if not token:
            return jsonify({"error": "Missing JWT token"}), 401

        try:
            decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            user = users_collection.find_one({"_id": ObjectId(decoded["user_id"])})
            if not user:
                return jsonify({"error": "User not found"}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except Exception as e:
            return jsonify({"error": "Token invalid", "details": str(e)}), 401

        return f(user, *args, **kwargs)
    decorator.__name__ = f.__name__
    return decorator


# -------------------------------------------------------------
# Disease Prediction Function
# -------------------------------------------------------------
# app.py
def predict_disease(mq2, mq3, mq135, temp, r0_mq2, r0_mq3, r0_mq135):
    if model is None or scaler is None or label_encoder is None:
        return {"prediction": "model_not_loaded"}

    # Create feature vector
    features = np.array([[mq2, mq3, mq135, temp]])

    features = np.nan_to_num(np.array([[mq2, mq3, mq135, temp]]))

    # Scale the features (same as training)
    scaled_features = scaler.transform(features)

    # Predict probabilities
    probs = model.predict(scaled_features)[0]

    # Decode prediction
    pred_idx = np.argmax(probs)
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # Prepare output
    probs_dict = {label_encoder.classes_[i]: round(float(probs[i] * 100), 2) for i in range(len(probs))}

    risk_score = round(float(np.max(probs) * 100), 2)

    return {
        "prediction": pred_label,
        "probabilities": probs_dict,
        "risk_score": risk_score
    }


# -------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------
@app.route("/")
def index():
    return jsonify({"status": "backend_running"}), 200


# ---------------- Registration ----------------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data or "email" not in data or "password" not in data:
        return jsonify({"error": "Email and password required"}), 400

    if users_collection.find_one({"email": data["email"]}):
        return jsonify({"error": "User already exists"}), 400

    hashed_pw = generate_password_hash(data["password"])
    new_user = {
        "email": data["email"],
        "password": hashed_pw,
        "created_at": datetime.datetime.utcnow()
    }
    users_collection.insert_one(new_user)
    return jsonify({"status": "registered"}), 201


# ---------------- Login ----------------
@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or "email" not in data or "password" not in data:
        return jsonify({"error": "Missing email or password"}), 400

    user = users_collection.find_one({"email": data["email"]})
    if not user or not check_password_hash(user["password"], data["password"]):
        return jsonify({"error": "Invalid credentials"}), 401

    token = jwt.encode(
        {
            "user_id": str(user["_id"]),
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        },
        SECRET_KEY,
        algorithm="HS256"
    )

    return jsonify({"token": token, "status": "login_success"}), 200


# ---------------- Logout ----------------
@app.route("/api/logout", methods=["POST"])
@token_required
def logout(user):
    return jsonify({"status": "logout_success"}), 200


# ---------------- ESP32 Sensor Endpoint ----------------
# ---------------- ESP32 Sensor Endpoint ----------------
@app.route("/api/sensor", methods=["POST"])
def receive_sensor():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Invalid JSON", "details": str(e)}), 400

    # default demo user
    user = {"_id": "demo_user", "email": "demo@app"}

    try:
        mq2 = float(data.get("mq2_adc", 0))
        mq3 = float(data.get("mq3_adc", 0))
        mq135 = float(data.get("mq135_adc", 0))
        temp = float(data.get("temp_c", 0))
        hum = float(data.get("humidity_pct", 0))
        r0_mq2 = float(data.get("r0_mq2", mq2))
        r0_mq3 = float(data.get("r0_mq3", mq3))
        r0_mq135 = float(data.get("r0_mq135", mq135))
    except Exception as e:
        return jsonify({"error": "Missing or invalid parameters", "details": str(e)}), 400

    # prediction
    pred = predict_disease(mq2, mq3, mq135, temp, r0_mq2, r0_mq3, r0_mq135)

    # store in DB
    record = {
        "user_id": str(user["_id"]),
        "email": user["email"],
        "timestamp": datetime.datetime.utcnow(),
        "mq2_adc": mq2,
        "mq3_adc": mq3,
        "mq135_adc": mq135,
        "temp_c": temp,
        "humidity_pct": hum,
        "r0_mq2": r0_mq2,
        "r0_mq3": r0_mq3,
        "r0_mq135": r0_mq135,
        "prediction": pred["prediction"],
        "probabilities": pred["probabilities"],
        "risk_score": pred["risk_score"],
    }
    sensor_collection.insert_one(record)

    return jsonify({
        "status": "success",
        "prediction": pred["prediction"],
        "risk_score": pred["risk_score"],
        "probabilities": pred["probabilities"]
    }), 200



# ---------------- Fetch latest reading ----------------
# ---------------- Fetch latest reading ----------------
@app.route("/api/user/latest", methods=["GET"])
def get_latest():
    record = sensor_collection.find_one(
        {"user_id": "demo_user"},
        sort=[("timestamp", -1)]
    )
    if not record:
        return jsonify({"message": "No readings yet"}), 404

    record["_id"] = str(record["_id"])
    return jsonify(record), 200


# ---------------- Fetch user history ----------------
@app.route("/api/user/history", methods=["GET"])
def get_history():
    limit = int(request.args.get("limit", 10))
    records = list(
        sensor_collection.find({"user_id": "demo_user"}).sort("timestamp", -1).limit(limit)
    )
    for r in records:
        r["_id"] = str(r["_id"])
    return jsonify({"count": len(records), "data": records}), 200

# ---------------- Download PDF Report ----------------
@app.route("/api/user/report", methods=["GET"])
def download_report():
    records = list(sensor_collection.find({"user_id": "demo_user"}).sort("timestamp", -1).limit(10))
    if not records:
        return jsonify({"error": "No data for report"}), 404

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    p.setFont("Helvetica-Bold", 16)
    p.drawString(200, height - 50, "VOC Health Summary Report")

    p.setFont("Helvetica", 12)
    y = height - 100
    for r in records:
        line = f"{r['timestamp'].strftime('%Y-%m-%d %H:%M')} | {r['prediction']} | Temp: {r['temp_c']}°C | MQ2: {r['mq2_adc']} | MQ3: {r['mq3_adc']} | MQ135: {r['mq135_adc']}"
        p.drawString(40, y, line)
        y -= 20
        if y < 100:
            p.showPage()
            y = height - 100

    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="VOC_Report.pdf", mimetype="application/pdf")

# -------------------------------------------------------------
# RUN SERVER
# -------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
