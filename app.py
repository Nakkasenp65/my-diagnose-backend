from flask import Flask, request, jsonify
import time
from threading import Lock
from flask_cors import CORS
from src.predict import predict_disease
import os

app = Flask(__name__)
CORS(app)

ip_last_call = {}
ip_lock = Lock()
COOLDOWN_SECONDS = 15


@app.route("/predict", methods=["GET"])
def predict_route():
    user_ip = request.remote_addr
    current_time = time.time()

    with ip_lock:
        last_time = ip_last_call.get(user_ip, 0)
        wait_time = COOLDOWN_SECONDS - (current_time - last_time)

        if wait_time > 0:
            time.sleep(wait_time)

        ip_last_call[user_ip] = time.time()

    symptom_str = request.args.get("symptoms", "")
    symptom_list = symptom_str.split(",") if symptom_str else []
    disease, probability, unknowns = predict_disease(symptom_list)

    return jsonify({
        "disease": disease,
        "confidence": round(probability, 4),
        "unknown_symptoms": unknowns,
        "symptoms_provided": symptom_list
    })

@app.route("/")
def index():
    return "Welcome to the Disease Prediction API! Use /predict endpoint."

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)

# GET request for predict, so the user will get the result