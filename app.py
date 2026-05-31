from flask import Flask, render_template, request, jsonify

from keras.models import load_model
import numpy as np
import mediapipe as mp
import joblib
import cv2
import base64

from io import BytesIO

app = Flask(__name__)

# =========================

# Load model & scaler

# =========================

model = load_model("best_sign_model.keras")

scaler = joblib.load("scaler.save")

# =========================

# Labels

# =========================

labels = {
"alsalam_kum": ["السلام عليكم", "Peace be upon you"],
"win_alhamam": ["وين دورة المياه", "Where is the restroom"],
"win_altawaree": ["وين الطوارئ", "Where is emergency"]
}

class_names = [
"alsalam_kum",
"win_alhamam",
"win_altawaree"
]

# =========================

# MediaPipe

# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
max_num_hands=2,
min_detection_confidence=0.3,
min_tracking_confidence=0.3
)

# =========================

# Sequence Memory

# =========================

sequence = []

SEQUENCE_LENGTH = 30

last_valid = np.zeros(126)

# =========================

# Normalize

# =========================

def normalize_hand(hand_array):
    hand = hand_array.reshape(21, 3)

    wrist = hand[0]

    hand = hand - wrist

    max_val = np.max(np.abs(hand))

    if max_val > 0:
        hand = hand / max_val

    return hand.flatten()


# =========================

# Pages

# =========================

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/translator")
def translator():
    return render_template("translator.html")

@app.route("/predict", methods=["POST"])
def predict():

    global sequence
    global last_valid

    data = request.get_json()

    image_data = data["image"]

    image_data = image_data.split(",")[1]

    image_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(image)

    if not results.multi_hand_landmarks:

        return jsonify({
            "arabic": "...",
            "english": "Waiting for gesture",
            "playAudio": False
        })

    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):

        label = handedness.classification[0].label

        hand_array = []

        for lm in hand_landmarks.landmark:

            hand_array.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        hand_array = np.array(hand_array)

        hand_array = normalize_hand(hand_array)

        if label == "Left":
            left_hand = hand_array

        if label == "Right":
            right_hand = hand_array

    frame_features = np.concatenate([
        left_hand,
        right_hand
    ])

    if np.all(frame_features == 0):

        frame_features = last_valid

    else:

        last_valid = frame_features

    sequence.append(frame_features)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    if len(sequence) < SEQUENCE_LENGTH:

        return jsonify({
            "arabic": "...",
            "english": "Reading gesture...",
            "playAudio": False
        })

    input_data = np.array(sequence)

    input_data = scaler.transform(input_data)

    input_data = input_data.reshape(
        1,
        30,
        126
    )

    prediction = model.predict(
        input_data,
        verbose=0
    )[0]

    predicted_class = np.argmax(
        prediction
    )

    predicted_label = class_names[
        predicted_class
    ]

    ar = labels[predicted_label][0]

    en = labels[predicted_label][1]

    return jsonify({
        "arabic": ar,
        "english": en,
        "playAudio": True
    })

if __name__ == "__main__":
    app.run(
host="0.0.0.0",
port=5000,
debug=True
)
