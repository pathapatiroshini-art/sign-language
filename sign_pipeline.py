import streamlit as st
import os
import shutil
import json
import numpy as np
import cv2
import glob
from collections import deque
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# -------------------------
# CONSTANTS
# -------------------------
SEQUENCE_LENGTH = 30
DATA_DIR = "sign_data"
DATASET_FILE = "dataset.npz"
LABEL_MAP_FILE = "label_map.json"
MODEL_FILE = "sign_model.h5"

MIN_DET_CONF = 0.5
MIN_TRACK_CONF = 0.5

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# --------------------------------------------------------------------------------
# Extracting Landmarks
# --------------------------------------------------------------------------------
def extract_landmarks(results):
    pose = np.zeros(33 * 3)
    left = np.zeros(21 * 3)
    right = np.zeros(21 * 3)

    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()

    if results.left_hand_landmarks:
        left = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()

    if results.right_hand_landmarks:
        right = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([pose, left, right])


# --------------------------------------------------------------------------------
# Collect Dataset Using Uploaded Images
# --------------------------------------------------------------------------------
def collect_from_upload(label, uploaded_images):
    if not label:
        st.error("Label required")
        return
    if not uploaded_images:
        st.error("Upload at least 1 image")
        return

    label_dir = os.path.join(DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    seq_dir = os.path.join(label_dir, f"{label}_0")
    os.makedirs(seq_dir, exist_ok=True)

    for i, file in enumerate(uploaded_images):
        img_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(seq_dir, f"frame_{i:03d}.jpg"), img)

    st.success(f"Saved {len(uploaded_images)} images for '{label}'")


# --------------------------------------------------------------------------------
# Build Dataset from Uploaded Images
# --------------------------------------------------------------------------------
def build_dataset():
    X, y = [], []

    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        st.error("❌ No sign data found! Please upload images first.")
        return

    with mp_holistic.Holistic(
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRACK_CONF
    ) as holistic:

        for label in sorted(os.listdir(DATA_DIR)):
            label_path = os.path.join(DATA_DIR, label)
            if not os.path.isdir(label_path):
                continue

            for seq_dir in sorted(os.listdir(label_path)):
                seq_path = os.path.join(label_path, seq_dir)
                frames = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))

                if len(frames) < SEQUENCE_LENGTH:
                    st.warning(f"Skipping incomplete sequence: {seq_dir}")
                    continue

                frames = frames[:SEQUENCE_LENGTH]
                seq_landmarks = []

                for img_file in frames:
                    img = cv2.imread(img_file)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = holistic.process(img_rgb)
                    keypoints = extract_landmarks(results)
                    seq_landmarks.append(keypoints)

                if len(seq_landmarks) == SEQUENCE_LENGTH:
                    X.append(seq_landmarks)
                    y.append(label)

    # -------------------------
    #       NEW SAFETY CHECKS
    # -------------------------
    if len(X) == 0 or len(y) == 0:
        st.error("❌ No valid sequences found. Make sure each label has at least 30 images.")
        return

    X = np.array(X)
    y = np.array(y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Additional safe check
    if len(y_enc) == 0:
        st.error("❌ Cannot encode labels — dataset empty.")
        return

    y_cat = to_categorical(y_enc)

    np.savez(DATASET_FILE, X=X, y=y_cat, classes=le.classes_)

    with open(LABEL_MAP_FILE, "w") as f:
        json.dump({"classes": le.classes_.tolist()}, f)

    st.success(f"Dataset created successfully! {len(X)} sequences saved.")


# --------------------------------------------------------------------------------
# Build Model
# --------------------------------------------------------------------------------
def build_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        LSTM(64),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# --------------------------------------------------------------------------------
# Train Model (FIXED FileNotFoundError)
# --------------------------------------------------------------------------------
def train_model(epochs):

    if not os.path.exists(DATASET_FILE):
        st.error("❌ dataset.npz not found! Please build dataset before training.")
        return

    data = np.load(DATASET_FILE, allow_pickle=True)
    X, y = data["X"], data["y"]

    # --------------------------
    # SAFETY: dataset size check
    # --------------------------
    num_samples = len(X)

    if num_samples == 0:
        st.error("❌ Cannot train — dataset is empty.")
        return

    if num_samples < 5:
        st.warning(f"⚠️ Very small dataset detected ({num_samples} samples). "
                   "Validation will be disabled automatically.")

        val_split = 0.0
    else:
        val_split = 0.2

    # --------------------------
    # Build & train model safely
    # --------------------------
    model = build_model((X.shape[1], X.shape[2]), y.shape[1])

    model.fit(
        X,
        y,
        epochs=epochs,
        validation_split=val_split
    )

    model.save(MODEL_FILE)
    st.success(f"Model trained successfully on {num_samples} samples and saved.")



# --------------------------------------------------------------------------------
# Load Labels
# --------------------------------------------------------------------------------
def load_label_map():

    if not os.path.exists(LABEL_MAP_FILE):
        st.error("label_map.json not found! Build dataset first.")
        return None

    with open(LABEL_MAP_FILE) as f:
        return json.load(f)["classes"]


# --------------------------------------------------------------------------------
# Real-Time Sign Detection via Webcam
# --------------------------------------------------------------------------------
def run_inference():

    if not os.path.exists(MODEL_FILE):
        st.error("Model not trained yet. Please train first.")
        return

    labels = load_label_map()
    if labels is None:
        return

    model = tf.keras.models.load_model(MODEL_FILE)

    cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    text_box = st.empty()

    seq = deque(maxlen=SEQUENCE_LENGTH)
    transcript = []

    with mp_holistic.Holistic(
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRACK_CONF
    ) as holistic:

        stop_btn = st.button("Stop Inference")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)

            mp_drawing.draw_landmarks(img_rgb, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(img_rgb, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            keypoints = extract_landmarks(results)
            seq.append(keypoints)

            if len(seq) == SEQUENCE_LENGTH:
                sample = np.expand_dims(seq, axis=0)
                pred = model.predict(sample, verbose=0)[0]
                idx = np.argmax(pred)

                # ---- FIX: Prevent repetition + Stop after first detection ----
                if pred[idx] > 0.7:
                    if len(transcript) == 0 or transcript[-1] != labels[idx]:
                        transcript.append(labels[idx])
                        break  # stop after detecting 1 sign

            st_frame.image(img_rgb, channels="RGB")
            text_box.write(" ".join(transcript))

            if stop_btn:
                break

    cap.release()
    st.success("Detection complete.")
    st.info(f"Recognized: {' '.join(transcript)}")


# --------------------------------------------------------------------------------
# Delete All Data
# --------------------------------------------------------------------------------
def delete_all_data():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    for file in [DATASET_FILE, LABEL_MAP_FILE, MODEL_FILE]:
        if os.path.exists(file):
            os.remove(file)

    st.warning("All stored data, models & datasets have been deleted!")


# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------
st.title("Sign Language → Real-Time Transcription System")

mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Upload Images", "Build Dataset", "Train Model", "Inference", "Delete All Data"]
)

# -------------------------
# Modes
# -------------------------
if mode == "Upload Images":
    label = st.text_input("Label for uploaded images:")
    uploaded_images = st.file_uploader("Upload sign images", type=["jpg", "png"], accept_multiple_files=True)
    if st.button("Save Images"):
        collect_from_upload(label, uploaded_images)

elif mode == "Build Dataset":
    if st.button("Build Dataset"):
        build_dataset()

elif mode == "Train Model":
    epochs = st.slider("Epochs", 5, 100, 20)
    if st.button("Train Model"):
        train_model(epochs)

elif mode == "Inference":
    run_inference()

elif mode == "Delete All Data":
    if st.button("DELETE EVERYTHING"):
        delete_all_data()

if st.button("Start Inference"):
        run_inference()
