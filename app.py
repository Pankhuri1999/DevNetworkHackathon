
import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import urllib.request
import bz2
import dlib
from imutils import face_utils
from scipy.spatial import distance
import matplotlib.pyplot as plt
from streamlit_media_recorder import media_recorder
import datetime

st.set_page_config(page_title="Lip Movement Comparison", layout="wide")

# Model download/extract
def download_model():
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_path = "shape_predictor_68_face_landmarks.dat.bz2"
    extracted_path = "shape_predictor_68_face_landmarks.dat"

    if not os.path.exists(extracted_path):
        if not os.path.exists(compressed_path):
            urllib.request.urlretrieve(url, compressed_path)
        with bz2.BZ2File(compressed_path, "rb") as f_in:
            with open(extracted_path, "wb") as f_out:
                f_out.write(f_in.read())
    return extracted_path

model_path = download_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)
LIP_IDX = list(range(48, 61))
LIP_POINTS = list(range(48, 61))

def extract_lip_movements(video_path):
    cap = cv2.VideoCapture(video_path)
    lip_movements = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            lips = shape[LIP_IDX]
            lips_centered = lips - lips.mean(axis=0)
            lip_movements.append(lips_centered.flatten())
    cap.release()
    return np.array(lip_movements)

def compare_lip_movements(movements1, movements2):
    min_len = min(len(movements1), len(movements2))
    movements1 = movements1[:min_len]
    movements2 = movements2[:min_len]
    distances = [distance.euclidean(a, b) for a, b in zip(movements1, movements2)]
    return np.mean(distances), distances

def extract_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if faces:
        shape = predictor(gray, faces[0])
        coords = np.array([[p.x, p.y] for p in shape.parts()])
        return coords
    return None

def get_lip_distance(landmarks):
    top_lip = np.mean([landmarks[i] for i in [50, 51, 52]], axis=0)
    bottom_lip = np.mean([landmarks[i] for i in [56, 57, 58]], axis=0)
    return np.linalg.norm(top_lip - bottom_lip)

def annotate_frame(frame, landmarks, label, value):
    if landmarks is not None:
        for (x, y) in landmarks[LIP_POINTS]:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    cv2.putText(frame, f"{label} Lip Open: {value:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame

st.title("👄 Lip Movement Comparison App")
st.markdown("Upload a reference video and compare it with a new recording or uploaded video.")

# Sidebar Uploads
st.sidebar.header("📁 Step 1: Upload Reference Video")
ref_video = st.sidebar.file_uploader("Upload Reference Video", type=["mp4", "avi"])

cmp_method = st.sidebar.radio("Step 2: Provide Comparison Video", ["Upload Video", "Record with Webcam"])
cmp_video = None

if cmp_method == "Upload Video":
    cmp_video = st.sidebar.file_uploader("Upload Comparison Video", type=["mp4", "avi"])
elif cmp_method == "Record with Webcam":
    st.sidebar.info("🎥 Use your webcam to record.")
    video_bytes = media_recorder("video")
    if video_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as vid_file:
            vid_file.write(video_bytes)
            cmp_video = vid_file

if ref_video and cmp_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as ref_temp:
        ref_temp.write(ref_video.read())
        ref_path = ref_temp.name

    cmp_path = cmp_video.name if hasattr(cmp_video, "name") else None

    st.video(ref_path, format="video/mp4")
    st.video(cmp_path, format="video/mp4")

    lips_actual = extract_lip_movements(ref_path)
    lips_other = extract_lip_movements(cmp_path)
    similarity, distances = compare_lip_movements(lips_actual, lips_other)

    st.subheader(f"Similarity Score: {similarity:.2f} (lower = more similar)")

    st.subheader("📊 Distance per Frame")
    fig, ax = plt.subplots()
    ax.plot(distances)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Euclidean Distance")
    ax.set_title("Frame-wise Lip Distance")
    st.pyplot(fig)

    cap1 = cv2.VideoCapture(ref_path)
    cap2 = cv2.VideoCapture(cmp_path)
    width, height = 640, 480
    out_path = "output_combined.avi"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (width * 2, height))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame1 = cv2.resize(frame1, (width, height))
        frame2 = cv2.resize(frame2, (width, height))
        lm1 = extract_landmarks(frame1)
        lm2 = extract_landmarks(frame2)
        d1 = get_lip_distance(lm1) if lm1 is not None else 0
        d2 = get_lip_distance(lm2) if lm2 is not None else 0
        frame1 = annotate_frame(frame1, lm1, "Video 1", d1)
        frame2 = annotate_frame(frame2, lm2, "Video 2", d2)
        combined = np.hstack((frame1, frame2))
        out.write(combined)

    cap1.release()
    cap2.release()
    out.release()

    # Download button
    with open(out_path, "rb") as f:
        st.download_button("📥 Download Combined Output Video", data=f.read(),
                           file_name="output_combined.avi", mime="video/avi")
else:
    st.info("Upload both reference and comparison videos to begin analysis.")
