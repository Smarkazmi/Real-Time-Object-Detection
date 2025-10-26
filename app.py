import streamlit as st
import cv2
from ultralytics import YOLO
import torch
import tempfile
import time

# 🧠 Load YOLO11n model (lightweight and fast)
model = YOLO("yolo11n.pt")

st.set_page_config(page_title="Real-Time Object Detection", layout="wide")
st.title("🎯 Real-Time Object Detection (using YOLO11n)")
st.markdown("Detect objects live with clean confidence scores!")
st.markdown("Created by: **Syed Muhammad Ali Raza Kazmi**")

st.markdown(
    """
    <a href="https://instagram.com/azmira_smarku" target="_blank">📸 Instagram</a> |
    <a href="https://www.linkedin.com/in/syed-muhammad-ali-raza-kazmi-a8b372308/" target="_blank">💼 LinkedIn</a> |
    <a href="https://youtube.com/@smarkazmii" target="_blank">▶️ YouTube</a>
    """,
    unsafe_allow_html=True
)

# Sidebar options
st.sidebar.header("⚙️ Options")
mode = st.sidebar.radio("Choose Mode:", ["🎥 Webcam", "🖼️ Demo Image/Video"])

# Use GPU if available
if torch.cuda.is_available():
    model.to("cuda")
    st.sidebar.success("✅ GPU acceleration enabled")
else:
    st.sidebar.warning("⚠️ Running on CPU (may be slower)")

# --------------------------------------------
# 🎥 WEBCAM MODE
# --------------------------------------------
if mode == "🎥 Webcam":
    run = st.checkbox('▶️ Start Detection')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    if run:
        with st.spinner("Loading camera and model..."):
            time.sleep(1)
        while run:
            ret, frame = camera.read()
            if not ret:
                st.warning("⚠️ Could not access webcam.")
                break

            results = model.predict(source=frame, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame)

    camera.release()
    st.write("🛑 Detection stopped.")

# --------------------------------------------
# 🖼️ DEMO IMAGE / VIDEO MODE
# --------------------------------------------
else:
    st.subheader("🧪 Try a Demo File")

    demo_type = st.radio("Choose a demo type:", ["Image", "Video"])
    uploaded_file = st.file_uploader("Upload your file", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            demo_path = tmp_file.name

        with st.spinner("Running detection..."):
            results = model.predict(source=demo_path, conf=0.25, stream=False, verbose=False)
            time.sleep(1)

        if demo_type == "Image":
            result_img = results[0].plot()
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)
        else:
            st.video(demo_path)

        st.success("✅ Detection complete!")
