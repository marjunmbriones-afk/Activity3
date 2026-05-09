import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
import time
import os

st.title("🎥 Live Object Detection, Counting & Alerts")

SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)

enable_alert = st.checkbox("Enable Alerts", True)
target_class = st.selectbox(
    "Select Object for Alert",
    ["person", "cell phone", "bottle"]
)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

last_saved_time = 0
SAVE_COOLDOWN = 5

# WebRTC Configuration (Fixes connection issue)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        global last_saved_time

        img = frame.to_ndarray(format="bgr24")

        # YOLO Detection + Tracking
        results = model.track(
            img,
            persist=True,
            conf=0.5,
            verbose=False
        )

        annotated_frame = results[0].plot()

        # Detect objects
        boxes = results[0].boxes

        detected_objects = []

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                detected_objects.append(class_name)

        # Alerts + Save image
        if enable_alert and target_class in detected_objects:
            current_time = time.time()

            if current_time - last_saved_time > SAVE_COOLDOWN:
                filename = f"{SAVE_DIR}/{target_class}_{int(current_time)}.jpg"

                cv2.imwrite(filename, annotated_frame)

                print(f"Saved detection: {filename}")

                last_saved_time = current_time

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )

# Start Webcam Stream
webrtc_streamer(
    key="yolo-detection",
    video_processor_factory=YOLOProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)
