import os
import tempfile
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="AI Muay Thai Analyzer", layout="wide")
st.title("🥊 AI Muay Thai Technique Analyzer")
st.write("อัปโหลดวิดีโอการฝึกซ้อมเพื่อตรวจจับจุดข้อต่อและวิเคราะห์ทักษะแม่ไม้มวยไทย")

# ---------------------------------------------------------------------------
# โมเดล (ลิงก์ทางการของ MediaPipe)
# ---------------------------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = os.path.join(tempfile.gettempdir(), "pose_landmarker_full.task")
MIN_MODEL_SIZE = 1_000_000


@st.cache_resource(show_spinner="กำลังเตรียมโมเดล AI ตรวจจับท่าทาง (เฉพาะครั้งแรก)...")
def load_model_buffer() -> bytes:
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < MIN_MODEL_SIZE:
        tmp_path = MODEL_PATH + ".part"
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        if os.path.getsize(tmp_path) < MIN_MODEL_SIZE:
            os.remove(tmp_path)
            raise RuntimeError("ไฟล์โมเดลที่ดาวน์โหลดมาไม่สมบูรณ์")
        os.replace(tmp_path, MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:  # อ่านเข้า memory เลี่ยงปัญหา path ภาษาไทย
        return f.read()


try:
    model_buffer = load_model_buffer()
except Exception as e:
    st.error(f"ดาวน์โหลดโมเดลไม่สำเร็จ: {e}\nกรุณาตรวจสอบอินเทอร์เน็ตแล้วรีเฟรชหน้าเว็บ")
    st.stop()

# ---------------------------------------------------------------------------
# ค่าคงที่
# ---------------------------------------------------------------------------
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
]
VISIBILITY_MIN = 0.5

# joints = (จุดแรก, จุดยอดมุม, จุดที่สาม) เป็น index ของ MediaPipe Pose
# mode "less": มุมน้อยกว่าเกณฑ์ = ดี | mode "more": มุมมากกว่าเกณฑ์ = ดี
SKILLS = {
    "1. การเก็บศอก (Elbow Flare)": {
        "joints": {"right": (24, 12, 14), "left": (23, 11, 13)},  # สะโพก-ไหล่-ศอก
        "default": 45, "range": (10, 90), "mode": "less",
        "rule": "มุมควรน้อยกว่าเกณฑ์ (ศอกไม่กาง)",
        "warn": "⚠️ ศอกเปิดมากเกินไป (Elbow Flare)",
        "good": "✅ Good Form: เก็บศอกชิดลำตัวได้ดี",
    },
    "2. การเหยียดหมัดตรง (Jab / Cross)": {
        "joints": {"right": (12, 14, 16), "left": (11, 13, 15)},  # ไหล่-ศอก-ข้อมือ
        "default": 160, "range": (100, 180), "mode": "more",
        "rule": "มุมควรมากกว่าเกณฑ์ (เหยียดแขนสุด)",
        "warn": "⚠️ หมัดยังงออยู่ ควรเหยียดแขนให้สุด",
        "good": "✅ Good Form: ออกหมัดเหยียดสุดแขน",
    },
    "3. การแทงเข่า (Knee Strike)": {
        "joints": {"right": (12, 24, 26), "left": (11, 23, 25)},  # ไหล่-สะโพก-เข่า
        "default": 75, "range": (20, 130), "mode": "less",
        "rule": "มุมควรน้อยกว่าเกณฑ์ (ยกเข่าสูง)",
        "warn": "⚠️ ยกเข่ายังต่ำอยู่ ควรดึงเข่าให้สูงขึ้น",
        "good": "✅ Good Form: ยกเข่าได้ระดับสูง",
    },
    "4. การเตะ / ถีบ (Kick / Teep)": {
        "joints": {"right": (24, 26, 28), "left": (23, 25, 27)},  # สะโพก-เข่า-ข้อเท้า
        "default": 155, "range": (90, 180), "mode": "more",
        "rule": "มุมควรมากกว่าเกณฑ์ (เหยียดขาตรง)",
        "warn": "⚠️ ขายังงอขณะเตะ/ถีบ ควรเหยียดขาให้ตรง",
        "good": "✅ Good Form: เหยียดขาเตะ/ถีบได้ตรง",
    },
}


def calculate_angle(a, b, c) -> float:
    """มุมที่จุด b (องศา 0-180) ส่งพิกัดเป็นพิกเซลเพื่อไม่ให้มุมเพี้ยนตามอัตราส่วนภาพ"""
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def is_visible(lm) -> bool:
    vis = lm.visibility if lm.visibility is not None else 1.0
    return vis >= VISIBILITY_MIN


# ---------------------------------------------------------------------------
# แถบตั้งค่า
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ ตั้งค่าการวิเคราะห์")
skill = st.sidebar.selectbox("เลือกทักษะมวยไทยที่ต้องการวิเคราะห์", list(SKILLS.keys()))
cfg = SKILLS[skill]

side = st.sidebar.radio("ข้างที่ต้องการวิเคราะห์", ["ขวา (Right)", "ซ้าย (Left)"])
side_key = "right" if side.startswith("ขวา") else "left"
i1, i2, i3 = cfg["joints"][side_key]

lo, hi = cfg["range"]
# key ผูกกับทักษะ เพื่อให้ slider รีเซ็ตเป็นค่าเริ่มต้นเมื่อเปลี่ยนทักษะ
threshold = st.sidebar.slider(
    f"เกณฑ์มุมมาตรฐาน (องศา) - {cfg['rule']}",
    lo, hi, cfg["default"], key=f"thr_{skill}",
)
frame_skip = st.sidebar.slider("ประมวลผลทุกกี่เฟรม (ยิ่งมาก ยิ่งเร็ว)", 1, 10, 2)
max_width = st.sidebar.slider("ความกว้างสูงสุดของวิดีโอ (px)", 320, 1280, 640)


def passes(angle: float) -> bool:
    return angle <= threshold if cfg["mode"] == "less" else angle >= threshold


# ---------------------------------------------------------------------------
# อัปโหลดและวิเคราะห์
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("เลือกไฟล์วิดีโอ (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.getbuffer())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    angle_history, time_history = [], []

    try:
        if not cap.isOpened():
            st.error("เปิดไฟล์วิดีโอไม่ได้ กรุณาลองไฟล์อื่น (แนะนำ MP4 แบบ H.264)")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        col1, col2 = st.columns(2)
        with col1:
            st_frame = st.empty()
        with col2:
            st.subheader(f"📊 ผลวิเคราะห์: {skill}")
            st_status = st.empty()
            st_angle_text = st.empty()
        progress = st.progress(0.0)

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=model_buffer),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        frame_idx = -1
        last_ts = -1

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                if not cap.grab():  # grab เร็วกว่า read สำหรับเฟรมที่ข้าม
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue
                ret, frame = cap.retrieve()
                if not ret:
                    break

                h, w = frame.shape[:2]
                if w > max_width:
                    frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
                    h, w = frame.shape[:2]

                timestamp_ms = int(frame_idx / fps * 1000)
                if timestamp_ms <= last_ts:  # ต้องเพิ่มขึ้นเสมอ
                    timestamp_ms = last_ts + 1
                last_ts = timestamp_ms

                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.pose_landmarks:
                    lm = results.pose_landmarks[0]  # คนแรกในภาพ

                    # โครงกระดูก
                    for s, e in POSE_CONNECTIONS:
                        if is_visible(lm[s]) and is_visible(lm[e]):
                            cv2.line(
                                frame,
                                (int(lm[s].x * w), int(lm[s].y * h)),
                                (int(lm[e].x * w), int(lm[e].y * h)),
                                (0, 0, 255), 2,
                            )
                    for i in range(11, 33):
                        if is_visible(lm[i]):
                            cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)), 4, (0, 255, 0), -1)

                    # มุมของทักษะที่เลือก
                    if all(is_visible(lm[i]) for i in (i1, i2, i3)):
                        p1 = (lm[i1].x * w, lm[i1].y * h)
                        p2 = (lm[i2].x * w, lm[i2].y * h)  # จุดยอดมุม
                        p3 = (lm[i3].x * w, lm[i3].y * h)

                        angle = calculate_angle(p1, p2, p3)
                        angle_history.append(angle)
                        time_history.append(round(frame_idx / fps, 2))

                        good = passes(angle)
                        cv2.putText(
                            frame, f"{int(angle)} deg",
                            (int(p2[0]) + 10, int(p2[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if good else (0, 165, 255), 2, cv2.LINE_AA,
                        )
                        st_angle_text.metric(f"มุมองศาปัจจุบัน ({side})", f"{int(angle)}°")
                        if good:
                            st_status.success(cfg["good"])
                        else:
                            st_status.warning(cfg["warn"])
                    else:
                        st_status.info(f"มองเห็นข้อต่อด้าน{side}ไม่ชัด ลองปรับมุมกล้องหรือเลือกอีกข้าง")
                else:
                    st_status.info("ไม่พบคนในภาพ")

                st_frame.image(frame, channels="BGR")
                if total_frames > 0:
                    progress.progress(min(frame_idx / total_frames, 1.0))
    finally:
        cap.release()
        try:
            os.unlink(video_path)
        except OSError:
            pass

    progress.progress(1.0)
    st.success("วิเคราะห์วิดีโอเสร็จสิ้น!")

    # -----------------------------------------------------------------------
    # สรุปผลทั้งคลิป
    # -----------------------------------------------------------------------
    if angle_history:
        st.divider()
        st.subheader("📈 สรุปผลทั้งคลิป")

        arr = np.array(angle_history)
        pass_rate = np.mean([passes(a) for a in arr]) * 100
        best = arr.min() if cfg["mode"] == "less" else arr.max()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("เฟรมที่วิเคราะห์ได้", f"{len(arr)}")
        m2.metric("มุมเฉลี่ย", f"{int(arr.mean())}°")
        m3.metric("มุมต่ำสุด", f"{int(arr.min())}°")
        m4.metric("มุมสูงสุด", f"{int(arr.max())}°")
        m5.metric("เฟรมที่ผ่านเกณฑ์", f"{int(pass_rate)}%")

        # ท่ามวยไทยแต่ละครั้งมีจังหวะ "จุดสูงสุดของท่า" จึงดูค่าที่ดีที่สุดในคลิปด้วย
        if passes(best):
            st.success(f"จุดที่ทำท่าได้ดีที่สุดในคลิป {int(best)}° ผ่านเกณฑ์ ({threshold}°)")
        else:
            st.warning(f"จุดที่ทำท่าได้ดีที่สุดในคลิป {int(best)}° ยังไม่ผ่านเกณฑ์ ({threshold}°)")

        df = pd.DataFrame({
            "เวลา (วินาที)": time_history,
            "มุมที่วัดได้ (องศา)": angle_history,
            "เกณฑ์มาตรฐาน": [threshold] * len(angle_history),
        }).set_index("เวลา (วินาที)")
        st.line_chart(df)
    else:
        st.warning("ไม่สามารถวัดมุมได้ในคลิปนี้ ลองถ่ายให้เห็นร่างกายส่วนที่วัดชัดขึ้น หรือเลือกอีกข้าง")