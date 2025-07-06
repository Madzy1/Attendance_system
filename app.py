import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# === Set paths ===
KNOWN_FACES_DIR = "known_faces"
DATE_STR = datetime.now().strftime("%Y-%m-%d")
ATTENDANCE_FILE = f"attendance_{DATE_STR}.csv"

# === Initialize attendance list safely ===
if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
    attendance_df = pd.read_csv(ATTENDANCE_FILE)
else:
    attendance_df = pd.DataFrame(columns=["Name", "Time"])

# === Load known faces from folders ===
print("[INFO] Loading known faces...")
known_faces = {}  # key = person name, value = list of images

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
    if os.path.isdir(person_folder):
        images = []
        for file in os.listdir(person_folder):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(person_folder, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
        if images:
            known_faces[person_name.lower()] = images

print(f"[INFO] Loaded faces for {len(known_faces)} people.")

# === Mark attendance ===
def mark_attendance(name):
    global attendance_df
    if name not in attendance_df["Name"].values:
        time_now = datetime.now().strftime("%H:%M:%S")
        new_entry = pd.DataFrame([[name, time_now]], columns=["Name", "Time"])
        attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
        attendance_df.to_csv(ATTENDANCE_FILE, index=False)

# === Start webcam ===
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)

frame_skip = 90  # Check every Nth frame
frame_count = 0
display_name = "Detecting..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip == 0:
        matched = False
        display_name = "Detecting..."

        for name, images in known_faces.items():
            for known_img in images:
                try:
                    result = DeepFace.verify(frame, known_img, model_name='SFace', enforce_detection=False)
                    if result['verified']:
                        display_name = f"Detected: {name}"
                        mark_attendance(name)
                        matched = True
                        break
                except Exception as e:
                    print(f"[!] Error verifying face for {name}: {e}")
            if matched:
                break

        if not matched:
            display_name = "User not found"

    # === Display status on webcam ===
    color = (0, 255, 0) if "Detected" in display_name else ((0, 0, 255) if display_name == "User not found" else (0, 255, 255))
    cv2.putText(frame, display_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Recognition Attendance", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
