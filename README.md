# Attendance_system

## 🔐 Smart Attendance System with Face Recognition & Safety Detection

This project is a real-time **face recognition-based attendance system** enhanced with **green safety coat detection** for safety compliance. Built using Python and DeepFace, it captures attendance only when a **registered face is recognized** *and* the **user is wearing a green safety coat (PPE)** — ensuring both **identity verification and workplace safety**.

### 🚀 Features

* ✅ Real-time face recognition using **DeepFace (SFace model)**
* 🟩 Safety compliance via **green coat detection**
* 📝 Automatic CSV-based attendance logging with timestamps
* 📷 Works with webcam feed
* 🔎 Detects unknown faces and rejects unsafe entries
* 🖼 Easy-to-manage folder structure for known faces

### 🧠 How It Works

1. Loads known faces from the `face_recognition/known_faces/` folder.
2. Captures webcam frames and compares them using DeepFace.
3. If a match is found and the user is wearing a green coat:

   * Marks attendance in `attendance_YYYY-MM-DD.csv`
   * Displays “✅ Name” on screen
4. If coat is missing: Shows warning “❌ Name, No coat”
5. If unknown: Shows “❌ User not found”

### 📁 Project Structure

```
attendance_system/
├── coat_detection/
│   └── detect_coat.py
├── combined/
│   └── smart_attendance.py
├── face_recognition/
│   └── known_faces/
│       ├── Alice/
│       ├── bob/
│       └── Harshraj/
├── app.py (basic version without coat detection)
```

### 📦 Requirements

* Python 3.7+
* OpenCV (`cv2`)
* NumPy
* Pandas
* DeepFace

Install all with:

```bash
pip install opencv-python numpy pandas deepface
```

