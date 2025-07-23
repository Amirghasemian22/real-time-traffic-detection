# 🚦 Real-Time Traffic Detection using YOLOv8n

Welcome to the official repository for Traffic Detection using YOLOv8, an open-source computer vision project that aims to detect and analyze road traffic congestion in real time, directly from camera footage — without relying on GPS or third-party services like Google Maps.

---

## 📌 Why This Project?

Most traffic analysis systems rely heavily on services like Google Maps, which can be manipulated or spoofed (e.g., using multiple low-speed smartphones). In fact, artist Simon Weckert famously created a fake traffic jam using 99 phones in a cart, tricking Google Maps into showing congestion where there was none.

💡 This project addresses this issue by offering an independent, vision-based traffic detection system using YOLOv8, Python, and OpenCV, making it ideal for smart city applications.

---

## 🎯 Project Goals

- ✅ Detect vehicles (cars, buses, motorcycles) from live camera streams
- ✅ Estimate traffic volume in real-time
- ✅ Provide a reliable alternative to GPS-based traffic systems
- ✅ Work with low-cost surveillance cameras
- ✅ Use custom-trained YOLOv8n model for high accuracy

---

## 🚧 How It Works

1. Video Feed Input  
   Accepts live streams or pre-recorded footage from traffic cameras.

2. Vehicle Detection  
   Runs a custom-trained YOLOv8 model (Ultralytics) to detect objects like cars, buses, and bikes.

3. Traffic Volume Estimation  
   Counts detected vehicles and classifies traffic as light, medium, or heavy.

4. Visualization  
   Annotates video frames with bounding boxes and congestion status in real time.

---

## 🧠 Technologies Used

| Component | Description |
|----------|-------------|
| 🐍 Python | Main programming language |
| 🎯 YOLOv8n | Object detection framework by Ultralytics |
| 🖼️ OpenCV | Frame processing and visualization |
| 🧪 labelImg | For custom dataset annotation |
| 📦 NumPy | Data analysis & preprocessing |
| 🧠 PyTorch | Backend for YOLOv8 training |

---

## 🗃️ Dataset

This project uses a custom dataset created with [labelImg] for:
- Cars 🚗

The model was trained using YOLOv8n training pipeline with data augmentation and evaluation metrics.

---

## 📸 Screenshots & Results

> *(Coming Soon: Sample outputs and detection previews)*

---

## 🏙️ Ideal Use Cases

- Smart city monitoring systems 🏙️  
- Traffic management dashboards 🚦  
- Low-resource local governments needing a GPS-independent solution 📉  
- Research in real-time object detection 🎓  

---
