# Vision-Based Syringe Manufacturing Inspection System

## 🚀 Overview
An automated, industrial-grade inspection system developed for the medical manufacturing sector. This project leverages **Computer Vision** and **Deep Learning** on a **Raspberry Pi** to ensure syringe needle bevel angles remain within a strict **10±2° tolerance range**, significantly reducing manual inspection errors.

## 🛠️ Key Technical Features
* **AI-Powered Localization:** Utilizes **YOLOv8** to identify and bound syringe needles in real-time.
* **Geometric Precision:** Implements **Canny Edge Detection** and **Probabilistic Hough Line Transforms** to calculate median bevel angles with high accuracy.
* **Quality Verification:** Employs **Template Matching** (`cv2.TM_CCOEFF_NORMED`) to verify surface consistency against "PASS" reference standards.
* **Industrial Monitoring GUI:** A custom-rendered OpenCV overlay designed for production-line operators, featuring a "REFRESH" manual trigger.

## 📁 Repository Structure
- `Raspi_Final_App.py`: The core execution script optimized for Raspberry Pi hardware.
- `models/`: Contains the trained YOLOv8 (`best.pt`) weights.
- `templates/`: Reference images for consistent pass/fail classification.
- `requirements.txt`: List of dependencies (OpenCV, NumPy, Ultralytics, PiCamera2).

## 💻 Tech Stack
- **Language:** Python
- **Libraries:** OpenCV, NumPy, Ultralytics, PiCamera2
- **Hardware:** Raspberry Pi 4, High-Resolution Camera Module
