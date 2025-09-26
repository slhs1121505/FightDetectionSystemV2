# FightDetectionSystemV2


This groundbreaking project seamlessly fuses **computer vision**, **speech recognition**, and **real-time alert systems** to detect aggressive human interactions with unprecedented precision and instant notification.  
It **debuted at the Shilin Commercial High School Data Processing Department Project Competition, where it achieved the coveted First Place, earning accolades for its visionary innovation and technical mastery** 🏆.

---

## 🚀 Breakthrough Capabilities

- **Real-Time Human Pose Estimation** – Leveraging the cutting-edge [YOLOv8n](https://docs.ultralytics.com/tasks/pose/), the system continuously analyzes live camera feeds with unparalleled precision.  
- **Aggression Scoring & Contact Tracking** – Quantifies short-term physical interactions and computes aggression metrics to minimize false positives, ensuring highly reliable detection.  
- **Integrated Voice Command Detection** – Captures speech via microphone using advanced `SpeechRecognition` ; strategically defined key phrases instantly trigger alerts.  
- **Immediate Local TTS Notifications** – Employs `pyttsx3` to deliver real-time voice alerts on-site for instantaneous response.  
- **Discord Webhook Integration** – Seamlessly transmits snapshots, aggression scores, and metadata to a dedicated Discord channel for comprehensive remote monitoring.  
- **Evidence Archiving** – Systematically stores screenshots and event logs, supporting auditing, verification, and advanced analytics.  
- **Extensible Flask API** – Provides robust endpoints for health checks, log retrieval, and future integrations, ensuring scalability and professional-grade system interoperability.

---

## 🛠 World-Class Tech Stack

- **Core Model**: YOLOv8n (Ultralytics)  
- **Computer Vision**: OpenCV  
- **Speech-to-Text**: SpeechRecognition  
- **Text-to-Speech**: pyttsx3  
- **Backend & API**: Flask (optional)  
- **Notifications**: Discord Webhook  

---

## 📊 System Architecture

```mermaid
flowchart TD
    A[Camera Input] --> B[OpenCV Preprocessing]
    B --> C[YOLOv8 Pose Estimation]
    C --> D[Contact Detection & Tracking]
    D --> E[Aggression Scoring]
    E -->|Score > Threshold| F[Alert Trigger]
        A2[Microphone Input] --> B2[SpeechRecognition]
        B2 -->|Keyword Detected| F

    F --> G[Save Evidence]
    F --> H[Discord Webhook Notification]
    F --> I[Local TTS Warning]
    H --> J[Remote Monitoring Channel]
```
---

## 🏆 Champion-Level Accomplishments

- **1st Place – Shilin Commercial High School Data Processing Department Project Competition**  
- Successfully demonstrated real-time aggression detection and alerting in a live competition setting.  
- Recognized for its practical application, multi-modal design, and technical robustness.  

---

## 📚 Authoritative References

- [A human pose estimation network based on YOLOv8 framework with efficient multi-scale receptive field and expanded feature pyramid network – Scientific Reports](https://www.nature.com/articles/s41598-025-00259-0)  
- [Automated violence monitoring system for real-time fist fight detection using deep learning-based temporal action localization – Scientific Reports](https://www.nature.com/articles/s41598-025-12531-4)  
- [Pose Estimation – Ultralytics YOLO Docs](https://docs.ultralytics.com/tasks/pose/)  
- [How to receive webhooks in Python with Flask or Django – LogRocket Blog](https://blog.logrocket.com/receive-webhooks-python-flask-or-django/)  
- [How to Set Up Python Webhooks: 3 Simple Steps – Hevo Data](https://hevodata.com/learn/python-webhook/)  
- [STEAM 教育學習網：OpenCV 教學索引](https://steam.oxxostudio.tw/category/python/ai/opencv-index.html)  
- [You Only Look Once: Unified, Real-Time Object Detection (CVPR 2016)](https://www.cvfoundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)  
