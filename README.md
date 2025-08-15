# AI-Powered Health Monitoring System

## Overview
This project is an AI-powered system that monitors users' health in real-time using data from wearable devices (or simulated data). It analyzes health metrics such as heart rate, blood oxygen levels, and activity levels to detect anomalies and provide insights.

## Features
- Real-time health data simulation
- Anomaly detection using machine learning
- Data visualization and reporting
- Modular, extensible codebase

## Project Structure
```
Medical_app/
│
├── app/
│   ├── index.py             # Main Streamlit app
│   ├── data_simulation.py   # Data simulation functions
│   ├── preprocessing.py     # Data preprocessing functions
│   ├── anomaly.py           # Anomaly detection and evaluation
│   └── __init__.py          # Package marker
│
├── requirements.txt         # Python dependencies
├── README.md                # Project overview and instructions
```

## Getting Started
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app:**
   ```bash
   streamlit run index.py
   ```

## Usage
- Use the sidebar to configure simulation parameters.
- View simulated data, detected anomalies, and evaluation metrics.
- Export anomaly reports as CSV.

## Future Enhancements
- Integrate with real wearable device APIs
- Add personalized health recommendations
- Deploy to cloud or as a mobile app

---
