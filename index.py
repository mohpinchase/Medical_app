# Streamlit AI Health Anomaly Detection System (Large-Scale Version)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta

# ---------------------------
# SECTION 1: Data Simulation
# ---------------------------
def simulate_health_data(num_users=5, minutes=500):
    user_ids = [f'user_{i+1}' for i in range(num_users)]
    start_time = datetime.now()
    data = []

    for user in user_ids:
        timestamp = start_time
        for _ in range(minutes):
            data.append({
                'user_id': user,
                'timestamp': timestamp,
                'heart_rate': np.random.randint(60, 100),
                'blood_oxygen': np.random.randint(90, 100),
                'temperature': np.random.normal(36.5, 0.5),
                'respiration_rate': np.random.randint(12, 20),
                'activity_level': np.random.choice(['low', 'moderate', 'high'])
            })
            timestamp += timedelta(minutes=1)

    df = pd.DataFrame(data)
    return df

# -----------------------------
# SECTION 2: Preprocessing
# -----------------------------
def preprocess_data(df):
    df = df.copy()
    df['activity_level'] = df['activity_level'].map({'low': 0, 'moderate': 1, 'high': 2})
    features = ['heart_rate', 'blood_oxygen', 'temperature', 'respiration_rate', 'activity_level']
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    return df, df_scaled, features

# -----------------------------
# SECTION 3: Anomaly Detection
# -----------------------------
def detect_anomalies(df_scaled, contamination=0.05):
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(df_scaled)
    return preds, model

# -----------------------------
# SECTION 4: Evaluation
# -----------------------------
def evaluate_model(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    return pd.DataFrame(report).transpose()

# -----------------------------
# SECTION 5: Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Health Anomaly Detection", layout="wide")
st.title("🧠 AI-Powered Health Anomaly Detection")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")
num_users = st.sidebar.slider("Number of Users", 1, 10, 3)
num_minutes = st.sidebar.slider("Minutes of Data per User", 100, 1000, 300)
contamination = st.sidebar.slider("Anomaly Rate (contamination)", 0.01, 0.2, 0.05)

# Data simulation and display
st.header("📊 Simulated Health Data")
df = simulate_health_data(num_users, num_minutes)
st.dataframe(df.head(100))

# Preprocessing and anomaly detection
st.header("🧪 Anomaly Detection")
df_processed, df_scaled, feature_cols = preprocess_data(df)
preds, model = detect_anomalies(df_scaled, contamination)
df_processed['anomaly'] = ['Anomaly' if x == -1 else 'Normal' for x in preds]
st.dataframe(df_processed[['user_id', 'timestamp', 'heart_rate', 'blood_oxygen', 'temperature', 'anomaly']].head(100))

# Evaluation setup
st.header("📈 Model Evaluation")
df_processed['anomaly_label'] = df_processed['anomaly'].map({'Normal': 0, 'Anomaly': 1})
X = df_scaled
y = df_processed['anomaly_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.fit(X_train, y_train).predict(X_test)
y_pred = [1 if x == -1 else 0 for x in y_pred]

# Display evaluation report
eval_df = evaluate_model(y_test, y_pred)
st.dataframe(eval_df)

# Visualize anomalies
st.header("📉 Anomaly Visualization")
fig, ax = plt.subplots()
anomaly_points = df_processed[df_processed['anomaly'] == 'Anomaly']
sns.lineplot(data=df_processed, x='timestamp', y='heart_rate', hue='user_id', ax=ax)
plt.scatter(anomaly_points['timestamp'], anomaly_points['heart_rate'], color='red', label='Anomaly')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)

# Optional save
st.sidebar.markdown("---")
if st.sidebar.button("💾 Export Anomaly Report"):
    df_processed.to_csv("anomaly_report.csv", index=False)
    st.success("Report saved as anomaly_report.csv")
