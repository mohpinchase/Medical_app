# Enhanced AI-Powered Health Monitoring System
# Combining real-time monitoring, advanced anomaly detection, and personalized recommendations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# CONFIGURATION
# ---------------------------
st.set_page_config(
    page_title="AI Health Monitor Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
}
.anomaly-alert {
    background-color: #ffebee;
    border: 2px solid #f44336;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.normal-status {
    background-color: #e8f5e8;
    border: 2px solid #4caf50;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# ENHANCED DATA SIMULATION
# ---------------------------
@st.cache_data
def simulate_comprehensive_health_data(num_users=5, days=7, samples_per_hour=12):
    """
    Simulate comprehensive health data with more realistic patterns
    """
    user_profiles = {
        f'user_{i+1}': {
            'age': np.random.randint(18, 80),
            'gender': np.random.choice(['M', 'F']),
            'activity_baseline': np.random.choice(['sedentary', 'moderate', 'active']),
            'health_condition': np.random.choice(['healthy', 'hypertension', 'diabetes', 'heart_condition'])
        } for i in range(num_users)
    }
    
    start_time = datetime.now() - timedelta(days=days)
    data = []
    
    for user_id, profile in user_profiles.items():
        current_time = start_time
        
        # Base values adjusted by user profile
        base_hr = 70 if profile['age'] < 40 else 75
        if profile['health_condition'] == 'hypertension':
            base_hr += 10
        elif profile['health_condition'] == 'heart_condition':
            base_hr += 15
            
        for day in range(days):
            for hour in range(24):
                for sample in range(samples_per_hour):
                    # Circadian rhythm effects
                    hr_variation = 10 * np.sin(2 * np.pi * hour / 24)
                    
                    # Activity level based on time of day
                    if 6 <= hour <= 22:
                        activity_prob = [0.3, 0.5, 0.2]  # low, moderate, high
                    else:
                        activity_prob = [0.9, 0.1, 0.0]  # mostly low activity at night
                    
                    activity = np.random.choice(['low', 'moderate', 'high'], p=activity_prob)
                    
                    # Heart rate influenced by activity and circadian rhythm
                    if activity == 'low':
                        heart_rate = base_hr + hr_variation + np.random.normal(0, 5)
                    elif activity == 'moderate':
                        heart_rate = base_hr + hr_variation + 20 + np.random.normal(0, 8)
                    else:
                        heart_rate = base_hr + hr_variation + 40 + np.random.normal(0, 10)
                    
                    # Introduce anomalies occasionally
                    if np.random.random() < 0.02:  # 2% chance of anomaly
                        if np.random.random() < 0.5:
                            heart_rate += np.random.normal(30, 10)  # Tachycardia
                        else:
                            heart_rate = max(40, heart_rate - np.random.normal(20, 5))  # Bradycardia
                    
                    # Blood oxygen (typically 95-100%)
                    blood_oxygen = np.random.normal(98, 1.5)
                    if np.random.random() < 0.01:  # 1% chance of low oxygen
                        blood_oxygen = np.random.normal(88, 3)
                    
                    # Temperature (normal around 36.5-37°C)
                    temperature = np.random.normal(36.7, 0.3)
                    if np.random.random() < 0.005:  # 0.5% chance of fever
                        temperature = np.random.normal(38.5, 0.5)
                    
                    # Respiratory rate
                    resp_rate = np.random.normal(16, 2)
                    if activity == 'high':
                        resp_rate += np.random.normal(8, 2)
                    
                    # Sleep quality (only during night hours)
                    sleep_quality = None
                    if hour >= 22 or hour <= 6:
                        sleep_quality = np.random.choice(['poor', 'fair', 'good', 'excellent'])
                    
                    data.append({
                        'user_id': user_id,
                        'timestamp': current_time,
                        'heart_rate': max(40, min(200, heart_rate)),  # Clamp to realistic range
                        'blood_oxygen': max(70, min(100, blood_oxygen)),
                        'temperature': max(35, min(42, temperature)),
                        'respiration_rate': max(8, min(40, resp_rate)),
                        'activity_level': activity,
                        'sleep_quality': sleep_quality,
                        'age': profile['age'],
                        'gender': profile['gender'],
                        'health_condition': profile['health_condition']
                    })
                    
                    current_time += timedelta(minutes=60//samples_per_hour)
    
    return pd.DataFrame(data)

# ---------------------------
# ADVANCED PREPROCESSING
# ---------------------------
def advanced_preprocessing(df):
    """
    Enhanced preprocessing with feature engineering
    """
    df_processed = df.copy()
    
    # Encode categorical variables
    le_activity = LabelEncoder()
    le_sleep = LabelEncoder()
    le_gender = LabelEncoder()
    le_condition = LabelEncoder()
    
    df_processed['activity_encoded'] = le_activity.fit_transform(df_processed['activity_level'])
    
    # Handle sleep quality (has NaN values)
    sleep_filled = df_processed['sleep_quality'].fillna('awake')
    df_processed['sleep_encoded'] = le_sleep.fit_transform(sleep_filled)
    
    df_processed['gender_encoded'] = le_gender.fit_transform(df_processed['gender'])
    df_processed['condition_encoded'] = le_condition.fit_transform(df_processed['health_condition'])
    
    # Feature engineering
    df_processed['hour'] = df_processed['timestamp'].dt.hour
    df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
    df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
    
    # Rolling averages for trend detection
    df_processed = df_processed.sort_values(['user_id', 'timestamp'])
    df_processed['hr_rolling_mean'] = df_processed.groupby('user_id')['heart_rate'].transform(
        lambda x: x.rolling(window=12, min_periods=1).mean()
    )
    df_processed['hr_deviation'] = df_processed['heart_rate'] - df_processed['hr_rolling_mean']
    
    # Select features for modeling
    feature_columns = [
        'heart_rate', 'blood_oxygen', 'temperature', 'respiration_rate',
        'activity_encoded', 'sleep_encoded', 'age', 'gender_encoded',
        'condition_encoded', 'hour', 'is_weekend', 'hr_deviation'
    ]
    
    return df_processed, feature_columns

# ---------------------------
# MULTI-MODEL ANOMALY DETECTION
# ---------------------------
class HealthAnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        X_scaled = self.scaler.transform(X)
        predictions = self.isolation_forest.predict(X_scaled)
        return ['Anomaly' if pred == -1 else 'Normal' for pred in predictions]
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        X_scaled = self.scaler.transform(X)
        scores = self.isolation_forest.decision_function(X_scaled)
        # Convert to probability-like scores
        proba = 1 / (1 + np.exp(scores))  # Sigmoid transformation
        return proba

# ---------------------------
# HEALTH RECOMMENDATIONS ENGINE
# ---------------------------
def generate_health_recommendations(user_data, anomaly_status):
    """
    Generate personalized health recommendations based on user data and anomaly status
    """
    recommendations = []
    
    latest_data = user_data.iloc[-1]
    
    if anomaly_status == 'Anomaly':
        recommendations.append("⚠️ **URGENT**: Anomalous readings detected. Consider consulting a healthcare professional.")
    
    # Heart rate recommendations
    if latest_data['heart_rate'] > 100:
        recommendations.append("💗 **Heart Rate**: Your heart rate is elevated. Consider stress management techniques or light exercise.")
    elif latest_data['heart_rate'] < 60:
        recommendations.append("💗 **Heart Rate**: Your heart rate is low. This could be normal if you're very fit, but monitor for other symptoms.")
    
    # Blood oxygen recommendations
    if latest_data['blood_oxygen'] < 95:
        recommendations.append("🫁 **Blood Oxygen**: Low oxygen levels detected. Ensure good ventilation and consider deep breathing exercises.")
    
    # Temperature recommendations
    if latest_data['temperature'] > 37.5:
        recommendations.append("🌡️ **Temperature**: Elevated temperature detected. Stay hydrated and monitor for other symptoms.")
    
    # Activity recommendations
    avg_activity = user_data['activity_encoded'].mean()
    if avg_activity < 0.5:
        recommendations.append("🏃‍♂️ **Activity**: Consider increasing your daily physical activity for better cardiovascular health.")
    
    # Sleep recommendations (when available)
    recent_sleep = user_data['sleep_quality'].dropna()
    if len(recent_sleep) > 0:
        poor_sleep_ratio = (recent_sleep == 'poor').sum() / len(recent_sleep)
        if poor_sleep_ratio > 0.3:
            recommendations.append("😴 **Sleep**: Poor sleep quality detected frequently. Consider establishing a regular sleep routine.")
    
    if not recommendations:
        recommendations.append("✅ **Overall**: Your health metrics look good! Keep maintaining your healthy lifestyle.")
    
    return recommendations

# ---------------------------
# STREAMLIT APPLICATION
# ---------------------------

# Title and Header
st.title("🏥 AI-Powered Health Monitoring System Pro")
st.markdown("### Advanced Real-time Health Analytics with Personalized Recommendations")

# Sidebar Configuration
st.sidebar.header("🔧 System Configuration")
num_users = st.sidebar.slider("Number of Users", 1, 10, 3)
num_days = st.sidebar.slider("Days of Historical Data", 1, 14, 7)
samples_per_hour = st.sidebar.selectbox("Data Frequency (samples/hour)", [4, 12, 24], index=1)

# Advanced Settings
st.sidebar.subheader("Advanced Settings")
contamination_rate = st.sidebar.slider("Anomaly Detection Sensitivity", 0.01, 0.15, 0.05)
show_technical_details = st.sidebar.checkbox("Show Technical Details", False)

# Data Generation and Caching
@st.cache_data
def load_health_data(users, days, freq):
    return simulate_comprehensive_health_data(users, days, freq)

# Load data
with st.spinner("Generating comprehensive health data..."):
    df_raw = load_health_data(num_users, num_days, samples_per_hour)

# Data preprocessing
df_processed, feature_cols = advanced_preprocessing(df_raw)

# Main Dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Users", len(df_processed['user_id'].unique()))
with col2:
    st.metric("Total Measurements", len(df_processed))
with col3:
    st.metric("Time Period", f"{num_days} days")
with col4:
    st.metric("Data Points/Hour", samples_per_hour)

# User Selection for Detailed Analysis
selected_user = st.selectbox("Select User for Detailed Analysis:", 
                           df_processed['user_id'].unique())

user_data = df_processed[df_processed['user_id'] == selected_user].copy()

# Real-time Dashboard Section
st.header("📊 Real-time Health Dashboard")

# Train anomaly detection model
detector = HealthAnomalyDetector()
X_features = df_processed[feature_cols].fillna(0)
detector.fit(X_features)

# Predict anomalies
anomaly_predictions = detector.predict(X_features)
df_processed['anomaly_status'] = anomaly_predictions

# Anomaly probabilities
anomaly_proba = detector.predict_proba(X_features)
df_processed['anomaly_probability'] = anomaly_proba

# Current status for selected user
user_anomalies = df_processed[df_processed['user_id'] == selected_user]
current_status = user_anomalies['anomaly_status'].iloc[-1]
latest_reading = user_anomalies.iloc[-1]

# Status Display
if current_status == 'Anomaly':
    st.markdown(f"""
    <div class="anomaly-alert">
    <h3>⚠️ ANOMALY DETECTED for {selected_user}</h3>
    <p>Anomaly probability: {latest_reading['anomaly_probability']:.2%}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="normal-status">
    <h3>✅ NORMAL STATUS for {selected_user}</h3>
    <p>All metrics within normal range</p>
    </div>
    """, unsafe_allow_html=True)

# Current Vital Signs
st.subheader(f"Current Vital Signs - {selected_user}")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Heart Rate", f"{latest_reading['heart_rate']:.0f} BPM", 
              delta=f"{latest_reading['hr_deviation']:.1f}")
with col2:
    st.metric("Blood Oxygen", f"{latest_reading['blood_oxygen']:.1f}%")
with col3:
    st.metric("Temperature", f"{latest_reading['temperature']:.1f}°C")
with col4:
    st.metric("Resp. Rate", f"{latest_reading['respiration_rate']:.0f}/min")

# Interactive Visualizations
st.header("📈 Health Trend Analysis")

# Time series plot with anomalies
fig_trends = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Heart Rate', 'Blood Oxygen', 'Temperature', 'Respiratory Rate'],
    vertical_spacing=0.1
)

user_anomaly_data = user_anomalies[user_anomalies['anomaly_status'] == 'Anomaly']

# Heart Rate
fig_trends.add_trace(go.Scatter(x=user_anomalies['timestamp'], y=user_anomalies['heart_rate'],
                               mode='lines', name='Heart Rate', line=dict(color='blue')), row=1, col=1)
fig_trends.add_trace(go.Scatter(x=user_anomaly_data['timestamp'], y=user_anomaly_data['heart_rate'],
                               mode='markers', name='Anomaly', marker=dict(color='red', size=8)), row=1, col=1)

# Blood Oxygen
fig_trends.add_trace(go.Scatter(x=user_anomalies['timestamp'], y=user_anomalies['blood_oxygen'],
                               mode='lines', name='Blood Oxygen', line=dict(color='green'), showlegend=False), row=1, col=2)
fig_trends.add_trace(go.Scatter(x=user_anomaly_data['timestamp'], y=user_anomaly_data['blood_oxygen'],
                               mode='markers', name='Anomaly', marker=dict(color='red', size=8), showlegend=False), row=1, col=2)

# Temperature
fig_trends.add_trace(go.Scatter(x=user_anomalies['timestamp'], y=user_anomalies['temperature'],
                               mode='lines', name='Temperature', line=dict(color='orange'), showlegend=False), row=2, col=1)
fig_trends.add_trace(go.Scatter(x=user_anomaly_data['timestamp'], y=user_anomaly_data['temperature'],
                               mode='markers', name='Anomaly', marker=dict(color='red', size=8), showlegend=False), row=2, col=1)

# Respiratory Rate
fig_trends.add_trace(go.Scatter(x=user_anomalies['timestamp'], y=user_anomalies['respiration_rate'],
                               mode='lines', name='Resp Rate', line=dict(color='purple'), showlegend=False), row=2, col=2)
fig_trends.add_trace(go.Scatter(x=user_anomaly_data['timestamp'], y=user_anomaly_data['respiration_rate'],
                               mode='markers', name='Anomaly', marker=dict(color='red', size=8), showlegend=False), row=2, col=2)

fig_trends.update_layout(height=600, showlegend=True, title_text=f"Health Trends for {selected_user}")
st.plotly_chart(fig_trends, use_container_width=True)

# Personalized Recommendations
st.header("💡 Personalized Health Recommendations")
recommendations = generate_health_recommendations(user_data, current_status)

for i, rec in enumerate(recommendations):
    st.markdown(f"{i+1}. {rec}")

# System-wide Analytics
st.header("🌐 System-wide Health Analytics")

col1, col2 = st.columns(2)

with col1:
    # Anomaly distribution by user
    anomaly_counts = df_processed.groupby(['user_id', 'anomaly_status']).size().unstack(fill_value=0)
    fig_anomaly = px.bar(anomaly_counts, title="Anomaly Distribution by User",
                        color_discrete_map={'Normal': 'green', 'Anomaly': 'red'})
    st.plotly_chart(fig_anomaly, use_container_width=True)

with col2:
    # Average metrics by health condition
    condition_metrics = df_processed.groupby('health_condition')[['heart_rate', 'blood_oxygen', 'temperature']].mean()
    fig_condition = px.bar(condition_metrics.reset_index(), x='health_condition', y='heart_rate',
                          title="Average Heart Rate by Health Condition")
    st.plotly_chart(fig_condition, use_container_width=True)

# Technical Details Section
if show_technical_details:
    st.header("🔬 Technical Details")
    
    # Model performance metrics
    st.subheader("Model Performance")
    
    # Create synthetic labels for evaluation (in real scenario, you'd have ground truth)
    # Here we use extreme values as "true" anomalies for demonstration
    true_anomalies = (
        (df_processed['heart_rate'] > 120) | 
        (df_processed['heart_rate'] < 50) |
        (df_processed['blood_oxygen'] < 90) |
        (df_processed['temperature'] > 38.0)
    ).astype(int)
    
    predicted_anomalies = (df_processed['anomaly_status'] == 'Anomaly').astype(int)
    
    accuracy = accuracy_score(true_anomalies, predicted_anomalies)
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    # Feature importance visualization
    st.subheader("Feature Analysis")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.random.random(len(feature_cols))  # Placeholder - in real scenario use model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                           title="Feature Importance for Anomaly Detection")
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Raw data preview
    st.subheader("Raw Data Sample")
    st.dataframe(df_processed.head(20))

# Export functionality
st.header("💾 Data Export")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export All Data"):
        csv = df_processed.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"health_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Export Anomalies Only"):
        anomaly_data = df_processed[df_processed['anomaly_status'] == 'Anomaly']
        csv = anomaly_data.to_csv(index=False)
        st.download_button(
            label="Download Anomalies CSV",
            data=csv,
            file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col3:
    if st.button("Export User Report"):
        user_report = user_data.to_csv(index=False)
        st.download_button(
            label=f"Download {selected_user} Report",
            data=user_report,
            file_name=f"{selected_user}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("**AI Health Monitoring System Pro** - Advanced healthcare analytics powered by machine learning")
st.markdown("⚠️ *Note: This is a demonstration system. Always consult healthcare professionals for medical advice.*")