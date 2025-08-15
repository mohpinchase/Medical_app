import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from app.data_simulation import simulate_health_data
from app.preprocessing import preprocess_data
from app.anomaly import detect_anomalies, evaluate_model

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
st.dataframe(df.head(101))

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
from sklearn.model_selection import train_test_split
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