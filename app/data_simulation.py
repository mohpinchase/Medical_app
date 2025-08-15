import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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