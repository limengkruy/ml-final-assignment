import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# Disable GPU (Optional)
tf.config.set_visible_devices([], 'GPU')  # Disable default GPU setting
print("Available devices:", tf.config.list_physical_devices())
    
# Load preprocessor and model
@st.cache_resource
def load_model():
    prep = joblib.load('deep_preprocessor.pkl')
    model = tf.keras.models.load_model('deep_model.keras')
    return prep, model

to_drop = ['session_id','unusual_time_access']
prep, model = load_model()

# Streamlit UI
st.title("🔍 Cybersecurity Intrusion Detection")
st.write("Manually input values and predict if an attack is detected.")

# Create input fields based on column types
session_id = st.text_input("Session ID (Optional)", "")
network_packet_size = st.number_input("Network Packet Size", min_value=0, step=1)
protocol_type = st.selectbox("Protocol Type", ['TCP', 'UDP', 'ICMP'])
login_attempts = st.number_input("Login Attempts", min_value=0, step=1)
session_duration = st.number_input("Session Duration (Seconds)", min_value=0.0, step=0.1)
encryption_used = st.selectbox("Encryption Used", ['DES', 'AES'])
ip_reputation_score = st.number_input("IP Reputation Score", min_value=0.0, max_value=1.0, step=0.01)
failed_logins = st.number_input("Failed Logins", min_value=0, step=1)
browser_type = st.selectbox("Browser Type", ['Edge', 'Firefox', 'Chrome', 'Unknown', 'Safari'])
unusual_time_access = st.number_input("Unusual Time Access (0 for No, 1 for Yes)", min_value=0, max_value=1, step=1)

# Collect all inputs into a dictionary
user_inputs = {
    "session_id": session_id,
    "network_packet_size": network_packet_size,
    "protocol_type": protocol_type,  # Directly pass raw categorical values
    "login_attempts": login_attempts,
    "session_duration": session_duration,
    "encryption_used": encryption_used,  # Directly pass raw categorical values
    "ip_reputation_score": ip_reputation_score,
    "failed_logins": failed_logins,
    "browser_type": browser_type,  # Directly pass raw categorical values
    "unusual_time_access": unusual_time_access
}

# Predict button
if st.button("Predict"):
    # Convert user input into a DataFrame
    input_df = pd.DataFrame([user_inputs])

    # Preprocess the input (Preprocessor handles categorical encoding automatically)
    input_transformed = prep.transform(input_df)
    # Make prediction
    y_pred = (model.predict(input_transformed)[0, 0] > 0.5).astype(int)

    # Display result
    st.subheader("🔮 Prediction Result")
    if y_pred == 1:
        st.error("⚠ **Attack Detected (1)**")
    else:
        st.success("✅ **No Attack Detected (0)**")
