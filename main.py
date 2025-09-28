# ===============================
# Streamlit Forest Fire Detection App (with SMS alert & forest theme)
# ===============================

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image
from twilio.rest import Client
from dotenv import load_dotenv
import streamlit as st
from twilio.rest import Client

ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]
AUTH_TOKEN  = st.secrets["TWILIO_AUTH_TOKEN"]
FROM_NUMBER = st.secrets["TWILIO_FROM_NUMBER"]
TO_NUMBER   = st.secrets["MY_PHONE_NUMBER"]

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# -------------------------------
# Page Config + CSS Styling
# -------------------------------
st.set_page_config(page_title="🌲 Forest Fire Detection", page_icon="🔥", layout="wide")

page_bg = """
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1509021436665-8f07dbf5bf1d");
    background-size: cover;
    background-attachment: fixed;
}
.block-container {
    background-color: rgba(0,0,0,0.6);
    padding: 2rem;
    border-radius: 15px;
    color: white;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.title("🌲 Forest Fire Detection System")
st.write("Upload an image and the system will detect if it contains a forest fire. 🔥")
st.write("If a fire is detected, you will also receive an **SMS alert** 📩")

# -------------------------------
# Load Model
# -------------------------------
model = load_model("model/forest_fire_model.h5")

# -------------------------------
# SMS Function
# -------------------------------
def send_sms_alert(message):
    try:
        client.messages.create(
            body=message,
            from_=FROM_NUMBER,
            to=TO_NUMBER
        )
        st.success("📩 SMS sent successfully!")
    except Exception as e:
        st.warning(f"⚠️ SMS could not be sent: {e}")

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader("📷 Upload a forest image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    img = Image.open(uploaded_file)
    img = img.convert("RGB")

    # Display uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Resize and preprocess
    img = img.resize((64, 64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    fire_prob = 1 - prediction   # invert logic if needed

    # Show confidence
    st.subheader(f"🔥 Fire Probability: {fire_prob*100:.2f}%")

    # Show result
    if fire_prob >= 0.5:
        st.error("⚠️ Forest Fire Detected!")
        send_sms_alert("🚨 ALERT: Forest fire detected in uploaded image!")
    else:
        st.success("✅ No Forest Fire Detected.")
