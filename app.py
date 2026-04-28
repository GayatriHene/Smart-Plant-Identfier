import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from plant_info import plant_details   # ✅ import plant care info

# Load model
model = tf.keras.models.load_model("plant_identifier_model.h5")

# Define class names (sequence same as your model training)
class_names = ['aloe_vera', 'money_plant', 'neem', 'tulsi']

st.title("🌿 Smart Plant Identifier")
st.write("Upload an image of a plant to identify whether it is Neem, Tulsi, Aloe Vera, or Money Plant.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"🌱 Predicted: **{predicted_class}** ({confidence:.2f}% confidence)")

    # ✅ Show plant care info
    if predicted_class in plant_details:
        details = plant_details[predicted_class]
        st.subheader(f"🌱 Plant Care Tips for {details['name']}")
        st.write(f"**Watering:** {details['watering']}")
        st.write(f"**Sunlight:** {details['sunlight']}")
        st.write(f"**Soil:** {details['soil']}")
        st.write(f"**Benefits:** {details['benefits']}")
