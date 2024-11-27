import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ëª¨ë¸ ë¡ë í¨ì
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.keras")

# ëª¨ë¸ ì´ê¸°í
model = load_model()

# Streamlit UI
st.title("Handwritten Digit Classification")
st.write("Upload a 28x28 grayscale image of a handwritten digit (0-9), and the model will classify it.")

# ì¬ì©ì ì´ë¯¸ì§ ìë¡ë
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # ì´ë¯¸ì§ ì´ê¸° ë° ì²ë¦¬
        image = Image.open(uploaded_file).convert("L")  # Grayscaleë¡ ë³í
        image = ImageOps.invert(image)  # ì ë°ì  (í° ë°°ê²½, ê²ì ê¸ì¨)
        image = image.resize((28, 28))  # 28x28 í¬ê¸°ë¡ ì¡°ì 
        image_array = np.array(image) / 255.0  # ì ê·í

        # ë°ì´í° íì ë³í
        image_array = np.expand_dims(image_array, axis=-1)  # (28, 28, 1)ë¡ ë³í
        image_array = np.expand_dims(image_array, axis=0)   # (1, 28, 28, 1)ë¡ ë³í

        # ìë ¥ ë°ì´í° í¬ê¸° íì¸
        st.write(f"Processed input shape: {image_array.shape}")

        # ëª¨ë¸ ìì¸¡
        prediction = model.predict(image_array, batch_size=1)
        predicted_label = np.argmax(prediction)

        # ê²°ê³¼ ì¶ë ¥
        st.subheader("Prediction")
        st.write(f"The model predicts this digit as: **{predicted_label}**")

        # ìì¸¡ íë¥  ìê°í
        st.bar_chart(prediction.flatten())

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

else:
    st.write("Please upload an image to get started.")
