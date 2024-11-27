import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import image,imageOps

@st.cache.resource
def load_model():
    return tf.keras.models.load_model("my_model.keras")

model = load_model()

st.title("Handwritten Digit Classfication")
st.write("Upload a 28x28 grayscale image of a handwritten digit (0-9), and the model will classify it.")

upload_file = st.file_uploader("Upload an image",type=["png","jpg","jpeg"])

if upload_file is not None:
    try:
        image = image.open(upload_file).convert("L")
        image = imageOps.invert(image)
        image = image.resize((28,28))
        image_array = np.array(image)

        image_array = np.expand_dims(image_array,axis=-1)
        image_array = np.expand_dims(image_array,axis=0)

        st.write("Processed input shape : {image_array.shape}")

        prediction = model.predict(image_array,batch_size=1)
        predicted_label = np.argmax(prediction)

        st.subheader("Prediction")
        st.write(f"The model predicts this is digit as: **{predicted_label}**")

        st.bar_chart(prediction.flatten())

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
else:
    st.write("Please Uploae an image to get started.")
