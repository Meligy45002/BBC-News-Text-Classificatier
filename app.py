# app.py

import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# Load model and encoders
# -------------------------
model = tf.keras.models.load_model("bbc_lstm_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

max_len = 200

# -------------------------
# Streamlit UI
# -------------------------
st.title("BBC News Text Classification")
st.write("Enter a news article text, and the model will predict its category.")

user_input = st.text_area("Enter text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Preprocess input
        seq = tokenizer.texts_to_sequences([user_input])
        pad_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

        # Predict
        pred = model.predict(pad_seq)
        class_idx = pred.argmax(axis=1)[0]
        category = le.inverse_transform([class_idx])[0]

        st.success(f"Predicted Category: **{category}**")
