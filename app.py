"""
Skin Lesion Classification — Streamlit Diagnostic Portal

Run with:  streamlit run app.py
"""

import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = "skin_lesion_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

CLASS_INFO = {
    "akiec": (
        "Actinic keratosis / intraepithelial carcinoma",
        "an early or pre-cancerous growth caused by sun damage",
    ),
    "bcc": (
        "Basal cell carcinoma",
        "a common, usually slow-growing form of skin cancer",
    ),
    "bkl": ("Benign keratosis-like lesion", "a non-cancerous growth on the skin"),
    "df": ("Dermatofibroma", "a benign, firm skin nodule"),
    "mel": ("Melanoma", "a potentially serious form of skin cancer"),
    "nv": ("Melanocytic nevus", "an ordinary mole"),
    "vasc": ("Vascular lesion", "a benign blood-vessel-related skin mark"),
}

UNCERTAIN_THRESHOLD = 0.50


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def preprocess_image(pil_image: Image.Image) -> np.ndarray:

    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize(IMG_SIZE)
    arr = np.array(pil_image).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr


st.set_page_config(
    page_title="Skin Lesion Classifier", page_icon="🩺", layout="centered"
)

st.title(" Skin Lesion Classifier")
st.markdown(
    "This tool analyses a **dermoscopic photo of a skin lesion** and predicts which of "
    "**7 common diagnostic categories** it most likely belongs to (e.g. mole, melanoma, "
    "basal cell carcinoma), based on the HAM10000 dataset."
)
st.warning(
    "⚠️ **This is a student coursework prototype, not a medical device.** It has not been "
    "clinically validated. Do not use it to make or delay any real medical decision,always "
    "consult a qualified clinician."
)

st.divider()

try:
    model, class_names = load_model_and_classes()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        f"Couldn't load the model files ({MODEL_PATH} / {CLASS_NAMES_PATH}). "
        f"Make sure they're in the same folder as this app. Details: {e}"
    )

uploaded_file = st.file_uploader(
    "Upload a dermoscopic image of a skin lesion",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Analysing image..."):
        processed = preprocess_image(image)
        predictions = model.predict(processed, verbose=0)[0]

    predicted_idx = int(np.argmax(predictions))
    predicted_class = class_names[predicted_idx]
    confidence = float(predictions[predicted_idx])
    full_name, plain_desc = CLASS_INFO.get(predicted_class, (predicted_class, ""))

    with col2:
        st.subheader("Result")
        st.metric(label=full_name, value=f"{confidence * 100:.1f}% confidence")

        if confidence < UNCERTAIN_THRESHOLD:
            st.info(
                " **The model is not confident about this image.** The result below is its "
                "best guess, but it looks less like a typical example from its training data. "
                "Treat this prediction with extra caution."
            )

        st.write(
            f"The model predicts this image most likely shows **{full_name.lower()}** "
            f"({plain_desc}), with **{confidence * 100:.1f}% confidence**."
        )

    st.divider()
    st.subheader("Full prediction breakdown")
    st.markdown(
        "How confident the model was in *each* possible category, not just the top pick:"
    )

    sorted_indices = np.argsort(predictions)[::-1]
    for idx in sorted_indices:
        cls = class_names[idx]
        name, _ = CLASS_INFO.get(cls, (cls, ""))
        st.progress(
            float(predictions[idx]), text=f"{name} — {predictions[idx] * 100:.1f}%"
        )

elif uploaded_file is not None and not model_loaded:
    st.stop()

st.divider()

with st.expander(" About this app / how to run it locally"):
    st.markdown(
        """
**What this app does:** Loads a trained MobileNetV2-based classifier
(`skin_lesion_model.keras`) and runs it on whatever dermoscopic image you upload, applying the
exact same preprocessing used during training (see notebook section A7).

**How to run locally:**
1. Install dependencies: `pip install -r requirements.txt`
2. Make sure `skin_lesion_model.keras` and `class_names.json` are in the same folder as `app.py`.
3. Run: `streamlit run app.py`
4. Your browser will open automatically at `http://localhost:8501`.

**Limitations:** trained only on the HAM10000 dermoscopic image dataset -performance on regular
phone-camera photos, other skin tones, or lesion types not represented in that dataset may be
significantly worse than shown during evaluation.
        """
    )
