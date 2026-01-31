import streamlit as st
import requests
import base64
from PIL import Image
import io

#  CONFIG 
st.set_page_config(
    page_title="MedLens",
    page_icon="🩺",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

#  CUSTOM CSS 
st.markdown("""
<style>
.main {
    background-color: #f9fbfd;
}
.title {
    font-size: 3rem;
    font-weight: 700;
    color: #1f4fd8;
}
.subtitle {
    font-size: 1.2rem;
    color: #555;
}
.card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.05);
}
.pred-normal {
    color: #1b9e77;
    font-size: 1.5rem;
    font-weight: 700;
}
.pred-pneumonia {
    color: #d62828;
    font-size: 1.5rem;
    font-weight: 700;
}
.confidence {
    font-size: 1.1rem;
    color: #333;
}
.disclaimer {
    font-size: 0.9rem;
    color: #888;
}
</style>
""", unsafe_allow_html=True)

#  HEADER 
st.markdown("<div class='title'>🩺 MedLens</div>", unsafe_allow_html=True)

#  MODALITY TAGLINES
taglines = {
    "XRAY": "Explainable AI for Chest X-ray Disease Detection",
    "ULTRASOUND": "AI-powered Ultrasound Analysis for Breast Conditions",
    "MRI": "Deep Learning MRI Analysis for Internal Abnormalities",
    "Choose Modality": "Multi-Modal Medical Imaging AI for Clinical Decision Support"
}

#  LAYOUT 
left, right = st.columns([1, 1])

#  LEFT: UPLOAD 
with left:
    st.markdown("###  Upload Medical Image")

    modality_display = st.selectbox(
        "Select Imaging Modality",
        ["Choose Modality", "XRAY", "ULTRASOUND", "MRI"]
    )

    st.markdown(
        f"<div class='subtitle'>{taglines[modality_display]}</div>",
        unsafe_allow_html=True
    )

    modality = modality_display.lower() if modality_display != "Choose Modality" else None

    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

st.markdown("---")

#  DISCLAIMER 
st.warning(
    " **Medical Disclaimer**: This system is for research and decision support only. "
    "It is NOT a diagnostic or clinical decision-making tool."
)

#  RIGHT: RESULTS 
with right:
    if uploaded_file and modality and st.button(" Analyze Image"):
        with st.spinner("Analyzing with AI model..."):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            response = requests.post(
                API_URL,
                params={"modality": modality},
                files=files
            )

        if response.status_code == 200:
            result = response.json()

            prediction = result["prediction"]
            confidence = result["confidence"]
            heatmap_b64 = result["heatmap"]

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            if prediction in ["PNEUMONIA", "COVID19", "TUBERCULOSIS", "MALIGNANT", "STONE"]:
                st.markdown(
                    f"<div class='pred-pneumonia'> Prediction: {prediction}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='pred-normal'> Prediction: {prediction}</div>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"<div class='confidence'>Confidence: {confidence * 100:.2f}%</div>",
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("###  Model Attention (Grad-CAM)")

            heatmap_bytes = base64.b64decode(heatmap_b64)
            heatmap_image = Image.open(io.BytesIO(heatmap_bytes))

            st.image(
                heatmap_image,
                caption="Highlighted regions influencing the model’s decision",
                use_column_width=True
            )

            st.markdown(
                "<div class='disclaimer'>Heatmap is an explainability aid, not a diagnosis.</div>",
                unsafe_allow_html=True
            )

        else:
            st.error(" Unable to communicate with backend API.")
