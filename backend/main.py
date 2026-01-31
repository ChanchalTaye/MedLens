from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
import base64
from pathlib import Path

#  APP 
app = FastAPI(title="MedLens – Multi-Modal Medical Imaging AI")

#  BASE DIR 
BASE_DIR = Path(__file__).resolve().parent.parent

#  LOAD MODELS 
xray_model = tf.keras.models.load_model(
    BASE_DIR / "model" / "multiclass_radiology_model.h5"
)

ultrasound_model = tf.keras.models.load_model(
    BASE_DIR / "model" / "ultrasound_breast_model.h5"
)

mri_model = tf.keras.models.load_model(
    BASE_DIR / "model" / "mri_kidney_stone_model.h5"
)

#  CLASS NAMES 
XRAY_CLASSES = ["NORMAL", "PNEUMONIA", "COVID19", "TUBERCULOSIS"]
ULTRASOUND_CLASSES = ["BENIGN", "MALIGNANT"]
MRI_CLASSES = ["NORMAL", "STONE"]

#  GRAD-CAM (MULTI-CLASS) 
def grad_cam_multiclass(model, img_array, class_index, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

#  GRAD-CAM (BINARY) 
def grad_cam_binary(model, img_array, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, prediction = grad_model(img_array)
        loss = prediction[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

#  API ENDPOINT 
@app.post("/predict")
async def predict(
    modality: str,
    file: UploadFile = File(...)
):
    contents = await file.read()

    # Decode image
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    img_resized = cv2.resize(image_rgb, (224, 224))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    #  ROUTING 
    if modality == "xray":
        preds = xray_model.predict(img_array)[0]
        class_index = int(np.argmax(preds))
        label = XRAY_CLASSES[class_index]
        confidence = float(preds[class_index])

        heatmap = grad_cam_multiclass(
            xray_model, img_array, class_index
        )

    elif modality == "ultrasound":
        pred = ultrasound_model.predict(img_array)[0][0]
        class_index = int(pred > 0.5)
        label = ULTRASOUND_CLASSES[class_index]
        confidence = float(pred)

        heatmap = grad_cam_binary(
            ultrasound_model, img_array
        )

    elif modality == "mri":
        pred = mri_model.predict(img_array)[0][0]
        class_index = int(pred > 0.5)
        label = MRI_CLASSES[class_index]
        confidence = float(pred)

        heatmap = grad_cam_binary(
            mri_model, img_array
        )

    else:
        return {"error": "Invalid modality. Use xray / ultrasound / mri"}

    #  HEATMAP OVERLAY 
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode(".png", overlay)
    heatmap_b64 = base64.b64encode(buffer).decode()

    #  RESPONSE 
    return {
        "modality": modality,
        "prediction": label,
        "confidence": round(confidence, 3),
        "heatmap": heatmap_b64,
        "disclaimer": "This is NOT a diagnostic tool. Research & decision support only."
    }
