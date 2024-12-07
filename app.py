from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image

app = FastAPI()

IMG_HEIGHT = 256
IMG_WIDTH = 256
CLASS_NAMES = ['Crossed_Eyes', 'Normal_eye', 'Red_eyes', 'eyebag']

model = load_model("model/trained_model.h5")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of the uploaded image.
    """
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
        
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0 
        img_array = np.expand_dims(img_array, axis=0)  
        
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])  
        predicted_class = min(predicted_class, len(CLASS_NAMES) - 1)
        predicted_label = CLASS_NAMES[predicted_class]
        
        return JSONResponse(
            status_code=200,
            content={"predicted_label": predicted_label}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
