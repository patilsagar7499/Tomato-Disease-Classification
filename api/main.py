from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json

app=FastAPI()

MODEL=tf.keras.models.load_model(r"C:\Users\SAGAR\Downloads\Tomato_disease\Saved_Model\2")
MODEL.save(r"C:\Users\SAGAR\Downloads\Tomato_disease\Saved_Model\MODEL.keras")
MODEL.save(r"C:\Users\SAGAR\Downloads\Tomato_disease\Saved_Model\MODEL.h5", save_format="h5")
CLASS_NAMES = [
    "Tomato__Target_Spot", "Tomato__Tomato_mosaic_virus", "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_healthy", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite"
]
CLASS_NAMES.sort()  # Ensure the order matches the training order

@app.get("/ping")
async def ping():
    return "Hello,I am alive"

def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file:UploadFile=File(...)
): 
    image=read_file_as_image(await file.read())
    img_batch=np.expand_dims(image,0)
    predictions=MODEL.predict(img_batch)
    predicted_class=CLASS_NAMES[np.argmax(predictions)]
    confidence=np.max(predictions)
    return{
        'class':predicted_class,
        'confidence':float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
