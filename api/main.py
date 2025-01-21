from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import subprocess
import time
import multiprocessing
import requests

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

MODEL=tf.keras.models.load_model(r"Saved_Model/2")
MODEL.save(r"Saved_Model/MODEL.keras")
MODEL.save(r"Saved_Model/MODEL.h5", save_format="h5")
CLASS_NAMES = [
    "Tomato__Target_Spot", "Tomato__Tomato_mosaic_virus", "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_healthy", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite"
]
CLASS_NAMES.sort()  

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
    try:
        image=read_file_as_image(await file.read())
        img_batch=np.expand_dims(image,0)
        predictions=MODEL.predict(img_batch)
        predicted_class=CLASS_NAMES[np.argmax(predictions)]
        confidence=np.max(predictions)
        return{
            'class':predicted_class,
            'confidence':float(confidence)
    }
    except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}

def run_fastapi():
    uvicorn.run(app, host="localhost", port=9000)

def run_flask():
    subprocess.Popen(["python", "flask_frontend.py"])

def is_fastapi_ready():
    try:
        response = requests.get("http://localhost:9000/ping")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

if __name__ == "__main__":
    fastapi_process = multiprocessing.Process(target=run_fastapi)
    fastapi_process.start()

    while not is_fastapi_ready():
        print("Waiting for FastAPI to start...")
        time.sleep(1) 

    print("FastAPI is ready, starting Flask...")
    run_flask()