# STEP 1: Import the necessary modules.

import easyocr
import cv2
import numpy as np
import mediapipe as mp
# STEP 2: Create an ObjectDetector object.
reader = easyocr.Reader(['ko','en'], gpu=False) # this needs to run only once to load the model into memory


# STEP 3: Create a Detector object.
from fastapi import FastAPI, UploadFile


app = FastAPI()


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()

    binary = np.fromstring(contents, dtype=np.uint8)
    cv_mat = cv2.imdecode(binary, cv2.IMREAD_COLOR)

    
    result = reader.readtext(cv_mat,detail = 0)
    print(result)

    return {"filename": result}