# main.py
# The web server: receives an image, returns the predicted number.

import base64                       # to decode the image that the web page will send as text
import io                           # lets us treat raw bytes like a file

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel      # describes the shape of the data the server expects

from utils import load_model, read_number   # the functions we just tested
from fastapi.responses import FileResponse   # lets us send a file (our HTML page) to the browser

app = FastAPI(title="Handwritten Number Reader")   # creates the web app

# ---- The main page: when someone opens the site, send them our HTML file ----
@app.get("/")
def home():
    return FileResponse("static/index.html")

# Load the trained model ONCE when the server starts (not on every request, that would be slow)
model = load_model("model.onnx")


def to_gray(pil_image):
    """Convert any image to a grayscale numpy array."""
    # A drawing from a web canvas can have a TRANSPARENT background.
    # If we converted it directly, transparent areas would turn black and ruin the picture.
    # So we paste the drawing on a white background first.
    if pil_image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", pil_image.size, (255, 255, 255))
        background.paste(pil_image, mask=pil_image.split()[-1])   # last channel = transparency
        pil_image = background
    return np.array(pil_image.convert("L"))


# ---- Endpoint 1: upload an image file (easy to test on the /docs page) ----
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    data = await file.read()                       # the raw bytes of the uploaded file
    try:
        gray = to_gray(Image.open(io.BytesIO(data)))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image file.")
    return read_number(gray, model)                # returns the number + details as JSON


# ---- Endpoint 2: image sent as text (this is what the web page's canvas will use) ----
class CanvasImage(BaseModel):
    image: str                                     # looks like "data:image/png;base64,iVBOR..."


@app.post("/predict")
def predict(body: CanvasImage):
    try:
        b64 = body.image.split(",")[-1]            # remove the "data:image/png;base64," prefix
        gray = to_gray(Image.open(io.BytesIO(base64.b64decode(b64))))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode the image.")
    return read_number(gray, model)


# ---- A tiny check that the server is alive ----
@app.get("/health")
def health():
    return {"status": "ok"}