# Handwritten Number Reader

Draw any number on a web canvas (2 digits, 5 digits, more) and the app reads it.
A small CNN trained on MNIST recognizes each digit, and OpenCV splits the drawing into separate digits.

![Demo](images/demo.png)

## How it works

1. **Training:** a CNN built with PyTorch is trained on MNIST (~99.11% test accuracy). See `notebook.ipynb`.
2. **Export:** the trained model is exported to ONNX, so the app runs with the light `onnxruntime` library instead of PyTorch.
3. **Cutting:** OpenCV finds each separate blob of ink in the drawing and sorts them left to right.
4. **Preprocessing:** each digit is converted to MNIST format (white on black, thickened strokes, resized to 20x20, centered by center of mass in a 28x28 image).
5. **Prediction:** the model predicts each digit, and the digits are joined into one number.
6. **Web app:** a FastAPI backend serves an HTML canvas page. The page sends the drawing to the API and shows the result.

## Features

- Reads numbers of any length
- Draws a box around each digit with its prediction and confidence
- Highlights digits the model is unsure about (under 80%) in red, with a warning
- Adjustable brush size and an Undo button

## Tech stack

PyTorch, ONNX Runtime, OpenCV, NumPy, FastAPI, HTML/CSS/JavaScript (canvas)

## Run it locally

```bash
# 1. Get the code (or download the ZIP from GitHub)
git clone https://github.com/YOUR-USERNAME/handwritten-number-reader.git
cd handwritten-number-reader

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install the libraries
pip install -r requirements.txt

# 4. Start the server
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000 and draw a number.

To retrain the model, open `notebook.ipynb` (it also needs `torch`, `torchvision` and `matplotlib`).

## API

| Endpoint | What it does |
|---|---|
| `POST /predict` | Takes a base64 PNG (`{"image": "..."}`) and returns the number plus each digit's confidence and box |
| `POST /predict-file` | Same, but takes an uploaded image file |
| `GET /health` | Returns `{"status": "ok"}` |
| `GET /docs` | Interactive API test page (FastAPI) |

## Known limitations

- Digits that touch each other are read as one blob, so leave small gaps between digits.
- A digit drawn in separate, unconnected strokes may be split into two pieces.
- The model only knows MNIST handwriting styles, so unusual styles get lower confidence.

## What I learned

- Preprocessing matters as much as the model: it scored ~99% on MNIST but failed on my own drawings until the strokes were thickened and centered by center of mass.
- Exporting to ONNX keeps the model's output (max difference about 4e-6) while making the app much lighter to deploy.
