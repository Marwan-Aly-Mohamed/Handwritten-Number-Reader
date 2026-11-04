# utils.py
# Image processing + prediction using the ONNX model (no PyTorch needed).

import cv2                       # image processing (crop, resize, find shapes)
import numpy as np               # grids of numbers
import onnxruntime as ort        # runs the exported ONNX model


def load_model(path="model.onnx"):
    # An InferenceSession is the loaded model, ready to make predictions
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def softmax(scores):
    # Turns the model's 10 raw scores into 10 percentages that add up to 1
    e = np.exp(scores - scores.max())     # subtracting the max keeps the numbers stable
    return e / e.sum()


def center_by_mass(img):
    """Shift a 28x28 image so the ink's balance point is in the middle (like MNIST)."""
    m = cv2.moments(img)
    if m["m00"] == 0:                     # no ink at all: nothing to shift
        return img
    cx = m["m10"] / m["m00"]              # average x position of the ink
    cy = m["m01"] / m["m00"]              # average y position of the ink
    M = np.float32([[1, 0, int(round(14 - cx))], [0, 1, int(round(14 - cy))]])
    return cv2.warpAffine(img, M, (28, 28))


def preprocess_digit(gray, thickness=0.10):
    """One digit (grayscale array) -> 28x28 MNIST-style image."""
    if gray.mean() > 127:                 # white background? flip so ink is white on black
        gray = 255 - gray

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(bw)
    if coords is None:
        raise ValueError("No digit found (blank image).")

    x, y, w, h = cv2.boundingRect(coords)   # tight box around the ink
    digit = bw[y:y + h, x:x + w]            # crop it

    k = max(3, int(max(w, h) * thickness))  # thicken the strokes, like MNIST
    digit = cv2.dilate(digit, np.ones((k, k), np.uint8), iterations=1)

    scale = 20 / max(w, h)                  # longest side becomes 20 pixels
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)               # empty black 28x28 image
    x_off, y_off = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = digit  # paste digit in the middle

    return center_by_mass(canvas)


def find_digit_boxes(gray):
    """Whole-number image -> (boxes sorted left to right, image with white ink on black)."""
    if gray.mean() > 127:
        gray = 255 - gray

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]

    if len(boxes) == 0:
        return [], gray

    tallest = max(h for (x, y, w, h) in boxes)
    boxes = [b for b in boxes if b[3] > 0.3 * tallest]   # remove tiny specks
    boxes.sort(key=lambda b: b[0])                       # left to right
    return boxes, gray


def read_number(gray, session):
    """
    gray    = grayscale numpy array with a handwritten number (any number of digits)
    session = the loaded ONNX model
    Returns {"number": "482", "digits": [{digit, confidence, box}, ...]}
    """
    boxes, oriented = find_digit_boxes(gray)
    if len(boxes) == 0:
        return {"number": "", "digits": []}

    results = []
    for (x, y, w, h) in boxes:
        pad = 10                                          # small margin around each digit
        crop = oriented[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
        processed = preprocess_digit(crop)

        # Shape [1, 1, 28, 28] with values 0-1, as float32, which is what the model expects
        tensor = (processed / 255.0).astype(np.float32)[None, None]

        scores = session.run(None, {"image": tensor})[0][0]   # the 10 scores for this digit
        probs = softmax(scores)

        guess = int(probs.argmax())
        results.append({
            "digit": guess,
            "confidence": round(float(probs[guess]) * 100, 1),
            "box": [int(x), int(y), int(w), int(h)],
        })

    number = "".join(str(r["digit"]) for r in results)    # [7, 7] -> "77"
    return {"number": number, "digits": results}