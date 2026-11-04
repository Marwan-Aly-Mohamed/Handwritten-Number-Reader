---
title: Handwritten Number Reader
emoji: ✍️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Handwritten Number Reader

Draw any number (any number of digits) and the app reads it.

- Model: small CNN trained on MNIST (PyTorch), ~99% test accuracy
- Digit cutting: OpenCV contours, each digit preprocessed to MNIST format
- Backend: FastAPI. Frontend: HTML canvas