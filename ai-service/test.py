# test.py
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI

print("✅ OpenCV:", cv2.__version__)
print("✅ MediaPipe:", mp.__version__)
print("✅ NumPy:", np.__version__)
print("✅ FastAPI: OK")
print("\n🎉 모든 라이브러리 설치 완료!")