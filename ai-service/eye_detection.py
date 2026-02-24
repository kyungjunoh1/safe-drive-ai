# eye_detection.py
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh

# 눈 랜드마크 인덱스
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def draw_eye_landmarks(image, landmarks, eye_indices, color=(0, 255, 0)):
    # image: 어느 사진에 그릴까?, landmarks: 468개 점 정보, eye_indices: 눈에 해당하는 점 번호들, color: 무슨 색? (기본값: 초록색)
    """눈 랜드마크를 이미지에 그리기"""
    h, w = image.shape[:2] # 이미지 크기

    points = []
    for idx in eye_indices: # 눈에 해당하는 각 점 번호마다
        lm = landmarks[idx]
        x = int(lm.x * w) # 비율을 실제 픽셀로 변환
        y = int(lm.y * h)
        points.append((x, y))
        cv2.circle(image, (x, y), 2, color, -1) # 반지름 2픽셀, 색깔 채우기(-1)

        # 눈 윤곽 그리기
        points = np.array(points, dtype=np.int32) # 점들을 배열로 정리, dtype=np.int32 = 정수형으로
        cv2.polylines(image, [points], True, color, 1) # 점들을 선으로 이어서 그리기

        return points

print("눈 영역 감지 시작...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break

       # RGB 변환
       image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       image.flags.writeable = False
        
       # 얼굴 감지
       results = face_mesh.process(image)
        
       # BGR 변환
       image.flags.writeable = True
       image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

       if results.multi_face_landmarks:
           for face_landmarks in results.multi_face_landmarks:
               landmarks = face_landmarks.landmark

               # 왼쪽 눈 그리기(초록색)
               left_eye_points = draw_eye_landmarks(
                   image, landmarks, LEFT_EYE, (0, 255, 0)
               )

               # 오른쪽 눈 그리기(초록색)
               right_eye_points = draw_eye_landmarks(
                   image, landmarks, RIGHT_EYE, (0, 255, 0)
               )

               # 텍스트 표시
               cv2.putText(image, "Left Eye",
                           tuple(left_eye_points[0]),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 1)
               
               cv2.putText(image, "Right Eye",
                           tuple(right_eye_points[0]),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 0, 0), 1)
               
           cv2.imshow('Eye Detection', image)

           if cv2.waitKey(1) & 0xFF == 27:
               break
           
cap.release()
cv2.destroyAllWindows()
print("눈 영역 감지 종료")           