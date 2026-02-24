# blink_detection.py
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh

# 눈 랜드마크 인덱스 (간소화 버전 - 6개 포인트)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

def calculate_ear(eye_points):
    """
    Eye Aspect Ratio (EAR) 계산
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    p1, p4: 눈의 좌우 끝점
    p2, p3, p5, p6: 눈의 상하 포인트
    """
    # 수직 거리
    A = distance.euclidean(eye_points[1], eye_points[5]) 
    B = distance.euclidean(eye_points[2], eye_points[4])
    
    # 수평 거리
    C = distance.euclidean(eye_points[0], eye_points[3])
    
    # EAR 계산
    ear = (A + B) / (2.0 * C)
    return ear

def get_eye_points(landmarks, indices, w, h):
    """랜드마크에서 눈 좌표 추출"""
    points = []
    for idx in indices:
        landmark = landmarks[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append([x, y])
    return np.array(points)

# EAR 임계값
EAR_THRESHOLD = 0.25  # 이 값보다 작으면 눈 감김
CONSECUTIVE_FRAMES = 2  # 연속 프레임 수

# 카운터
blink_counter = 0 # 눈 감은 프레임 카운터
total_blinks = 0 # 총 깜빡인 횟수

print("눈 감김 감지 시작...")
print(f"EAR 임계값: {EAR_THRESHOLD}")
print("ESC 키로 종료")

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
        
        h, w = frame.shape[:2]
        
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
                
                # 왼쪽 눈 EAR 계산
                left_eye = get_eye_points(landmarks, LEFT_EYE_INDICES, w, h)
                left_ear = calculate_ear(left_eye)
                
                # 오른쪽 눈 EAR 계산
                right_eye = get_eye_points(landmarks, RIGHT_EYE_INDICES, w, h)
                right_ear = calculate_ear(right_eye)
                
                # 평균 EAR
                avg_ear = (left_ear + right_ear) / 2.0
                
                # 눈 윤곽 그리기
                cv2.polylines(image, [left_eye], True, (0, 255, 0), 1)
                cv2.polylines(image, [right_eye], True, (255, 0, 0), 1)
                
                # 눈 감김 감지
                if avg_ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= CONSECUTIVE_FRAMES:
                        total_blinks += 1
                        print(f"깜빡임 감지! 총 {total_blinks}번")
                    blink_counter = 0
                
                # EAR 값 표시
                cv2.putText(image, f"EAR: {avg_ear:.2f}", 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                # 깜빡임 횟수 표시
                cv2.putText(image, f"Blinks: {total_blinks}", 
                           (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2) # 글자 크기, 색깔, 글자 두께
                
                # 눈 감김 상태 표시
                status = "CLOSED" if avg_ear < EAR_THRESHOLD else "OPEN"
                color = (0, 0, 255) if status == "CLOSED" else (0, 255, 0)
                cv2.putText(image, f"Status: {status}", 
                           (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
        
        cv2.imshow('Blink Detection', image)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
print(f"\n총 깜빡임 횟수: {total_blinks}")
print("눈 감김 감지 종료")