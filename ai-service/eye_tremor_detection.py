# eye_tremor_detection.py
import cv2
import mediapipe as mp
import numpy as np
from collections import deque # 양쪽 끝에서 추가/삭제 가능한 큐

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 눈 영역 랜드마크
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_eye_center(landmarks, eye_indices, w, h):
    """눈 중심점 계산"""
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])
    
    # 중심점 = 모든 점의 평균
    center_x = int(np.mean([p[0] for p in points]))
    center_y = int(np.mean([p[1] for p in points]))
    
    return (center_x, center_y)

def calculate_tremor(position_history):
    """
    눈 떨림 계산
    
    떨림 = 위치 변화량의 표준편차
    정상: 작은 움직임
    떨림: 큰 움직임
    """
    if len(position_history) < 2:
        return 0.0
    
    # 위치 변화량 계산
    movements = []
    for i in range(1, len(position_history)):
        prev_x, prev_y = position_history[i-1] # 이전 위치 (언패킹)
        curr_x, curr_y = position_history[i] # 현재 위치
        
        # 유클리드 거리(피타고라스 정리)
        distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        movements.append(distance)
    
    # 표준편차 = 떨림 정도
    tremor_score = np.std(movements)
    
    return tremor_score

# 떨림 임계값
TREMOR_THRESHOLD = 2.0  # 표준편차 2.0 이상이면 떨림

# 위치 기록 (최근 30프레임)
left_eye_history = deque(maxlen=30)
right_eye_history = deque(maxlen=30)

print("눈 떨림 감지 시작...")
print(f"떨림 임계값: {TREMOR_THRESHOLD}")
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
                
                # 왼쪽 눈 중심
                left_center = get_eye_center(landmarks, LEFT_EYE, w, h)
                left_eye_history.append(left_center)
                
                # 오른쪽 눈 중심
                right_center = get_eye_center(landmarks, RIGHT_EYE, w, h)
                right_eye_history.append(right_center)
                
                # 떨림 계산
                left_tremor = calculate_tremor(left_eye_history)
                right_tremor = calculate_tremor(right_eye_history)
                avg_tremor = (left_tremor + right_tremor) / 2.0
                
                # 떨림 판정
                is_tremor = avg_tremor > TREMOR_THRESHOLD
                
                # 눈 중심점 표시
                color = (0, 0, 255) if is_tremor else (0, 255, 0)
                cv2.circle(image, left_center, 5, color, -1)
                cv2.circle(image, right_center, 5, color, -1)
                
                # 움직임 경로 표시 (최근 10개 점)
                if len(left_eye_history) > 1: # 위치가 2개 이상 있어야 선 그리기 가능
                    points = list(left_eye_history)[-10:] # 리스트로 변환
                    for i in range(1, len(points)): # 1부터 마지막까지
                        cv2.line(image, points[i-1], points[i], (255, 255, 0), 1) # 선 그리기
                
                if len(right_eye_history) > 1:
                    points = list(right_eye_history)[-10:]
                    for i in range(1, len(points)):
                        cv2.line(image, points[i-1], points[i], (255, 255, 0), 1)
                
                # 정보 표시
                cv2.putText(image, f"Tremor: {avg_tremor:.3f}", 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                status = "TREMOR!" if is_tremor else "Stable"
                status_color = (0, 0, 255) if is_tremor else (0, 255, 0)
                cv2.putText(image, f"Status: {status}", 
                           (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, status_color, 2)
        
        cv2.imshow('Eye Tremor Detection', image)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("눈 떨림 감지 종료")