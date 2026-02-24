# alcohol_probability.py
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 랜드마크 인덱스
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, # 얼굴 홍조에 사용
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# === 이전 함수들 재사용 ===

def get_eye_region(image, landmarks, eye_indices, w, h):
    """눈 영역 추출"""
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    x, y, w_box, h_box = cv2.boundingRect(points)
    
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w_box = min(w - x, w_box + 2 * margin)
    h_box = min(h - y, h_box + 2 * margin)
    
    eye_roi = image[y:y+h_box, x:x+w_box]
    return eye_roi, (x, y, w_box, h_box)

def calculate_redness(eye_roi):
    """눈 충혈도 계산"""
    if eye_roi.size == 0:
        return 0.0
    
    rgb = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2RGB)
    r_mean = np.mean(rgb[:, :, 0])
    g_mean = np.mean(rgb[:, :, 1])
    b_mean = np.mean(rgb[:, :, 2])
    
    total = r_mean + g_mean + b_mean
    if total == 0:
        return 0.0
    
    redness_ratio = r_mean / total
    return redness_ratio

def get_face_region(image, landmarks, face_indices, w, h):
    """얼굴 영역 추출"""
    points = []
    for idx in face_indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    x, y, w_box, h_box = cv2.boundingRect(points)
    
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w_box = min(w - x, w_box + 2 * margin)
    h_box = min(h - y, h_box + 2 * margin)
    
    face_roi = image[y:y+h_box, x:x+w_box]
    return face_roi, (x, y, w_box, h_box), points

def calculate_flush(face_roi):
    """얼굴 홍조 계산"""
    if face_roi.size == 0:
        return 0.0
    
    rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    r_mean = np.mean(rgb[:, :, 0])
    g_mean = np.mean(rgb[:, :, 1])
    b_mean = np.mean(rgb[:, :, 2])
    
    if g_mean == 0:
        return 0.0
    
    flush_ratio = r_mean / g_mean
    return flush_ratio

def get_eye_center(landmarks, eye_indices, w, h):
    """눈 중심점 계산"""
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])
    
    center_x = int(np.mean([p[0] for p in points]))
    center_y = int(np.mean([p[1] for p in points]))
    
    return (center_x, center_y)

def calculate_tremor(position_history):
    """눈 떨림 계산"""
    if len(position_history) < 2:
        return 0.0
    
    movements = []
    for i in range(1, len(position_history)):
        prev_x, prev_y = position_history[i-1]
        curr_x, curr_y = position_history[i]
        
        distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        movements.append(distance)
    
    tremor_score = np.std(movements)
    return tremor_score

# === 음주 확률 계산 ===

def calculate_alcohol_probability(redness, flush, tremor):
    """
    음주 확률 계산
    
    3가지 지표를 종합하여 0~100% 확률 계산
    """
    # 각 지표별 점수 (0~100)
    
    # 1. 눈 충혈 점수
    if redness < 0.38:
        redness_score = 0
    elif redness < 0.40:
        redness_score = 30
    elif redness < 0.45:
        redness_score = 60
    else:
        redness_score = 100
    
    # 2. 얼굴 홍조 점수
    if flush < 1.2:
        flush_score = 0
    elif flush < 1.3:
        flush_score = 30
    elif flush < 1.4:
        flush_score = 60
    else:
        flush_score = 100
    
    # 3. 눈 떨림 점수
    if tremor < 1.5:
        tremor_score = 0
    elif tremor < 2.0:
        tremor_score = 30
    elif tremor < 3.0:
        tremor_score = 60
    else:
        tremor_score = 100
    
    # 가중 평균 (눈 충혈 40%, 홍조 40%, 떨림 20%)
    probability = (redness_score * 0.4 + 
                   flush_score * 0.4 + 
                   tremor_score * 0.2)
    
    return probability

def get_risk_level(probability):
    """위험도 분류"""
    if probability < 30:
        return "안전", (0, 255, 0)  # 초록
    elif probability < 60:
        return "주의", (0, 255, 255)  # 노랑
    elif probability < 80:
        return "경고", (0, 165, 255)  # 주황
    else:
        return "위험", (0, 0, 255)  # 빨강

# 위치 기록
left_eye_history = deque(maxlen=30)
right_eye_history = deque(maxlen=30)

print("음주 확률 감지 시작...")
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
                
                # === 1. 눈 충혈 분석 ===
                left_eye_roi, _ = get_eye_region(image, landmarks, LEFT_EYE, w, h)
                right_eye_roi, _ = get_eye_region(image, landmarks, RIGHT_EYE, w, h)
                
                left_redness = calculate_redness(left_eye_roi)
                right_redness = calculate_redness(right_eye_roi)
                avg_redness = (left_redness + right_redness) / 2.0
                
                # === 2. 얼굴 홍조 분석 ===
                face_roi, _, face_points = get_face_region(
                    image, landmarks, FACE_OVAL, w, h
                )
                flush_ratio = calculate_flush(face_roi)
                
                # === 3. 눈 떨림 분석 ===
                left_center = get_eye_center(landmarks, LEFT_EYE, w, h)
                left_eye_history.append(left_center)
                
                right_center = get_eye_center(landmarks, RIGHT_EYE, w, h)
                right_eye_history.append(right_center)
                
                left_tremor = calculate_tremor(left_eye_history)
                right_tremor = calculate_tremor(right_eye_history)
                avg_tremor = (left_tremor + right_tremor) / 2.0
                
                # === 4. 음주 확률 계산 ===
                probability = calculate_alcohol_probability(
                    avg_redness, flush_ratio, avg_tremor
                )
                risk_level, risk_color = get_risk_level(probability)
                
                # === 5. 얼굴 윤곽 표시 ===
                cv2.polylines(image, [face_points], True, risk_color, 2)
                
                # === 6. 정보 표시 ===
                # 개별 지표
                cv2.putText(image, f"Redness: {avg_redness:.3f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
                
                cv2.putText(image, f"Flush: {flush_ratio:.3f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
                
                cv2.putText(image, f"Tremor: {avg_tremor:.3f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
                
                # 음주 확률
                cv2.putText(image, f"Alcohol: {probability:.1f}%", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, risk_color, 3)
                
                # 위험도
                cv2.putText(image, f"Level: {risk_level}", 
                           (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, risk_color, 3)
        
        cv2.imshow('Alcohol Probability Detection', image)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("음주 확률 감지 종료")