# face_flush_detection.py
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 얼굴 영역 랜드마크 (볼, 이마, 코)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def get_face_region(image, landmarks, face_indices, w, h):
    """얼굴 영역 이미지 추출"""
    points = []
    for idx in face_indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    
    # 얼굴 영역 바운딩 박스
    x, y, w_box, h_box = cv2.boundingRect(points)
    
    # 여유 공간 추가
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w_box = min(w - x, w_box + 2 * margin)
    h_box = min(h - y, h_box + 2 * margin)
    
    # 얼굴 영역 추출
    face_roi = image[y:y+h_box, x:x+w_box]
    
    return face_roi, (x, y, w_box, h_box), points

def calculate_flush(face_roi):
    """
    얼굴 홍조 계산
    
    홍조 = 빨간색 강도
    정상: 낮은 빨강
    홍조: 높은 빨강
    """
    if face_roi.size == 0:
        return 0.0
    
    # BGR을 RGB로 변환
    rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    
    # 각 채널 평균
    r_mean = np.mean(rgb[:, :, 0])
    g_mean = np.mean(rgb[:, :, 1])
    b_mean = np.mean(rgb[:, :, 2])
    
    # 빨간색 강도 계산
    # 홍조는 빨간색이 초록색보다 월등히 높음
    if g_mean == 0:
        return 0.0
    
    flush_ratio = r_mean / g_mean
    
    return flush_ratio

# 홍조 임계값
FLUSH_THRESHOLD = 1.3  # R/G 비율이 1.3 이상이면 홍조

print("얼굴 홍조 감지 시작...")
print(f"홍조 임계값: {FLUSH_THRESHOLD}")
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
                
                # 얼굴 영역 분석
                face_roi, face_box, face_points = get_face_region(
                    image, landmarks, FACE_OVAL, w, h
                )
                flush_ratio = calculate_flush(face_roi)
                
                # 홍조 판정
                is_flushed = flush_ratio > FLUSH_THRESHOLD
                
                # 얼굴 윤곽 표시
                color = (0, 0, 255) if is_flushed else (0, 255, 0)
                cv2.polylines(image, [face_points], True, color, 2)
                
                # 정보 표시
                cv2.putText(image, f"Flush: {flush_ratio:.3f}", 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                status = "FLUSHED!" if is_flushed else "Normal"
                status_color = (0, 0, 255) if is_flushed else (0, 255, 0)
                cv2.putText(image, f"Status: {status}", 
                           (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, status_color, 2)
        
        cv2.imshow('Face Flush Detection', image)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("얼굴 홍조 감지 종료")