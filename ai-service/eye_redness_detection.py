# eye_redness_detection.py
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 눈 영역 랜드마크
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_eye_region(image, landmarks, eye_indices, w, h):
    """눈 영역 이미지 추출"""
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])

    points = np.array(points, dtype=np.int32)
    
    # 눈 영역 바운딩 박스
    x, y, w_box, h_box = cv2.boundingRect(points)

    # 여유 공간 추가
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w_box = min(w - x, w_box + 2 * margin)
    h_box = min(h - y, h_box + 2 * margin)    

    # 눈 영역 추출
    eye_roi = image[y:y+h_box, x:x+w_box]

    return eye_roi, (x, y, w_box, h_box)

def calculate_redness(eye_roi):
    """
    눈 충혈도 계산

    빨간색 비율 = R / (R + G + B)
    충혈되면 빨간색 성분 증가
    """
    if eye_roi.size == 0:
        return 0.0
    
    # BGR을 RGB로 변환
    rgb = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2RGB)

    # 각 채널 평균
    r_mean = np.mean(rgb[:, :, 0])
    g_mean = np.mean(rgb[:, :, 1])
    b_mean = np.mean(rgb[:, :, 2])

    # 빨간색 비율 계산
    total = r_mean + g_mean + b_mean
    if total == 0:
        return 0.0
    
    redness_ratio = r_mean / total

    return redness_ratio

# 충혈 임계값
REDNESS_THRESHOLD = 0.40 # 40% 이상이면 충혈

print("눈 충혈 감지 시작...")
print(f"충혈 임계값: {REDNESS_THRESHOLD}")
print("ESC 키로 종료")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 웹캠 열기

with mp_face_mesh.FaceMesh( # FaceMesh() = 얼굴 인식기 생성
    max_num_faces=1, # 최대 1개 얼굴만 찾기
    refine_landmarks=True, # 눈, 입술 더 정밀하게
    min_detection_confidence=0.5, # 50% 이상 확신할 때만 "얼굴이다!" 인정
    min_tracking_confidence=0.5 # 추적 중인 얼굴, 50% 이상 확신
) as face_mesh: # 사용 끝나면 자동 정리

    while cap.isOpened():
       ret, frame = cap.read() # 웹캠에서 1장의 사진 가져오기
       if not ret:
           break
       
       h, w = frame.shape[:2] # 앞의 2개만 가져오기

       # RGB 변환
       image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       image.flags.writeable = False # 이미지 수정 금지

       # 얼굴 감지
       results = face_mesh.process(image)

       # BGR 변환
       image.flags.writeable = True
       image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

       if results.multi_face_landmarks:
           for face_landmarks in results.multi_face_landmarks: # 찾은 얼굴마다 반복
               landmarks = face_landmarks.landmark

               # 왼쪽 눈 분석
               left_eye_roi, left_box = get_eye_region(
                   image, landmarks, LEFT_EYE, w, h
               )
               left_redness = calculate_redness(left_eye_roi)

               # 오른쪽 눈 분석
               right_eye_roi, right_box = get_eye_region(
                   image, landmarks, RIGHT_EYE, w, h
               )
               right_redness = calculate_redness(right_eye_roi)

               # 평균 충혈도
               avg_redness = (left_redness + right_redness) / 2.0

               # 충혈 판정
               is_red = avg_redness > REDNESS_THRESHOLD

               # 눈 영역 표시
               color = (0, 0, 255) if is_red else (0, 255, 0)

               # 왼쪽 눈 박스
               x, y, w_box, h_box = left_box
               cv2.rectangle(image, (x, y), (x+w_box, y+h_box), color, 2)

               # 오른쪽 눈 박스
               x, y, w_box, h_box = right_box
               cv2.rectangle(image, (x, y), (x+w_box, y+h_box), color, 2)

               # 정보 표시
               cv2.putText(image, f"Redness: {avg_redness:.3f}",
                           (10, 30), # 텍스트 위치 (x=10, y=30)
                           cv2.FONT_HERSHEY_SIMPLEX, # 폰트 종류
                           0.7, (0, 255, 0), 2) # 0.7:글자 크기, 2:글자 두께
               
               status = "RED EYES!" if is_red else "Normal"
               status_color = (0, 0, 255) if is_red else (0, 255, 0)
               cv2.putText(image, f"Status: {status}",
                           (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, status_color, 2)
               
               cv2.imshow('Eye Redness Detection', image)

               if cv2.waitKey(1) & 0xFF == 27:
                   break
               
cap.release() # 웹캠 사용 종료
cv2.destroyAllWindows() # OpenCV로 연 모든 창 닫기 
print("눈 충혈 감지 종료")               
