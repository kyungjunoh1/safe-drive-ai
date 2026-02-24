# face_mesh_basic.py
import cv2
import mediapipe as mp
import os
import sys

# MediaPipe 경로 문제 해결
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils  
mp_drawing_styles = mp.solutions.drawing_styles

print("MediaPipe Face Mesh 시작...")

# 웹캠 연결
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다!")
    exit()

print("✅ 웹캠 연결 성공!")

# Face Mesh 객체 생성 (with 문 밖에서)
try:
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✅ FaceMesh 초기화 성공!")
except Exception as e:
    print(f"❌ FaceMesh 초기화 실패: {e}")
    cap.release()
    exit()

print("ESC 키를 누르면 종료됩니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # BGR을 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Face Mesh 감지
    results = face_mesh.process(image)
    
    # 다시 BGR로 변환
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 얼굴이 감지되면 랜드마크 그리기
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 전체 얼굴 메쉬 그리기
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # 눈 윤곽 그리기
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
    
    # 화면 표시
    cv2.imshow('MediaPipe Face Mesh', image)
    
    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 정리
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
print("Face Mesh 종료")