# test_camera.py
import cv2 # OpenCV 라이브러리 (컴퓨터 비전)

print("웹캠 테스트 시작...")

# 웹캠 연결(0번 = 기본 웹캠)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다!")
    exit()

print("✅ 웹캠 연결 성공!")
print("ESC 키를 누르면 종료됩니다.")  

while True:
    # 프레임 읽기
    ret, frame = cap.read() # 웹캠에서 1장의 사진(프레임) 가져오기

    if not ret:
        print("프레임을 읽을 수 없습니다")
        break

    # 화면에 표시
    cv2.imshow('Webcam Test', frame) # cv2.imshow(창이름, 이미지)

    # ESC 키 (27)로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 정리
cap.release() # 웹캠 해제
cv2.destroyAllWindows() # 모든 창 닫기
print("웹캠 테스트 종료")    