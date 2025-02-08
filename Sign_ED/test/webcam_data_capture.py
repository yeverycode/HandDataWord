import cv2
import numpy as np
import mediapipe as mp
import os
import time

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 저장 경로
dataset_path = "./Sign_ED/test/testdata"
os.makedirs(dataset_path, exist_ok=True)

# 제스처 목록
GESTURES = [
    '안녕,안부',
    '실수',
    '살다,삶,생활',
    '가족,식구,세대,가구',
    '취미',
    '아빠,부,부친,아비,아버지',
    '건강,기력,강건하다,튼튼하다',
    '꿈,포부,꿈꾸다',
    '병원,의원',
    '어머니,모친,어미,엄마',
    '노래,음악,가요',
    '검사',
    '쉬다,휴가,휴게,휴식,휴양',
    '바쁘다,분주하다',
    '여행',
    '주무시다,자다,잠들다,잠자다',
    '고모',
    '치료',
    '자유,임의,마구,마음껏,마음대로,멋대로,제멋대로,함부로',
    '다니다',
    '이기다,승리,승리하다,(경쟁 상대를) 제치다',
    '낫다,치유',
    '성공',
    '죽다,돌아가다,사거,사망,서거,숨지다,죽음,임종',
    '오빠,오라버니',
    '누나,누님',
    '머무르다,존재,체류,계시다,묵다',
    '형,형님',
    '남편,배우자,서방',
    '축구,차다',
    '실패',
    '입원',
    '양치질,양치',
    '아들',
    '형제',
    '몸살',
    '약',
    '결혼,혼인,화혼',
    '남동생',
    '감기',
    '안과',
    '습관,버릇',
    '상하다,다치다,부상,상처,손상',
    '수술',
    '동생',
    '모자(관계)',
    '시동생',
    '편찮다,아프다',
    '신기록',
    '할머니,조모'
]

# 저장할 제스처 선택
print("📝 저장 가능한 제스처 목록:")
for idx, gesture in enumerate(GESTURES, 1):
    print(f"{idx}. {gesture}")

selected_indices = input("\n👉 저장할 제스처의 번호를 입력하세요 (쉼표로 구분, 예: 1,2,3): ").split(',')
selected_indices = [int(idx.strip()) - 1 for idx in selected_indices]  # 1부터 시작하는 번호를 0부터 시작하는 인덱스로 변환
manual_labels = [GESTURES[idx] for idx in selected_indices]

# 웹캠 열기
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

for manual_label in manual_labels:
    sequence = []  # 라벨당 한 개의 npy 파일을 저장하기 위해 초기화
    print(f"📸 현재 촬영 중: {manual_label} (손을 올리고 기다려 주세요...)")

    start_time = time.time()
    while time.time() - start_time < 3:  # 3초 동안 촬영
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # 손 랜드마크 그리기
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 랜드마크 데이터 수집
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                sequence.append(landmarks)

        else:
            sequence.append([0] * 63)  # 손이 감지되지 않을 경우 0으로 패딩

        # 화면에 표시
        cv2.putText(frame, f"Label: {manual_label} (종료: Q)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Webcam Feed", frame)

        # 종료키 (Q 누르면 종료)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("🚪 프로그램 종료!")
            exit()

    # 데이터 저장
    np.save(os.path.join(dataset_path, f"test_landmarks_{manual_label}.npy"), np.array(sequence))
    np.save(os.path.join(dataset_path, f"test_label_{manual_label}.npy"), np.array([manual_label]))
    print(f"✅ {manual_label} 데이터 저장 완료!")

    # 다음 촬영을 위한 3초 대기
    print("⏳ 다음 제스처로 이동 중... (손을 내렸다가 다시 올려주세요)")
    time.sleep(3)

cap.release()
cv2.destroyAllWindows()
print("🚀 모든 데이터 저장 완료!")
