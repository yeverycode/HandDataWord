import numpy as np
import tensorflow as tf
import os

# 모델 로드
model = tf.keras.models.load_model("./model/Sign_ED_best.keras")

# 저장된 데이터 경로
dataset_path = "./Sign_ED/test/testdata"

# 제스처 클래스 리스트 수정
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

# 저장된 데이터 파일 확인
files = [f for f in os.listdir(dataset_path) if f.startswith("test_landmarks_")]
for file in files:
    label = file.replace("test_landmarks_", "").replace(".npy", "")  # 라벨 추출
    test_data = np.load(os.path.join(dataset_path, file))  # 저장된 데이터 로드
    test_data = np.expand_dims(test_data, axis=0)  # 모델 입력 크기 맞추기

    # 모델 예측
    predictions = model.predict(test_data)[0]
    max_index = np.argmax(predictions)
    predicted_label = GESTURES[max_index]  # GESTURES 리스트 사용
    confidence = predictions[max_index] * 100

    # 결과 출력
    print(f"📌 저장된 데이터 라벨: {label}")
    print(f"✅ 모델 예측 결과: {predicted_label} ({confidence:.2f}%)")
    print("-" * 50)