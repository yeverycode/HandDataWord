import numpy as np
import tensorflow as tf
import os

# 모델 로드
model = tf.keras.models.load_model("./model/Sign_ED_best.keras")

# 저장된 데이터 경로
dataset_path = "./Sign_ED/test/testdata"

# 새로운 제스처 클래스 리스트
actions = [
        '안녕,안부', '실수', '살다,삶,생활', '취미',
        '아빠,부,부친,아비,아버지', '건강,기력,강건하다,튼튼하다', '쉬다,휴가,휴게,휴식,휴양',
        '고모', '다니다', '죽다,돌아가다,사거,사망,서거,숨지다,죽음,임종',
        '누나,누님', '남편,배우자,서방', '양치질,양치',
        '몸살', '결혼,혼인,화혼', '남동생', '상하다,다치다,부상,상처,손상', '동생', '모자(관계)', '신기록'
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
    predicted_label = actions[max_index]
    confidence = predictions[max_index] * 100

    # 결과 출력
    print(f"📌 저장된 데이터 라벨: {label}")
    print(f"✅ 모델 예측 결과: {predicted_label} ({confidence:.2f}%)")
    print("-" * 50)