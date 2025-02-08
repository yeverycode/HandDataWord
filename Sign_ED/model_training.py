import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, Masking, TimeDistributed, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import pickle

# 제스처 클래스 정의 
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

# 데이터 로드
x_train = np.load('./dataset/x_train.npy')
y_train = np.load('./dataset/y_train.npy')
x_val = np.load('./dataset/x_val.npy')
y_val = np.load('./dataset/y_val.npy')

# sequence_length 적용
sequence_length = 15

# 슬라이딩 윈도우 적용
def sliding_window(data, seq_length):
    new_sequences = []
    for seq in data:
        for i in range(len(seq) - seq_length + 1):
            new_sequences.append(seq[i:i+seq_length])
    return np.array(new_sequences, dtype='float32')

# 학습 데이터 변환
x_train = sliding_window(x_train, sequence_length)
x_val = sliding_window(x_val, sequence_length)

# y_train, y_val 슬라이딩 적용 후 데이터 개수 맞추기
y_train = np.repeat(y_train, x_train.shape[0] // y_train.shape[0], axis=0)
y_val = np.repeat(y_val, x_val.shape[0] // y_val.shape[0], axis=0)

print(f"🔍 x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
print(f"🔍 x_val.shape: {x_val.shape}, y_val.shape: {y_val.shape}")

# F1-score 커스텀 지표 정의
def metric_F1score(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    F1score = 2 * precision * recall / (precision + recall + 1e-7)

    return F1score

# 모델 정의
model = Sequential([
    Masking(mask_value=0.0, input_shape=(x_train.shape[1], x_train.shape[2])),
    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    TimeDistributed(Flatten()),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(GESTURES), activation='softmax')  # GESTURES 길이로 변경
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# 학습 콜백 설정
early_stopping = EarlyStopping(monitor='val_acc', patience=20, mode='max')
callbacks = [
    ModelCheckpoint('./model/Sign_ED_best.keras', monitor='val_acc', save_best_only=True),
    ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10),
    early_stopping
]

# 모델 학습
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks
)

# 모델 저장
model.save("./model/Sign_ED.keras")

# 모델 구조 확인
model.summary()

# 모델 요약 정보를 파일에 저장
with open("./model/model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# 학습 history 저장
with open('./model/history.pkl', 'wb') as f:
    pickle.dump(history.history, f)