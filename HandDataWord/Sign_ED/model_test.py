import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import pickle

# history 불러오기
with open('./model/history.pkl', 'rb') as f:
    history = pickle.load(f)

# 그래프 출력 등 활용 가능
print(history['accuracy'])  # 학습 정확도 기록

# F1 Score 함수 정의
def metric_F1score(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP + 1e-7)  # epsilon 추가
    recall = TP / (TP + FN + 1e-7)     # epsilon 추가
    F1score = 2 * precision * recall / (precision + recall + 1e-7)  # epsilon 추가
    return F1score

# 모델 로드 - .keras 확장자로 변경
model = load_model('./model/Sign_ED.keras', custom_objects={'metric_F1score': metric_F1score})

# Test data load 
x_test = np.load('./dataset/x_test.npy')
y_test = np.load('./dataset/y_test.npy')

# 그래프 스타일 설정
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12

# 손실 함수 및 정확도 그래프
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 20))

# 손실 그래프
ax1.plot(history['loss'], label='Train Loss', color='#FF9999')
ax1.plot(history['val_loss'], label='Validation Loss', color='#FF0000')
ax1.set_title('Model Loss', pad=20)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# 정확도 그래프
ax2.plot(history['acc'], label='Train Accuracy', color='#99FF99')
ax2.plot(history['val_acc'], label='Validation Accuracy', color='#009900')
ax2.set_title('Model Accuracy', pad=20)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 테스트 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)
test_f1score = metric_F1score(y_test, y_pred)

# 결과 출력 (보기 좋게 포맷팅)
print("\n" + "="*50)
print("📊 테스트 결과")
print("="*50)
print(f"✓ 테스트 정확도: {test_acc*100:.2f}%")
print(f"✓ 테스트 F1 Score: {test_f1score:.4f}")
print(f"✓ 테스트 손실: {test_loss:.4f}")
print("="*50)

# 전체 성능 요약 그래프
plt.figure(figsize=(16, 10))
epochs = range(1, len(history['acc']) + 1)

plt.plot(epochs, history['acc'], 'b-', label='Training Acc')
plt.plot(epochs, history['val_acc'], 'g-', label='Validation Acc')
plt.axhline(y=test_acc, color='r', linestyle='--', 
           label=f'Test Acc: {test_acc*100:.2f}%')

plt.title('Training, Validation, and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
