import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import pickle

# history 불러오기
with open('./model/history.pkl', 'rb') as f:
    history = pickle.load(f)

# F1 Score 함수 정의
def metric_F1score(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1score = 2 * precision * recall / (precision + recall)
    return F1score

# 모델 로드
model = load_model('./model/Sign_ED_best.keras', custom_objects={'metric_F1score': metric_F1score})

# 데이터 로드
x_train = np.load('./dataset/x_train.npy')
x_val = np.load('./dataset/x_val.npy')
y_train = np.load('./dataset/y_train.npy')
y_val = np.load('./dataset/y_val.npy')

# 손실 함수 및 정확도 그래프 그리기
fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history['loss'], 'y', label='train loss')
loss_ax.plot(history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history['acc'], 'b', label='train acc')
acc_ax.plot(history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper right')

plt.title('Model Loss and Accuracy')
plt.show()

# validation 데이터에 대한 평가
val_scores = model.evaluate(x_val, y_val, verbose=0)
y_pred = model.predict(x_val)
val_f1score = metric_F1score(y_val, y_pred)

print(f"Validation Loss: {val_scores[0]:.4f}")
print(f"Validation Accuracy: {val_scores[1]*100:.2f}%")
print(f"Validation F1 Score: {val_f1score:.4f}")

# validation 결과를 포함한 최종 accuracy 그래프
plt.figure(figsize=(16, 10))
plt.plot(history['acc'], 'b', label='train acc')
plt.plot(history['val_acc'], 'g', label='val acc')
plt.axhline(y=val_scores[1], color='r', linestyle='--', 
            label=f'final val acc: {val_scores[1]*100:.2f}%')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Model Accuracy with Validation Results')
plt.legend(loc='lower right')
plt.show()

# Loss 추이 그래프
plt.figure(figsize=(16, 10))
plt.plot(history['loss'], 'b', label='train loss')
plt.plot(history['val_loss'], 'g', label='val loss')
plt.axhline(y=val_scores[0], color='r', linestyle='--',
            label=f'final val loss: {val_scores[0]:.4f}')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Model Loss with Validation Results')
plt.legend(loc='upper right')
plt.show()