import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ✅ 한글 폰트 설정 (Windows 환경)
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# 🔹 F1 Score metric 정의
def metric_F1score(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    F1score = 2 * precision * recall / (precision + recall + 1e-7)
    return F1score

# 🔹 모델 로드 함수
def load_sign_model(model_path):
    return load_model(model_path, custom_objects={'metric_F1score': metric_F1score})

# 🔹 모델 평가 함수
def evaluate_model(model, x_val, y_val, actions):
    # 모델 입력 크기 확인
    print("\n=== 모델 입력 형태 ===")
    print(f"모델 기대 입력: {model.input_shape}")
    print(f"데이터 입력: {x_val.shape}")

    # 입력 크기 조정 (30 프레임을 사용하는 모델)
    if x_val.shape[1] != 30:
        print(f"⚠ 입력 크기 조정: {x_val.shape} → ({x_val.shape[0]}, 30, {x_val.shape[2]})")
        x_val = x_val[:, :30, :]

    # 예측 수행
    y_pred = model.predict(x_val)

    # 원-핫 인코딩된 결과를 클래스 인덱스로 변환
    y_val_classes = np.argmax(y_val, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification Report 생성
    report = classification_report(y_val_classes, y_pred_classes, target_names=actions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    print("\n=== Classification Report ===")
    print(df_report.round(3))

    # Confusion Matrix 시각화
    cm = confusion_matrix(y_val_classes, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
    plt.title('Confusion Matrix')
    plt.xlabel('예측값')
    plt.ylabel('실제값')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return df_report

# 🔹 특정 클래스(단어)의 예측 성능 확인
def analyze_class_performance(y_val_classes, y_pred_classes, actions, target_word):
    target_index = actions.index(target_word)

    report = classification_report(y_val_classes, y_pred_classes, target_names=actions, output_dict=True)
    print(f"\n=== 성능 확인: {target_word} ===")
    print(f"Precision: {report[target_word]['precision']:.3f}")
    print(f"Recall: {report[target_word]['recall']:.3f}")
    print(f"F1-score: {report[target_word]['f1-score']:.3f}")
    print(f"Support(샘플 수): {report[target_word]['support']}")

    # 오분류된 경우 찾기
    misclassified_samples = [(actions[y_val_classes[i]], actions[y_pred_classes[i]]) for i in range(len(y_val_classes))
                             if y_val_classes[i] == target_index and y_pred_classes[i] != target_index]
    
    if misclassified_samples:
        df_misclassified = pd.DataFrame(misclassified_samples, columns=['실제 클래스', '예측 클래스'])
        print("\n=== 오분류 샘플 ===")
        print(df_misclassified)

    # Confusion Matrix에서 특정 클래스 시각화
    cm = confusion_matrix(y_val_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm[:, target_index].reshape(-1, 1), annot=True, fmt='d', cmap='Blues',
                yticklabels=actions, xticklabels=[target_word])
    plt.title(f"Confusion Matrix - '{target_word}'")
    plt.ylabel("실제 클래스")
    plt.xlabel("예측된 클래스")
    plt.show()

# 🔹 데이터 로드 및 평가 실행
if __name__ == "__main__":
    # ✅ 모델 로드
    model = load_sign_model('./model/Sign_ED_best.keras')

    # ✅ 데이터 로드
    x_val = np.load('./dataset/x_val.npy')
    y_val = np.load('./dataset/y_val.npy')

    # 제스처 정의
    actions = [
        '안녕,안부', '실수', '살다,삶,생활', '취미',
        '아빠,부,부친,아비,아버지', '건강,기력,강건하다,튼튼하다', '쉬다,휴가,휴게,휴식,휴양',
        '고모', '다니다', '죽다,돌아가다,사거,사망,서거,숨지다,죽음,임종',
        '누나,누님', '남편,배우자,서방', '양치질,양치',
        '몸살', '결혼,혼인,화혼', '남동생', '상하다,다치다,부상,상처,손상', '동생', '모자(관계)', '신기록'
    ]

    # ✅ 모델 평가 수행
    report = evaluate_model(model, x_val, y_val, actions)

    # ✅ 특정 단어 분석 (예: '축구,차다')
    analyze_class_performance(np.argmax(y_val, axis=1), np.argmax(model.predict(x_val), axis=1), actions, '축구,차다')
