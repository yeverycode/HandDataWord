import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 제스처 정의
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

# 데이터 경로 및 설정 
dataset_path = "./sign_language_data"
max_frames = 30
feature_size = 63

def normalize_sequence_length(data, target_length, feature_size):
   """데이터의 길이를 target_length로 맞춤 (짧으면 패딩, 길면 자름)"""
   if len(data) < target_length:
       padding = np.zeros((target_length - len(data), feature_size), dtype='float32')
       return np.vstack((data, padding)).astype('float32')
   elif len(data) > target_length:
       return data[:target_length].astype('float32')
   return data.astype('float32')

def process_gesture_data(gestures, dataset_path, max_frames, feature_size):
   data = []
   labels = []
   
   for idx, gesture in enumerate(gestures):
       folder_path = os.path.join(dataset_path, gesture)
       if not os.path.exists(folder_path):
           print(f"문제 있음: {gesture} 폴더가 없습니다.")
           continue

       for file_name in os.listdir(folder_path):
           if file_name.endswith(".npy"):
               file_path = os.path.join(folder_path, file_name)
               sequence_data = np.load(file_path, allow_pickle=True)

               if not isinstance(sequence_data, np.ndarray):
                   print(f"오류: {file_name} 데이터 타입이 ndarray가 아님")
                   continue

               # 데이터 길이 정규화
               normalized_data = normalize_sequence_length(sequence_data, max_frames, feature_size)
               normalized_data = normalized_data.astype('float32')

               # 원본 데이터 추가
               data.append(normalized_data)
               labels.append(idx)  # gesture 문자열 대신 인덱스 사용

               # 좌우반전 데이터 추가  
               flipped_data = normalized_data.copy()
               flipped_data[:, 0] = -flipped_data[:, 0]
               data.append(flipped_data)
               labels.append(idx)  # gesture 문자열 대신 인덱스 사용

   return np.array(data, dtype='float32'), np.array(labels)

def save_preprocessed_data(x_data, y_data, save_dir="./dataset"):
   if not os.path.exists(save_dir):
       os.makedirs(save_dir)
   
   y_data = to_categorical(y_data, num_classes=len(GESTURES))
   x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2020)

   np.save(os.path.join(save_dir, 'x_train.npy'), x_train)
   np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
   np.save(os.path.join(save_dir, 'x_val.npy'), x_val)
   np.save(os.path.join(save_dir, 'y_val.npy'), y_val)

   print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
   print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")

# 실행
x_data, labels = process_gesture_data(GESTURES, dataset_path, max_frames, feature_size)
save_preprocessed_data(x_data, labels)