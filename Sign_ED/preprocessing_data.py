import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 제스처 정의
actions = [
        '안녕,안부', '실수', '살다,삶,생활', '취미',
        '아빠,부,부친,아비,아버지', '건강,기력,강건하다,튼튼하다', '쉬다,휴가,휴게,휴식,휴양',
        '고모', '다니다', '죽다,돌아가다,사거,사망,서거,숨지다,죽음,임종', '남편,배우자,서방',
        '몸살', '결혼,혼인,화혼', '노래,음악,가요', '동생', '모자(관계)', '신기록'
    ]

# 데이터 경로 및 설정
dataset_path = "./dataset"
max_frames = 30  # 모든 시퀀스의 프레임 길이를 30으로 고정
feature_size = 63  # 각 프레임의 특징 개수

def normalize_sequence_length(data, target_length, feature_size):
    """데이터의 길이를 target_length로 맞추고, 필요한 경우 feature size도 조정"""
    data = np.array(data)
    
    # feature size가 126인 경우 63으로 줄임 (절반으로)
    if data.shape[1] == 126:
        data = data[:, :63]  # 첫 63개의 특징만 사용
    elif data.shape[1] != 63:
        raise ValueError(f"지원하지 않는 특징 크기: {data.shape[1]}")
    
    # 프레임 수 조정
    if len(data) < target_length:
        padding = np.zeros((target_length - len(data), feature_size), dtype='float32')
        return np.vstack((data, padding))
    elif len(data) > target_length:
        return data[:target_length]
    
    return data.astype('float32')

def process_gesture_data(actions, dataset_path, max_frames, feature_size):
    data = []
    labels = []
    
    total_gestures = len(actions)
    
    for gesture_idx, gesture in enumerate(actions):
        print(f"\n처리 중: {gesture} ({gesture_idx + 1}/{total_gestures})")
        folder_path = os.path.join(dataset_path, gesture)
        
        if not os.path.exists(folder_path):
            print(f"경고: {gesture} 폴더가 없습니다.")
            continue

        files_processed = 0
        files_skipped = 0
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".npy"):
                try:
                    file_path = os.path.join(folder_path, file_name)
                    sequence_data = np.load(file_path, allow_pickle=True)
                    
                    normalized_data = normalize_sequence_length(sequence_data, max_frames, feature_size)
                    
                    # 원본 데이터 추가
                    data.append(normalized_data)
                    labels.append(gesture_idx)
                    
                    # 좌우반전 데이터 추가
                    flipped_data = normalized_data.copy()
                    flipped_data[:, 0] = -flipped_data[:, 0]
                    data.append(flipped_data)
                    labels.append(gesture_idx)
                    
                    files_processed += 1
                    
                except Exception as e:
                    print(f"경고: {file_name} 처리 중 오류 발생: {str(e)}")
                    files_skipped += 1
                    continue
        
        print(f"완료: {files_processed}개 파일 처리됨, {files_skipped}개 파일 건너뜀")

    if not data:
        raise ValueError("처리된 데이터가 없습니다!")
        
    data = np.array(data, dtype='float32')
    labels = np.array(labels)
    
    print(f"\n최종 데이터 shape: {data.shape}")
    print(f"최종 레이블 shape: {labels.shape}")
    
    return data, labels

def save_preprocessed_data(x_data, y_data, save_dir="./dataset"):
    """전처리된 데이터를 저장하는 함수"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 데이터 분할 전 shape 출력
    print(f"전체 데이터 shape - x_data: {x_data.shape}, y_data: {y_data.shape}")
    
    # 레이블 원-핫 인코딩
    y_data = to_categorical(y_data, num_classes=len(actions))
    
    # 학습/검증 데이터 분할
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, 
        test_size=0.2, 
        random_state=2020
    )

    # 저장 전 각 데이터셋의 shape 출력
    print(f"\n분할된 데이터 shape:")
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_val: {x_val.shape}, y_val: {y_val.shape}")

    # 데이터 저장
    np.save(os.path.join(save_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'x_val.npy'), x_val)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)

    print("\n데이터 저장 완료!")

if __name__ == "__main__":
    try:
        print("데이터 처리 시작...")
        x_data, labels = process_gesture_data(actions, dataset_path, max_frames, feature_size)
        print("\n데이터 처리 완료, 저장 시작...")
        save_preprocessed_data(x_data, labels)
        print("전체 프로세스 완료!")
    except Exception as e:
        print(f"처리 중 오류 발생: {str(e)}")