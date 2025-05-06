
# 수어 인식 시스템 (Sign Language Recognition System)

본 프로젝트는 수어(한국어 손 모양) 인식을 위한 데이터 수집부터 모델 학습, 테스트, 실시간 인식까지 전 과정을 포함하는 통합 프레임워크입니다.  
MediaPipe를 활용한 데이터 수집, LSTM 기반 학습, 실시간 인식 인터페이스 등이 포함되어 있습니다.

## 주요 기능

### 1. 데이터 수집 (making_dataset.py)
- MediaPipe로 손 관절 포인트 추출
- 자음, 모음, 겹받침 등 다양한 수어 제스처 데이터 수집 가능
- 제스처별 폴더로 자동 정리 및 .npy 파일 저장
- 사용자 입력으로 동적 라벨링 지원

### 2. 데이터 전처리 (preprocessing_data.py)
- 모든 제스처 시퀀스를 고정 프레임(30)으로 정규화
- 데이터 증강(좌우 반전 등) 적용
- 학습/검증 데이터 자동 분할

### 3. 모델 학습 (model_training.py)
- Bidirectional LSTM + Conv1D 모델 구조
- F1-score 기반 성능 평가 및 최적 모델 저장
- 슬라이딩 윈도우 기반 데이터 효율화

### 4. 모델 평가 (model_test.py & simplemodeltest.py)
- 학습 이력 시각화 (Loss, Accuracy, F1-score)
- 저장된 모델을 이용한 빠른 테스트 기능

### 5. 실시간 수어 인식 (model_test_webcam.py)
- PyQt6 기반 사용자 인터페이스 제공
- 퀴즈 모드 및 실시간 피드백 기능 포함
- 제스처 예측 결과와 정확도 실시간 표시

### 6. 테스트 및 디버깅 도구
- 저장된 데이터셋 기반 테스트 (test_saved_data.py)
- 웹캠으로 추가 데이터 수집 가능 (webcam_data_capture.py)

## 프로젝트 구조

```
AI_NEW/
├── dataset/             원본 데이터셋 (이미지, JSON, npy)
├── suyoun_dataset/      커스텀 수어 데이터셋
├── model/               학습된 모델 및 로그
├── Sign_ED/             테스트 및 실시간 인식 스크립트
├── making_dataset.py    데이터 수집
├── preprocessing_data.py 데이터 전처리
├── model_training.py    모델 학습
├── model_test.py        모델 평가
└── README.md
```

## 사용 방법

```
# 데이터 수집
python making_dataset.py

# 데이터 전처리
python preprocessing_data.py

# 모델 학습
python model_training.py

# 모델 평가
python model_test.py

# 실시간 인식
python model_test_webcam.py
```
