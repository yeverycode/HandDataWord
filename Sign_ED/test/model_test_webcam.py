import sys
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import random
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt

# ✅ 모델 로드
model = tf.keras.models.load_model("./model/Sign_ED_best.keras")

# ✅ 수어 단어 리스트
ACTIONS = [
        '안녕,안부', '실수', '살다,삶,생활', '취미',
        '아빠,부,부친,아비,아버지', '건강,기력,강건하다,튼튼하다', '쉬다,휴가,휴게,휴식,휴양',
        '고모', '다니다', '죽다,돌아가다,사거,사망,서거,숨지다,죽음,임종',
        '누나,누님', '남편,배우자,서방', '양치질,양치',
        '몸살', '결혼,혼인,화혼', '남동생', '상하다,다치다,부상,상처,손상', '동생', '모자(관계)', '신기록'
    ]

# ✅ MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()

        # ✅ UI 설정
        self.setWindowTitle("✨ Sign Language Quiz ✨")
        self.setStyleSheet("background-color: #eeeael;")  
        self.showFullScreen()  # 전체 화면 설정

        # ✅ 상단 제목
        self.title_label = QLabel("✨ Sign Language Quiz ✨", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 45px; color: #flac85; font-weight: bold; font-family: 'Times New Roman';")

        # ✅ 웹캠 화면 표시
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 3px solid #368f5f; background-color: #ffffff;")
        self.video_label.setFixedSize(640, 480)

        # ✅ 문제 표시
        self.quiz_label = QLabel("", self)
        self.quiz_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.quiz_label.setSizePolicy(QLabel().sizePolicy())  
        self.quiz_label.setStyleSheet("font-size: 35px; color: #368f5f; font-family: 'Times New Roman'; font-weight: bold; margin-bottom: 30px; text-align: left;")

        # ✅ 예측된 단어 & 정확도 표시
        self.prediction_label = QLabel("", self)
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.quiz_label.setSizePolicy(QLabel().sizePolicy())  
        self.prediction_label.setStyleSheet("font-size: 30px; color: #368f5f; font-family: 'Times New Roman'; font-weight: bold; margin-bottom: 30px; text-align: left;")

        # ✅ 결과 메시지
        self.result_label = QLabel("🖐 Try to follow the sign shown!", self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.quiz_label.setSizePolicy(QLabel().sizePolicy())  
        self.result_label.setStyleSheet("font-size: 30px; color: #368f5f; font-weight: bold; font-family: 'Times New Roman'; font-weight: bold; margin-bottom: 30px; text-align: left;")

        # ✅ 남은 시간 표시
        self.timer_label = QLabel("", self)
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.quiz_label.setSizePolicy(QLabel().sizePolicy())  
        self.timer_label.setStyleSheet("font-size: 27px; color: #cc7fb5; font-family: 'Times New Roman'; font-weight: bold; text-align: left;")

        # ✅ 레이아웃 설정
        main_layout = QVBoxLayout()

        # 상단 제목 중앙 정렬
        main_layout.addSpacing(50)
        main_layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        main_layout.addSpacing(25)

        # 가운데 웹캠과 텍스트를 나란히 배치
        center_layout = QHBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 왼쪽에 텍스트 정보 배치
        text_layout = QVBoxLayout()
        text_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        quiz_layout = QHBoxLayout()
        quiz_layout.addWidget(self.quiz_label, alignment=Qt.AlignmentFlag.AlignLeft)
        text_layout.addLayout(quiz_layout)

        prediction_layout = QHBoxLayout()
        prediction_layout.addWidget(self.prediction_label, alignment=Qt.AlignmentFlag.AlignLeft)
        text_layout.addLayout(prediction_layout)

        result_layout = QHBoxLayout()
        result_layout.addWidget(self.result_label, alignment=Qt.AlignmentFlag.AlignLeft)
        text_layout.addLayout(result_layout)

        timer_layout = QHBoxLayout()
        timer_layout.addWidget(self.timer_label, alignment=Qt.AlignmentFlag.AlignLeft)
        text_layout.addLayout(timer_layout)

        # 오른쪽에 웹캠 배치
        webcam_layout = QVBoxLayout()
        webcam_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 텍스트와 웹캠 레이아웃 합치기
        center_layout.addLayout(text_layout)
        center_layout.addLayout(webcam_layout)

        # 메인 레이아웃에 추가
        main_layout.addLayout(center_layout)
        main_layout.addSpacing(170)

        self.setLayout(main_layout)

        # ✅ 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # ✅ 문제 제한 시간 타이머
        self.quiz_timer = QTimer()
        self.quiz_timer.timeout.connect(self.time_up)
        self.time_left = 10

        # ✅ 시퀀스 데이터 저장
        self.sequence = []
        self.current_quiz = None
        self.correct_count = 0

        # ✅ 퀴즈 진행 플래그
        self.quiz_in_progress = False

        # ✅ 초기화: 남은 동작 관리
        self.remaining_actions = ACTIONS.copy()
        random.shuffle(self.remaining_actions)

        # ✅ 웹캠 초기화
        self.cap = None
        self.set_new_quiz()

        # ✅ 웹캠 자동 실행
        self.start_webcam()

    def start_webcam(self):
        """웹캠을 시작하는 함수"""
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.time_left = 10
        self.quiz_timer.start(1000)

    def stop_webcam(self):
        """웹캠을 중지하는 함수"""
        self.timer.stop()
        self.quiz_timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.video_label.clear()
        self.result_label.setText("🖐 Try to follow the sign shown!")
        self.quiz_label.setText("")
        self.timer_label.setText("")
        self.prediction_label.setText("")

    def keyPressEvent(self, event):
        """키보드 입력 처리: 'q'를 누르면 종료"""
        if event.key() == Qt.Key.Key_Q:
            self.close()

    def set_new_quiz(self):
        """새로운 퀴즈 출제"""
        if self.quiz_in_progress:
            return

        self.quiz_in_progress = True

        # 남은 동작이 없으면 초기화
        if not self.remaining_actions:
            self.remaining_actions = ACTIONS.copy()
            random.shuffle(self.remaining_actions)

        # 리스트에서 하나 선택
        self.current_quiz = self.remaining_actions.pop(0)

        # 화면에 남은 문제 표시
        remaining_count = len(self.remaining_actions)
        total_count = len(ACTIONS)
        self.quiz_label.setText(
            f"📝 Follow this sign: {self.current_quiz} (Remaining: {remaining_count}/{total_count})"
        )

        # 화면 초기화
        self.result_label.setText("🖐 Try to follow the sign shown!")
        self.sequence = []
        self.time_left = 10
        self.timer_label.setText(f"⏳ Time left: {self.time_left} seconds")

        QTimer.singleShot(3000, self.reset_quiz_status)

    def reset_quiz_status(self):
        """퀴즈 상태 플래그를 해제하는 함수"""
        self.quiz_in_progress = False

    def time_up(self):
        """시간 초과 시 동작"""
        self.time_left -= 1
        self.timer_label.setText(f"⏳ Time left: {self.time_left} seconds")

        if self.time_left == 0:
            self.result_label.setText("❌ Time's up! Try again.")
            self.time_left = 10
            self.sequence = []
            self.quiz_timer.start(1000)

    def update_frame(self):
        """웹캠 프레임 캡처 및 예측"""
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe 손 인식
        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            results = hands.process(frame)

        # 랜드마크 데이터 저장
        landmarks = np.zeros((63,), dtype=np.float32)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

        # 시퀀스 데이터 저장
        self.sequence.append(landmarks)
        if len(self.sequence) > 30:
            self.sequence = self.sequence[1:]

        # 예측 실행
        if len(self.sequence) == 30:
            input_data = np.expand_dims(self.sequence, axis=0)
            predictions = model.predict(input_data)[0]
            max_index = np.argmax(predictions)
            predicted_action = ACTIONS[max_index]
            confidence = predictions[max_index] * 100

            # 예측된 단어 & 정확도 표시
            self.prediction_label.setText(f"🎯 Prediction: {predicted_action} ({confidence:.2f}%)")

            # 정답 확인
            if predicted_action == self.current_quiz and confidence > 80:
                self.result_label.setText(f"✅ Correct! {predicted_action} ({confidence:.2f}%)")
                self.quiz_in_progress = False
                QTimer.singleShot(2000, self.set_new_quiz)
            elif confidence > 80:
                self.result_label.setText(f"❌ Incorrect! Try again! ({predicted_action}, {confidence:.2f}%)")

        # OpenCV → PyQt 변환 및 화면 출력
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

# ✅ 실행
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec())
