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

# ✅ 제스처 리스트
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

# ✅ MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()

        # ✅ UI 설정
        self.setWindowTitle("✨ Sign Language Quiz ✨")
        self.setStyleSheet("background-color: #eeeael;")  
        self.showFullScreen()  

        # ✅ 상단 제목
        self.title_label = QLabel("✨ Sign Language Quiz ✨", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 45px; color: #f1ac85; font-weight: bold; font-family: 'Times New Roman';")

        # ✅ 웹캠 화면 표시
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 3px solid #368f5f; background-color: #ffffff;")
        self.video_label.setFixedSize(640, 480)

        # ✅ 문제 표시
        self.quiz_label = QLabel("", self)
        self.quiz_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.quiz_label.setSizePolicy(QLabel().sizePolicy())
        self.quiz_label.setStyleSheet(
            "font-size: 35px; color: #368f5f; font-family: 'Times New Roman'; "
            "font-weight: bold; margin-bottom: 30px; text-align: left;"
        )

        # ✅ 예측된 단어 & 정확도 표시
        self.prediction_label = QLabel("", self)
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.prediction_label.setSizePolicy(QLabel().sizePolicy())
        self.prediction_label.setStyleSheet(
            "font-size: 30px; color: #368f5f; font-family: 'Times New Roman'; "
            "font-weight: bold; margin-bottom: 30px; text-align: left;"
        )

        # ✅ 결과 메시지
        self.result_label = QLabel("🖐 Try to follow the sign shown!", self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.result_label.setSizePolicy(QLabel().sizePolicy())
        self.result_label.setStyleSheet(
            "font-size: 30px; color: #368f5f; font-weight: bold; font-family: 'Times New Roman'; "
            "font-weight: bold; margin-bottom: 30px; text-align: left;"
        )

        # ✅ 남은 시간 표시
        self.timer_label = QLabel("", self)
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.timer_label.setSizePolicy(QLabel().sizePolicy())
        self.timer_label.setStyleSheet(
            "font-size: 27px; color: #cc7fb5; font-family: 'Times New Roman'; "
            "font-weight: bold; text-align: left;"
        )

        # ✅ 레이아웃 설정
        main_layout = QVBoxLayout()
        main_layout.addSpacing(50)
        main_layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        main_layout.addSpacing(25)

        # 중앙 레이아웃
        center_layout = QHBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 왼쪽 텍스트 레이아웃
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

        # 오른쪽 웹캠 레이아웃
        webcam_layout = QVBoxLayout()
        webcam_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 레이아웃 합치기
        center_layout.addLayout(text_layout)
        center_layout.addLayout(webcam_layout)

        main_layout.addLayout(center_layout)
        main_layout.addSpacing(170)

        self.setLayout(main_layout)

        # ✅ 초기화
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.quiz_timer = QTimer()
        self.quiz_timer.timeout.connect(self.time_up)
        self.time_left = 10

        self.sequence = []
        self.current_quiz = None
        self.correct_count = 0
        self.quiz_in_progress = False

        self.remaining_gestures = GESTURES.copy()
        random.shuffle(self.remaining_gestures)

        self.cap = None
        self.set_new_quiz()
        self.start_webcam()

    def start_webcam(self):
        """웹캠 시작"""
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.time_left = 10
        self.quiz_timer.start(1000)

    def stop_webcam(self):
        """웹캠 중지"""
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
        """키 입력 처리"""
        if event.key() == Qt.Key.Key_Q:
            self.close()

    def set_new_quiz(self):
        """새 퀴즈 설정"""
        if self.quiz_in_progress:
            return

        self.quiz_in_progress = True

        if not self.remaining_gestures:
            self.remaining_gestures = GESTURES.copy()
            random.shuffle(self.remaining_gestures)

        self.current_quiz = self.remaining_gestures.pop(0)
        
        remaining_count = len(self.remaining_gestures)
        total_count = len(GESTURES)
        self.quiz_label.setText(
            f"📝 Follow this sign: {self.current_quiz} (Remaining: {remaining_count}/{total_count})"
        )

        self.result_label.setText("🖐 Try to follow the sign shown!")
        self.sequence = []
        self.time_left = 10
        self.timer_label.setText(f"⏳ Time left: {self.time_left} seconds")

        QTimer.singleShot(3000, self.reset_quiz_status)

    def reset_quiz_status(self):
        """퀴즈 상태 초기화"""
        self.quiz_in_progress = False

    def time_up(self):
        """시간 초과 처리"""
        self.time_left -= 1
        self.timer_label.setText(f"⏳ Time left: {self.time_left} seconds")

        if self.time_left == 0:
            self.result_label.setText("❌ Time's up! Try again.")
            self.time_left = 10
            self.sequence = []
            self.quiz_timer.start(1000)

    def update_frame(self):
        """프레임 업데이트 및 예측"""
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 손 랜드마크 추출
        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            results = hands.process(frame)

        landmarks = np.zeros((126,), dtype=np.float32)
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmarks = np.append(landmarks, [lm.x, lm.y, lm.z])

        # 시퀀스 데이터 관리
        self.sequence.append(landmarks)
        if len(self.sequence) > 30:
            self.sequence = self.sequence[1:]

        # 예측
        if len(self.sequence) == 30:
            input_data = np.expand_dims(self.sequence, axis=0)
            predictions = model.predict(input_data, verbose=0)[0]
            max_index = np.argmax(predictions)
            predicted_gesture = GESTURES[max_index]
            confidence = predictions[max_index] * 100

            self.prediction_label.setText(f"🎯 Prediction: {predicted_gesture} ({confidence:.2f}%)")

            # 정답 확인
            if predicted_gesture == self.current_quiz and confidence > 80:
                self.result_label.setText(f"✅ Correct! {predicted_gesture} ({confidence:.2f}%)")
                self.quiz_in_progress = False
                QTimer.singleShot(2000, self.set_new_quiz)
            elif confidence > 80:
                self.result_label.setText(f"❌ Incorrect! Try again! ({predicted_gesture}, {confidence:.2f}%)")

        # 화면 업데이트
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec())