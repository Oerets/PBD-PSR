import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox, QSlider, QPlainTextEdit, QWidget, QFrame, QFileDialog, QProgressBar, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
import requests
from pydicom import dcmread
import numpy as np

def normalize_pixel_array(pixel_array):
    # 최소값과 최대값으로 정규화
    min_val = np.min(pixel_array)
    max_val = np.max(pixel_array)
    pixel_array = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return pixel_array

def dicom_to_pixmap(dicom_path):
    # DICOM 파일 읽기
    dicom_data = dcmread(dicom_path)

    # Pixel 데이터 추출
    pixel_array = dicom_data.pixel_array

    # DICOM 데이터를 8비트로 정규화 (0-255)
    pixel_array = normalize_pixel_array(pixel_array)

    # Numpy 배열을 QImage로 변환
    height, width = pixel_array.shape
    bytes_per_line = width  # 그레이스케일이므로 width 그대로 사용
    qimage = QImage(
        pixel_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8
    )

    # QImage를 QPixmap으로 변환
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

class StreamWorker(QThread):
    """
    Flask 서버에서 실시간 스트리밍 데이터를 받아오는 백그라운드 스레드
    """
    data_received = pyqtSignal(str)  # 데이터를 메인 스레드(UI)로 전달하는 시그널

    def __init__(self, url, settings):
        super().__init__()  # QThread 초기화
        self.url = url  # Flask 서버 URL
        self.settings = settings  # JSON 설정 데이터

    def run(self):
        """
        서버와 통신을 실행하는 메서드 (이 메서드는 백그라운드에서 실행됨)
        """
        try:
            with requests.post(self.url, json=self.settings, stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith("data: "):  # 서버에서 오는 데이터 필터링
                            plain_text = json.dumps(json.loads(decoded_line[6:]), indent=4, ensure_ascii=False)
                            self.data_received.emit(plain_text)  # 📌 메인 스레드(UI)로 데이터 전달
        except Exception as e:
            self.data_received.emit(f"Error: {str(e)}")  # 오류 발생 시 UI에 출력

class BMDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings_file = 'save_settings.txt'
        self.image_files = []
        self.current_image_index = -1
        self.detection_finished = False
        self.initUI()
        self.load_settings()  # Load settings on application start.
        self.url = "http://localhost:5000/analyze"

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main horizontal layout
        main_layout = QHBoxLayout()

        #Image 디스플레이 관련 항목들 (후반부에 관련 함수도 주석처리 되어있음음)
        '''# Left layout for image display
        self.image_label = QLabel("No Image")
        self.image_label.setFixedSize(300, 400)  # Define a fixed size for the image display
        self.image_label.setAlignment(Qt.AlignCenter)

        # Previous and Next image buttons
        self.current_image_name_label = QLabel("No Image")
        self.prev_image_button = QPushButton("Previous Image")
        self.prev_image_button.clicked.connect(self.show_previous_image)
        self.next_image_button = QPushButton("Next Image")
        self.next_image_button.clicked.connect(self.show_next_image)

        # Add image widgets to layout
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.current_image_name_label)
        image_layout.addWidget(self.prev_image_button)
        image_layout.addWidget(self.next_image_button)
        main_layout.addLayout(image_layout)'''

        # Right layout for controls
        controls_layout = QVBoxLayout()

        ####################################################################

        # QFrame 생성
        frame1 = QFrame(self)
        frame2 = QFrame(self)
        frame3 = QFrame(self)
        frame4 = QFrame(self)
        frame5 = QFrame(self)

        frame1.setObjectName("frame1")
        frame2.setObjectName("frame2")
        frame3.setObjectName("frame3")
        frame4.setObjectName("frame4")
        frame5.setObjectName("frame5")

        frame1.setFrameShape(QFrame.Box)  # 테두리 박스 설정
        frame2.setFrameShape(QFrame.Box)  # 테두리 박스 설정
        frame3.setFrameShape(QFrame.Box)  # 테두리 박스 설정
        frame4.setFrameShape(QFrame.Box)  # 테두리 박스 설정
        frame5.setFrameShape(QFrame.Box)  # 테두리 박스 설정

        frame1.setStyleSheet("#frame1 {border: 1px solid gray;}")
        frame2.setStyleSheet("#frame2 {border: 1px solid gray;}")
        frame3.setStyleSheet("#frame3 {border: 1px solid gray;}")
        frame4.setStyleSheet("#frame4 {border: 1px solid gray;}")
        frame5.setStyleSheet("#frame5 {border: 1px solid gray;}")

        frame1.setFixedSize(350, 150)  # 크기 설정
        frame2.setFixedSize(350, 120)  # 크기 설정
        frame3.setFixedSize(350, 120)  # 크기 설정
        frame4.setFixedSize(350, 110)  # 크기 설정
        frame5.setFixedSize(350, 110)  # 크기 설정

        # Detection and Regression model paths and options
        self.det_model_path = QLineEdit()
        self.det_model_select_button = QPushButton("Select Detection Model Path")
        self.Detection_model_type = QComboBox()
        self.Detection_model_type.addItems(["Yolov5", "Yolov8", "Yolov11-Hip"])
        self.det_model_select_button.clicked.connect(self.select_det_model_path)

        self.reg_model_path = QLineEdit()
        self.reg_model_select_button = QPushButton("Select Regression Model Path")
        self.Regression_model_type = QComboBox()
        self.Regression_model_type.addItems(["resnet50", "resnet18", "vgg16", "squeezenet", "efficientnet", "resnet50-Hip" ])
        self.reg_model_select_button.clicked.connect(self.select_reg_model_path)

        # Mode selection
        # self.mode_combo = QComboBox()
        # self.mode_combo.addItems(['Train(unavailable)', 'Validation(unavailable)', 'Test'])

        # Mode selection 백도어
        self.bodypart_combo = QComboBox()
        self.bodypart_combo.addItems(['Vertebra', 'Hip'])

        # Weighted Mode checkbox
        self.weighted_mode_checkbox = QCheckBox("Use Weighted Mode")

        # Z-score threshold slider
        self.z_threshold_slider = QSlider(Qt.Horizontal)
        self.z_threshold_slider.setRange(-50, 50)
        self.z_threshold_slider.setValue(-20)
        self.z_threshold_label = QLabel('Z-Threshold: -2.0')
        self.z_threshold_slider.valueChanged.connect(self.update_z_threshold_label)

        #JPG 이미지를 이제 사용하지 않음.
        '''# Data, Excel, and DICOM path
        self.data_path = QLineEdit()
        self.data_path_button = QPushButton("Select Data Path")
        self.data_path_button.clicked.connect(self.select_data_path)'''

        self.excel_path = QLineEdit()
        self.excel_path_button = QPushButton("Select Excel File Path")
        self.excel_path_button.clicked.connect(self.select_excel_path)

        self.dicom_path = QLineEdit()
        self.dicom_path_button = QPushButton("Select DICOM Path")
        self.dicom_path_button.clicked.connect(self.select_dicom_path)

        # Start button
        self.start_button = QPushButton("Test Start")
        self.start_button.clicked.connect(self.run_application)
        self.start_button.setFixedSize(100, 50)

        # ProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        #Hip 백도어어
        frame_layout1 = QVBoxLayout()

        frame_layout1.addWidget(QLabel("Body part"))
        frame_layout1.addWidget(self.bodypart_combo)

        frame_layout1.addWidget(self.weighted_mode_checkbox)

        frame_layout1.addWidget(self.z_threshold_label)
        frame_layout1.addWidget(self.z_threshold_slider)

        frame1.setLayout(frame_layout1)
        controls_layout.addWidget(frame1)

        # Add widgets to controls layout
        frame_layout2 = QVBoxLayout()

        frame_layout2.addWidget(QLabel("Detection Model Select"))
        frame_layout2.addWidget(self.Detection_model_type)
        frame_layout2.addWidget(self.det_model_path)
        frame_layout2.addWidget(self.det_model_select_button)

        frame2.setLayout(frame_layout2)
        controls_layout.addWidget(frame2)

        ###
        frame_layout3 = QVBoxLayout()

        frame_layout3.addWidget(QLabel("Regression Model Select"))
        frame_layout3.addWidget(self.Regression_model_type)
        frame_layout3.addWidget(self.reg_model_path)
        frame_layout3.addWidget(self.reg_model_select_button)

        frame3.setLayout(frame_layout3)
        controls_layout.addWidget(frame3)

        ###JPG는 이제 사용하지 않음
        '''controls_layout.addWidget(QLabel("Data Path"))
        controls_layout.addWidget(self.data_path)
        controls_layout.addWidget(self.data_path_button)'''

        ###
        frame_layout4 = QVBoxLayout()

        frame_layout4.addWidget(QLabel("DICOM Path"))
        frame_layout4.addWidget(self.dicom_path)
        frame_layout4.addWidget(self.dicom_path_button)

        frame4.setLayout(frame_layout4)
        controls_layout.addWidget(frame4)

        ###
        frame_layout5 = QVBoxLayout()
        
        frame_layout5.addWidget(QLabel("Excel File Path"))
        frame_layout5.addWidget(self.excel_path)
        frame_layout5.addWidget(self.excel_path_button)
        
        frame5.setLayout(frame_layout5)
        controls_layout.addWidget(frame5)

        controls_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)
        controls_layout.addWidget(QLabel("-Test Mode Supported Only-"), alignment=Qt.AlignCenter)
        # Add controls layout to the right side of the main layout
        main_layout.addLayout(controls_layout)

        # Add image widgets to layout

        # Results display
        self.result_display = QPlainTextEdit()
        self.result_display.setReadOnly(True)

        result_layout = QVBoxLayout()

        result_layout.addWidget(QLabel("Results"))
        result_layout.addWidget(self.result_display)
        result_layout.addWidget(self.progress_bar)
        main_layout.addLayout(result_layout)

        central_widget.setLayout(main_layout)
        self.setWindowTitle("BMD Analysis")
        self.setGeometry(300, 500, 700, 500)
        self.setFixedSize(700,800)

    def update_image(self):
        if 0 <= self.current_image_index < len(self.image_files):
            image_path = self.image_files[self.current_image_index]
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
                self.current_image_name_label.setText(f"{os.path.basename(image_path)} ({self.current_image_index+1}/{len(self.image_files)})")

            else:
                self.image_label.setText("Invalid Image Path")

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image()

    def show_next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.update_image()

    # Methods for path selection and settings
    def select_det_model_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Detection Model Path")
        if path:
            self.det_model_path.setText(path)
            self.save_settings()

    def select_reg_model_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Regression Model Path")
        if path:
            self.reg_model_path.setText(path)
            self.save_settings()

    def update_z_threshold_label(self, value):
        z_threshold_value = value / 10.0
        self.z_threshold_label.setText(f'Z-Threshold: {z_threshold_value}')
        self.save_settings()

    def draw_display_image(self, image_path, label_path):
        # DICOM 파일인지 확인 및 처리
        if image_path.lower().endswith(".dcm"):
            pixmap = dicom_to_pixmap(image_path)
        else:
            pixmap = QPixmap(image_path)
        
        if pixmap.isNull():
            self.image_label.setText("Invalid Image")
            return

        if self.detection_finished:
            painter = QPainter(pixmap)
            pen = QPen(Qt.red, 5)
            painter.setPen(pen)

            try:
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        values = list(map(float, line.strip().split()))
                        if len(values) < 9:
                            continue
                        _, x1, y1, x2, y2, x3, y3, x4, y4 = values

                        width = pixmap.width()
                        height = pixmap.height()

                        # Convert normalized coordinates to pixel coordinates
                        x1, y1 = int(x1 * width), int(y1 * height)
                        x2, y2 = int(x2 * width), int(y2 * height)
                        x3, y3 = int(x3 * width), int(y3 * height)
                        x4, y4 = int(x4 * width), int(y4 * height)

                        painter.drawLine(x1, y1, x2, y2)
                        painter.drawLine(x2, y2, x3, y3)
                        painter.drawLine(x3, y3, x4, y4)
                        painter.drawLine(x4, y4, x1, y1)

            except FileNotFoundError:
                self.display_update_with_clear(f"Label file not found for image: {image_path}")
            painter.end()

        # Pixmap 리사이즈
        pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def display_update_with_clear(self, text):
        self.result_display.clear()
        #self.result_display.appendPlainText(text)
        QTimer.singleShot(0, lambda: self.result_display.setPlainText(text))

    #JPG 및 이미지 영사 기능을 이제 사용하지 않음.
    '''def select_data_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if path:
            self.data_path.setText(path)
            self.save_settings()

            # Scan directory and count image files
            self.image_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.image_files.append(os.path.join(root, file))

            file_count = len(self.image_files)

            # Display result in the Results display
            self.display_update_with_clear(f"Data Path: {path}\nImage File Count: {file_count}\n")

            # Initialize the image viewer if images are found
            if self.image_files:
                self.current_image_index = 0
                self.update_image()
            else:
                self.image_label.setText("No Images Found")'''

    def select_excel_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Excel File Path")
        if path:
            self.excel_path.setText(path)
            self.save_settings()

    def select_dicom_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select DICOM Directory")
        if path:
            self.dicom_path.setText(path)
            self.save_settings()

    # Settings save/load methods
    def save_settings(self):
        settings = {
            'det_model_path': self.det_model_path.text(),
            'reg_model_path': self.reg_model_path.text(),
            #'data_path': self.data_path.text(),
            'excel_path': self.excel_path.text(),
            'dicom_path': self.dicom_path.text(),
            'z_threshold': self.z_threshold_slider.value(),
            'weighted_mode': self.weighted_mode_checkbox.isChecked(),
            'bodypart_mode': self.bodypart_combo.currentText()
        }
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f)

    def load_settings(self):
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                self.det_model_path.setText(settings.get('det_model_path', ''))
                self.reg_model_path.setText(settings.get('reg_model_path', ''))
                #self.data_path.setText(settings.get('data_path', ''))
                self.excel_path.setText(settings.get('excel_path', ''))
                self.dicom_path.setText(settings.get('dicom_path', ''))
                self.z_threshold_slider.setValue(settings.get('z_threshold', -20))
                self.weighted_mode_checkbox.setChecked(settings.get('weighted_mode', False))
                self.bodypart_combo.setCurrentText(settings.get('bodypart_mode', ''))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def handle_stream_message(self, message_str):
        try:
            message = json.loads(message_str)
            status = message.get("status")
            progress = message.get("progress")

            if status:
                self.result_display.appendPlainText(status)

            if progress is not None:
                self.progress_bar.setValue(progress)
        except Exception as e:
            self.result_display.appendPlainText(message_str)

    def run_application(self):
        """
        UI 버튼이 클릭될 때 실행되는 메서드 (메인 스레드에서 실행)
        """
        self.result_display.clear()  # 기존 데이터 삭제

        url = self.url  # 서버 URL
        settings = {
            'bodypart_mode': self.bodypart_combo.currentText(),
            'weighted_mode': self.weighted_mode_checkbox.isChecked(),
            'z_threshold': self.z_threshold_slider.value() / 10.0,
            'det_model_type': self.Detection_model_type.currentText(),
            'reg_model_type': self.Regression_model_type.currentText(),
            'det_model_path': self.det_model_path.text(),
            'reg_model_path': self.reg_model_path.text(),
            #'data_path': self.data_path.text(),
            'excel_path': self.excel_path.text(),
            'dicom_path': self.dicom_path.text(),
        }

        self.result_display.setPlainText("Starting process...")

        # 백그라운드 스레드 실행
        self.worker = StreamWorker(url, settings)
        self.worker.data_received.connect(self.handle_stream_message)
        # self.worker.data_received.connect(self.result_display.appendPlainText)  # UI 업데이트 연결
        self.worker.start()  # 스레드 시작
    
    #Image 디스플레이 관련 함수들
    '''def process_images(self):
        if self.current_progress < len(self.image_files):
            # Update progress bar
            self.progress_bar.setValue(self.current_progress + 1)
            self.current_progress += 1

            # Update tqdm-like progress
            percentage = int((self.current_progress / len(self.image_files)) * 100)
            num_hashes = percentage // 2
            progress_bar = "|" + "#" * num_hashes + " " * (50 - num_hashes) + "|"
            self.result_display.setPlainText(f"Progress: {progress_bar} {percentage}%")
        else:
            # Stop the timer when done
            self.timer.stop() 
            self.result_display.setPlainText("\nDetection complete.")
            # Initialize image display with bounding boxes
            self.current_image_index = 0
            self.update_image_with_bbox()

    def update_image_with_bbox(self):
        print('update_image_with_bbox')
        if 0 <= self.current_image_index < len(self.image_files):
            image_path = self.image_files[self.current_image_index]
            label_path = image_path.replace("images", "labels").replace(
                os.path.splitext(image_path)[1], ".txt"
            )
            self.current_image_name_label.setText(f"{os.path.basename(image_path)} ({self.current_image_index+1}/{len(self.image_files)})")
            # Draw bounding boxes on the resized image
            self.draw_display_image(image_path, label_path)

        # Update the progress bar to reflect image changes
        self.progress_bar.setValue(self.current_image_index + 1)

    def show_previous_image(self):
        print('show_previous_image')
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image_with_bbox()

    def show_next_image(self):
        print('show_next_image')
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.update_image_with_bbox()'''

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BMDApp()
    ex.show()
    sys.exit(app.exec_())