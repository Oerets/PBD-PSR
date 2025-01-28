import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox, QSlider, QPlainTextEdit, QWidget, QFileDialog, QProgressBar, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen
import requests

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

        # Left layout for image display
        self.image_label = QLabel("No Image")
        self.image_label.setFixedSize(300, 400)  # Define a fixed size for the image display
        self.image_label.setAlignment(Qt.AlignCenter)

        # Previous and Next image buttons
        self.prev_image_button = QPushButton("Previous Image")
        self.prev_image_button.clicked.connect(self.show_previous_image)
        self.next_image_button = QPushButton("Next Image")
        self.next_image_button.clicked.connect(self.show_next_image)

        # Add image widgets to layout
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.prev_image_button)
        image_layout.addWidget(self.next_image_button)
        main_layout.addLayout(image_layout)

        # Right layout for controls
        controls_layout = QVBoxLayout()

        # ProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Detection and Regression model paths and options
        self.det_model_path = QLineEdit()
        self.det_model_select_button = QPushButton("Select Detection Model Path")
        self.Detection_model_type = QComboBox()
        self.Detection_model_type.addItems(["Yolov5", "Yolov8"])
        self.det_model_select_button.clicked.connect(self.select_det_model_path)

        self.reg_model_path = QLineEdit()
        self.reg_model_select_button = QPushButton("Select Regression Model Path")
        self.Regression_model_type = QComboBox()
        self.Regression_model_type.addItems(["resnet18", "resnet50", "vgg16", "squeezenet", "efficientnet"])
        self.reg_model_select_button.clicked.connect(self.select_reg_model_path)

        # Mode selection
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Train', 'Validation', 'Test'])

        # Weighted Mode checkbox
        self.weighted_mode_checkbox = QCheckBox("Use Weighted Mode")

        # Z-score threshold slider
        self.z_threshold_slider = QSlider(Qt.Horizontal)
        self.z_threshold_slider.setRange(-50, 50)
        self.z_threshold_slider.setValue(-20)
        self.z_threshold_label = QLabel('Z-Threshold: -2.0')
        self.z_threshold_slider.valueChanged.connect(self.update_z_threshold_label)

        # Data, Excel, and DICOM path
        self.data_path = QLineEdit()
        self.data_path_button = QPushButton("Select Data Path")
        self.data_path_button.clicked.connect(self.select_data_path)

        self.excel_path = QLineEdit()
        self.excel_path_button = QPushButton("Select Excel File Path")
        self.excel_path_button.clicked.connect(self.select_excel_path)

        self.dicom_path = QLineEdit()
        self.dicom_path_button = QPushButton("Select DICOM Path")
        self.dicom_path_button.clicked.connect(self.select_dicom_path)

        # Start button
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.run_application)

        # Results display
        self.result_display = QPlainTextEdit()
        self.result_display.setReadOnly(True)

        # ProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Add widgets to controls layout
        controls_layout.addWidget(QLabel("Detection Model Path"))
        controls_layout.addWidget(self.det_model_path)
        controls_layout.addWidget(self.Detection_model_type)
        controls_layout.addWidget(self.det_model_select_button)

        controls_layout.addWidget(QLabel("Regression Model Path"))
        controls_layout.addWidget(self.reg_model_path)
        controls_layout.addWidget(self.Regression_model_type)
        controls_layout.addWidget(self.reg_model_select_button)

        controls_layout.addWidget(QLabel("Mode"))
        controls_layout.addWidget(self.mode_combo)

        controls_layout.addWidget(self.weighted_mode_checkbox)

        controls_layout.addWidget(self.z_threshold_label)
        controls_layout.addWidget(self.z_threshold_slider)

        controls_layout.addWidget(QLabel("Data Path"))
        controls_layout.addWidget(self.data_path)
        controls_layout.addWidget(self.data_path_button)

        controls_layout.addWidget(QLabel("Excel File Path"))
        controls_layout.addWidget(self.excel_path)
        controls_layout.addWidget(self.excel_path_button)

        controls_layout.addWidget(QLabel("DICOM Path"))
        controls_layout.addWidget(self.dicom_path)
        controls_layout.addWidget(self.dicom_path_button)

        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(QLabel("Results"))
        controls_layout.addWidget(self.result_display)

        # Add controls layout to the right side of the main layout
        main_layout.addLayout(controls_layout)
        
        central_widget.setLayout(main_layout)
        self.setWindowTitle("BMD Analysis")
        self.setGeometry(300, 500, 700, 500)
        self.setFixedSize(700,700)

    def update_image(self):
        if 0 <= self.current_image_index < len(self.image_files):
            pixmap = QPixmap(self.image_files[self.current_image_index])
            if not pixmap.isNull():
                pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
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
                self.result_display.appendPlainText(f"Label file not found for image: {image_path}")
            painter.end()

        #여기서 pixmap resize하는 과정이 빠져있다. 다시 하자.
        pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)        
        self.image_label.setPixmap(pixmap)

    def select_data_path(self):
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
            self.result_display.appendPlainText(f"Data Path: {path}\nImage File Count: {file_count}\n")

            # Initialize the image viewer if images are found
            if self.image_files:
                self.current_image_index = 0
                self.update_image()
            else:
                self.image_label.setText("No Images Found")

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
            'data_path': self.data_path.text(),
            'excel_path': self.excel_path.text(),
            'dicom_path': self.dicom_path.text(),
            'z_threshold': self.z_threshold_slider.value(),
            'weighted_mode': self.weighted_mode_checkbox.isChecked(),
            'mode': self.mode_combo.currentText()
        }
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f)

    def load_settings(self):
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                self.det_model_path.setText(settings.get('det_model_path', ''))
                self.reg_model_path.setText(settings.get('reg_model_path', ''))
                self.data_path.setText(settings.get('data_path', ''))
                self.excel_path.setText(settings.get('excel_path', ''))
                self.dicom_path.setText(settings.get('dicom_path', ''))
                self.z_threshold_slider.setValue(settings.get('z_threshold', -20))
                self.weighted_mode_checkbox.setChecked(settings.get('weighted_mode', False))
                mode = settings.get('mode', 'Train')
                if mode in ['Train', 'Validation', 'Test']:
                    self.mode_combo.setCurrentText(mode)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def run_application(self):
        # Display current settings in the result display
        self.result_display.clear()
        url = self.url
        settings = {
            'Mode': self.mode_combo.currentText(),
            'Weighted Mode': self.weighted_mode_checkbox.isChecked(),
            'Z-Threshold': self.z_threshold_slider.value() / 10.0,
            'Detection Model Type': self.Detection_model_type.currentText(),
            'Regression Model Type': self.Regression_model_type.currentText(),
            'Detection Model Path': self.det_model_path.text(),
            'Regression Model Path': self.reg_model_path.text(),
            'Data Path': self.data_path.text(),
            'Excel Path': self.excel_path.text(),
            'DICOM Path': self.dicom_path.text(),
        }

        try:
            response = requests.post(url, json=settings, timeout=5)  # 타임아웃 추가
            if response.status_code == 200:
                # JSON 응답을 문자열로 변환 후 출력
                self.result_display.setPlainText(f"Response from server: {response.json()}")
            else:
                # 실패한 경우 상태 코드와 메시지를 출력
                self.result_display.setPlainText(f"Failed: {response.status_code}, {response.text}")
                print("Failed:", response.status_code, response.text)
        except requests.exceptions.ConnectionError:
            # 서버 연결 오류
            self.result_display.setPlainText("Error: Unable to connect to the server. Is the server running?")
        except requests.exceptions.Timeout:
            # 요청 타임아웃 오류
            self.result_display.setPlainText("Error: The request timed out.")
        except Exception as e:
            # 기타 예외 처리
            self.result_display.setPlainText(f"An error occurred: {str(e)}")

        
        '''
        for key, value in settings.items():
            self.result_display.appendPlainText(f"{key}: {value}")

        # Show progress message
        self.result_display.appendPlainText("\nDetection in progress...")

        # Set up progress bar
        image_count = len(self.image_files)
        if image_count == 0:
            self.result_display.setPlainText("\nNo images found in the selected data path.")
            return

        self.progress_bar.setMaximum(image_count)
        self.progress_bar.setValue(0)

        # Display tqdm-like progress in the result display
        self.result_display.setPlainText("Progress: |" + " " * 50 + "| 0%")
        self.tqdm_progress = 0

        # Simulate processing with a QTimer
        self.current_progress = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_images)
        self.timer.start(50)  # 0.01 seconds per image

        self.result_display.appendPlainText("\nDetection Completed")
        self.detection_finished = True
        self.update_image_with_bbox()
        '''
    def process_images(self):
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
            self.update_image_with_bbox()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BMDApp()
    ex.show()
    sys.exit(app.exec_())