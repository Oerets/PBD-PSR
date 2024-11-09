import sys
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox, QSlider, QPlainTextEdit, QWidget, QFileDialog, QProgressBar
)
from PyQt5.QtCore import Qt
import requests

class BMDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings_file = 'save_settings.txt'
        self.initUI()
        self.load_settings()  # 애플리케이션 시작 시 세팅을 불러옵니다.

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()

        # ProgressBar 추가
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Detection 및 Regression 모델 경로 설정
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

        # 모드 선택
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Train', 'Validation', 'Test'])

        # Weighted Mode 체크박스
        self.weighted_mode_checkbox = QCheckBox("Use Weighted Mode")

        # Z-score 문턱치 조정 슬라이더
        self.z_threshold_slider = QSlider(Qt.Horizontal)
        self.z_threshold_slider.setRange(-50, 50)
        self.z_threshold_slider.setValue(-20)
        self.z_threshold_label = QLabel('Z-Threshold: -2.0')
        self.z_threshold_slider.valueChanged.connect(self.update_z_threshold_label)

        # 데이터 경로 설정
        self.data_path = QLineEdit()
        self.data_path_button = QPushButton("Select Data Path")
        self.data_path_button.clicked.connect(self.select_data_path)

        # Excel 및 DICOM 경로 설정
        self.excel_path = QLineEdit()
        self.excel_path_button = QPushButton("Select Excel File Path")
        self.excel_path_button.clicked.connect(self.select_excel_path)

        self.dicom_path = QLineEdit()
        self.dicom_path_button = QPushButton("Select DICOM Path")
        self.dicom_path_button.clicked.connect(self.select_dicom_path)

        # Start 버튼
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.run_analysis)

        # 결과 출력 창
        self.result_display = QPlainTextEdit()
        self.result_display.setReadOnly(True)

        # 레이아웃에 위젯 추가
        main_layout.addWidget(QLabel("Detection Model Path"))
        main_layout.addWidget(self.det_model_path)
        main_layout.addWidget(self.Detection_model_type)
        main_layout.addWidget(self.det_model_select_button)

        main_layout.addWidget(QLabel("Regression Model Path"))
        main_layout.addWidget(self.reg_model_path)
        main_layout.addWidget(self.Regression_model_type)
        main_layout.addWidget(self.reg_model_select_button)

        main_layout.addWidget(QLabel("Mode"))
        main_layout.addWidget(self.mode_combo)

        main_layout.addWidget(self.weighted_mode_checkbox)

        main_layout.addWidget(self.z_threshold_label)
        main_layout.addWidget(self.z_threshold_slider)

        main_layout.addWidget(QLabel("Data Path"))
        main_layout.addWidget(self.data_path)
        main_layout.addWidget(self.data_path_button)

        main_layout.addWidget(QLabel("Excel File Path"))
        main_layout.addWidget(self.excel_path)
        main_layout.addWidget(self.excel_path_button)

        main_layout.addWidget(QLabel("DICOM Path"))
        main_layout.addWidget(self.dicom_path)
        main_layout.addWidget(self.dicom_path_button)

        main_layout.addWidget(self.start_button)
        main_layout.addWidget(QLabel("Results"))
        main_layout.addWidget(self.result_display)
        
        central_widget.setLayout(main_layout)
        self.setWindowTitle("BMD Analysis")
        self.setGeometry(300, 300, 600, 500)

    # 경로 선택 및 세팅 저장
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

    def select_data_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if path:
            self.data_path.setText(path)
            self.save_settings()

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

    # 세팅을 파일에 저장
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

    # 세팅을 파일에서 불러오기
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
            # 파일이 없거나 JSON 형식이 잘못된 경우 기본값 사용
            pass

    def start_container(self):
        # 입력 경로 확인
        data_path = self.data_path.text()
        model_path = self.model_path.text()
        if not data_path or not model_path:
            self.result_display.setPlainText("Please select both data and model paths.")
            return

        # Docker 실행 명령 구성
        docker_command = [
            "docker", "run", "--rm", "-p", "5000:5000",
            "-v", f"{data_path}:/app/data",
            "-v", f"{model_path}:/app/models",
            "bmd_backend"
        ]

    def run_analysis(self):
        # 설정 값 가져오기
        data = {
            'mode': self.mode_combo.currentText(),
            'weighted_mode': self.weighted_mode_checkbox.isChecked(),
            'z_threshold': self.z_threshold_slider.value() / 10.0,
            'det_model_name': self.Detection_model_type.currentText(),
            'reg_model_name': self.Regression_model_type.currentText(),

            'reg_model_path': self.reg_model_path.text(),
            'det_model_path': self.det_model_path.text(),
            'data_path': self.data_path.text(),
            'excel_path': self.excel_path.text(),
            'dicom_path': self.dicom_path.text(),
        }

        print(data)



        # Flask API로 요청을 보내고 진행 상태 업데이트
        response = requests.post("http://localhost:5000/analyze", json=data, stream=True)
        for line in response.iter_lines():
            if line:
                status_update = json.loads(line.decode('utf-8')[5:])
                self.progress_bar.setValue(status_update["progress"])
                self.result_display.setPlainText(status_update.get("status", ""))
                if status_update["progress"] == 100:
                    self.result_display.setPlainText(status_update.get("result", ""))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BMDApp()
    ex.show()
    sys.exit(app.exec_())
