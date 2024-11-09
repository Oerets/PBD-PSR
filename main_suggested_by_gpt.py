import sys
import json
import requests
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QPlainTextEdit, QWidget
)
from PyQt5.QtCore import Qt

class BMDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings_file = 'save_settings.txt'
        self.initUI()
        self.load_settings()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()

        # 경로 설정 UI
        self.data_path = QLineEdit()
        self.data_path_button = QPushButton("Select Data Path")
        self.data_path_button.clicked.connect(self.select_data_path)

        self.model_path = QLineEdit()
        self.model_path_button = QPushButton("Select Model Path")
        self.model_path_button.clicked.connect(self.select_model_path)

        # Start 버튼
        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.run_docker_and_analysis)

        # 결과 출력 창
        self.result_display = QPlainTextEdit()
        self.result_display.setReadOnly(True)

        # 레이아웃에 위젯 추가
        main_layout.addWidget(QLabel("Data Path"))
        main_layout.addWidget(self.data_path)
        main_layout.addWidget(self.data_path_button)

        main_layout.addWidget(QLabel("Model Path"))
        main_layout.addWidget(self.model_path)
        main_layout.addWidget(self.model_path_button)

        main_layout.addWidget(self.start_button)
        main_layout.addWidget(QLabel("Results"))
        main_layout.addWidget(self.result_display)
        
        central_widget.setLayout(main_layout)
        self.setWindowTitle("BMD Analysis")
        self.setGeometry(300, 300, 600, 500)

    # 경로 선택 메서드
    def select_data_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if path:
            self.data_path.setText(path)

    def select_model_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if path:
            self.model_path.setText(path)

    # Docker 컨테이너 실행 및 분석 수행 메서드
    def run_docker_and_analysis(self):
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

        # Docker 컨테이너 실행
        try:
            subprocess.Popen(docker_command)
            self.result_display.setPlainText("Docker container started. Running analysis...")
        except Exception as e:
            self.result_display.setPlainText(f"Failed to start Docker container: {e}")
            return

        # 분석 요청 보내기
        self.send_analysis_request()

    # Flask API에 분석 요청
    def send_analysis_request(self):
        data = {
            'mode': 'Test',
            'det_model_path': "/app/models/detection_model.pt",
            'reg_model_path': "/app/models/regression_model.pt",
            'data_path': "/app/data",
            'z_threshold': -2.0
        }
        
        try:
            response = requests.post("http://localhost:5000/analyze", json=data, timeout=120)
            if response.status_code == 200:
                self.result_display.setPlainText(response.json().get("result", "No result found"))
            else:
                self.result_display.setPlainText("Error: " + response.text)
        except requests.exceptions.RequestException as e:
            self.result_display.setPlainText(f"Failed to connect to Flask server: {e}")

    # 세팅을 파일에서 불러오기
    def load_settings(self):
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                self.data_path.setText(settings.get('data_path', ''))
                self.model_path.setText(settings.get('model_path', ''))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BMDApp()
    ex.show()
    sys.exit(app.exec_())
