import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QSlider, QPlainTextEdit, QWidget, QFileDialog
)
from PyQt5.QtCore import Qt
from image_qt_backend import bmd_analysis  # bmd_analysis 함수 임포트

class BMDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()

        # Detection 및 Regression 모델 경로 설정
        self.det_model_path = QLineEdit()
        self.det_model_select_button = QPushButton("Select Detection Model Path")
        self.det_model_select_button.clicked.connect(self.select_det_model_path)

        self.reg_model_path = QLineEdit()
        self.reg_model_select_button = QPushButton("Select Regression Model Path")
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
        main_layout.addWidget(self.det_model_select_button)

        main_layout.addWidget(QLabel("Regression Model Path"))
        main_layout.addWidget(self.reg_model_path)
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

    def select_det_model_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Detection Model Path")
        if path:
            self.det_model_path.setText(path)

    def select_reg_model_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Regression Model Path")
        if path:
            self.reg_model_path.setText(path)

    def update_z_threshold_label(self, value):
        z_threshold_value = value / 10.0
        self.z_threshold_label.setText(f'Z-Threshold: {z_threshold_value}')

    def select_data_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if path:
            self.data_path.setText(path)

    def select_excel_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Excel File Path")
        if path:
            self.excel_path.setText(path)

    def select_dicom_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select DICOM Directory")
        if path:
            self.dicom_path.setText(path)

    def run_analysis(self):
        # 설정 값 가져오기
        det_model_path = self.det_model_path.text()
        reg_model_path = self.reg_model_path.text()
        mode = self.mode_combo.currentText()
        weighted_mode = self.weighted_mode_checkbox.isChecked()
        z_threshold = self.z_threshold_slider.value() / 10.0
        data_path = self.data_path.text()
        excel_path = self.excel_path.text()
        dicom_path = self.dicom_path.text()

        # bmd_analysis 함수 호출
        result = bmd_analysis(
            mode=mode,
            weighted_mode=weighted_mode,
            det_model_path=det_model_path,
            det_model_name="yolo",  # det_model_name 예시로 지정
            reg_model_path=reg_model_path,
            reg_model_name="resnet18",  # reg_model_name 예시로 지정
            data_path=data_path,
            excel_path=excel_path,
            dicom_path=dicom_path,
            z_threshold=z_threshold
        )
        
        # 결과 출력
        self.result_display.setPlainText(result)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BMDApp()
    ex.show()
    sys.exit(app.exec_())