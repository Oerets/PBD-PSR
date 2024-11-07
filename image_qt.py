import sys, os
import PyQt5
from PyQt5 import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class Main(QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        pic = QLabel(self)
        pixmap = QPixmap("c:/Users/hyunoh/Documents/Codes/BMD_code/data/images/test/2938560_3.jpg")
        
        if pixmap.isNull():
            print("Image failed to load!")
        else:
            # 이미지 크기 조정: 비율을 유지하면서 500x500으로 크기 조정
            pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)

            # 이미지 중앙 정렬
            pic.setPixmap(pixmap)
            pic.setAlignment(Qt.AlignCenter)  # 가로, 세로 중앙 정렬

        main_layout.addWidget(pic)  # Layout에 이미지 추가

        sub_layout = QFormLayout()
        Empty_layout = QHBoxLayout()

        Model_select_layout = QFormLayout()
        Model_select_detmodel_widget = QComboBox()
        Model_select_regmodel_widget = QComboBox()
        Model_select_layout.addRow("Detection", Model_select_detmodel_widget)
        Model_select_layout.addRow("Regression", Model_select_regmodel_widget)

        Mode_select_radio_layout = QHBoxLayout()
        Mode_select_radio_widget_1 = QRadioButton("Train")
        Mode_select_radio_widget_2 = QRadioButton("Validate")
        Mode_select_radio_widget_3 = QRadioButton("Detect")
        Mode_select_radio_layout.addWidget(Mode_select_radio_widget_1)
        Mode_select_radio_layout.addWidget(Mode_select_radio_widget_2)
        Mode_select_radio_layout.addWidget(Mode_select_radio_widget_3)


        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(-50, 50)  # 슬라이더의 범위 (0부터 100까지)
        self.slider.setValue(-20)  # 초기 값 50으로 설정
        self.slider.setTickInterval(1)  # 눈금 간격을 10으로 설정
        self.slider.setTickPosition(QSlider.TicksBelow)  # 눈금을 슬라이더 아래에 표시

        self.slider.valueChanged.connect(self.update_label)

        self.label = QLabel('Value: -0.2', self)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.label)


        metrics_layout = QGridLayout()
        metric_widgets = [QCheckBox("Separate Correlation Coefficient (Pearson)"), 
                          QCheckBox("Mean Squared Error (MSE)"),
                          QCheckBox("Mean Absolute Error (MAE)"),
                          QCheckBox("Root Mean Squared Error (RMSE)"),
                          QCheckBox("Bland-Altman Bias"),
                          QCheckBox("R²"),
                          QCheckBox("Calibration Slope (CITL)"),
                          QCheckBox("Accuracy"),
                          QCheckBox("Sensitivity"),
                          QCheckBox("Specificity")]

        for i in range(len(metric_widgets)):
            metrics_layout.addWidget(metric_widgets[i], i//4, i%4)


        save_path_layout = QFormLayout()
        save_path_sub_layout_1 = QHBoxLayout()
        save_path_sub_layout_2 = QHBoxLayout()

        excel_save_path_widget = QLineEdit()
        excel_save_path_search = QPushButton("search")

        image_save_path_widget = QLineEdit()
        image_save_path_type_combo = QComboBox()
        image_save_path_type_combo.addItems(['jpg', 'jpeg', 'png', 'gif'])
        image_save_path_search_button = QPushButton("search")

        save_path_sub_layout_1.addWidget(excel_save_path_widget)
        save_path_sub_layout_1.addWidget(excel_save_path_search)

        save_path_sub_layout_2.addWidget(image_save_path_widget)
        save_path_sub_layout_2.addWidget(image_save_path_type_combo)
        save_path_sub_layout_2.addWidget(image_save_path_search_button)

        save_path_layout.addRow("Excel save folder path", save_path_sub_layout_1)
        save_path_layout.addRow("Image save folder path", save_path_sub_layout_2)


        save_cancel_layout = QHBoxLayout()
        save_button = QPushButton('Start')
        cancel_button = QPushButton('Cancel')
        save_cancel_layout.addWidget(save_button)
        save_cancel_layout.addWidget(cancel_button)

        self_intro = QPlainTextEdit()

        sub_layout.addRow("Model Select", Empty_layout)
        sub_layout.addRow(" ", Model_select_layout)
        
        sub_layout.addRow("Mode Select", Empty_layout)
        sub_layout.addRow(" ", Mode_select_radio_layout)
        
        sub_layout.addRow("Preference", Empty_layout)
        
        sub_layout.addRow("* z-score threshold", Empty_layout)
        sub_layout.addRow("", slider_layout)

        sub_layout.addRow("* metrics to use (Supported at Train / Validate)", Empty_layout)
        sub_layout.addRow("", metrics_layout)

        sub_layout.addRow("* result saving method", Empty_layout)
        sub_layout.addRow("", save_path_layout)
        sub_layout.addRow(save_cancel_layout)
        sub_layout.addRow(self_intro)

        main_layout.addLayout(sub_layout)

        self.setLayout(main_layout)
        self.resize(700, 700)
        self.show()

    def update_label(self, value):
        float_value = value / 100.0
        self.label.setText(f'Value: {float_value:.2f}')

    def update_slider(self, value):
        slider_value = int(value * 100)
        self.slider.setValue(slider_value)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    sys.exit(app.exec_())