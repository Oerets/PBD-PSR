    def update_image(self):
        if 0 <= self.current_image_index < len(self.image_files):
            pixmap = QPixmap(self.image_files[self.current_image_index])
            if not pixmap.isNull():
                pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
            else:
                self.image_label.setText("Invalid Image Path")


'''
실제로 구현해야하는거는 다음과 같다.
1. detection in progress
2. 서버로 일단 모든 정보 보내기. (서버에 데이터 보내놓은채로 쓸지, 아니면 데이터를 그때그때 보낼지는 고민좀 해봐야할 듯)



최종 : 왼쪽의 이미지 부분에 bbox 띄우고, regression 결과 각 박스별로 적어둘 수 있도록. 이 정보를 따로 excel파일로 만들어서 저장할 필요도 있음.

(그럼 두 단계로 진행해야하는데..)
detection할 때 정보 보내고, 서버쪽에서 정보 기반으로 세팅해서 detection 돌리고, detection 결과 반환하고, 그대로 가지고있던 정보 가지고 regression돌리고, regression 결과 반환하고. 깔끔하죠?

근데 세팅이 여러가지가 있을 수 있어서 여기서부터 좀 헷갈리기 시작하는거.
(도그푸딩을 좀 하자 - eat your own dog food ㅋㅋㅋㅋㅋㅋ)

존재할 수 있는 세팅이 뭐가있을까?
일단 test/train/validation?(val은 빼자)

지금 train할때 dicom데이터도 필요하고~ excel파일도 필요하고~ jpg파일도 필요하고~ 그냥 간소화 해서 dicom버전 하나, jpg 버전 하나 이렇게 만들어서 따로 돌리자. 그러면 data mode로 하나 설정하고 바로 쓸 수 있으니까 조~금 간소화됨.

결과 이미지 저장 모드 유/무
결과 엑셀 저장 모드 유/무
결과 bbox 좌표 저장 모드 유/무

생각해보니 지금 데이터 위치 할당해주면.. 그거 내부에서 어떻게 쓸건데? (그냥 특정 폴더 지정해서 사용하게 만들면 되겠다. 이거 사용자 매뉴얼을 좀 빡세게 만들어야겠는데?)

정사각형 레이아웃으로 잘 만들어보자.
'''