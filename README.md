# PBD-PSR 사용 가이드

# To-Do
- [x] block으로 인한 프리즈 현상 해결
- [x] Hip code 이식
- [x] UI 개선 (Progress bar, 전체적인 직관성 개선)
- [-] Train 모드 개선 (25/3/27 기준 Test만 구현 예정)
- [x] 준비물 간소화 (jpg 준비할 필요 없이 dicom만 사용, 엑셀파일 -> txt파일로 형식 개선 : 의논 필요)
- [ ] GT 사용 / 미사용 모드 추가
- [ ] 프로그램 휴대성 확장

## 1. 개요
이 프로젝트는 Flask 서버와 PyQt 클라이언트를 사용하여 실시간 영상 분석을 수행한다.  
- Flask 서버는 데이터를 분석하고, 스트리밍 방식으로 클라이언트에 전달한다.  
- PyQt 클라이언트는 서버로부터 데이터를 받아 UI에서 실시간으로 표시한다.  

---

## 2. 설치 방법

### 1) 가상 환경 생성 (선택 사항)
```sh
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows
```

### 2) 필수 패키지 설치
```sh
pip install -r requirements.txt
```

### 3) 실행 방법

#### 1) Docker image 다운로드 및 Container 생성
- 해당 프로그램은 Docker을 필요로합니다.
  - https://www.docker.com/

- dockerfile을 통해 이미지 빌드
```docker
docker build -t my_custom_image .
```
- image를 활용해 container 생성
```sh
docker run --name bmd_container --gpus all -it --entrypoint /bin/bash -p 8080:80 -p 5000:5000 -v .:/app/workspace bmd_backend 
```

#### 2) Flask 서버 실행

```sh
python server.py
```

#### 3) PyQt 클라이언트 실행
```sh
python main.py
```

### 4) 프로젝트 구조

```bash
BMD_code/
├── server.py               # Flask 서버
├── main.py                 # PyQt 클라이언트
├── bmd_analysis.py         # 분석 로직
├── utils/                  # 유틸리티 함수
│   ├── regression_model.py
│   ├── ultralytics_custom_utils.py
├── assets/                 # UI 리소스 (아이콘, 이미지 등)
├── requirements.txt        # 필요한 패키지 목록
├── README.md               # 프로젝트 설명
└── github.md               # 이 문서
```

### 5) 기능 설명

실시간 분석
- PyQt에서 UI 버튼 클릭 시 Flask 서버에 요청을 보내고 실시간 분석을 수행함.
백그라운드 스레드 처리 (QThread)
- 네트워크 요청을 백그라운드에서 실행하여 UI가 멈추지 않도록 함.
스트리밍 데이터 처리
- 서버가 yield를 사용하여 데이터를 지속적으로 전송함.
- 클라이언트가 requests.post(..., stream=True)로 데이터를 실시간으로 받아 UI에 표시함.