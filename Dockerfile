# Base image 선택
FROM ultralytics/ultralytics
RUN apt-get update && apt-get install -y libgl1

# 작업 디렉터리 생성
WORKDIR /app

# 백엔드 코드 및 유틸리티 복사
COPY image_qt_backend.py /app/
COPY bmd_analysis.py /app/
COPY utils/ /app/utils/
COPY requirements.txt /app/

# 필요한 Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# torch와 torchvision 설치 (CUDA 11.3 지원 버전)
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Flask 서버 실행
CMD ["python", "image_qt_backend.py"]