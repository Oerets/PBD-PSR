def process_label_file(txt_file_path):
    """
    주어진 txt 파일을 분석하여 사용할 라벨을 선택하고 반환하는 함수
    
    Args:
        txt_file_path (str): 라벨이 포함된 텍스트 파일 경로
    
    Returns:
        list: 선택된 라벨 박스 정보 (리스트 형태)
    """
    with open(txt_file_path, "r") as f:
        lines = f.readlines()
    
    label_dict = {}
    for line in lines:
        parts = line.strip().split()
        label = int(parts[0])
        bbox = list(map(float, parts[1:]))
        if label not in label_dict:
            label_dict[label] = bbox
    
    for priority_label in [0, 1, 2, 3, 4, 5]:  # 0 우선, 없으면 1, 그 외는 고려 안함
        if priority_label in label_dict:
            return label_dict[priority_label]
    
    return None  # 사용 가능한 라벨이 없는 경우

def crop_dicom_image(input_image, bbox):
    """
    정규화된 DICOM 이미지에서 주어진 바운딩 박스를 기준으로 잘라낸 이미지를 반환하는 함수
    """
    normalized_image = input_image
    h, w = normalized_image.shape
    
    x_center, y_center, width, height = bbox
    x_min = int((x_center - width / 2) * w)
    x_max = int((x_center + width / 2) * w)
    y_min = int((y_center - height / 2) * h)
    y_max = int((y_center + height / 2) * h)
    
    cropped_image = normalized_image[y_min:y_max, x_min:x_max]
    return cropped_image