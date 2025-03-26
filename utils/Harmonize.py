import numpy as np
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
#from boxcutter import *

########################################################################

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

################################################################################

def normalize_dicom_image(dicom_path):
    """
    DICOM 데이터를 가져와 이미지 처리를 거쳐 0~1로 정규화된 numpy 배열을 반환하는 함수
    """
    dicom_data = pydicom.dcmread(dicom_path)
    pixel_array = dicom_data.pixel_array.astype(np.float32)

    # DICOM 기본정보 추출
    rescale_slope = getattr(dicom_data, "RescaleSlope", 1)
    rescale_intercept = getattr(dicom_data, "RescaleIntercept", 0)
    window_center = getattr(dicom_data, "WindowCenter", None)
    window_width = getattr(dicom_data, "WindowWidth", None)
    photometric_interpretation = getattr(dicom_data, "PhotometricInterpretation", "MONOCHROME2")
    
    # Rescale 적용
    image = pixel_array * rescale_slope + rescale_intercept

    # Photometric Interpretation 적용
    if photometric_interpretation == "MONOCHROME1":
        image = np.max(image) - image

    # Window Level 적용
    if isinstance(window_center, pydicom.multival.MultiValue):
        window_center = window_center[0]
    if isinstance(window_width, pydicom.multival.MultiValue):
        window_width = window_width[0]
    
    if window_center and window_width:
        lower_bound = window_center - (window_width / 2)
        upper_bound = window_center + (window_width / 2)
        image = np.clip(image, lower_bound, upper_bound)
    
    # 정규화 (0~1 범위로 변환)
    image_min, image_max = np.min(image), np.max(image)
    if image_max != image_min:
        image = (image - image_min) / (image_max - image_min) * 255
    else:
        image = np.zeros_like(image)
    
    return image

def dicom_image_to_pil(input_image, mode="image", output_path=None):
    """
    DICOM 데이터를 받아서 PIL 이미지 또는 히스토그램을 저장하는 함수
    
    Args:
        dicom_data: DICOM 파일 데이터
        mode (str): "image" 또는 "histogram" 선택
        output_path (str, optional): 저장할 파일 경로
    """
    normalized_image = input_image
    image_8bit = (normalized_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_8bit)
    
    if mode == "image":
        if output_path:
            pil_image.save(output_path)
        return pil_image
    elif mode == "histogram":
        plt.figure()
        plt.hist(normalized_image.ravel(), bins=256, color='black', alpha=0.7)
        plt.title("Histogram of DICOM Image")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show(block=True)
    else:
        raise ValueError("Invalid mode. Choose 'image' or 'histogram'.")

# 예제 사용
if __name__ == "__main__":
    dicom_file_path = "C:/Users/hyunoh/Documents/Codes/BMD_code/data/Hip_data/dcm/test/1542964_190329.dcm"  # DICOM 파일 경로
    label_file = "C:/Users/hyunoh/Documents/Codes/BMD_code/data/Hip_data/bbox_labels/test/1542964_190329.txt"
    
    dicom_data = pydicom.dcmread(dicom_file_path)
    image = normalize_dicom_image(dicom_data)
    label = process_label_file(label_file)
    cropped_image = crop_dicom_image(image, label)
    
    # 이미지 저장 및 출력 
    # pil_image = dicom_image_to_pil(image, mode="image", output_path="output.png")
    # pil_image2 = dicom_image_to_pil(cropped_image, mode="image", output_path="cropped_output.png")
    # pil_image.show()
    # pil_image2.show()
    # 히스토그램 저장 및 출력
    # dicom_image_to_pil(image, mode="histogram", output_path="histogram.png")
    # dicom_image_to_pil(cropped_image, mode="histogram", output_path="cropped_histogram.png")