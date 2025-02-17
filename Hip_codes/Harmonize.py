import numpy as np
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
from boxcutter import *

def normalize_dicom_image(dicom_data):
    """
    DICOM 데이터를 가져와 이미지 처리를 거쳐 0~1로 정규화된 numpy 배열을 반환하는 함수
    """
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
        image = (image - image_min) / (image_max - image_min)
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