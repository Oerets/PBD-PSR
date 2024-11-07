import os
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from .regression_model import load_regression_model
import cv2
import warnings
import logging
logging.getLogger('SimpleITK').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

#input : txt_dir, dicom_dir, excel_dir, mode, regression_dir
#output : accuracy

def process_image(image, margin_modifier, transform=None):
    # 이미지 정규화
    image = (image - image.min()) / (image.max() - image.min())

    # Margin 적용
    modifiedX = int(image.shape[0] * margin_modifier)
    modifiedY = int(image.shape[1] * margin_modifier)
    image = image[modifiedX:-modifiedX, modifiedY:-modifiedY]
    image = np.repeat(image[:, :, None], 3, axis=-1)

    # 변환 적용
    if transform:
        image = transform(image=image)["image"]
        
    return image

def torch_image(image, margin):
    margin_modifier = ((1 - margin) / ((1 + 1) * 2))
    processed_image = process_image(image, margin_modifier)
    image_tensor = torch.tensor(processed_image).unsqueeze(0).float()
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    
    return image_tensor

def polygon_to_bbox(poly):

    xs = [x for x in poly[::2]]
    ys = [y for y in poly[1::2]]

    return np.array((min(xs), min(ys), max(xs), max(ys)))

def crop_roi_with_margin(img, bbox, margin=1):
    # Image's height and width
    img_h, img_w = img.shape[0:2]

    if isinstance(bbox, list):
        bbox = np.array(bbox)

    # Bounding box coordinates
    x1, y1, x2, y2 = bbox.astype(int)
    
    width = x2 - x1
    height = y2 - y1
    
    w_margin = int(width * margin) // 2
    h_margin = int(height * margin) // 2
    
    # Adding margin but also ensuring that it doesn't exceed the image boundaries
    x1 = max(x1 - w_margin, 0)
    y1 = max(y1 - h_margin, 0)
    x2 = min(x2 + w_margin, img_w)
    y2 = min(y2 + h_margin, img_h)
    
    return img[y1:y2, x1:x2]

def find_bounding_box(coord_list):
    
    x1, y1, x2, y2, x3, y3, x4, y4 = coord_list
    
    # 주어진 점들을 numpy 배열로 변환
    points = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

    # 각 점들의 x, y 좌표를 따로 추출
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # MBB를 구성하는 왼쪽 상단 모서리와 오른쪽 하단 모서리 계산
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)

    # MBB의 너비와 높이 계산
    width = max_x - min_x
    height = max_y - min_y

    # MBB의 중심점 계산
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    return center_x, center_y, width, height

def parse_bbox_file(file_path, mode, r_shape):
    bboxes = []  # 바운딩 박스를 저장할 리스트
    height, width = r_shape  # 이미지 높이와 너비

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if mode == 'obb':  # c, x1, y1, x2, y2, x3, y3, x4, y4
                # 클래스 레이블(c)을 제외하고 좌표만 추출하여 float 형태로 변환
                coords = list(map(float, parts[1:]))  # x1, y1, x2, y2, x3, y3, x4, y4
                # 정규화된 좌표를 실제 좌표로 변환
                bbox = [coord * width if i % 2 == 0 else coord * height for i, coord in enumerate(coords)]
                bboxes.append(bbox)
            elif mode == 'coco': # c, x, y, w, h
                # 클래스 레이블(c)을 제외하고 좌표만 추출하여 float 형태로 변환
                predicted_bbox = list(map(float, parts[1:]))
                if len(predicted_bbox) == 8:
                    x, y, w, h = find_bounding_box(predicted_bbox)
                else : 
                    x, y, w, h = predicted_bbox  # x, y, w, h
                # 정규화된 중심 좌표와 크기를 실제 좌표로 변환
                x, y, w, h = x * width, y * height, w * width, h * height
                x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
                bboxes.append([x1, y1, x2, y2])
    return bboxes


def load_dicom_image(dicom_path):
    """ DICOM 이미지를 불러오는 함수 """
    dicom_image = sitk.GetArrayFromImage(sitk.ReadImage(dicom_path))[0]
    return dicom_image


def crop_image(image, coord, mode):
    """ 이미지를 주어진 좌표에 따라 자르는 함수 """
    if mode == 'obb':
        coord = polygon_to_bbox(coord)
    cropped_image = crop_roi_with_margin(image, coord, margin=0.3)
    return cropped_image

def crop_masked_image(image, coord, mode):
    """ 이미지를 주어진 좌표에 따라 자르는 함수 """
    if mode == 'obb':
        coord = polygon_to_bbox(coord)
    cropped_image = crop_roi_with_margin(image, coord, margin=0.3)
    return cropped_image

def mask_polygon_area(image, polygon):
    """ 주어진 다각형 내부 영역을 마스킹하는 함수 """
    # Create a mask with the same dimensions as the image
    mask = np.zeros_like(image)
    
    # Create a white filled polygon on the mask
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], (255,) * image.shape[2])
    
    # Mask the image using the created mask
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def Final_pipeline_yolov8(test_name, model_name, r_shape, txt_dir, dicom_dir, bmd_data, mode, regression_dir, z_threshold=-2):
    
    bmd_list = []
    bmd_size_list = []
    gt_bmd_list = []
    
    # Load Regression model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    regression_model = load_regression_model(model_name, regression_dir, device)
    
    #read single txt file
    bboxes = parse_bbox_file(txt_dir, mode, r_shape)
    #read single dicom file
    image_basename = os.path.basename(txt_dir).split('.')[0]
    dicom_dir = dicom_dir + '/' + image_basename + '.dcm'
    dcm_img = load_dicom_image(dicom_dir)

    
    #read single excel file
    row = bmd_data[bmd_data['ID'].astype(str) == image_basename]
    ref_mean = row['Mean'].values[0] if not row.empty else None
    ref_std = row['SD'].values[0] if not row.empty else None
    
    gt_bmd_score = row['Total'].values[0] if not row.empty else None
    gt_z_score = row['Zscore_Total'].values[0] if not row.empty else None
    
    
    gt_bmd_l1 = row['L1'].values[0] if not row.empty else None
    gt_bmd_l2 = row['L2'].values[0] if not row.empty else None
    gt_bmd_l3 = row['L3'].values[0] if not row.empty else None
    gt_bmd_l4 = row['L4'].values[0] if not row.empty else None
    
    gt_bmd_list.append(gt_bmd_l1)
    gt_bmd_list.append(gt_bmd_l2)
    gt_bmd_list.append(gt_bmd_l3)
    gt_bmd_list.append(gt_bmd_l4)
    
    #crop image
    bboxes = sorted(bboxes, key=lambda x: x[1])
    
    for i, bbox in enumerate(bboxes):
        if mode == "obb":
            cropped_image  = crop_masked_image(dcm_img, bbox, mode)
        else :    
            cropped_image = crop_image(dcm_img, bbox, mode)
        # cropped_image = crop_image(dcm_img, bbox, mode)
        regression_image = torch_image(cropped_image, 0.5)
        regression_image = regression_image.to(device)
        bmd_size_list.append(regression_image.shape[2]*regression_image.shape[3])
        
        #predict with the model
        prediction = regression_model(regression_image)
        bmd_list.append(prediction[0][0])
    
    #mean, weighted mean
    bmd_list_cpu = [x.detach().cpu().numpy() if x.is_cuda else x.detach().numpy() for x in bmd_list]

    # bmd_size_list의 각 항목에 대해, 텐서인 경우 CPU로 이동 및 NumPy 배열로 변환
    bmd_size_list_cpu = [x.detach().cpu().numpy() if torch.is_tensor(x) and x.is_cuda else x.numpy() if torch.is_tensor(x) else x for x in bmd_size_list]

    #simple mean
    pred_bmd_score_mean = np.mean(bmd_list_cpu)
    #weighted mean
    pred_bmd_score_weighted_mean = np.sum([x * y for x, y in zip(bmd_list_cpu, bmd_size_list_cpu)]) / np.sum(bmd_size_list_cpu)

    columns = [ test_name+'_L1', test_name+'_L2', test_name+'_L3', test_name+'_L4']
    
    # 필요한 열이 없으면 생성
    for col in columns:
        if col not in bmd_data.columns:
            bmd_data[col] = None
    
    indices = bmd_data[bmd_data['ID'].astype(str) == image_basename].index
    
    # indices의 길이를 검사하여 하나의 값만 있을 때만 item()을 사용
    # if len(indices) == 1:
    #     print(indices.item())
        
    # 각 예측값을 해당 열에 저장
    for i, prediction in enumerate(bmd_list):
        # 필요한 열이 없으면 생성
        if columns[i] not in bmd_data.columns:
            bmd_data[columns[i]] = None
        # 예측값을 열에 저장
        bmd_data.at[indices.item(), columns[i]] = prediction.item()

    # 평균 및 가중 평균 계산
    avgBMD = bmd_data.loc[indices, columns[:len(bmd_list)]].mean(axis=1).values
    w_avgBMD = ((bmd_data.loc[indices, columns[:len(bmd_list)]] * bmd_size_list).sum(axis=1) / sum(bmd_size_list)).values

    pred_z_score_mean = (pred_bmd_score_mean - ref_mean) / ref_std
    pred_z_score_weighted_mean = (pred_bmd_score_weighted_mean - ref_mean) / ref_std
    
    bmd_data.at[indices.item(), '_a'] = avgBMD.item()
    bmd_data.at[indices.item(), '_w_a'] = w_avgBMD.item()
    bmd_data.at[indices.item(), '_z'] = pred_z_score_mean.item()
    bmd_data.at[indices.item(), '_w_z'] = pred_z_score_weighted_mean.item()
    
    #z-score : simple mean
    z_class = 0 if pred_z_score_mean < z_threshold else 1
    z_gt_class = 0 if gt_z_score < z_threshold else 1
    class_result_mean = 1 if z_class == z_gt_class else 0

    #z-score : weighted mean
    z_class_w = 0 if pred_z_score_weighted_mean < z_threshold else 1
    z_gt_class = 0 if gt_z_score < z_threshold else 1
    class_result_weighted_mean = 1 if z_class_w == z_gt_class else 0
    
    # print(" ")
    # print(image_basename)
    # print(bmd_list_cpu)
    # print(f"gt : {gt_bmd_score}")
    # print(pred_z_score_mean)
    # print(pred_z_score_weighted_mean)
    
    if len(bmd_list_cpu) < 4:
        for i in range(0, 4 - len(bmd_list_cpu)):
            bmd_list_cpu.append(np.array(0,dtype='float32'))
    
    return image_basename, pred_bmd_score_weighted_mean, class_result_mean, class_result_weighted_mean, pred_bmd_score_mean, gt_bmd_score, bmd_data, z_class, z_gt_class, z_class_w, gt_bmd_list, bmd_list_cpu