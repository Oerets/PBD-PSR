import os
import logging
import warnings
import numpy as np
import pandas as pd
from glob import glob
from ultralytics import YOLO
from scipy.stats import pearsonr
from utils.metric_utils import *
from utils.regression_model import load_regression_model
from utils.ultralytics_custom_utils import Vertebra_regression_process, Hip_regression_process

import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pydicom
from PIL import Image
from utils.Harmonize import *

logging.getLogger('SimpleITK').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def update_path(original_path):
    # 기존 경로에서 'C:/Users/hyunoh/Documents/Codes/'를 '/app/workspace'로 변경
    base_path = 'C:/Users/hyunoh/Documents/Codes/BMD_code'
    target_base_path = '/app/workspace/'

    # 경로가 'C:/Users/hyunoh/Documents/Codes/'로 시작하면 바꿔준다.
    if original_path.startswith(base_path):
        updated_path = target_base_path + original_path[len(base_path):]
    else:
        # 경로가 해당 부분으로 시작하지 않으면 그대로 반환
        updated_path = original_path
    
    return updated_path


def bmd_analysis(settings_data):

    #모델 경로로
    det_model_path = settings_data.get('det_model_path', None)
    reg_model_path = settings_data.get('reg_model_path', None)
    reg_model_name = settings_data.get('reg_model_type', None)

    #데이터 경로
    excel_path = settings_data.get('excel_path', None)
    dicom_path = settings_data.get('dicom_path', None)

    #Hip/Vertebra 종류 지정
    bodypart_mode = settings_data.get('bodypart_mode', None)
    z_threshold = settings_data.get('z_threshold', None)

    #Vertebra에서만 사용하는 것
    weighted_mode = settings_data.get('weighted_mode', None)

    #여기까지 하면, 이제 코드 실행 부분.

    #모델 지정
    det_model_path = update_path(det_model_path)
    reg_model_path = update_path(reg_model_path)
    
    excel_path = update_path(excel_path)
    print(excel_path)
    dicom_path = update_path(dicom_path)

    test_name = f'{det_model_path}_{reg_model_path}'
    det_model = YOLO(det_model_path)
    dcm_files = glob(f'{dicom_path}/*')

    bmd_data = pd.read_excel(excel_path)

    box_mode = 'obbox' if 'obb' in os.path.basename(det_model_path).lower() else 'bbox'

    # DICOM 이미지들을 한 번에 로드하여 리스트로 만듦
    images = [cv2.cvtColor(normalize_dicom_image(dcm_path), cv2.COLOR_GRAY2RGB) for dcm_path in dcm_files]

    #나중에 이 부분 메모리 문제 해결해야함.
    # YOLO에 배치 입력
    if bodypart_mode == "Vertebra":
        max_det_num = 4
        results_df = pd.DataFrame(columns=['basename', 'pred_bmd_score_mean', 'weighted_mean', 'bmd_list'])
    else:
        max_det_num = 2
        results_df = pd.DataFrame(columns=['basename', 'bmd_score'])

    total = len(images)
    results = []
    for i, result in enumerate(det_model(images, max_det=max_det_num, save=True, save_txt=True, stream=True)):
        progress = int((i + 1) / total * 50)
        yield {
            "progress": progress,
            "status": f"{i+1}/{total} detection completed"
        }
        results.append(result)

    # DataFrame 생성용 리스트
    rows = []

    for i, r in enumerate(results):
        data_shape = r.orig_shape
        data_basename = os.path.basename(r.path).split('.')[0]
        data_label_path = os.path.join(r.save_dir, "labels", data_basename + ".txt")
        rows.append([data_shape, data_label_path, dcm_files[i]])

    # DataFrame 생성
    detection_df = pd.DataFrame(rows, columns=['r_shape', 'r_label', 'dcm_path'])

    for i, row in detection_df.iterrows():
        
        progress = 50 + int((i + 1) / total * 50)
        yield {
            "progress": progress,
            "status": f"{i+1}/{total} regression completed"
        }

        r_label = row['r_label']
        r_shape = row['r_shape']
        dcm_path = row['dcm_path']
        
        if bodypart_mode == "Vertebra":
            result = Vertebra_regression_process(
                box_mode, reg_model_name, reg_model_path, r_shape, r_label, dcm_path, bmd_data, z_threshold
            )
            results_df.loc[i] = [
                result['image_basename'],
                result['pred_bmd_score_mean'],
                result['pred_bmd_score_weighted_mean'],
                result['bmd_list']
            ]
            subject_name = result['image_basename']
            result_bmd_clean = [float(x) for x in result['bmd_list']]
            yield f"{subject_name} : {result_bmd_clean}"
            
        elif bodypart_mode == "Hip":
            result = Hip_regression_process(reg_model_name, reg_model_path, r_shape, r_label, dcm_path, bmd_data, z_threshold)
            results_df.loc[i] = [
                result['image_basename'],
                result['pred_bmd_score']
            ]
            subject_name = result['image_basename']
            result_bmd_clean = float(result['pred_bmd_score'])
            yield f"{subject_name} : {result_bmd_clean}"

        else:
            yield "Unsupported body part"

    # 메트릭 계산 (Train 및 Validation 모드에서만) : Train/Val없는 현재는 생략
    # if mode in ['Train', 'Validation']:
    #     mse, mae, rmse, r2, bias, calibration_slope = calculate_metrics(
    #         results_df['pred_bmd_score_mean'], results_df['gt_bmd_score']
    #     )
    #     separate_mse, separate_mae, separate_rmse, separate_r2, separate_bias, separate_calibration_slope = calculate_metrics(
    #         results_df['separate_pred'].explode(), results_df['separate_gt'].explode()
    #     )
    #     # 진행 상황 업데이트 - 메트릭 계산 완료
    #     yield {"progress": 90, "status": "Metric calculation completed"}

    # print(f'summary result : {result_summary}')

    # # 분석 요약 반환
    # result_summary = "결과 요약..."  # (예제 요약 텍스트)
    # yield {"progress": 100, "status": "Analysis complete", "result": result_summary}

