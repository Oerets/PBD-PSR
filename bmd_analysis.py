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
from utils.ultralytics_custom_utils import Regression_process

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

    #여기서 settings에 따라 실행하고 결과 돌려주는 식으로. 새로 짜는게 더 효율적일 것 같다.
    if any(mode in settings_data.get('mode', []) for mode in ['Train', 'Validation']):
        print("Unavailable mode")
        return ["!![Warning]!! : This mode is unavailable for now."]

    mode = settings_data.get('mode', None)
    det_model_path = settings_data.get('det_model_path', None)
    reg_model_path = settings_data.get('reg_model_path', None)
    data_path = settings_data.get('data_path', None)
    excel_path = settings_data.get('excel_path', None)
    reg_model_name = settings_data.get('reg_model_type', None)
    dicom_path = settings_data.get('dicom_path', None)
    z_threshold = settings_data.get('z_threshold', None)
    weighted_mode = settings_data.get('weighted_mode', None)

    det_model_path = update_path(det_model_path)
    reg_model_path = update_path(reg_model_path)
    data_path = update_path(data_path)
    excel_path = update_path(excel_path)
    dicom_path = update_path(dicom_path)

    test_name = f'{mode}_{det_model_path}_{reg_model_path}'
    print(test_name)
    det_model = YOLO(det_model_path)
    test_files = glob(f'{data_path}/*')

    bmd_data = pd.read_excel(excel_path)

    box_mode = 'obbox' if 'obb' in os.path.basename(det_model_path).lower() else 'bbox'
    
    results_df = pd.DataFrame(columns=[
        'basename', 'pred_bmd_score_mean', 'gt_bmd_score', 'boolean_mean', 'z_class', 'gt_z_class',
        'weighted_mean', 'separate_gt', 'separate_pred'
    ])

    detection_results = det_model(test_files, max_det=4, save_txt=True, save=True)

    # 진행 상황 업데이트 - 감지 완료
    yield {"progress": 20, "status": "Detection completed, Regression starting"}

    for i, r in enumerate(detection_results):

        print(f'r : {r}')
        image_basename = os.path.basename(r.path).split('.')[0]
        save_dir = r.save_dir
        r_label = save_dir + '/labels/' + image_basename + '.txt'
        r_shape = r.orig_shape
        
        result = Regression_process(
            mode, box_mode, reg_model_name, reg_model_path, r_shape, r_label, dicom_path, bmd_data, z_threshold
        )
        
        new_index = len(results_df)
        
        print(f"results_df index length : {new_index}")
        
        if weighted_mode:
            results_df.loc[new_index] = [
                result['image_basename'],
                result['pred_bmd_score_mean'],
                result['gt_bmd_score'],
                result['class_result_weighted_mean'],
                result['z_class_w'],
                result['gt_z_class'],
                result['pred_bmd_score_weighted_mean'],
                result['gt_bmd_list'],
                result['bmd_list_cpu']
            ]
        else:
            results_df.loc[new_index] = [
                result['image_basename'],
                result['pred_bmd_score_mean'],
                result['gt_bmd_score'],
                result['class_result_weighted_mean'],
                result['z_class_w'],
                result['gt_z_class'],
                result['pred_bmd_score_weighted_mean'],
                result['gt_bmd_list'],
                result['bmd_list_cpu']
            ]

            yield result['gt_bmd_list']

            # if train mode
            # result['gt_bmd_score'],
            # result['z_gt_class'],
        
        print('yielding')

        # 진행 상황 업데이트 - 각 이미지 처리 완료
        yield {"progress": 20 + int((i + 1) / len(detection_results) * 60), "status": f"Processing image {i + 1}/{len(detection_results)}"}

    print('yielding loop ended')
    # 메트릭 계산 (Train 및 Validation 모드에서만)
    if mode in ['Train', 'Validation']:
        mse, mae, rmse, r2, bias, calibration_slope = calculate_metrics(
            results_df['pred_bmd_score_mean'], results_df['gt_bmd_score']
        )
        separate_mse, separate_mae, separate_rmse, separate_r2, separate_bias, separate_calibration_slope = calculate_metrics(
            results_df['separate_pred'].explode(), results_df['separate_gt'].explode()
        )
        # 진행 상황 업데이트 - 메트릭 계산 완료
        yield {"progress": 90, "status": "Metric calculation completed"}

    # print(f'summary result : {result_summary}')

    # # 분석 요약 반환
    # result_summary = "결과 요약..."  # (예제 요약 텍스트)
    # yield {"progress": 100, "status": "Analysis complete", "result": result_summary}