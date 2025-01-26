import os
import logging
import warnings
import numpy as np
import pandas as pd
from glob import glob
from ultralytics import YOLO
from scipy.stats import pearsonr
from utils.metric_utils import calculate_metrics
from utils.regression_model import load_regression_model
from utils.ultralytics_custom_utils import Regression_process

logging.getLogger('SimpleITK').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def bmd_analysis(mode, weighted_mode, det_model_path, det_model_name, reg_model_path, reg_model_name, data_path, excel_path, dicom_path, z_threshold):
    test_name = f'{mode}_{det_model_path}_{reg_model_path}'
    print(test_name)
    det_model = YOLO(det_model_path)
    test_files = glob(f'{data_path}/*')

    # Excel 데이터 로드
    bmd_data = pd.read_excel(excel_path)

    box_mode = 'obbox' if 'obb' in os.path.basename(det_model_path).lower() else 'bbox'
    
    results_df = pd.DataFrame(columns=[
        'basename', 'pred_bmd_score_mean', 'gt_bmd_score', 'boolean_mean', 'z_class', 'z_gt_class',
        'weighted_mean', 'separate_gt', 'separate_pred'
    ])
    
    detection_results = det_model(test_files, max_det=4, save_txt=True, save=True)
    print("Detection completed")

    for i, r in enumerate(detection_results):
        image_basename = os.path.basename(r.path).split('.')[0]
        save_dir = r.save_dir
        r_label = save_dir + '/labels/' + image_basename + '.txt'
        r_shape = r.orig_shape
        
        result = Regression_process(
            mode, box_mode, reg_model_name, reg_model_path, r_shape, r_label, dicom_path, bmd_data, z_threshold
        )
        
        new_index = len(results_df)
        if mode in ['Train', 'Validation']:
            if weighted_mode:
                results_df.loc[new_index] = [
                    result['image_basename'],
                    result['pred_bmd_score_mean'],
                    result['gt_bmd_score'],
                    result['class_result_weighted_mean'],
                    result['z_class_w'],
                    result['z_gt_class'],
                    result['pred_bmd_score_weighted_mean'],
                    result['gt_bmd_list'],
                    result['bmd_list_cpu']
                ]
            else:
                results_df.loc[new_index] = [
                    result['image_basename'],
                    result['pred_bmd_score_mean'],
                    result['gt_bmd_score'],
                    result['class_result_mean'],
                    result['z_class'],
                    result['z_gt_class'],
                    result['pred_bmd_score_weighted_mean'],
                    result['gt_bmd_list'],
                    result['bmd_list_cpu']
                ]
        else:
            if weighted_mode:
                results_df.loc[new_index] = [
                    result['image_basename'],
                    result['pred_bmd_score_mean'],
                    None,
                    result['class_result_weighted_mean'],
                    result['z_class_w'],
                    None,
                    result['pred_bmd_score_weighted_mean'],
                    None,
                    None
                ]
            else:
                results_df.loc[new_index] = [
                    result['image_basename'],
                    result['pred_bmd_score_mean'],
                    None,
                    result['class_result_mean'],
                    result['z_class'],
                    None,
                    result['pred_bmd_score_mean'],
                    None,
                    None
                ]
        
        print(f"Processing image {i + 1}/{len(detection_results)}")

    # 메트릭 계산 (Train 및 Validation 모드에서만)
    if mode in ['Train', 'Validation']:
        mse, mae, rmse, r2, bias, calibration_slope = calculate_metrics(
            results_df['pred_bmd_score_mean'], results_df['gt_bmd_score']
        )
        separate_mse, separate_mae, separate_rmse, separate_r2, separate_bias, separate_calibration_slope = calculate_metrics(
            results_df['separate_pred'].explode(), results_df['separate_gt'].explode()
        )
        print("Metric calculation completed")

    # 분석 요약 출력
    result_summary = "결과 요약..."  # (예제 요약 텍스트)
    print("Analysis complete")
    print(result_summary)

# 테스트 실행
if __name__ == "__main__":
    # 인자를 직접 설정하여 함수 실행
    mode = 'Test'
    weighted_mode = False
    det_model_path = 'C:/Users/hyunoh/Documents/Codes/BMD_code/weights/obb_yolo_last.pt'
    det_model_name = 'YOLOv8'
    reg_model_path = 'C:/Users/hyunoh/Documents/Codes/BMD_code/weights/res50.pt'
    reg_model_name = 'resnet50'
    data_path = 'C:/Users/hyunoh/Documents/Codes/BMD_code/data/images/test'
    excel_path = 'C:/Users/hyunoh/Documents/Codes/BMD_code/updated_data.xlsx'
    dicom_path = 'C:/Users/hyunoh/Documents/Codes/BMD_code/data/dxa_data'
    z_threshold = 1.0

    bmd_analysis(mode, weighted_mode, det_model_path, det_model_name, reg_model_path, reg_model_name, data_path, excel_path, dicom_path, z_threshold)