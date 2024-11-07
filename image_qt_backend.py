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
from utils.ultralytics_custom_utils import Regression_Process

logging.getLogger('SimpleITK').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def bmd_analysis(mode, weighted_mode, det_model_path, det_model_name, reg_model_path, reg_model_name, data_path, excel_path, dicom_path, z_threshold):
    """
    Bone Mineral Density (BMD) Analysis Function
    
    Parameters:
    - mode (str): 'Train', 'Validation', or 'Test' mode
    - weighted_mode (bool): Whether to use weighted mode
    - det_model_path (str): Path to detection model
    - det_model_name (str): Name of detection model
    - reg_model_path (str): Path to regression model
    - reg_model_name (str): Name of regression model
    - data_path (str): Path to test data
    - excel_path (str): Path to Excel file with ground truth BMD data (used in Train/Validation)
    - dicom_path (str): Path to DXA data (used in Train/Validation)
    - z_threshold (float): Z-score threshold
    
    Returns:
    - str: Analysis result summary
    """
    
    test_name = f'{mode}_{det_model_path}_{reg_model_path}'

    print(mode)
    print(f'{det_model_name} : {det_model_path}')
    print(f'{reg_model_name} : {reg_model_path}')
    print(weighted_mode)
    
    det_model = YOLO(det_model_path)
    test_files = glob(f'{data_path}/*')

    # Load Excel data only if in Train or Validation mode
    if mode in ['Train', 'Validation']:
        bmd_data = pd.read_excel(excel_path)
    else:
        bmd_data = None

    results_df = pd.DataFrame(columns=[
        'basename', 'pred_bmd_score_mean', 'gt_bmd_score', 'boolean_mean', 'z_class', 'z_gt_class',
        'weighted_mean', 'separate_gt', 'separate_pred'
    ])

    detection_results = det_model(test_files, max_det=4, save_txt=True, save=True)

    for r in detection_results:
        image_basename = os.path.basename(r.path).split('.')[0]
        save_dir = r.save_dir
        r_label = save_dir + '/labels/' + image_basename + '.txt'
        r_shape = r.orig_shape
        
        result = Regression_Process(
            reg_model_path, r_shape, r_label, dicom_path, bmd_data, mode, z_threshold
        )
        
        new_index = len(results_df)

        # Select weighted or unweighted data based on mode
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
    
    # Calculate metrics if in Train or Validation mode
    if mode in ['Train', 'Validation']:
        mse, mae, rmse, r2, bias, calibration_slope = calculate_metrics(
            results_df['pred_bmd_score_mean'], results_df['gt_bmd_score']
        )

        separate_mse, separate_mae, separate_rmse, separate_r2, separate_bias, separate_calibration_slope = calculate_metrics(
            results_df['separate_pred'].explode(), results_df['separate_gt'].explode()
        )

        # Calculate correlation
        plot_correlation(results_df['pred_bmd_score_mean'], results_df['gt_bmd_score'])
        corr, p_value = pearsonr(results_df['pred_bmd_score_mean'], results_df['gt_bmd_score'])

        # Separate correlation
        plot_correlation(results_df['separate_pred'].explode(), results_df['separate_gt'].explode())
        separate_corr, separate_p_value = pearsonr(
            results_df['separate_pred'].explode(), results_df['separate_gt'].explode()
        )

        # Calculate average scores and accuracy
        pred_bmd_score_mean = results_df['pred_bmd_score_mean'].mean()
        gt_bmd_score_mean = results_df['gt_bmd_score'].mean()
        accuracy_mean = results_df['boolean_mean'].mean()

        # Sensitivity and specificity
        sensitivity, specificity = calculate_sensitivity_specificity(
            results_df['z_gt_class'], results_df['z_class'], 0.5
        )

    # Compile result summary
    result_summary = (
        f"<{test_name}>\n"
        f"##########################################################################################\n"
        f"*****Separate Version*****\n"
        f"Separate Correlation Coefficient (Pearson): {separate_corr}\n"
        f"correlation_p_value : {separate_p_value}\n"
        f"Mean Squared Error (MSE) : {separate_mse}\n"
        f"Mean Absolute Error (MAE) : {separate_mae}\n"
        f"Root Mean Squared Error (RMSE) : {separate_rmse}\n"
        f"Bland-Altman Bias : {separate_bias}\n"
        f"R² (결정계수) : {separate_r2}\n"
        f"Calibration Slope (CITL) : {separate_calibration_slope}\n"
        f"*****Averaged Version*****\n"
        f"Correlation Coefficient (Pearson) : {corr}\n"
        f"correlation_p_value : {p_value}\n"
        f"Mean Squared Error (MSE) : {mse}\n"
        f"Mean Absolute Error (MAE) : {mae}\n"
        f"Root Mean Squared Error (RMSE) : {rmse}\n"
        f"Bland-Altman Bias : {bias}\n"
        f"R² (결정계수) : {r2}\n"
        f"Calibration Slope (CITL) : {calibration_slope}\n"
        f"Accuracy_mean : {accuracy_mean * 100:.2f}%\n"
        f"Sensitivity : {sensitivity:.2f}\n"
        f"Specificity : {specificity:.2f}\n"
        f"##########################################################################################\n\n"
    )

    # 결과를 파일에 저장
    with open("result_save.txt", "a") as file:
        file.write(result_summary + "\n")
    
    return result_summary

# Example usage
if __name__ == "__main__":
    result = bmd_analysis(
        mode='Train',
        weighted_mode=True,
        det_model_path='path/to/detection/model',
        det_model_name='yolo',
        reg_model_path='path/to/regression/model',
        reg_model_name='resnet18',
        data_path='path/to/data',
        excel_path='path/to/excel',
        dicom_path='path/to/dicom',
        z_threshold=-2.0
    )
    print(result)
