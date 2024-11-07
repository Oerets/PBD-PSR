#organized code
from scipy.stats import pearsonr
from ultralytics import YOLO
import os
from glob import glob
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils.ultralytics_custom_utils import Final_pipeline_yolov8
import warnings
import logging
import pandas as pd
import openpyxl
import xlsxwriter
logging.getLogger('SimpleITK').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Load YOLO model
def load_model(mode, model_name):
    if mode == 'obb':
        model = YOLO('yolov8n-obb.pt')
        model = YOLO('weights/obb_yolo_last.pt')
    elif mode == 'coco':
        model = YOLO('yolov8n.pt')
        model = YOLO('weights/coco_yolo_best.pt')
    return model

# Load regression model path
def get_regression_model_path(model_name):
    paths = {
        'resnet18': 'weights/res18.pt',
        'resnet50': 'weights/res50.pt',
        'vgg16': 'weights/vgg16.pt',
        'squeezenet': 'weights/squeezenet.pt',
        'efficientnet': 'weights/effnet.pt'
    }
    return paths.get(model_name)

# Calculate metrics
def calculate_metrics(gt_list, pred_list):
    mse = mean_squared_error(gt_list, pred_list)
    mae = mean_absolute_error(gt_list, pred_list)
    rmse = np.sqrt(mse)
    r2 = r2_score(gt_list, pred_list)
    bias = np.mean(np.array(gt_list) - np.array(pred_list))
    reg = LinearRegression().fit(np.array(pred_list).reshape(-1, 1), np.array(gt_list).reshape(-1, 1))
    calibration_slope = reg.coef_[0][0]
    
    return mse, mae, rmse, r2, bias, calibration_slope

# Calculate sensitivity and specificity
def calculate_sensitivity_specificity(gt_scores, pred_scores, threshold):
    gt_scores = np.array(gt_scores)
    pred_scores = np.array(pred_scores)
    
    tp = np.sum((pred_scores > threshold) & (gt_scores > threshold))
    fp = np.sum((pred_scores > threshold) & (gt_scores <= threshold))
    tn = np.sum((pred_scores <= threshold) & (gt_scores <= threshold))
    fn = np.sum((pred_scores <= threshold) & (gt_scores > threshold))
    
    sensitivity = tp / (tp + fn) if tp + fn != 0 else 0
    specificity = tn / (tn + fp) if tn + fp != 0 else 0
    
    return sensitivity, specificity

# Plot and save correlation graph
def plot_correlation(pred_list, gt_list):
    plt.scatter(pred_list, gt_list)
    plt.title('Correlation')
    plt.xlabel('Predicted BMD Score')
    plt.ylabel('Ground Truth BMD Score')
    
    z = np.polyfit(pred_list, gt_list, 1)
    p = np.poly1d(z)
    plt.plot(pred_list, p(pred_list), "r--")
    
    plt.savefig('correlation_graph.png')
    plt.close()

# Main function
def main():
    
    #############################################
    #엑셀 생성 코드
    # 새 XLSX 파일을 생성하고 workbook 객체를 생성합니다.
    workbook = xlsxwriter.Workbook('final.xlsx')
    # workbook 객체 내에 worksheet 객체를 추가합니다.
    worksheet = workbook.add_worksheet('bmds')
    worksheet2 = workbook.add_worksheet('weighted_mean')
    #############################################
    
    mode_list = ['obb', 'coco']
    model_list = ['resnet18','vgg16','squeezenet','efficientnet']
    for index, mode in enumerate(mode_list):
        for index_2, model_name in enumerate(model_list):
            test_name = f"{mode},{model_name}"
            #############################################
            #엑셀 열 이름 추가
            worksheet.write(0, index * 2 * len(model_list) + index_2 * 2 + 1, f"{test_name}_pred")
            worksheet.write(0, index * 2 * len(model_list) + index_2 * 2 + 2, f"{test_name}_gt")
            worksheet2.write(0, index * len(model_list) + index_2 + 1, f"{test_name}_pred")
            #############################################
    
            model = load_model(mode, model_name)
            regression_model_path = get_regression_model_path(model_name)
            
            test_files = glob('data/images/test/*')
            excel_dir = 'updated_data.xlsx'
            dicom_dir = 'data/dxa_data'
            
            results = model(test_files, max_det=4, save_txt=True, save=True)
            
            bmd_data = pd.read_excel(excel_dir)
            pred_bmd_score_mean_list = []
            gt_bmd_score_list = []
            boolean_list_mean = []
            boolean_list_weighted_mean = []
            z_class_list = []
            z_class_w_list = []
            z_gt_class_list = []
            separate_gt_list = []
            separate_pred_list = []
            basename_list = []
            weighted_mean_list = []
            z_threshold = -2
            
            for r in results:
                image_basename = os.path.basename(r.path).split('.')[0]
                save_dir = r.save_dir
                r_label = save_dir + '/labels/' + image_basename + '.txt'
                r_shape = r.orig_shape
                
                image_basename, pred_bmd_score_weighted_mean, z_mean, z_weighted_mean, pred_bmd_score_mean, gt_bmd_score, bmd_data, z_class, z_gt_class, z_class_w, gt_bmd_list, bmd_list_cpu = Final_pipeline_yolov8(
                    test_name, model_name, r_shape, r_label, dicom_dir, bmd_data, mode, regression_model_path, z_threshold
                )
                
                for pred in bmd_list_cpu:
                    separate_pred_list.append(pred)
                for gt in gt_bmd_list:
                    separate_gt_list.append(gt)
                pred_bmd_score_mean_list.append(pred_bmd_score_mean)
                gt_bmd_score_list.append(gt_bmd_score)
                boolean_list_mean.append(z_mean)
                boolean_list_weighted_mean.append(z_weighted_mean)
                z_class_list.append(z_class)
                z_class_w_list.append(z_class_w)
                z_gt_class_list.append(z_gt_class)
                basename_list.append(image_basename)
                weighted_mean_list.append(pred_bmd_score_weighted_mean)
            
            mse, mae, rmse, r2, bias, calibration_slope = calculate_metrics(pred_bmd_score_mean_list, gt_bmd_score_list)
            separate_mse, separate_mae, separate_rmse, separate_r2, separate_bias, separate_calibration_slope = calculate_metrics(separate_pred_list, separate_gt_list)
            
            #correlation
            plot_correlation(pred_bmd_score_mean_list, gt_bmd_score_list)
            corr, p_value = pearsonr(pred_bmd_score_mean_list, gt_bmd_score_list)
            #separate correlation
            plot_correlation(separate_pred_list, separate_gt_list)
            separate_corr, separate_p_value = pearsonr(separate_pred_list, separate_gt_list)

            print("save excel")
            bmd_data.to_excel(f'updated_data_{test_name}.xlsx', index=False)
            
            #############################################
            #엑셀코드
            excel_data = [separate_pred_list, separate_gt_list]
            
            for row_num, value in enumerate(basename_list):
                    worksheet.write(row_num + 1, 0, value)
                    
            for col_num, column_data in enumerate(excel_data):
                for row_num, value in enumerate(column_data):
                    worksheet.write(row_num + 1, index * 2 * len(model_list) + index_2 * 2 + col_num + 1, value)
            
            for row_num, value in enumerate(basename_list):
                    worksheet2.write(row_num + 1, 0, value)
            
            excel_data2 = [weighted_mean_list]
            
            for col_num, column_data in enumerate(excel_data2):
                for row_num, value in enumerate(column_data):
                    worksheet2.write(row_num + 1, index * len(model_list) + index_2 + col_num + 1, value)
                    
            print("excel_printed")
            #############################################
            
            pred_bmd_score_mean = np.mean(pred_bmd_score_mean_list)
            gt_bmd_score_mean = np.mean(gt_bmd_score_list)
            accuracy_mean = sum(boolean_list_mean) / len(boolean_list_mean)
            accuracy_weighted_mean = sum(boolean_list_weighted_mean) / len(boolean_list_weighted_mean)
            
            #sensitivity, specificity
            sensitivity, specificity = calculate_sensitivity_specificity(z_gt_class_list, z_class_list, 0.5)
            sensitivity_w, specificity_w = calculate_sensitivity_specificity(z_gt_class_list, z_class_w_list, 0.5)
            
            result = (
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
                #f"Predicted BMD Score Mean : {pred_bmd_score_mean}\n"
                #f"Ground Truth BMD Score Mean : {gt_bmd_score_mean}\n"
                f"Accuracy_mean : {accuracy_mean * 100:.2f}%\n"
                f"Accuracy_weighted_mean : {accuracy_weighted_mean * 100:.2f}%\n"
                f"Sensitivity : {sensitivity:.2f}\n"
                f"Specificity : {specificity:.2f}\n"
                f"Sensitivity_w : {sensitivity_w:.2f}\n"
                f"Specificity_w : {specificity_w:.2f}\n"
                f"##########################################################################################\n\n"
            )
            with open("result_save.txt", "a") as file:
                file.write(result + "\n")                
    workbook.close()
if __name__ == "__main__":
    main()