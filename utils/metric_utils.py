import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import PIL as plt

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