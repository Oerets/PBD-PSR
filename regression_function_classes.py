#General
import os
import cv2
import random
import numpy as np
from glob import glob
#Visual
import pandas as pd
from sklearn.metrics import mean_squared_error

#Torch
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR, CyclicLR, OneCycleLR

#Augmentation
import albumentations as A
from albumentations import *
from albumentations.pytorch import ToTensorV2

#metrics
import wandb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

class UniformRandomResize(DualTransform):
    def __init__(self, scale_range=(0.9, 1.1), interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.scale_range = scale_range
        self.interpolation = interpolation

    def get_params_dependent_on_targets(self, params):
        scale = random.uniform(*self.scale_range)
        height = int(round(params['image'].shape[0] * scale))
        width = int(round(params['image'].shape[1] * scale))
        return {'new_height': height, 'new_width': width}

    def apply(self, img, new_height=0, new_width=0, interpolation=cv2.INTER_LINEAR, **params):
        return resize(img, height=new_height, width=new_width, interpolation=interpolation)

    def apply_to_keypoint(self, keypoint, new_height=0, new_width=0, **params):
        scale_x = new_width / params["cols"]
        scale_y = new_height / params["rows"]
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return "scale_range", "interpolation"

    @property
    def targets_as_params(self):
        return ["image"]
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_experiment_results(experiment_settings, test_results, excel_path, index):
    # 새로운 실험 결과 데이터 프레임 생성
    new_data_dict = {}
    new_data_dict.update(experiment_settings)  # experiment_settings의 키와 값을 추가
    new_data_dict.update({f'test_{key}': value for key, value in test_results.items()})

    new_data = pd.DataFrame([new_data_dict])

    # 기존 데이터 불러오기 또는 새 데이터프레임 생성
    try:
        existing_data = pd.read_excel(excel_path)
    except FileNotFoundError:
        # 모든 가능한 열 이름을 포함하는 새 데이터프레임 생성
        columns = list(experiment_settings.keys()) + [f'test_{key}' for key in test_results.keys()]
        existing_data = pd.DataFrame(columns=columns)

    # 새로운 데이터 추가
    if index == 2:
        empty_row = pd.DataFrame({col: '' for col in df.columns}, index=[0])
        new_data = pd.concat([new_data, empty_row], ignore_index=True)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)    

    # 업데이트된 데이터를 엑셀 파일로 저장
    updated_data.to_excel(excel_path, index=False)
        
def set_random_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if deterministic:
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
    
def get_gt_mean_std(bmd_excel_path):
    male_bmd_df = pd.read_excel(bmd_excel_path, sheet_name="Sheet2")
    female_bmd_df = pd.read_excel(bmd_excel_path, sheet_name="Sheet3")

    # 남성 및 여성 BMD 데이터를 사전 형태로 변환
    bmd_dict = {
        'male': male_bmd_df.set_index('Age').to_dict(orient='index'),
        'female': female_bmd_df.set_index('Age').to_dict(orient='index')
    }
    
    return bmd_dict

class BMDDataset(Dataset):
    def __init__(self, data_root, bmd_excel_path, enable_group_regression=0, margin=0, original_margin=0.5, transform=None):
        
        self.enable_group_regression = enable_group_regression
        self.margin = margin
        self.original_margin = original_margin
        self.transform = transform
        self.bmd_data = pd.read_excel(bmd_excel_path)
        img_list = sorted(glob(os.path.join(data_root, '*.npy')))
        
        if self.enable_group_regression == 0:
            self.img_list = img_list
        else:
            self.img_dict = {}
            for img_path in img_list:
                fname = os.path.basename(img_path)
                ID = fname.split('_L')[0]
                if ID not in self.img_dict:
                    self.img_dict[ID] = []
                self.img_dict[ID].append(img_path)
        try:
            self.male_bmd_df = pd.read_excel(bmd_excel_path, sheet_name="Sheet2")
            self.female_bmd_df = pd.read_excel(bmd_excel_path, sheet_name="Sheet3")
            self.bmd_data['ID'] = self.bmd_data['ID'].astype(str)
        except Exception as e:
            print(f"Error loading Excel file: {e}")

    def __len__(self):
        if self.enable_group_regression == 0:
            return len(self.img_list)
        else:
            return len(self.img_dict.keys())
    
    def process_image(self, img_path):
        image = np.load(img_path)

        # Margin 적용
        #modifiedX = int(image.shape[0] * self.margin_modifier)
        #modifiedY = int(image.shape[1] * self.margin_modifier)
        #image = image[modifiedX:-modifiedX, modifiedY:-modifiedY]
        #image = np.repeat(image[:, :, None], 3, axis=-1)

        self.transform = A.Compose([
            A.Resize(300, 300),
            A.Normalize(mean=(0.5,), std=(0.5,)),  # 정규화
            ToTensorV2()
        ])
        
        if self.transform:
            image = self.transform(image=image)["image"]

        return image
    
    def __getitem__(self, idx):
        self.margin_modifier = ((self.original_margin - self.margin) / ((self.original_margin + 1) * 2))
        if self.enable_group_regression == 1:
            ID = list(self.img_dict.keys())[idx]
            img_paths = self.img_dict[ID]
            images = [self.process_image(path) for path in img_paths]
            images_tensor = torch.stack(images, dim=0)  # 이미지를 스택하여 텐서로 변환
            bmds = torch.tensor([self.get_bmd(os.path.basename(path), ID) for path in img_paths], dtype=torch.float)
            meta_data = self.get_meta_data(ID)
            bmds = [bmd for bmd in bmds if bmd is not None] #L5와 같은 필요없는 데이터 필터링
            if meta_data is None:
                return None  # 조건에 맞지 않는 경우 None 반환

            return images_tensor, bmds, meta_data
    
        else:
            img_path = self.img_list[idx]
            image = self.process_image(img_path)
            ID = os.path.basename(img_path).split('_L')[0]
            bmd = self.get_bmd(os.path.basename(img_path), ID)
            return image, bmd, img_path

    def get_bmd(self, fname, ID):
        vert_name = fname.split('_')[-1].replace('.npy', '') if "_L" in fname else 'Total'
        try:
            bmd = self.bmd_data[self.bmd_data['ID'].astype(str) == ID][vert_name].values[0]
        except:
            print("out of bound vert detected")
            bmd = None
            
        return bmd
        
    def get_meta_data(self, ID):
        
        row = self.bmd_data[self.bmd_data['ID'].astype(str) == ID]
        
        if row.empty:
            return None  # 조건에 맞지 않으면 None 반환
        
        gender = row['Sex'].values[0] if not row.empty else None
        age = row['Age'].values[0] if not row.empty else None
        total_bmd = row['Total'].values[0]
        
        if gender == 'M':
            gender = 'male'
        elif gender == 'F':
            gender = 'female'

        z_score_total_gt = row['Zscore_Total'].values[0]
        
        if age < 10:
            return [z_score_total_gt, 1000, 1000, total_bmd, ID]
        
        if gender == 'male' :
            gender_age_bmd_mean = self.male_bmd_df.set_index('Age').loc[age,'Mean']
            gender_age_bmd_std = self.male_bmd_df.set_index('Age').loc[age,'SD']
        elif gender == 'female' :
            gender_age_bmd_mean = self.female_bmd_df.set_index('Age').loc[age,'Mean']
            gender_age_bmd_std = self.female_bmd_df.set_index('Age').loc[age,'SD']
         
        return [z_score_total_gt, gender_age_bmd_mean, gender_age_bmd_std, total_bmd, ID]

def evaluate_test(result_df):
    GT = np.array([t[0].item() for t in result_df['GT']])
    Prediction = np.array([t[0][0].item() for t in result_df['Prediction']])
    
    ID = list(result_df['ID'])

    
    z_gt = np.array([t[0].item() for t in result_df['z_gt']])
    z_pred = np.array([t[0][0].item() for t in result_df['z_pred']])

    # Convert GT and Prediction to binary classes
    GT_class = np.where(z_gt < -2, 0, 1)
    Prediction_class = np.where(z_pred < -2, 0, 1)
    
    # Rest of the code...
    differences = GT - Prediction
    mean_difference = np.mean(differences)
    std_dev_difference = np.std(differences)

    # MSE, MAE, RMSE
    MSE = mean_squared_error(GT, Prediction)
    MAE = np.mean(np.abs(differences))
    RMSE = np.sqrt(MSE)

    # Correlation, r2
    try:
        Corr = np.corrcoef(GT, Prediction)[0, 1]
    except Exception as e:
        print("Error in correlation calculation:", e)
        Corr = None
        
    try:
        z_Corr = np.corrcoef(z_gt, z_pred)
    except Exception as e:
        print("Error in z_correlation calculation:", e)
        z_Corr = None

    try:
        R2 = r2_score(GT, Prediction)
    except Exception as e:
        print("Error in R2 calculation:", e)
        R2 = None

    # F1 score
    F1 = f1_score(GT_class, Prediction_class)
    cm = confusion_matrix(Prediction_class, GT_class)
    
    eval_info = {
        "Mean Difference": mean_difference,
        "Standard Deviation of Difference": std_dev_difference,
        "MSE": MSE,
        "MAE": MAE,
        "RMSE": RMSE,
        "Correlation": Corr,
        "Correlation of Z-score": z_Corr,
        "R2": R2,
        "F1 Score": F1,
        "Upper 1.96SD": mean_difference + 1.96 * std_dev_difference,
        "Lower -1.96SD": mean_difference - 1.96 * std_dev_difference,
        "Upper 1SD": mean_difference + std_dev_difference,
        "Lower -1SD": mean_difference - std_dev_difference,
        "cm": cm
    }    
    return eval_info

def initialize_optimizer_scheduler(net, epochs, optimizer_type, scheduler_type, learning_rate, weight_decay = 1e-2, steps_per_epoch = 1):
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
    elif optimizer_type == 'Nadam':
        optimizer = optim.Nadam(net.parameters(), lr=learning_rate)

    if scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type == 'CyclicLR':
        scheduler = CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=steps_per_epoch * 5)
    elif scheduler_type == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=steps_per_epoch, epochs=epochs)

    return optimizer, scheduler