import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pydicom
from PIL import Image
from scipy.stats import pearsonr
from Harmonize import *
from boxcutter import *

# 데이터셋 클래스 정의
class DICOMDataset(Dataset):
    def __init__(self, dicom_dir, label_file, excel_file, model_type='resnet50', transform=None):
        self.dicom_dir = dicom_dir
        self.transform = transform
        self.label_data = pd.read_excel(excel_file)
        self.label_file = label_file
        self.dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
        self.model_type = model_type
        
    def _find_target_from_excel(self, dicom_filename):
        """
        주어진 DICOM 파일 이름에서 ID와 시행일자를 추출하여 엑셀 파일에서 해당하는 BMD[Total] 값을 찾는다.
        """
        parts = dicom_filename.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid DICOM filename format: {dicom_filename}")
        
        dicom_id, dicom_date = parts[0], parts[1]
        dicom_date = f"20{dicom_date[:2]}-{dicom_date[2:4]}-{dicom_date[4:]}"  # YYMMDD -> YYYY-MM-DD 형식 변환
        
        matched_row = self.label_data[(self.label_data['ID'] == int(dicom_id)) & (self.label_data['시행일자'] == dicom_date)]
        
        if matched_row.empty:
            raise ValueError(f"No matching entry found in Excel for: {dicom_filename}")
        
        return matched_row.iloc[0]['BMD[Total]']
    
    def __len__(self):
        return len(self.dicom_files)
    
    def __getitem__(self, idx):
        dicom_path = os.path.join(self.dicom_dir, self.dicom_files[idx])
        dicom_data = pydicom.dcmread(dicom_path)
        image = normalize_dicom_image(dicom_data)
        label_file_path = os.path.join(self.label_file, os.path.splitext(os.path.basename(dicom_path))[0] + ".txt")
        label = process_label_file(label_file_path)
        cropped_image = crop_dicom_image(image, label)
        
        if self.transform:
            cropped_image = Image.fromarray((cropped_image * 255).astype(np.uint8))
            cropped_image = self.transform(cropped_image)
            if cropped_image.shape[0] == 1:  # 채널 차원이 1이면 3채널로 복사
                cropped_image = cropped_image.repeat(3, 1, 1)
        
        file_key = os.path.splitext(self.dicom_files[idx])[0]
        target = self._find_target_from_excel(file_key)
        
        return cropped_image, torch.tensor(target, dtype=torch.float32)

# 모델 학습 및 평가 함수

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=20, val_time=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_correlation = -1
    best_model_path = "best_model.pth"
    last_model_path = "last_model.pth"
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss / len(train_loader):.4f}")
        
        if (epoch + 1) % val_time == 0:
            model.eval()
            val_loss = 0.0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs).squeeze()
                    val_loss += criterion(outputs, targets).item()
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(targets.cpu().numpy())
            
            correlation, _ = pearsonr(predictions, actuals)
            print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Correlation Coefficient: {correlation:.4f}")
            
            if correlation > best_correlation:
                best_correlation = correlation
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model updated with correlation: {correlation:.4f}")
    
    torch.save(model.state_dict(), last_model_path)
    print("Last model saved.")

    # 테스트 데이터 평가
    model.eval()
    test_loss = 0.0
    test_predictions = []
    test_actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            test_loss += criterion(outputs, targets).item()
            test_predictions.extend(outputs.cpu().numpy())
            test_actuals.extend(targets.cpu().numpy())
    
    test_correlation, _ = pearsonr(test_predictions, test_actuals)
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Correlation Coefficient: {test_correlation:.4f}")

    return model

# 주요 설정
BATCH_SIZE = 16
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
MODEL_TYPE = 'resnet50'  # 'resnet50' 또는 'resnet18' 선택 가능

# 데이터 변환
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 데이터셋 및 데이터로더 생성
dcm_path = "C:/Users/hyunoh/Documents/Codes/BMD_code/data/Hip_data/dcm/train"
label_path = "C:/Users/hyunoh/Documents/Codes/BMD_code/data/Hip_data/bbox_labels/train"
excel_path = "C:/Users/hyunoh/Documents/Codes/Updated_usables_final.xlsx"
train_dataset = DICOMDataset(dcm_path, label_path, excel_path, MODEL_TYPE, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dcm_path = "C:/Users/hyunoh/Documents/Codes/BMD_code/data/Hip_data/dcm/val"
val_label_path = "C:/Users/hyunoh/Documents/Codes/BMD_code/data/Hip_data/bbox_labels/val"
val_dataset = DICOMDataset(val_dcm_path, val_label_path, excel_path, MODEL_TYPE, transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dcm_path = "C:/Users/hyunoh/Documents/Codes/BMD_code/data/Hip_data/dcm/test"
test_label_path = "C:/Users/hyunoh/Documents/Codes/BMD_code/data/Hip_data/bbox_labels/test"
test_dataset = DICOMDataset(test_dcm_path, test_label_path, excel_path, MODEL_TYPE, transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 선택
if MODEL_TYPE == 'resnet50':
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
elif MODEL_TYPE == 'resnet18':
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
else:
    raise ValueError("Invalid model type. Choose 'resnet50' or 'resnet18'")

# 회귀를 위해 출력층 조정
model.fc = nn.Linear(model.fc.in_features, 1)

# 손실 함수 및 최적화 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Validation 수행주기 설정정
val_time = 1

# 모델 학습
trained_model = train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, NUM_EPOCHS, val_time)