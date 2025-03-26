# def process_dicom_data(data_path, excel_path, dicom_path, resnet_path, model_type='resnet50'):

#     class DICOMTestDataset(Dataset):
#         def __init__(self, dicom_dir, label_file, excel_file, transform=None):
#             self.dicom_dir = dicom_dir
#             self.transform = transform
#             self.label_data = pd.read_excel(excel_file)
#             self.label_file = label_file
#             self.dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
        
#         def __len__(self):
#             return len(self.dicom_files)
        
#         def __getitem__(self, idx):
#             dicom_path = os.path.join(self.dicom_dir, self.dicom_files[idx])
#             dicom_data = pydicom.dcmread(dicom_path)
#             image = normalize_dicom_image(dicom_data)
#             label_file_path = os.path.join(self.label_file, os.path.splitext(os.path.basename(dicom_path))[0] + ".txt")
#             label = process_label_file(label_file_path)
#             cropped_image = crop_dicom_image(image, label)
            
#             if self.transform:
#                 cropped_image = Image.fromarray((cropped_image * 255).astype(np.uint8))
#                 cropped_image = self.transform(cropped_image)
#                 if cropped_image.shape[0] == 1:
#                     cropped_image = cropped_image.repeat(3, 1, 1)
            
#             return cropped_image, self.dicom_files[idx]

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])

#     test_dataset = DICOMTestDataset(dicom_path, data_path, excel_path, transform)
#     test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#     if model_type == 'resnet50':
#         model = models.resnet50()
#     elif model_type == 'resnet18':
#         model = models.resnet18()
#     else:
#         raise ValueError("Invalid model type. Choose 'resnet50' or 'resnet18'")

#     model.fc = torch.nn.Linear(model.fc.in_features, 1)
#     model.load_state_dict(torch.load(resnet_path))
#     model.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     results = []
#     with torch.no_grad():
#         for inputs, filenames in test_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs).squeeze()
#             outputs = outputs.cpu().numpy()
            
#             for filename, prediction in zip(filenames, outputs):
#                 results.append([filename, prediction])

#     df_results = pd.DataFrame(results, columns=["DICOM File", "Predicted BMD"])
#     return df_results

# def process_label_file(txt_file_path):
#     """
#     주어진 txt 파일을 분석하여 사용할 라벨을 선택하고 반환하는 함수
    
#     Args:
#         txt_file_path (str): 라벨이 포함된 텍스트 파일 경로
    
#     Returns:
#         list: 선택된 라벨 박스 정보 (리스트 형태)
#     """
#     with open(txt_file_path, "r") as f:
#         lines = f.readlines()
    
#     label_dict = {}
#     for line in lines:
#         parts = line.strip().split()
#         label = int(parts[0])
#         bbox = list(map(float, parts[1:]))
#         if label not in label_dict:
#             label_dict[label] = bbox
    
#     for priority_label in [0, 1, 2, 3, 4, 5]:  # 0 우선, 없으면 1, 그 외는 고려 안함
#         if priority_label in label_dict:
#             return label_dict[priority_label]
    
#     return None  # 사용 가능한 라벨이 없는 경우

# def crop_dicom_image(input_image, bbox):
#     """
#     정규화된 DICOM 이미지에서 주어진 바운딩 박스를 기준으로 잘라낸 이미지를 반환하는 함수
#     """
#     normalized_image = input_image
#     h, w = normalized_image.shape
    
#     x_center, y_center, width, height = bbox
#     x_min = int((x_center - width / 2) * w)
#     x_max = int((x_center + width / 2) * w)
#     y_min = int((y_center - height / 2) * h)
#     y_max = int((y_center + height / 2) * h)
    
#     cropped_image = normalized_image[y_min:y_max, x_min:x_max]
#     return cropped_image