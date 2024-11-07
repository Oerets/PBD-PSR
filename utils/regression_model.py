import torch
import torch.nn as nn
from torchvision import models

def load_resnet18(device):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 1, bias=True)
    return model.to(device)

def load_resnet50(device):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 1, bias=True)
    return model.to(device)

def load_vgg16(device):
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 1, bias=True)
    return model.to(device)

def load_squeezenet(device):
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = 1
    return model.to(device)

def load_efficientnet(device):
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)  # 분류기를 회귀 출력으로 변경
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device)

def load_regression_model(model_name, regression_dir, device):
    if model_name == 'resnet18':
        model = load_resnet18(device)
    elif model_name == 'resnet50':
        model = load_resnet50(device)
    elif model_name == 'vgg16':
        model = load_vgg16(device)
    elif model_name == 'squeezenet':
        model = load_squeezenet(device)
    elif model_name == 'efficientnet':
        model = load_efficientnet(device)
    else:
        raise ValueError("Unsupported model name")
    
    model.load_state_dict(torch.load(regression_dir))
    model.eval()
    return model
