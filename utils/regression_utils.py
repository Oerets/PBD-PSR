import torch
import numpy as np

def process_image(image, margin_modifier, transform=None):
    # 이미지 정규화
    image = (image - image.min()) / (image.max() - image.min())

    # Margin 적용
    modifiedX = int(image.shape[0] * margin_modifier)
    modifiedY = int(image.shape[1] * margin_modifier)
    image = image[modifiedX:-modifiedX, modifiedY:-modifiedY]
    image = np.repeat(image[:, :, None], 3, axis=-1)

    # 변환 적용
    if transform:
        image = transform(image=image)["image"]
        
    return image

def torch_image(image, margin):
    margin_modifier = ((1 - margin) / ((1 + 1) * 2))
    processed_image = process_image(image, margin_modifier)
    image_tensor = torch.tensor(processed_image).unsqueeze(0).float()
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    
    return image_tensor