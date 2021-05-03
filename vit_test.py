import json
from PIL import Image

import torch
from torchvision import transforms

from pytorch_pretrained_vit import ViT
import matplotlib.pyplot as plt

model_name = 'B_16_imagenet1k'
model = ViT(model_name, pretrained=True)

img = Image.open('C:/Users/kmsli/Desktop/CSE 382/project/img1.jpg')
img

# Preprocess image
tfms = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
img = tfms(img).unsqueeze(0)

# Load class names
labels_map = json.load(open('C:/Users/kmsli/Desktop/CSE 382/project/labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img).squeeze(0)
print('-----')
for idx in torch.topk(outputs, k=3).indices.tolist():
    prob = torch.softmax(outputs, -1)[idx].item()
    print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))