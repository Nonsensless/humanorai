
import os
import numpy as np
from my_swin_transformer import Swin_Tiny
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from PIL import Image
from my_cnn import Net



model = Net()
state_dict = torch.load("swin_transformer_model4_epoch_97%.pth", map_location=torch.device('cpu'), weights_only=True)
# model.load_state_dict(state_dict)
# 删除不匹配的权重
if 'fc1.0.weight' in state_dict:
    del state_dict['fc1.0.weight']
if 'fc1.0.bias' in state_dict:
    del state_dict['fc1.0.bias']
model.load_state_dict(state_dict, strict=False)

model.eval()  # 将模型设置为评估模式

# 定义类别标签列表
class_labels = ['Human','AI']  # 根据你的模型实际情况替换类别名称

# 定义预处理操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图片大小调整为256x256
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])


# 加载图像
image_path = 'img.png'  # 替换为你的图像路径
image = Image.open(image_path)
image = image.convert('RGB')

# 应用预处理
image = transform(image).unsqueeze(0)  # 增加批次维度，因为模型预期的输入是批次的形式

# 使用模型进行预测
with torch.no_grad():  # 确保在预测时不计算梯度
    outputs = model(image)
_, predicted = torch.max(outputs, 1)
# 输出预测概率
probabilities = torch.nn.functional.softmax(outputs, dim=1)
print(f'AI与人制作的概率值分别是: {probabilities}')
# 输出预测结果
print('我推断这张图片是由 [%s] 制作的'%class_labels[predicted.item()])

