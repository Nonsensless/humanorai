import os

import timm as timm
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from io import BytesIO
from my_cnn import Net

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 移除文件大小限制
app.config['MAX_CONTENT_LENGTH'] = None

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型
model = Net()  # 确保从my_cnn导入
state_dict = torch.load("swin_transformer_model4_epoch_97%.pth", map_location=torch.device('cpu'), weights_only=True)

# 删除不匹配的权重
if 'fc1.0.weight' in state_dict:
    del state_dict['fc1.0.weight']
if 'fc1.0.bias' in state_dict:
    del state_dict['fc1.0.bias']

model.load_state_dict(state_dict, strict=False)
model.eval()

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compress_image(image, max_size=5*1024*1024):
    """
    压缩图像，确保图像不会过大
    :param image: PIL Image对象
    :param max_size: 最大文件大小（字节）
    :return: 压缩后的图像
    """
    # 如果图像已经很小，直接返回
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    if buffer.tell() <= max_size:
        return image

    # 逐步降低质量
    for quality in [95, 90, 85, 80, 75, 70, 65, 60, 55, 50]:
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        if buffer.tell() <= max_size:
            return Image.open(buffer)

    # 如果无法压缩到指定大小，返回调整大小后的图像
    base_width = int(image.width * (max_size / buffer.tell()) ** 0.5)
    base_height = int(image.height * (max_size / buffer.tell()) ** 0.5)
    return image.resize((base_width, base_height), Image.LANCZOS)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    error = None
    filename = None
    human_prob = None
    ai_prob = None

    if request.method == 'POST':
        # 检查是否有文件被上传
        if 'file' not in request.files:
            error = '没有文件部分'
            return render_template('index.html', error=error)

        file = request.files['file']

        # 如果用户没有选择文件，浏览器也会提交一个空文件
        if file.filename == '':
            error = '没有选择文件'
            return render_template('index.html', error=error)

        # 如果文件存在且是允许的类型
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # 打开并处理图像
            try:
                image = Image.open(file)

                # 转换为RGB模式
                image = image.convert('RGB')

                # 保存文件
                image.save(filepath)

                # 预处理图像用于模型
                image_tensor = transform(image).unsqueeze(0)

                # 模型预测
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)

                # 获取概率值
                human_prob = f"{probabilities[0][0].item()*100:.2f}%"
                ai_prob = f"{probabilities[0][1].item()*100:.2f}%"

            except Exception as e:
                error = f"图像处理错误：{str(e)}"

    return render_template('index.html',
                           filename=filename,
                           human_prob=human_prob,
                           ai_prob=ai_prob,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True)
