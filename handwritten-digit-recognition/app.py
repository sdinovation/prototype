"""
Flask Web应用 - 手写数字识别
提供交互式手写画板, 实时识别手写数字
"""
import os
import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

from model.neural_network import NeuralNetwork

app = Flask(__name__)

# 全局模型变量
model = None


def load_model(model_path='./model/mnist_model_quick.pkl'):
    """加载预训练模型"""
    global model
    
    # 如果快速训练模型不存在, 尝试完整训练模型
    if not os.path.exists(model_path):
        alt_path = './model/mnist_model.pkl'
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print("警告: 未找到预训练模型, 请先运行 train.py 进行训练")
            print("使用: python train.py --mode quick")
            return None
    
    model = NeuralNetwork.load(model_path)
    print(f"模型加载成功: {model_path}")
    return model


def preprocess_image(image_data):
    """
    预处理前端传来的手写图像
    
    步骤:
        1. 解码Base64图像
        2. 转换为灰度图
        3. 缩放到 28x28
        4. 像素展开 + 归一化
    """
    # 1. 解码Base64
    image_data = image_data.split(',')[1]  # 去掉 "data:image/png;base64," 前缀
    image_bytes = base64.b64decode(image_data)
    
    # 2. 打开图像并转换为灰度
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # 3. 缩放到 28x28 (与MNIST一致)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 4. 转换为numpy数组并归一化
    img_array = np.array(img, dtype=np.float32)
    
    # MNIST是白字黑底, 而画板通常是黑字白底, 需要反转颜色
    img_array = 255.0 - img_array
    
    # 像素展开 + 归一化 (与PPT一致)
    img_flat = img_array.reshape(1, -1) / 255.0  # (1, 784), 归一化到 0~1
    
    return img_flat


@app.route('/')
def index():
    """首页 - 手写画板"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """接收手写图像并返回识别结果"""
    global model
    
    if model is None:
        return jsonify({
            'success': False,
            'error': '模型未加载, 请先训练模型'
        })
    
    try:
        # 获取前端传来的图像数据
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': '未收到图像数据'})
        
        # 预处理图像
        img_processed = preprocess_image(image_data)
        
        # 前向传播预测
        probabilities = model.predict_proba(img_processed)[0]  # (10,)
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit])
        
        # 返回所有类别的概率
        all_probs = [
            {'digit': i, 'probability': float(probabilities[i])}
            for i in range(10)
        ]
        all_probs.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'success': True,
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'all_probabilities': all_probs
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/status')
def status():
    """检查模型状态"""
    return jsonify({
        'model_loaded': model is not None,
        'model_info': {
            'input_size': model.input_size if model else None,
            'hidden1_size': model.hidden1_size if model else None,
            'hidden2_size': model.hidden2_size if model else None,
            'output_size': model.output_size if model else None,
        }
    })


def init_app():
    """初始化应用, 加载模型"""
    load_model()


if __name__ == '__main__':
    init_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
