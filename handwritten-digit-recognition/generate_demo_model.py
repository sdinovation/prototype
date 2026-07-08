"""
生成演示模型
用于在没有训练数据的情况下快速启动Web应用进行界面演示
"""
import numpy as np
import os
from model.neural_network import NeuralNetwork


def generate_demo_model(save_path='./model/mnist_model_quick.pkl'):
    """生成一个随机初始化的演示模型"""
    print("生成演示模型...")
    
    nn = NeuralNetwork(
        input_size=784,
        hidden1_size=128,
        hidden2_size=64,
        output_size=10,
        learning_rate=0.01,
        seed=42
    )
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    nn.save(save_path)
    
    print("演示模型生成完成!")
    print(f"模型路径: {save_path}")
    print("\n注意: 这是未经训练的演示模型, 识别结果是随机的。")
    print("要获得可用的模型, 请运行: python train.py --mode quick")
    
    return nn


if __name__ == '__main__':
    generate_demo_model()
