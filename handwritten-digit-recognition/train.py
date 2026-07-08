"""
模型训练脚本
按照PPT中的训练循环进行模型训练
"""
import numpy as np
import os
import sys

from model.neural_network import NeuralNetwork
from utils.data_loader import load_mnist, create_batches


def train_model(epochs=20, batch_size=64, learning_rate=0.01,
                subset_size=None, save_path='./model/mnist_model.pkl',
                seed=42):
    """
    训练神经网络
    
    训练循环 (对应PPT):
        1. 取下一张训练图片 (小批量)
        2. 重复 1-19 步 (前向传播 + 反向传播 + 参数更新)
        3. 直到收敛或达到最大轮数
        目标: 最小化 Loss, 提高识别准确率
    """
    print("=" * 60)
    print("手写数字识别神经网络训练")
    print("=" * 60)
    
    # 1. 加载并预处理数据
    print("\n[1/4] 加载MNIST数据集...")
    (X_train, y_train_onehot, y_train_labels,
     X_test, y_test_onehot, y_test_labels) = load_mnist(
        data_dir='./data', subset_size=subset_size
    )
    
    # 2. 初始化神经网络
    print("\n[2/4] 初始化神经网络...")
    print("  网络结构: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)")
    print(f"  学习率: {learning_rate}")
    print(f"  批次大小: {batch_size}")
    print(f"  训练轮数: {epochs}")
    
    nn = NeuralNetwork(
        input_size=784,
        hidden1_size=128,
        hidden2_size=64,
        output_size=10,
        learning_rate=learning_rate,
        seed=seed
    )
    
    # 3. 训练循环
    print("\n[3/4] 开始训练...")
    print("-" * 60)
    
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        # 遍历每个小批量
        for batch_idx, (X_batch, y_batch) in enumerate(
            create_batches(X_train, y_train_onehot, batch_size, shuffle=True), 1
        ):
            # 单步训练: 前向传播 -> 计算损失 -> 反向传播 -> 更新参数
            loss = nn.train_step(X_batch, y_batch)
            epoch_loss += loss
            
            # 每100个batch打印一次进度
            if batch_idx % 100 == 0 or batch_idx == n_batches:
                print(f"  Epoch {epoch}/{epochs} | Batch {batch_idx}/{n_batches} | Loss: {loss:.6f}")
        
        # 计算epoch平均损失
        avg_loss = epoch_loss / n_batches
        
        # 计算训练集和测试集准确率
        train_acc = nn.accuracy(X_train, y_train_labels)
        test_acc = nn.accuracy(X_test, y_test_labels)
        
        print(f"  >>> Epoch {epoch} 完成 | Avg Loss: {avg_loss:.6f} | "
              f"Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
        print("-" * 60)
    
    # 4. 保存模型
    print("\n[4/4] 保存模型...")
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    nn.save(save_path)
    
    # 最终评估
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最终训练准确率: {nn.accuracy(X_train, y_train_labels)*100:.2f}%")
    print(f"最终测试准确率: {nn.accuracy(X_test, y_test_labels)*100:.2f}%")
    print(f"模型已保存至: {save_path}")
    print("=" * 60)
    
    return nn


def quick_train():
    """快速训练 (使用较少数据, 用于快速验证)"""
    return train_model(
        epochs=5,
        batch_size=128,
        learning_rate=0.1,
        subset_size=10000,
        save_path='./model/mnist_model_quick.pkl',
        seed=42
    )


def full_train():
    """完整训练 (使用全部60000张训练图片)"""
    return train_model(
        epochs=20,
        batch_size=64,
        learning_rate=0.01,
        subset_size=None,
        save_path='./model/mnist_model.pkl',
        seed=42
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练手写数字识别神经网络')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full'],
                        help='训练模式: quick(快速) 或 full(完整)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        kwargs = {}
        if args.epochs: kwargs['epochs'] = args.epochs
        if args.lr: kwargs['learning_rate'] = args.lr
        if args.batch_size: kwargs['batch_size'] = args.batch_size
        quick_train(**kwargs) if kwargs else quick_train()
    else:
        kwargs = {}
        if args.epochs: kwargs['epochs'] = args.epochs
        if args.lr: kwargs['learning_rate'] = args.lr
        if args.batch_size: kwargs['batch_size'] = args.batch_size
        full_train(**kwargs) if kwargs else full_train()
