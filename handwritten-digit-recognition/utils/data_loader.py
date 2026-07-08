"""
MNIST数据加载与预处理
完全按照PPT第1章的数学原理实现
"""
import numpy as np
import os
import urllib.request


# MNIST npz 文件 URL (TensorFlow Keras 数据集镜像)
MNIST_NPZ_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'


def download_mnist_npz(data_dir='./data'):
    """下载MNIST npz文件"""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'mnist.npz')
    
    if not os.path.exists(filepath):
        print(f"正在下载 MNIST 数据集...")
        print(f"来源: {MNIST_NPZ_URL}")
        try:
            urllib.request.urlretrieve(MNIST_NPZ_URL, filepath)
            print(f"下载完成: {filepath}")
        except Exception as e:
            print(f"下载失败: {e}")
            print("尝试使用备用源...")
            # 备用源
            alt_url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
            urllib.request.urlretrieve(alt_url, filepath)
            print(f"备用源下载完成: {filepath}")
    else:
        print(f"数据集已存在: {filepath}")
    
    return filepath


def load_mnist(data_dir='./data', subset_size=None):
    """
    加载并预处理MNIST数据
    
    Returns:
        (X_train, y_train_onehot, y_train_labels,
         X_test, y_test_onehot, y_test_labels)
    """
    # 下载并加载数据
    filepath = download_mnist_npz(data_dir)
    
    print("正在加载数据...")
    with np.load(filepath) as data:
        X_train = data['x_train'].astype(np.float32)
        y_train = data['y_train'].astype(np.int32)
        X_test = data['x_test'].astype(np.float32)
        y_test = data['y_test'].astype(np.int32)
    
    print(f"原始数据加载完成:")
    print(f"  训练集: {X_train.shape[0]} 张")
    print(f"  测试集: {X_test.shape[0]} 张")
    
    # 预处理
    return preprocess_data(X_train, y_train, X_test, y_test, subset_size)


def preprocess_data(X_train, y_train, X_test, y_test, subset_size=None):
    """
    数据预处理 (对应PPT 1-2, 1-3, 1-4)
    
    步骤:
        1. 像素展开 (1-2): 28x28 -> 784维向量
        2. 归一化 (1-3): 像素值从 0~255 缩放到 0~1
        3. One-Hot编码 (1-4): 标签转换为One-Hot向量
    
    Args:
        X_train: 训练图像, shape (N, 28, 28)
        y_train: 训练标签, shape (N,)
        X_test: 测试图像, shape (M, 28, 28)
        y_test: 测试标签, shape (M,)
        subset_size: 如果指定, 只使用部分训练数据 (用于快速测试)
    
    Returns:
        预处理后的数据和原始标签
    """
    # 1-2 像素展开: 将 28x28 的像素矩阵展开成长度为 784 的向量
    # 按行从左到右、从上到下展开
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # (N, 784)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)      # (M, 784)
    
    # 1-3 归一化: 将像素值从 0~255 缩放到 0~1
    # x_norm = x / 255
    X_train_norm = X_train_flat / 255.0
    X_test_norm = X_test_flat / 255.0
    
    # 1-4 One-Hot编码
    # 将标签 0~9 转换为长度为10的One-Hot向量
    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train]  # (N, 10)
    y_test_onehot = np.eye(num_classes)[y_test]    # (M, 10)
    
    # 如果使用子集
    if subset_size is not None and subset_size < len(X_train_norm):
        indices = np.random.choice(len(X_train_norm), subset_size, replace=False)
        X_train_norm = X_train_norm[indices]
        y_train_onehot = y_train_onehot[indices]
        y_train = y_train[indices]
    
    print(f"\n预处理完成:")
    print(f"  训练数据形状: {X_train_norm.shape}")
    print(f"  训练标签形状: {y_train_onehot.shape}")
    print(f"  测试数据形状: {X_test_norm.shape}")
    print(f"  测试标签形状: {y_test_onehot.shape}")
    
    return (X_train_norm, y_train_onehot, y_train,
            X_test_norm, y_test_onehot, y_test)


def create_batches(X, y, batch_size=64, shuffle=True):
    """创建小批量数据"""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]
