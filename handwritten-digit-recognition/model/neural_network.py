"""
手写数字识别神经网络 - NumPy从头实现
完全按照PPT《手写识别神经网络数学原理》中的公式实现

网络结构: 输入层(784) -> 隐藏层1(128, ReLU) -> 隐藏层2(64, ReLU) -> 输出层(10, Softmax)
"""
import numpy as np
import pickle


class NeuralNetwork:
    """
    全连接前馈神经网络
    
    架构 (与PPT一致):
        - 输入层: 784 个神经元 (对应 28x28 像素)
        - 隐藏层1: 128 个神经元, ReLU 激活
        - 隐藏层2: 64 个神经元, ReLU 激活
        - 输出层: 10 个神经元, Softmax 激活 (对应数字 0~9)
    """

    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10,
                 learning_rate=0.01, seed=None):
        """
        初始化网络参数
        
        按照PPT 1-6: 权重随机初始化(接近0), 偏置初始化为0
        """
        if seed is not None:
            np.random.seed(seed)

        self.lr = learning_rate
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        # 1-6 初始参数
        # 权重: 随机初始化, 接近0 (例如 w ~ N(0, 0.01))
        # 偏置: 初始化为0
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.01   # (784, 128)
        self.b1 = np.zeros((1, hidden1_size))                        # (1, 128)

        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01 # (128, 64)
        self.b2 = np.zeros((1, hidden2_size))                        # (1, 64)

        self.W3 = np.random.randn(hidden2_size, output_size) * 0.01  # (64, 10)
        self.b3 = np.zeros((1, output_size))                         # (1, 10)

        # 缓存前向传播中间结果, 用于反向传播
        self.cache = {}

    # ==================== 激活函数 ====================

    @staticmethod
    def relu(z):
        """ReLU激活函数: ReLU(z) = max(0, z)"""
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        """ReLU导数: f'(z) = 1 if z > 0 else 0"""
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z):
        """Softmax函数: ŷ_i = e^{z_i} / Σ_m e^{z_m}"""
        # 数值稳定性处理: 减去最大值
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # ==================== 前向传播 ====================

    def forward(self, X):
        """
        前向传播 (对应PPT第2章)
        
        Args:
            X: 输入数据, shape (batch_size, 784)
        
        Returns:
            y_hat: 预测概率分布, shape (batch_size, 10)
        """
        # 2-1 输入 -> 隐藏层1
        # z^(1) = W^(1)x + b^(1)
        self.z1 = np.dot(X, self.W1) + self.b1  # (batch, 128)

        # 2-2 隐藏层1激活 (ReLU)
        # a^(1) = ReLU(z^(1)) = max(0, z^(1))
        self.a1 = self.relu(self.z1)            # (batch, 128)

        # 2-3 隐藏层1 -> 隐藏层2
        # z^(2) = W^(2)a^(1) + b^(2)
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # (batch, 64)

        # 2-4 隐藏层2激活 (ReLU)
        # a^(2) = ReLU(z^(2)) = max(0, z^(2))
        self.a2 = self.relu(self.z2)            # (batch, 64)

        # 2-5 隐藏层2 -> 输出层 (得分)
        # z^(3) = W^(3)a^(2) + b^(3)
        self.z3 = np.dot(self.a2, self.W3) + self.b3  # (batch, 10)

        # 2-6 输出层 (softmax概率)
        # ŷ_i = softmax(z^(3)_i) = e^{z^(3)_i} / Σ_m e^{z^(3)_m}
        self.y_hat = self.softmax(self.z3)      # (batch, 10)

        # 缓存中间结果用于反向传播
        self.cache = {
            'X': X, 'z1': self.z1, 'a1': self.a1,
            'z2': self.z2, 'a2': self.a2,
            'z3': self.z3, 'y_hat': self.y_hat
        }

        return self.y_hat

    # ==================== 损失计算 ====================

    @staticmethod
    def cross_entropy_loss(y_hat, y_true):
        """
        交叉熵损失 (对应PPT 3-1)
        
        Loss = -Σ_i y_i log(ŷ_i)
        
        由于y是One-Hot向量, 只有真实类别对应的y_k=1, 其余为0
        因此简化为: Loss = -log(ŷ_k)
        """
        batch_size = y_true.shape[0]
        # 防止log(0)导致数值问题
        epsilon = 1e-12
        y_hat = np.clip(y_hat, epsilon, 1. - epsilon)

        # 计算交叉熵
        loss = -np.sum(y_true * np.log(y_hat)) / batch_size
        return loss

    # ==================== 反向传播 ====================

    def backward(self, y_true):
        """
        反向传播计算梯度 (对应PPT第3章)
        
        Args:
            y_true: 真实标签的One-Hot编码, shape (batch_size, 10)
        
        Returns:
            grads: 各参数梯度字典
        """
        batch_size = y_true.shape[0]
        X = self.cache['X']
        a1 = self.cache['a1']
        a2 = self.cache['a2']
        z1 = self.cache['z1']
        z2 = self.cache['z2']
        y_hat = self.cache['y_hat']

        # 3-2 输出层误差 δ^(3)
        # δ^(3) = ŷ - y  (向量减法, 逐元素相减)
        delta3 = y_hat - y_true  # (batch, 10)

        # 3-3 输出层梯度
        # ∂Loss/∂W^(3) = a^(2) · δ^(3)T
        # ∂Loss/∂b^(3) = δ^(3)
        dW3 = np.dot(a2.T, delta3) / batch_size   # (64, 10)
        db3 = np.sum(delta3, axis=0, keepdims=True) / batch_size  # (1, 10)

        # 3-4 误差传回隐藏层2: δ^(2)
        # δ^(2) = f'(z^(2)) ⊙ (W^(3)T · δ^(3))
        # 其中 f'(z) 为ReLU导数 (PPT中标注有误, 按ReLU实现)
        delta2 = np.dot(delta3, self.W3.T) * self.relu_derivative(z2)  # (batch, 64)

        # 3-5 隐藏层2梯度
        # ∂Loss/∂W^(2) = a^(1) · δ^(2)T
        # ∂Loss/∂b^(2) = δ^(2)
        dW2 = np.dot(a1.T, delta2) / batch_size   # (128, 64)
        db2 = np.sum(delta2, axis=0, keepdims=True) / batch_size  # (1, 64)

        # 3-6 误差传回隐藏层1: δ^(1)
        # δ^(1) = f'(z^(1)) ⊙ (W^(2)T · δ^(2))
        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(z1)  # (batch, 128)

        # 3-7 隐藏层1梯度
        # ∂Loss/∂W^(1) = X · δ^(1)T
        # ∂Loss/∂b^(1) = δ^(1)
        dW1 = np.dot(X.T, delta1) / batch_size   # (784, 128)
        db1 = np.sum(delta1, axis=0, keepdims=True) / batch_size  # (1, 128)

        grads = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dW3': dW3, 'db3': db3
        }

        return grads

    # ==================== 参数更新 (梯度下降) ====================

    def update_parameters(self, grads):
        """
        使用梯度下降更新所有参数 (对应PPT第4章)
        
        W_new = W_old - η * ∂Loss/∂W
        b_new = b_old - η * ∂Loss/∂b
        """
        self.W1 -= self.lr * grads['dW1']
        self.b1 -= self.lr * grads['db1']
        self.W2 -= self.lr * grads['dW2']
        self.b2 -= self.lr * grads['db2']
        self.W3 -= self.lr * grads['dW3']
        self.b3 -= self.lr * grads['db3']

    # ==================== 训练步骤 ====================

    def train_step(self, X, y_true):
        """单步训练: 前向传播 -> 计算损失 -> 反向传播 -> 更新参数"""
        y_hat = self.forward(X)
        loss = self.cross_entropy_loss(y_hat, y_true)
        grads = self.backward(y_true)
        self.update_parameters(grads)
        return loss

    # ==================== 预测 ====================

    def predict(self, X):
        """预测类别"""
        y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)

    def predict_proba(self, X):
        """预测概率"""
        return self.forward(X)

    def accuracy(self, X, y_true_labels):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y_true_labels)

    # ==================== 模型保存/加载 ====================

    def save(self, filepath):
        """保存模型参数"""
        params = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'input_size': self.input_size,
            'hidden1_size': self.hidden1_size,
            'hidden2_size': self.hidden2_size,
            'output_size': self.output_size,
            'learning_rate': self.lr
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """加载模型参数"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        nn = cls(
            input_size=params['input_size'],
            hidden1_size=params['hidden1_size'],
            hidden2_size=params['hidden2_size'],
            output_size=params['output_size'],
            learning_rate=params['learning_rate']
        )
        nn.W1 = params['W1']
        nn.b1 = params['b1']
        nn.W2 = params['W2']
        nn.b2 = params['b2']
        nn.W3 = params['W3']
        nn.b3 = params['b3']
        
        print(f"Model loaded from {filepath}")
        return nn
