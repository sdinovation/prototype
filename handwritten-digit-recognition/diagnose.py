"""诊断脚本：分析为什么模型总是预测同一个数字"""
import numpy as np
from model.neural_network import NeuralNetwork

nn = NeuralNetwork.load('./model/mnist_model_quick.pkl')

# 测试不同输入
print("=" * 50)
print("不同输入下的预测结果:")
print("=" * 50)
for i in range(10):
    X = np.random.rand(1, 784)
    probs = nn.predict_proba(X)[0]
    pred = np.argmax(probs)
    top3 = np.argsort(probs)[-3:][::-1]
    print(f"  测试{i+1}: 预测={pred}, 前3={top3}, 概率={probs[pred]:.4f}")

# 输出层偏置
print()
print("=" * 50)
print("输出层偏置 b3:")
print("=" * 50)
for i, val in enumerate(nn.b3[0]):
    bar = "#" * max(1, int(50 * (val - nn.b3.min()) / (nn.b3.max() - nn.b3.min() + 1e-10)))
    print(f"  数字{i}: {val:+.6f}  {bar}")

# 100个随机输入的平均概率
print()
print("=" * 50)
print("100个随机输入的平均预测概率:")
print("=" * 50)
X_test = np.random.rand(100, 784)
avg_probs = nn.predict_proba(X_test).mean(axis=0)
for i, p in enumerate(avg_probs):
    bar = "#" * int(p * 50)
    print(f"  数字{i}: {p:.4f}  {bar}")

print()
print("结论: 模型未经训练，权重是随机的。")
print("输出层偏置 b3 的不均匀导致某个类别总是获得最高分。")
print("要解决此问题，请先训练模型: python train.py --mode quick")