"""
手写数字识别 - 交互式GUI应用
使用 matplotlib 提供交互式绘制界面
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from model.neural_network import NeuralNetwork


class HandwritingGUI:
    """交互式手写数字识别GUI"""
    
    def __init__(self, model_path='./model/mnist_model_quick.pkl'):
        # 加载模型
        self.model = None
        if self.load_model(model_path):
            print(f"模型加载成功: {model_path}")
        else:
            print("警告: 模型未加载，识别将不可用")
        
        # 画布状态
        self.canvas_size = 280
        self.grid_size = 28
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.image = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # 创建画布点数组用于绘制
        self.fig, self.ax = None, None
        
    def load_model(self, model_path):
        """加载模型"""
        import os
        if not os.path.exists(model_path):
            alt_path = './model/mnist_model.pkl'
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                print(f"错误: 模型文件不存在: {model_path}")
                print("请先运行: python train.py --mode quick")
                return False
        
        self.model = NeuralNetwork.load(model_path)
        return True
    
    def on_press(self, event):
        """鼠标按下事件: 开始绘制"""
        if event.inaxes != self.ax:
            return
        self.drawing = True
        self.last_x = event.xdata
        self.last_y = event.ydata
    
    def on_release(self, event):
        """鼠标松开事件: 停止绘制并预测"""
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.predict()
    
    def on_move(self, event):
        """鼠标移动事件: 绘制"""
        if not self.drawing or event.inaxes != self.ax:
            return
        
        x = event.xdata
        y = event.ydata
        
        if self.last_x is not None:
            # Bresenham直线算法插值绘制
            self.draw_line(self.last_x, self.last_y, x, y)
        
        self.last_x = x
        self.last_y = y
        self.update_canvas()
    
    def draw_line(self, x0, y0, x1, y1, brush_size=1.5):
        """在图像上绘制线条（反锯齿）"""
        # 坐标转换: 画布坐标 -> grid坐标 (0~27)
        x0_grid = int(x0 * self.grid_size / self.canvas_size)
        y0_grid = int(y0 * self.grid_size / self.canvas_size)
        x1_grid = int(x1 * self.grid_size / self.canvas_size)
        y1_grid = int(y1 * self.grid_size / self.canvas_size)
        
        # 简单的线绘制，在附近点添加强度
        steps = max(int(np.hypot(x1_grid - x0_grid, y1_grid - y0_grid)), 1)
        for t in np.linspace(0, 1, steps):
            x = int(round(x0_grid + t * (x1_grid - x0_grid)))
            y = int(round(y0_grid + t * (y1_grid - y0_grid)))
            
            # 在当前点和周围增加强度，实现画笔效果
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        dist = np.sqrt(dx*dx + dy*dy)
                        intensity = np.exp(-dist / brush_size)
                        self.image[ny, nx] = min(1.0, self.image[ny, nx] + intensity)
    
    def update_canvas(self):
        """更新画布显示"""
        self.im.set_data(self.image)
        self.fig.canvas.draw_idle()
    
    def predict(self):
        """使用模型预测当前手写数字"""
        if self.model is None:
            self.result_text.set_text("请先训练模型\npython train.py --mode quick")
            self.fig.canvas.draw_idle()
            return
        
        # 预处理图像: MNIST是黑底白字，我们需要反转
        # 图像已经是 0(黑)~1(白)，但输入需要展开并归一化
        img_input = (1 - self.image).reshape(1, -1) / 1.0
        
        # 预测
        probs = self.model.predict_proba(img_input)[0]
        prediction = np.argmax(probs)
        confidence = probs[prediction]
        
        # 更新概率条
        for i in range(10):
            self.prob_bars[i].set_width(probs[i])
            if i == prediction:
                self.prob_bars[i].set_facecolor('#48bb78')
            else:
                self.prob_bars[i].set_facecolor('#667eea')
        
        # 更新结果文本
        self.result_text.set_text(f"预测: {prediction}\n置信度: {confidence*100:.1f}%")
        
        self.fig.canvas.draw_idle()
    
    def clear_canvas(self, event):
        """清空画布"""
        self.image = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.update_canvas()
        self.result_text.set_text("请绘制数字\n然后松开鼠标")
        for i in range(10):
            self.prob_bars[i].set_width(0)
            self.prob_bars[i].set_facecolor('#667eea')
        self.fig.canvas.draw_idle()
    
    def show(self):
        """显示GUI窗口"""
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        self.fig = plt.figure(figsize=(12, 8), dpi=100)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
        
        # 左侧: 绘制区域
        ax_draw = self.fig.add_subplot(gs[0, 0])
        ax_draw.set_title('手写画板 (请在此绘制 0~9)', fontsize=14, fontweight='bold')
        ax_draw.set_xlim(0, self.canvas_size)
        ax_draw.set_ylim(self.canvas_size, 0)  # y轴反转，使0在顶部
        ax_draw.set_xticks([])
        ax_draw.set_yticks([])
        
        # 显示初始图像
        self.im = ax_draw.imshow(
            self.image, 
            extent=[0, self.canvas_size, 0, self.canvas_size], 
            cmap='gray', 
            vmin=0, 
            vmax=1,
            origin='upper'
        )
        
        # 添加清空按钮
        from matplotlib.widgets import Button
        ax_clear = plt.axes([0.22, 0.02, 0.16, 0.05])
        btn_clear = Button(ax_clear, '清空画板', color='#e2e8f0', hovercolor='#cbd5e0')
        btn_clear.on_clicked(self.clear_canvas)
        
        # 添加预测按钮（可选，松开自动预测）
        ax_predict = plt.axes([0.62, 0.02, 0.16, 0.05])
        btn_predict = Button(ax_predict, '识别数字', color='#667eea', hovercolor='#764ba2')
        btn_predict.label.set_color('white')
        def predict_click(event):
            self.predict()
        btn_predict.on_clicked(predict_click)
        
        # 右侧: 结果和概率
        ax_result = self.fig.add_subplot(gs[0, 1])
        ax_result.set_title('识别结果', fontsize=14, fontweight='bold')
        ax_result.set_xlim(0, 1)
        ax_result.set_ylim(0, 10.5)
        ax_result.set_xticks([])
        ax_result.grid(False)
        
        # 结果文本框
        result_box = Rectangle((0.1, 8), 0.8, 2.0, fill=True, color='#f0f0f0', zorder=0)
        ax_result.add_patch(result_box)
        self.result_text = ax_result.text(
            0.5, 9.0, "请绘制数字\n然后松开鼠标",
            ha='center', va='center', fontsize=18
        )
        
        # 概率条
        self.prob_bars = []
        bar_height = 0.8
        bar_margin = 0.2
        for i in range(10):
            y = 7 - i
            # 背景
            bg = Rectangle((0.1, y - bar_height/2 + bar_margin/2), 0.8, bar_height, 
                           fill=True, color='#edf2f7', zorder=0)
            ax_result.add_patch(bg)
            # 概率条
            bar = Rectangle((0.1, y - bar_height/2 + bar_margin/2), 0, bar_height, 
                           fill=True, color='#667eea', zorder=1)
            ax_result.add_patch(bar)
            # 数字标签
            ax_result.text(0.05, y, str(i), ha='right', va='center', fontsize=12, fontweight='bold')
            self.prob_bars.append(bar)
        
        self.ax = ax_draw
        
        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        
        plt.suptitle('手写数字识别 - 神经网络数学原理', fontsize=18, fontweight='bold', y=0.98)
        
        self.fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.show()


def main():
    """主函数"""
    app = HandwritingGUI()
    app.show()


if __name__ == '__main__':
    main()
