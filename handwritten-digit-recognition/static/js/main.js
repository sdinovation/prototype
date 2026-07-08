/**
 * 手写数字识别 - 前端交互
 */

// 获取DOM元素
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultArea = document.getElementById('resultArea');
const probBars = document.getElementById('probBars');

// 画板状态
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// 初始化画板
function initCanvas() {
    // 设置画笔样式
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // 填充黑色背景 (与MNIST一致)
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// 获取鼠标/触摸在canvas上的坐标
function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    let clientX, clientY;
    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }
    
    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}

// 开始绘制
function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    const coords = getCoordinates(e);
    lastX = coords.x;
    lastY = coords.y;
}

// 绘制
function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    
    const coords = getCoordinates(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
    
    lastX = coords.x;
    lastY = coords.y;
}

// 停止绘制
function stopDrawing() {
    isDrawing = false;
}

// 清空画板
function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 重置结果区域
    resultArea.innerHTML = '<div class="placeholder">请先手写数字并点击识别</div>';
    probBars.innerHTML = '';
}

// 发送预测请求
async function predictDigit() {
    // 获取画板图像数据
    const imageData = canvas.toDataURL('image/png');
    
    // 显示加载状态
    resultArea.innerHTML = '<div class="loading"></div>';
    predictBtn.disabled = true;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResult(result);
        } else {
            resultArea.innerHTML = `<div class="placeholder" style="color: #e53e3e;">识别失败: ${result.error}</div>`;
        }
    } catch (error) {
        resultArea.innerHTML = `<div class="placeholder" style="color: #e53e3e;">请求错误: ${error.message}</div>`;
    } finally {
        predictBtn.disabled = false;
    }
}

// 显示识别结果
function displayResult(result) {
    // 显示预测数字
    resultArea.innerHTML = `
        <div>
            <div class="result-digit">${result.predicted_digit}</div>
            <div class="result-confidence">置信度: ${(result.confidence * 100).toFixed(2)}%</div>
        </div>
    `;
    
    // 显示概率条
    probBars.innerHTML = '';
    result.all_probabilities.forEach((item, index) => {
        const isTop = index === 0;
        const percentage = (item.probability * 100).toFixed(1);
        
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';
        probItem.innerHTML = `
            <div class="prob-digit">${item.digit}</div>
            <div class="prob-bar-container">
                <div class="prob-bar ${isTop ? 'highlight' : ''}" style="width: ${percentage}%">
                    <span class="prob-value">${percentage}%</span>
                </div>
            </div>
        `;
        probBars.appendChild(probItem);
    });
}

// 初始化概率条 (空状态)
function initProbBars() {
    probBars.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';
        probItem.innerHTML = `
            <div class="prob-digit">${i}</div>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: 0%"></div>
            </div>
        `;
        probBars.appendChild(probItem);
    }
}

// 事件监听
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// 触摸事件支持
canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing);

clearBtn.addEventListener('click', clearCanvas);
predictBtn.addEventListener('click', predictDigit);

// 初始化
initCanvas();
initProbBars();

// 检查模型状态
async function checkModelStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        
        if (!status.model_loaded) {
            resultArea.innerHTML = `
                <div class="placeholder" style="color: #e53e3e;">
                    模型未加载, 请先运行训练脚本<br>
                    <code>python train.py --mode quick</code>
                </div>
            `;
        }
    } catch (error) {
        console.error('无法检查模型状态:', error);
    }
}

checkModelStatus();
