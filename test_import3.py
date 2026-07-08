# 正确的导入顺序：torch 先，PyQt5 后
from ultralytics import YOLO
print("ultralytics OK")

from PyQt5.QtWidgets import QApplication
print("PyQt5 OK")

import torch
print("torch OK:", torch.__version__)
