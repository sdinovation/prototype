import sys
print("Python:", sys.executable)

# 先导入 PyQt5，再导入 torch
try:
    from PyQt5.QtWidgets import QApplication
    print("PyQt5 OK")
except Exception as e:
    print("PyQt5 FAILED:", e)

try:
    import torch
    print("torch OK:", torch.__version__)
except Exception as e:
    print("torch FAILED:", e)

try:
    from ultralytics import YOLO
    print("ultralytics OK")
except Exception as e:
    print("ultralytics FAILED:", e)
