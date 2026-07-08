import sys
print("Python:", sys.executable)
print("CWD:", sys.path[0])

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
