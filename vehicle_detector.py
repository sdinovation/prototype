# -*- coding: utf-8 -*-
"""
车辆识别系统 - 基于 PyQt5 + YOLOv8
功能：选择视频 → YOLOv8 检测车辆 → 右侧多功能面板（统计/跟踪/违规/控制/趋势）
"""

import sys
import cv2
import numpy as np
import time
import collections
from pathlib import Path

# NOTE: torch 必须在 PyQt5 之前导入，否则 DLL 会冲突
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox, QStatusBar, QMessageBox,
    QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QPainter, QColor, QPen,
    QPainterPath
)


# ──────────────── COCO 80类中与车辆相关的类别 ────────────────
VEHICLE_CLASSES = {
    2: "汽车",
    3: "摩托车",
    5: "公交车",
    7: "卡车",
}

VEHICLE_COLORS = {
    2:  (0, 255, 0),
    3:  (255, 165, 0),
    5:  (255, 0, 0),
    7:  (0, 255, 255),
}


# ═══════════════════════════════════════════════
#  简单 IOU 跟踪器
# ═══════════════════════════════════════════════
class SimpleTracker:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = {}
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.frame_idx = 0

    def iou(self, a, b):
        x1, y1, x2, y2 = a
        x3, y3, x4, y4 = b
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        """
        detections: [(cls_id, conf, x1, y1, x2, y2), ...]
        返回: {track_id: {...}, ...}
        """
        self.frame_idx += 1
        matched = set()
        new_tracks = {}

        for tid, track in self.tracks.items():
            best_iou = self.iou_threshold
            best_idx = None
            for i, (cls_id, conf, x1, y1, x2, y2) in enumerate(detections):
                if i in matched:
                    continue
                if cls_id != track.get('class_id', cls_id):
                    continue
                val = self.iou(track['boxes'][-1], (x1, y1, x2, y2))
                if val > best_iou:
                    best_iou = val
                    best_idx = i

            if best_idx is not None:
                cls_id, conf, x1, y1, x2, y2 = detections[best_idx]
                track['boxes'].append((x1, y1, x2, y2))
                track['last_seen'] = self.frame_idx
                track['conf'] = conf
                new_tracks[tid] = track
                matched.add(best_idx)

        for i, (cls_id, conf, x1, y1, x2, y2) in enumerate(detections):
            if i not in matched:
                new_tracks[self.next_id] = {
                    'boxes': [(x1, y1, x2, y2)],
                    'last_seen': self.frame_idx,
                    'class_id': cls_id,
                    'conf': conf,
                    'created': self.frame_idx,
                }
                self.next_id += 1

        self.tracks = {
            tid: t for tid, t in new_tracks.items()
            if self.frame_idx - t['last_seen'] <= self.max_age
        }
        return self.tracks


# ═══════════════════════════════════════════════
#  违规检测器
# ═══════════════════════════════════════════════
class ViolationDetector:
    def __init__(self):
        self.illegal_park_count = 0

    def detect_illegal_parking(self, tracks, 静止阈值=20):
        """检测违停：最近N帧几乎没移动"""
        park = []
        for tid, track in tracks.items():
            boxes = track['boxes']
            if len(boxes) < 静止阈值:
                continue
            recent = boxes[-静止阈值:]
            max_dx = max(b[0] for b in recent) - min(b[0] for b in recent)
            max_dy = max(b[1] for b in recent) - min(b[1] for b in recent)
            if max_dx < 8 and max_dy < 8:
                park.append(tid)
        return park


# ═══════════════════════════════════════════════
#  历史折线图 (PyQt5 原生绘制)
# ═══════════════════════════════════════════════
class HistoryChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = collections.deque(maxlen=60)
        self.setMinimumHeight(130)

    def add_point(self, count):
        self.data.append(count)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor("#0d1117"))

        if len(self.data) < 2:
            painter.setPen(QColor("#888"))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待数据...")
            return

        # 网格
        painter.setPen(QPen(QColor("#333"), 1, Qt.DotLine))
        for i in range(5):
            y = h - 22 - (h - 44) * i // 4
            painter.drawLine(28, y, w - 8, y)

        max_val = max(self.data) if max(self.data) > 0 else 1
        points = []
        for i, v in enumerate(self.data):
            x = 28 + (w - 36) * i // (len(self.data) - 1)
            y = h - 22 - int((h - 44) * v / max_val)
            points.append(QPoint(x, y))

        # 填充
        if len(points) >= 2:
            path = QPainterPath()
            path.moveTo(points[0].x(), h - 22)
            for p in points:
                path.lineTo(p)
            path.lineTo(points[-1].x(), h - 22)
            path.closeSubpath()
            painter.fillPath(path, QColor("#0f3460"))

        # 折线
        painter.setPen(QPen(QColor("#00ff88"), 2))
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

        # 数据点
        painter.setBrush(QColor("#00ff88"))
        for p in points:
            painter.drawEllipse(p, 3, 3)

        # 标签
        painter.setPen(QColor("#888"))
        painter.setFont(QFont("Microsoft YaHei", 8))
        painter.drawText(2, h - 20, "0")
        painter.drawText(2, 14, str(max_val))
        painter.end()


# ═══════════════════════════════════════════════
#  检测线程
# ═══════════════════════════════════════════════
class DetectionThread(QThread):
    # 信号: 帧, 总数, 分类计数, 跟踪, 违停ID
    frame_ready = pyqtSignal(np.ndarray, int, dict, dict, list)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    history_ready = pyqtSignal(int)

    def __init__(self, video_path, model_path="yolov8n.pt"):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self._running = False
        self._paused = False
        self._step = False
        self.speed_multiplier = 1.0

    def run(self):
        try:
            self.model = YOLO(self.model_path)
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.error_signal.emit(f"无法打开视频: {self.video_path}")
                return

            tracker = SimpleTracker()
            violation = ViolationDetector()
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            frame_time = 1000 / fps
            last_history = time.time()

            self._running = True
            while self._running:
                if self._paused and not self._step:
                    self.msleep(50)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    break
                self._step = False

                results = self.model(frame, classes=list(VEHICLE_CLASSES.keys()), verbose=False)

                detections = []
                class_counts = {2: 0, 3: 0, 5: 0, 7: 0}
                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append((cls_id, conf, x1, y1, x2, y2))
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

                tracks = tracker.update(detections)
                park_ids = violation.detect_illegal_parking(tracks)
                annotated = self._draw_frame(frame, detections, tracks, park_ids)

                total = sum(class_counts.values())
                self.frame_ready.emit(annotated, total, class_counts, tracks, park_ids)

                now = time.time()
                if now - last_history >= 1:
                    self.history_ready.emit(total)
                    last_history = now

                delay = int(frame_time / self.speed_multiplier)
                self.msleep(max(1, delay))

            self.cap.release()
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))

    def _draw_frame(self, frame, detections, tracks, park_ids):
        annotated = frame.copy()
        for cls_id, conf, x1, y1, x2, y2 in detections:
            color = VEHICLE_COLORS.get(cls_id, (255, 255, 255))
            label = VEHICLE_CLASSES.get(cls_id, "?")
            text = f"{label} {conf:.0%}"

            tid = None
            for t_id, track in tracks.items():
                if track['boxes']:
                    if self._box_match(track['boxes'][-1], (x1, y1, x2, y2)):
                        tid = t_id
                        break

            if tid in park_ids:
                color = (255, 0, 255)
                text = f"[违停] {text}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, text, (x1, y1 - 3), font, 0.6, (255, 255, 255), 2)
            if tid:
                cv2.putText(annotated, f"ID:{tid}", (x1, y2 + 15), font, 0.5, color, 1)
        return annotated

    def _box_match(self, a, b):
        x1, y1, x2, y2 = a
        x3, y3, x4, y4 = b
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter
        iou = inter / union if union > 0 else 0
        return iou > 0.4

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False
        self._step = False

    def step_frame(self):
        self._paused = True
        self._step = True

    def set_speed(self, m):
        self.speed_multiplier = m

    def stop(self):
        self._running = False
        self.wait()


# ═══════════════════════════════════════════════
#  主窗口
# ═══════════════════════════════════════════════
class VehicleDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("车辆识别系统 - YOLOv8")
        self.setMinimumSize(1280, 760)
        self.resize(1360, 820)
        self.video_path = ""
        self.detect_thread = None
        self.is_detecting = False
        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QWidget { background-color: #1a1a2e; color: #e0e0e0; }
            QGroupBox {
                font-size: 13px; font-weight: bold;
                border: 1px solid #3a3a5c;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 14px;
                color: #a0c4ff;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton {
                background-color: #0f3460; color: white;
                border: none; border-radius: 5px;
                padding: 8px 16px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #16213e; }
            QPushButton:pressed { background-color: #533483; }
            QPushButton:disabled { background-color: #2a2a4a; color: #666; }
            QLabel { font-size: 12px; }
            QScrollArea { border: none; background: transparent; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # ═══════ 左侧：视频 + 底部控制 ═══════
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)

        title = QLabel("🚗  车辆识别系统")
        title.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        title.setStyleSheet("color: #a0c4ff;")
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        video_group = QGroupBox("视频画面")
        video_layout = QVBoxLayout(video_group)
        self.video_label = QLabel("请选择视频文件开始识别")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(480)
        self.video_label.setStyleSheet("""
            background-color: #0d1117;
            border: 2px dashed #3a3a5c;
            border-radius: 6px;
            font-size: 15px; color: #888;
        """)
        video_layout.addWidget(self.video_label)
        left_layout.addWidget(video_group, stretch=1)

        bottom = QHBoxLayout()
        self.btn_browse = QPushButton("📁 浏览视频")
        self.btn_browse.clicked.connect(self._on_browse)
        bottom.addWidget(self.btn_browse)

        self.btn_detect = QPushButton("▶ 开始识别")
        self.btn_detect.clicked.connect(self._on_toggle_detect)
        self.btn_detect.setEnabled(False)
        self.btn_detect.setStyleSheet("""
            QPushButton { background-color: #e94560; padding: 8px 28px; }
            QPushButton:hover { background-color: #ff6b6b; }
            QPushButton:disabled { background-color: #2a2a4a; }
        """)
        bottom.addWidget(self.btn_detect)

        self.path_label = QLabel("未选择视频")
        self.path_label.setStyleSheet("color: #888;")
        bottom.addWidget(self.path_label, stretch=1)

        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: #00ff88; font-weight: bold; font-size: 14px;")
        bottom.addWidget(self.count_label)

        left_layout.addLayout(bottom)
        main_layout.addWidget(left_widget, stretch=1)

        # ═══════ 右侧：功能面板 ═══════
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMaximumWidth(320)
        right_scroll.setMinimumWidth(300)
        right_scroll.setStyleSheet("QScrollArea { border: none; }")

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setAlignment(Qt.AlignTop)

        # ── 1. 统计面板 ──
        stats_group = QGroupBox("📊 车辆类型统计")
        stats_layout = QVBoxLayout(stats_group)
        self.stat_labels = {}
        for cls_id, name in VEHICLE_CLASSES.items():
            row = QHBoxLayout()
            dot = QLabel("●")
            c = VEHICLE_COLORS[cls_id]
            dot.setStyleSheet(f"color: rgb({c[2]},{c[1]},{c[0]}); font-size: 16px;")
            row.addWidget(dot)
            row.addWidget(QLabel(name))
            cnt = QLabel("0")
            cnt.setStyleSheet("font-size: 16px; font-weight: bold; color: #fff;")
            cnt.setAlignment(Qt.AlignRight)
            row.addWidget(cnt)
            self.stat_labels[cls_id] = cnt
            stats_layout.addLayout(row)
        self.stat_total = QLabel("总计: 0")
        self.stat_total.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #00ff88; "
            "border-top: 1px solid #3a3a5c; padding-top: 6px;"
        )
        self.stat_total.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.stat_total)
        right_layout.addWidget(stats_group)

        # ── 2. 车辆跟踪 ──
        track_group = QGroupBox("🔍 车辆跟踪")
        track_layout = QVBoxLayout(track_group)
        self.track_list = QLabel("暂无跟踪目标")
        self.track_list.setStyleSheet("color: #888; font-size: 11px;")
        self.track_list.setWordWrap(True)
        track_layout.addWidget(self.track_list)
        right_layout.addWidget(track_group)

        # ── 3. 违停检测 ──
        viol_group = QGroupBox("⚠️ 违停检测")
        viol_layout = QVBoxLayout(viol_group)
        self.viol_park = QLabel("违停车辆: 0")
        self.viol_park.setStyleSheet("color: #ff00ff; font-size: 13px;")
        viol_layout.addWidget(self.viol_park)
        right_layout.addWidget(viol_group)

        # ── 4. 播放控制 ──
        ctrl_group = QGroupBox("⏯ 播放控制")
        ctrl_layout = QHBoxLayout(ctrl_group)
        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_pause.setEnabled(False)
        ctrl_layout.addWidget(self.btn_pause)

        self.btn_speed = QPushButton("⏩ 快进")
        self.btn_speed.clicked.connect(self._on_speed)
        self.btn_speed.setEnabled(False)
        ctrl_layout.addWidget(self.btn_speed)

        self.btn_step = QPushButton("⏭ 逐帧")
        self.btn_step.clicked.connect(self._on_step)
        self.btn_step.setEnabled(False)
        ctrl_layout.addWidget(self.btn_step)
        right_layout.addWidget(ctrl_group)

        # ── 5. 历史折线图 ──
        chart_group = QGroupBox("📈 历史趋势 (辆/秒)")
        chart_layout = QVBoxLayout(chart_group)
        self.history_chart = HistoryChart()
        chart_layout.addWidget(self.history_chart)
        right_layout.addWidget(chart_group)

        right_layout.addStretch()
        right_scroll.setWidget(right_widget)
        main_layout.addWidget(right_scroll)

        self.statusBar().setStyleSheet("color: #888; font-size: 11px;")
        self.statusBar().showMessage("就绪 - 请选择视频文件")

    # ──────────────── 事件处理 ────────────────
    def _on_browse(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择监控视频", "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv);;所有文件 (*)"
        )
        if fp:
            self.video_path = fp
            self.path_label.setText(Path(fp).name)
            self.path_label.setStyleSheet("color: #a0c4ff;")
            self.btn_detect.setEnabled(True)
            cap = cv2.VideoCapture(fp)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self._display_frame(frame)

    def _on_toggle_detect(self):
        if self.is_detecting:
            self._stop_detect()
        else:
            self._start_detect()

    def _start_detect(self):
        if not self.video_path:
            QMessageBox.warning(self, "提示", "请先选择视频文件！")
            return
        self.is_detecting = True
        self.btn_detect.setText("⏹ 停止识别")
        self.btn_browse.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_speed.setEnabled(True)
        self.btn_step.setEnabled(True)
        self.statusBar().showMessage("正在加载模型...")

        self.detect_thread = DetectionThread(self.video_path)
        self.detect_thread.frame_ready.connect(self._on_frame_ready)
        self.detect_thread.finished_signal.connect(self._on_detect_finished)
        self.detect_thread.error_signal.connect(self._on_detect_error)
        self.detect_thread.history_ready.connect(self._on_history)
        self.detect_thread.start()

    def _stop_detect(self):
        self.is_detecting = False
        if self.detect_thread:
            self.detect_thread.stop()

    def _on_frame_ready(self, frame, total, class_counts, tracks, park_ids):
        self._display_frame(frame)
        self.count_label.setText(f"检测到 {total} 辆车")

        # 统计面板
        for cls_id, lbl in self.stat_labels.items():
            lbl.setText(str(class_counts.get(cls_id, 0)))
        self.stat_total.setText(f"总计: {total}")

        # 跟踪列表
        if tracks:
            lines = []
            for tid, track in tracks.items():
                name = VEHICLE_CLASSES.get(track['class_id'], '?')
                st = " [违停]" if tid in park_ids else ""
                lines.append(f"ID{tid}: {name}{st}")
            txt = "\n".join(lines[:8])
            if len(tracks) > 8:
                txt += f"\n...共 {len(tracks)} 个目标"
            self.track_list.setText(txt)
            self.track_list.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        else:
            self.track_list.setText("暂无跟踪目标")
            self.track_list.setStyleSheet("color: #888; font-size: 11px;")

        # 违停
        self.viol_park.setText(f"违停车辆: {len(park_ids)}")

    def _on_history(self, count):
        self.history_chart.add_point(count)

    def _on_detect_finished(self):
        self.is_detecting = False
        self.btn_detect.setText("▶ 开始识别")
        self.btn_detect.setEnabled(True)
        self.btn_browse.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_speed.setEnabled(False)
        self.btn_step.setEnabled(False)
        self.count_label.setText("")
        self.statusBar().showMessage("识别完成 - 视频播放结束")

    def _on_detect_error(self, msg):
        self.is_detecting = False
        self.btn_detect.setText("▶ 开始识别")
        self.btn_detect.setEnabled(True)
        self.btn_browse.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_speed.setEnabled(False)
        self.btn_step.setEnabled(False)
        QMessageBox.critical(self, "错误", msg)
        self.statusBar().showMessage("识别出错")

    def _on_pause(self):
        if not self.detect_thread:
            return
        if self.btn_pause.text() == "⏸ 暂停":
            self.detect_thread.pause()
            self.btn_pause.setText("▶ 继续")
        else:
            self.detect_thread.resume()
            self.btn_pause.setText("⏸ 暂停")

    def _on_speed(self):
        if not self.detect_thread:
            return
        if self.btn_speed.text() == "⏩ 快进":
            self.detect_thread.set_speed(3.0)
            self.btn_speed.setText("⏪ 正常")
        else:
            self.detect_thread.set_speed(1.0)
            self.btn_speed.setText("⏩ 快进")

    def _on_step(self):
        if self.detect_thread:
            self.detect_thread.step_frame()
            self.btn_pause.setText("▶ 继续")

    def _display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)
        self.video_label.setStyleSheet(
            "background-color: #0d1117; border: 2px solid #3a3a5c; border-radius: 6px;"
        )

    def closeEvent(self, event):
        if self.detect_thread and self.is_detecting:
            self.detect_thread.stop()
        event.accept()


# ──────────────── 入口 ────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = VehicleDetectorApp()
    window.show()
    sys.exit(app.exec_())
