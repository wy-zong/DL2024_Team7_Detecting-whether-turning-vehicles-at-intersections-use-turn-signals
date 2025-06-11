# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import argparse

class VehicleTurnDetector:
    """車輛轉彎 + 方向燈偵測（YOLO 框標示版）"""

    def __init__(self, model_path="yolov8n.pt", signal_model_path=None):
        # 車輛偵測 & 追蹤模型
        self.model = YOLO(model_path)
        # 自訓練方向燈偵測模型（可選）
        self.signal_model = YOLO(signal_model_path) if signal_model_path else None

        # 歷史緩存
        self.track_history   = defaultdict(lambda: deque(maxlen=30))   # 軌跡
        self.turn_status     = defaultdict(str)                         # 目前狀態
        self.turn_angles     = defaultdict(float)                       # 角度
        self.status_history  = defaultdict(lambda: deque(maxlen=5))     # 狀態序列
        self.turn_signal_history = defaultdict(lambda: deque(maxlen=30))# 方向燈方向

        self.vehicle_classes = [2, 3, 5, 7] # COCO 車輛類別 ID：car, motorcycle, bus, truck
        self.signal_classes_map = {0: "LEFT_SIGNAL", 1: "RIGHT_SIGNAL"} # 方向燈類別 ID 與文字
        # 轉彎偵測參數
        self.min_track_length    = 15
        self.angle_threshold     = 10
        self.smoothing_window    = 8
        self.stability_threshold = 3

    # ------------------------------------------------------------------
    # 轉彎角度計算
    # ------------------------------------------------------------------
    def calculate_direction_angle(self, points):
        if len(points) < self.smoothing_window:
            return 0
        pts = np.array(list(points))
        # 簡易移動平均平滑
        if len(pts) >= 3:
            smoothed = []
            for i in range(len(pts)):
                start = max(0, i-1)
                end   = min(len(pts), i+2)
                smoothed.append(np.mean(pts[start:end], axis=0))
            pts = np.array(smoothed)
        seg_angles = []
        seg_len = max(3, len(pts)//3)
        for i in range(0, len(pts)-seg_len, seg_len//2):
            seg = pts[i:i+seg_len]
            if len(seg) >= 3:
                v1 = seg[len(seg)//2] - seg[0]
                v2 = seg[-1] - seg[len(seg)//2]
                if np.linalg.norm(v1) < 1 or np.linalg.norm(v2) < 1:
                    continue
                cos_th = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                cos_th = np.clip(cos_th, -1, 1)
                ang = np.degrees(np.arccos(cos_th))
                cross = np.cross(v1, v2)
                seg_angles.append(ang if cross >= 0 else -ang)
        return np.median(seg_angles) if seg_angles else 0

    # ------------------------------------------------------------------
    # 轉彎偵測
    # ------------------------------------------------------------------
    def detect_turn(self, track_id, center_point):
        self.track_history[track_id].append(center_point)
        if len(self.track_history[track_id]) < self.min_track_length:
            self.turn_status[track_id] = "STRAIGHT"
            return "STRAIGHT"
        ang_delta = self.calculate_direction_angle(self.track_history[track_id])
        self.turn_angles[track_id] = ang_delta
        # 粗判斷
        if abs(ang_delta) < self.angle_threshold:
            cur = "STRAIGHT"
        elif ang_delta > 0:
            cur = "RIGHT"
        else:
            cur = "LEFT"
        # 穩定化
        self.status_history[track_id].append(cur)
        recent = list(self.status_history[track_id])
        if len(recent) >= self.stability_threshold:
            last_n = recent[-self.stability_threshold:]
            if len(set(last_n)) == 1:
                stable = last_n[0]
            else:
                counts = {s:last_n.count(s) for s in set(last_n)}
                stable = max(counts, key=counts.get)
        else:
            stable = "STRAIGHT"
        # 防抖
        if track_id in self.turn_status:
            prev = self.turn_status[track_id]
            if prev != "STRAIGHT" and stable == "STRAIGHT" and abs(ang_delta) > self.angle_threshold*0.7:
                stable = prev
        self.turn_status[track_id] = stable
        return stable

    # ------------------------------------------------------------------
    # 工具：方向燈框是否落在車輛框內
    # ------------------------------------------------------------------
    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(areaA + areaB - inter)

    def check_signal_match(self, veh_box, detections, iou_thresh=0.3):
        """如偵測到方向燈框落在車輛框內，回傳 LEFT / RIGHT"""
        for box, cls_id in detections:
            if self._iou(veh_box, box) >= iou_thresh:
                return "LEFT"  if cls_id == 0 else "RIGHT"
        return None

    def is_correct_turn(self, track_id, turn_dir):
        """如果在歷史方向燈中對應方向曾出現過，即視為正確轉彎"""
        if turn_dir not in ("LEFT", "RIGHT"):
            return False
        history = self.turn_signal_history[track_id]
        return turn_dir in history

    # ------------------------------------------------------------------
    # 視覺化：YOLO 風格車輛框
    # ------------------------------------------------------------------
    def draw_turn_result(self, frame, bbox, correct):
        x1,y1,x2,y2 = map(int, bbox)
        color = (0,0,255) if correct else (0,255,0)   # 綠 = 正確，紅 = 不正確
        label = "INCORRECT_TURN" if correct else "CORRECT_TURN"
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    def draw_vehicle_bbox(self, frame, bbox, status, correct):
        x1, y1, x2, y2 = map(int, bbox)
        # 顏色：綠=正確轉彎 / 紅=未打燈轉彎 / 白=直行
        if status in ("LEFT", "RIGHT"):
            color = (0, 255, 0) if correct else (0, 0, 255)
            label = f"{status}_OK" if correct else f"{status}_NO_SIG"
        else:
            color = (255, 255, 255)
            label = "STRAIGHT"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ------------------------------------------------------------------
    # 方向燈偵測
    # ------------------------------------------------------------------
    def detect_signals(self, frame):
        if self.signal_model is None:
            return []
        results = self.signal_model.predict(frame, verbose=False)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.int().cpu().numpy()
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                if conf < 0.01:
                    continue
                dets.append((box, cls_id))
        return dets

    # ------------------------------------------------------------------
    # 主迴圈
    # ------------------------------------------------------------------
    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        count = 0
        print(f"Processing {input_path} -> {output_path}  (Frames: {total}, FPS: {fps})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1

            # 方向燈偵測先跑，供車輛偵測後比對
            signal_dets = self.detect_signals(frame)

            # 車輛偵測 + 追蹤
            results = self.model.track(frame, persist=True, classes=self.vehicle_classes)
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                ids   = results[0].boxes.id.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, tid, conf in zip(boxes, ids, confs):
                    if conf < 0.5:
                        continue
                    xc, yc, w_box, h_box = box
                    x1, y1 = int(xc - w_box / 2), int(yc - h_box / 2)
                    x2, y2 = int(xc + w_box / 2), int(yc + h_box / 2)
                    veh_bbox = [x1, y1, x2, y2]

                    # 取得當前車輛中心點並判斷轉彎
                    status = self.detect_turn(tid, (int(xc), int(yc)))

                    # 僅關注轉彎車輛，直行不畫
                    if status == "STRAIGHT": continue

                    # 檢查方向燈是否與當前車輛匹配
                    sig_dir = self.check_signal_match(veh_bbox, signal_dets)
                    if sig_dir:
                        self.turn_signal_history[tid].append(sig_dir)

                    # 判斷是否正確轉彎
                    correct = self.is_correct_turn(tid, status)
                    self.draw_turn_result(frame, veh_bbox, correct)# 判斷 Correct / Incorrect 並畫框
                    # ---- 只畫 YOLO 框（軌跡線/文字保留待用） ----
                    # self.draw_vehicle_bbox(frame, veh_bbox, status, correct)

                    # 若仍想顯示軌跡線與文字，可取消下行註解
                    self.draw_track_and_status(frame, tid, veh_bbox, status)

            # 方向燈框（採白框）—— 若不需要，也可整段註解
            for box, cls_id in signal_dets:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                text = self.signal_classes_map.get(cls_id, f"CLS{cls_id}")
                cv2.putText(frame, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # # Progress log
            # if count % 30 == 0:
            #     print(f"Progress {count/total*100:.1f}% ({count}/{total})")

            out.write(frame)
            cv2.imshow('Turn & Signal (YOLO Box)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Done! Saved to", output_path)

    # ------------------------------------------------------------------
    # （保留）原先軌跡線 + 英文標籤函式 — 目前未被呼叫
    # ------------------------------------------------------------------
    def draw_track_and_status(self, frame, track_id, bbox, status):
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        track_pts = list(self.track_history[track_id])
        # 白色軌跡線
        if len(track_pts) > 1:
            for i in range(1, len(track_pts)):
                cv2.line(frame, track_pts[i - 1], track_pts[i], (255, 255, 255), 2)
        # 文字標籤
        cv2.putText(frame, status, (cx - 30, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Vehicle Turn & Signal Detector (YOLO Box)')
    parser.add_argument('-i', '--input', required=True, help='Input video path')
    parser.add_argument('-o', '--output', required=True, help='Output video path')
    parser.add_argument('-m', '--model', default='yolov8n.pt', help='Vehicle YOLO model')
    parser.add_argument('-s', '--signal_model', default=None, help='Turn-signal YOLO model')
    args = parser.parse_args()
    det = VehicleTurnDetector(model_path=args.model, signal_model_path=args.signal_model)
    det.process_video(args.input, args.output)

if __name__ == '__main__':
    # 範例 (可自行修改)
    INPUT  = r"D:\深度學習概論_熊博安\期末專題\wy\測試用\測試用.mp4"
    OUTPUT = r"D:\深度學習概論_熊博安\期末專題\wy\video\測試用_out.mp4"
    VEHICLE_MODEL = 'yolov8n.pt'
    SIGNAL_MODEL  = r"D:\深度學習概論_熊博安\期末專題\wy\wy_custom_v3_model\weights\best.pt"

    print("=== Vehicle Turn & Signal Detector (YOLO Box) ===")
    det = VehicleTurnDetector(model_path=VEHICLE_MODEL, signal_model_path=SIGNAL_MODEL)
    det.process_video(INPUT, OUTPUT)
