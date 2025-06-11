import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import defaultdict, deque
import argparse
import os

class VehicleTurnDetector:
    def __init__(self, model_path='yolov8n.pt',
                 snapshot_dir='snapshots',
                 snapshot_offset=0,
                 cooldown_frames=5):
        """
        初始化車輛轉彎偵測器
        
        Args:
            model_path: YOLO模型路徑，預設使用yolov8n
        """
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: deque(maxlen=30))  # 儲存軌跡歷史
        self.turn_status = defaultdict(str)  # 儲存轉彎狀態
        self.turn_angles = defaultdict(float)  # 儲存轉彎角度
        self.status_history = defaultdict(lambda: deque(maxlen=5))  # 狀態歷史用於穩定判斷
        self.turn_confidence = defaultdict(int)  # 轉彎信心度計數器
        
        # === 新增成員 ===
        self.snapshot_dir = r"D:\深度學習概論_熊博安\期末專題\wy\save"           # 存圖資料夾
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.snapshot_offset = snapshot_offset     # offset 幀
        self.cooldown_frames = cooldown_frames     # 每車輛冷卻
        self.last_snapshot_frame = defaultdict(lambda: -1)  # 上一次已截圖幀
        self.pending_snapshots = deque()           # (frame_no, track_id, tag)
        # 車輛相關的類別ID (COCO數據集)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # 轉彎偵測參數
        self.min_track_length = 15  # 增加最小軌跡長度
        self.angle_threshold = 5   # 提高角度閾值(度)
        self.smoothing_window = 10   # 增加平滑窗口大小
        self.stability_threshold = 4  # 穩定性閾值：需要連續N次相同判斷才確認
        
    def calculate_direction_angle(self, points):
        """
        計算軌跡的方向角度變化（加強版：多重平滑）
        
        Args:
            points: 軌跡點列表 [(x, y), ...]
            
        Returns:
            angle_change: 角度變化量(度)
        """
        if len(points) < self.smoothing_window:
            return 0
            
        # 轉換為numpy數組便於計算
        points_array = np.array(list(points))
        
        # 1. 先對軌跡點進行平滑濾波
        if len(points_array) >= 3:
            # 簡單移動平均平滑
            smoothed_points = []
            for i in range(len(points_array)):
                start_idx = max(0, i-1)
                end_idx = min(len(points_array), i+2)
                avg_point = np.mean(points_array[start_idx:end_idx], axis=0)
                smoothed_points.append(avg_point)
            points_array = np.array(smoothed_points)
        
        # 2. 計算多個片段的角度變化
        segment_angles = []
        segment_length = max(3, len(points_array) // 3)
        
        for i in range(0, len(points_array) - segment_length, segment_length // 2):
            end_idx = min(i + segment_length, len(points_array))
            segment = points_array[i:end_idx]
            
            if len(segment) >= 3:
                # 計算該片段的方向變化
                start_vector = segment[len(segment)//2] - segment[0]
                end_vector = segment[-1] - segment[len(segment)//2]
                
                # 避免零向量
                if np.linalg.norm(start_vector) < 1 or np.linalg.norm(end_vector) < 1:
                    continue
                    
                # 計算角度
                cos_angle = np.dot(start_vector, end_vector) / (
                    np.linalg.norm(start_vector) * np.linalg.norm(end_vector))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                
                # 判斷方向
                cross_product = np.cross(start_vector, end_vector)
                signed_angle = angle_deg if cross_product >= 0 else -angle_deg
                segment_angles.append(signed_angle)
        
        # 3. 返回角度變化的中位數（更穩定）
        if segment_angles:
            return np.median(segment_angles)
        return 0
    
    def detect_turn(self, track_id, center_point):
        """
        偵測車輛是否轉彎（加強穩定性版本）
        
        Args:
            track_id: 追蹤ID
            center_point: 車輛中心點 (x, y)
            
        Returns:
            turn_status: "直行", "左轉", "右轉"
        """
        # 更新軌跡歷史
        self.track_history[track_id].append(center_point)
        
        # 軌跡點不足時返回直行
        if len(self.track_history[track_id]) < self.min_track_length:
            self.turn_status[track_id] = "直行"
            return "直行"
        
        # 計算方向角度變化
        angle_change = self.calculate_direction_angle(self.track_history[track_id])
        self.turn_angles[track_id] = angle_change
        
        # 初步判斷轉彎狀態
        if abs(angle_change) < self.angle_threshold:
            current_status = "直行"
        elif angle_change > self.angle_threshold:
            current_status = "turn_righ"  # 正角度為右轉
        else:
            current_status = "turn_left"  # 負角度為左轉
        
        # 更新狀態歷史
        self.status_history[track_id].append(current_status)
        
        # 穩定性檢查：需要連續多次相同判斷才確認
        recent_statuses = list(self.status_history[track_id])
        if len(recent_statuses) >= self.stability_threshold:
            # 檢查最近幾次判斷是否一致
            recent_same = recent_statuses[-self.stability_threshold:]
            if len(set(recent_same)) == 1:  # 全部相同
                stable_status = recent_same[0]
            else:
                # 使用多數決
                status_counts = {}
                for status in recent_same:
                    status_counts[status] = status_counts.get(status, 0) + 1
                stable_status = max(status_counts, key=status_counts.get)
        else:
            stable_status = "直行"  # 資料不足時預設直行
        
        # 額外穩定性檢查：避免直行和轉彎之間頻繁切換
        if track_id in self.turn_status:
            prev_status = self.turn_status[track_id]
            # 如果從轉彎狀態回到直行，需要更強的證據
            if prev_status != "直行" and stable_status == "直行":
                if abs(angle_change) > self.angle_threshold * 0.7:  # 還有一定角度變化
                    stable_status = prev_status  # 保持原狀態
        
        self.turn_status[track_id] = stable_status
        return stable_status
    
    def draw_track_and_status(self, frame, track_id, bbox, status, angle):
        """
        繪製軌跡和轉彎狀態 (簡化版 - 只顯示軌跡和狀態)
        
        Args:
            frame: 視頻幀
            track_id: 追蹤ID
            bbox: 邊界框 [x1, y1, x2, y2]
            status: 轉彎狀態
            angle: 轉彎角度
        """
        x1, y1, x2, y2 = map(int, bbox)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # 設定顏色
        color_map = {
            "直行": (0, 255, 0),    # 綠色
            "左轉": (0, 0, 255),    # 紅色  
            "右轉": (255, 0, 0)     # 藍色
        }
        color = color_map.get(status, (128, 128, 128))
        
        # 只繪製軌跡線
        track_points = list(self.track_history[track_id])
        if len(track_points) > 1:
            for i in range(1, len(track_points)):
                cv2.line(frame, track_points[i-1], track_points[i], color, 3)
        
        # 只在轉彎時顯示狀態文字
        if status != "直行":
            label = f"{status}"
            cv2.putText(frame, label, (center_x-20, center_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    def process_video(self, input_path, output_path):
        """
        處理視頻文件
        
        Args:
            input_path: 輸入視頻路徑
            output_path: 輸出視頻路徑
        """
        cap = cv2.VideoCapture(input_path)
        
        # 獲取視頻屬性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 設定視頻編寫器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        print(f"開始處理視頻: {input_path}")
        print(f"總幀數: {total_frames}, FPS: {fps}")
        print(f"輸出路徑: {output_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # === NEW 1️⃣：先檢查是否有排程要存的圖 ===
            while self.pending_snapshots and self.pending_snapshots[0][0] == frame_count:
                _, t_id, tag = self.pending_snapshots.popleft()
                fname = f"{t_id}_{tag}_{frame_count}.jpg"
                cv2.imwrite(os.path.join(self.snapshot_dir, fname), frame)
            
            # YOLO偵測和追蹤
            results = self.model.track(frame, persist=True, classes=self.vehicle_classes)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                # 獲取偵測結果
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()
                
                for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                    if conf > 0.5:  # 信心度閾值
                        # 轉換坐標格式
                        x_center, y_center, w, h = box
                        x1 = int(x_center - w/2)
                        y1 = int(y_center - h/2)
                        x2 = int(x_center + w/2)
                        y2 = int(y_center + h/2)
                        
                        center_point = (int(x_center), int(y_center))
                        
                        # 偵測轉彎
                        turn_status = self.detect_turn(track_id, center_point)
                        turn_angle = self.turn_angles[track_id]
                        # === NEW 2️⃣：決定是否安排截圖 ===
                        if turn_status in ("turn_left", "turn_right", "左轉", "右轉"):
                            last_f = self.last_snapshot_frame[track_id]
                            if frame_count - last_f >= self.cooldown_frames:
                                # 立即存第一張
                                fname_now = f"{track_id}_{turn_status}_{frame_count}.jpg"
                                cv2.imwrite(os.path.join(self.snapshot_dir, fname_now), frame)

                                # # 排程 offset 幀後再存一張
                                # future_frame = frame_count + self.snapshot_offset
                                # self.pending_snapshots.append((future_frame, track_id, turn_status))
                                offsets = [-10, -5, 0, 5, 10]
                                for offset in offsets:
                                    future_frame = frame_count + offset
                                    self.pending_snapshots.append((future_frame, track_id, turn_status))

                                # 更新冷卻計時
                                self.last_snapshot_frame[track_id] = frame_count
                        
                        # 繪製結果
                        self.draw_track_and_status(frame, track_id, [x1, y1, x2, y2], 
                                                 turn_status, turn_angle)
            
            # 顯示處理進度
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"處理進度: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # 寫入輸出視頻
            out.write(frame)
            
            # 顯示畫面
            # cv2.imshow('Vehicle Turn Detection', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        # 釋放資源
        cap.release()
        out.release()
        #cv2.destroyAllWindows()
        
        print(f"處理完成! 輸出視頻已保存至: {output_path}")

def main():
    """
    主函數
    """
    # 設定參數解析器
    parser = argparse.ArgumentParser(description='車輛轉彎偵測系統')
    parser.add_argument('--input', '-i', required=True, help='輸入視頻路徑')
    parser.add_argument('--output', '-o', required=True, help='輸出視頻路徑')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='YOLO模型路徑')
    
    args = parser.parse_args()
    
    # 創建偵測器
    detector = VehicleTurnDetector(model_path=args.model)
    
    # 處理視頻
    try:
        detector.process_video(args.input, args.output)
    except Exception as e:
        print(f"錯誤: {e}")

if __name__ == "__main__":
    # 直接執行的範例
    # 請修改以下路徑為您的實際路徑
    
    INPUT_VIDEO = r"D:\深度學習概論_熊博安\期末專題\wy\data\datav2_merged_output.mp4"      # 輸入視頻路徑
    OUTPUT_VIDEO = r"D:\深度學習概論_熊博安\期末專題\wy\data\output_video.mp4"    # 輸出視頻路徑
    
    print("=== 車輛轉彎偵測系統 ===")
    print(f"輸入視頻: {INPUT_VIDEO}")
    print(f"輸出視頻: {OUTPUT_VIDEO}")
    
    # 創建偵測器實例
    detector = VehicleTurnDetector(model_path='yolov8n.pt')
    
    # 開始處理
    try:
        detector.process_video(INPUT_VIDEO, OUTPUT_VIDEO)
    except FileNotFoundError:
        print("錯誤: 找不到輸入視頻文件，請檢查路徑是否正確")
    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")
        
    print("\n使用說明:")
    print("1. 請確保已安裝所需套件:")
    print("   pip install ultralytics opencv-python")
    print("2. 修改 INPUT_VIDEO 和 OUTPUT_VIDEO 變數為您的實際路徑")
    print("3. 或使用命令列參數:")
    print("   python script.py --input input.mp4 --output output.mp4")
    print("4. 按 'q' 鍵可提前終止處理")