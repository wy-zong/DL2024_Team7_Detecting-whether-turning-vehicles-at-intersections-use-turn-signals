# 深度學習概論期末專案  
## 偵測路口轉彎車輛是否正確使用方向燈

### 程式檔案功能
| 檔案 | 功能說明 |
|------|---------|
| **merge_video.py** | 將多段原始路口影片合併為單一影片，方便後續統一處理。 |
| **data_process.py** | 使用 `YOLOv8` 的 `model.track` 追蹤車輛軌跡，判斷直行／左轉／右轉；當偵測到左／右轉時，擷取該時刻 ±N 幀並存圖，作為後續人工標註素材。 |
| **train.py** | 讀取 `data.yaml` 與標註影像資料，訓練 YOLOv8 模型；最佳權重輸出至 `best.pt`。 |
| **judge_turn_correctness.py** | 重複 `data_process.py` 的轉向判斷，同時以 **訓練後模型** 辨識方向燈；若轉向方向與方向燈相符，記錄為為正確轉彎。 |
| **data.yaml** | Roboflow 產生的資料集設定檔，定義三類別：`left_signal`、`right_signal`。 |
| **best.pt** | 已訓練完成、效果最佳的 YOLOv8 權重檔。 |

---

### 工作流程
1. **影片蒐集**  
   - 取得多段監視器或行車記錄器影片，放入 `datasets/raw_videos/`。

2. **影片合併**  
   ```bash
   python merge_video.py \
     --input_dir datasets/raw_videos \
     --ext mp4 \
     --output merged.mp4
