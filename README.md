# 深度學習概論期末專案  
## 偵測路口轉彎車輛是否正確使用方向燈

> 使用 **YOLOv8** + 視覺後處理邏輯，自動判斷轉彎車輛有無開啟方向燈，並提供一條龍的資料擴充與再訓練流程。

---

## 目錄
- [專案特色](#專案特色)
- [快速開始](#快速開始)
- [資料準備](#資料準備)
- [模型訓練](#模型訓練)
- [使用流程](#使用流程)
  - [1. 多影片合併 (`merge_video.py`)](#1-多影片合併-merge_videopy)
  - [2. 產生軌跡與擷取標註影格 (`data_process.py`)](#2-產生軌跡與擷取標註影格-data_processpy)
  - [3. 判斷轉彎正確性 (`judge_turn_correctness.py`)](#3-判斷轉彎正確性-judge_turn_correctnesspy)
- [檔案結構](#檔案結構)
- [套件需求](#套件需求)
- [貢獻方式](#貢獻方式)
- [License](#license)
- [致謝](#致謝)

---

## 專案特色
| 功能 | 說明 |
|------|------|
| **即時車輛偵測** | 以 YOLOv8 低延遲偵測車輛與方向燈狀態 |
| **軌跡追蹤與方向判斷** | `model.track` + 自定邏輯判斷直行／左轉／右轉 |
| **自動擷取標註影格** | 偵測轉彎時擷取前後影格，快速產生 Roboflow 標註素材 |
| **轉彎正確性比對** | 轉向方向 vs. 方向燈狀態，自動輸出違規清單 |
| **資料循環再訓練** | `data_process` → 標註 → `train.py` → 更新 `best.pt` |

---

## 快速開始

```bash
# 1. 下載專案
git clone https://github.com/<your-name>/turn-signal-detection.git
cd turn-signal-detection

# 2. 建立虛擬環境
conda create -n turn-signal python=3.10 -y
conda activate turn-signal

# 3. 安裝套件
pip install -r requirements.txt

# 4. 放置資料 / 權重
#   ├── datasets/          (影片與標註資料)
#   └── weights/best.pt    (已訓練好的模型)

# 5. 執行範例流程
python merge_video.py --input_dir datasets/raw_videos --output merged.mp4
python data_process.py --video merged.mp4 --weights weights/best.pt
python judge_turn_correctness.py --video merged.mp4 --weights weights/best.pt




