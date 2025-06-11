from moviepy import VideoFileClip, concatenate_videoclips
import os

# 設定你的資料夾路徑
folder_path = r"D:\深度學習概論_熊博安\期末專題\wy\測試用"  # 例如 "C:/Users/你的名字/Videos"

# 收集所有 mp4 檔案
mp4_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
mp4_files.sort()  # 若檔案名稱有順序性，可排序確保正確順序

# 載入影片
clips = [VideoFileClip(os.path.join(folder_path, f)) for f in mp4_files]

# 合併影片
final_clip = concatenate_videoclips(clips)

# 輸出結果
output_path = os.path.join(r"D:\深度學習概論_熊博安\期末專題\wy\測試用\測試用.mp4")
final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
