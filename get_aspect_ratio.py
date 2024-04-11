import os
import cv2
import numpy as np

# 圖片數據資料夾路徑
image_folder_path = "C:/project/yolov7/dataset/images/train/"
# 標籤數據資料夾路徑
label_folder_path = "C:/project/yolov7/dataset/labels/train/"

# 初始化一個列表來存儲所有標籤框的寬高比
aspect_ratios = []

# 讀取圖像文件和對應的標籤文件
image_files = os.listdir(image_folder_path)
label_files = os.listdir(label_folder_path)

for label_file in label_files:
    # 產生標籤檔案的完整路徑
    label_file_path = os.path.join(label_folder_path, label_file)
    
    # 生成對應圖像文件的完整路徑
    image_file = label_file.replace(".txt", ".jpg")
    image_file_path = os.path.join(image_folder_path, image_file)

    # 載入圖像
    image = cv2.imread(image_file_path)
    height, width, _ = image.shape

    # 讀取標籤文件中的標籤信息
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        # 解析標籤信息
        class_id, norm_x, norm_y, norm_w, norm_h = map(float, line.strip().split())
        
        # 將歸一化的坐標轉換為實際坐標
        x_min = int((norm_x - norm_w / 2) * width)
        y_min = int((norm_y - norm_h / 2) * height)
        x_max = int((norm_x + norm_w / 2) * width)
        y_max = int((norm_y + norm_h / 2) * height)

        # 計算標籤框的寬高比
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        aspect_ratio = bbox_width / bbox_height
        aspect_ratios.append(aspect_ratio)

# 將寬高比轉換為NumPy數組以便進行統計
aspect_ratios = np.array(aspect_ratios)

# 統計寬高比的分佈情況
ratio_counts, bin_edges = np.histogram(aspect_ratios, bins=10)  # 分成10個區間進行統計，你可以根據需要調整
print("寬高比的分佈情況：")
for i in range(len(ratio_counts)):
    print(f"區間 {bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {ratio_counts[i]}")

# 找出最大的寬高比
max_aspect_ratio = np.max(aspect_ratios)
print(f"最大寬高比是：{max_aspect_ratio:.2f}")
