import argparse
import time
from pathlib import Path

import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

#------標點程式----------------------------------------------------------------
horizontal_point_list = []

def mouse_handler(event, x, y, flags, data):
    """
    滑鼠事件處理函式，用於在圖像上選取點並顯示

    Args:
        event (int): 滑鼠事件類型
        x (int): 滑鼠點擊位置的 x 座標
        y (int): 滑鼠點擊位置的 y 座標
        flags (int): 額外的參數，例如按下的按鍵
        data (dict): 包含圖像和已選取點的資料字典

    Returns:
        None
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # 標記點位置
        cv2.circle(data['img'], (x,y), 3, (0,0,255), 5, 16) 

        # 改變顯示 window 的內容
        cv2.imshow("Image", data['img'])
        
        # 顯示 (x,y) 並儲存到 list中
        print("獲取點位: (x, y) = ({}, {})".format(x, y))
        data['points'].append((x,y))
        
def get_horizontal_points(im):
    """
    選擇水平點位的範圍

    參數:
        im (numpy.ndarray): 輸入圖像

    返回:
        tuple: 包含選取的點的座標和更新後的點列表
    """
    while True:
        # 建立 data dict, img:存放圖片, points:存放點
        data = {}
        data['img'] = im.copy()
        data['points'] = []

        cv2.namedWindow("Image", 0)

        # 改變 window 成為適當圖片大小
        h, w, dim = im.shape
        h = int(h * 0.9)
        w = int(w * 0.9)
        cv2.resizeWindow("Image", w, h)
            
        # 顯示圖片在 window 中
        cv2.imshow('Image', im)
        
        # 利用滑鼠回傳值，資料皆保存於 data dict中
        cv2.setMouseCallback("Image", mouse_handler, data)
        
        # 等待按下任意鍵，藉由 OpenCV 內建函數釋放資源
        cv2.waitKey()
        cv2.destroyAllWindows()

        # 將座標位置轉換為整數
        data['points'] = [(int(x), int(y)) for x, y in data['points']]

        # 判斷選擇了兩個點還是四個點
        if len(data['points']) == 2:
            # 如果選擇了兩個點
            # 执行相应操作，比如計算水平線或者其他操作
            return data['points'], True
        elif len(data['points']) == 4:
            # 如果選擇了四個點
            # 執行执行相应操作，比如計算選擇區域的特徵或者其他操作
            return data['points'], False
        else:
            # 如果選擇的點數不符合預期，可以執行錯誤處理
            print("請選擇兩個或四個點，重新選擇。")

def get_range_points(im, point_list):
    """
    選擇需要警示的範圍

    參數:
        im (numpy.ndarray): 輸入圖像
        point_list (list): 存放已選取點的列表

    返回:
        tuple: 包含選取的點的座標和更新後的點列表
    """
    while True:
        # 建立 data dict, img:存放圖片, points:存放點
        data = {}
        data['img'] = im.copy()
        data['points'] = []

        cv2.namedWindow("Image", 0)

        # 改變 window 成為適當圖片大小
        h, w, dim = im.shape
        h = int(h * 0.9)
        w = int(w * 0.9)
        cv2.resizeWindow("Image", w, h)
            
        # 顯示圖片在 window 中
        cv2.imshow('Image',im)
        
        # 利用滑鼠回傳值，資料皆保存於 data dict中
        cv2.setMouseCallback("Image", mouse_handler, data)
        
        # 等待按下任意鍵，藉由 OpenCV 內建函數釋放資源
        cv2.waitKey()
        cv2.destroyAllWindows()

        # 將座標位置轉換為整數
        data['points'] = [(int(x), int(y)) for x, y in data['points']]

        if len(data['points']) <= 2:
            print("選擇的點數不足，請重新選擇。")
        else:
            point_list.append(data['points'])
            return data['points'], point_list


#------判斷平交道的狀態----------------------------------------------------------------

def check_detection_position(x_position, y_position, top_points, bottom_points):
    """
    判斷 x_position, y_position 是否在 top_points, bottom_points 範圍內

    Args:
        x_position (int): 要判斷的 x 座標值
        y_position (int): 要判斷的 y 座標值
        top_points (list): 最上面兩個點的座標列表
        bottom_points (list): 最下面兩個點的座標列表

    Returns:
        str: 結果字符串，'none' 表示沒有火車要經過，'Approaching' 表示火車接近中，'Passing' 表示火車通過中
    """
    # 計算 top_points 兩點之間的直線方程式
    k_top = (top_points[1][1] - top_points[0][1]) / (top_points[1][0] - top_points[0][0])
    b_top = top_points[0][1] - k_top * top_points[0][0]

    # 計算 bottom_points 兩點之間的直線方程式
    k_bottom = (bottom_points[1][1] - bottom_points[0][1]) / (bottom_points[1][0] - bottom_points[0][0])
    b_bottom = bottom_points[0][1] - k_bottom * bottom_points[0][0]

    # 計算 x_position 對應的 top_points 和 bottom_points 上的 y 座標
    top_line_y = k_top * x_position + b_top
    bottom_line_y = k_bottom * x_position + b_bottom

    if y_position < top_line_y:  # 在 top_points 兩點連線的上方
        return 'none'
    elif top_line_y <= y_position <= bottom_line_y:  # 在 top_points 兩點連線和 bottom_points 兩點連線之間
        return 'Approaching'
    else:  # 在 bottom_points 兩點連線的下方
        return 'Passing'

def determine_detection_status(cls3_count, cls3_positions, is_two_points, prev_status):
    """
    根據檢測到的 cls=3 目標的數量、位置狀態和 is_two_points 的值確定最終的狀態。

    參數:
        cls3_count (int): cls=3 目標的數量
        cls3_positions (list): cls=3 目標的位置狀態列表
        is_two_points (bool): 是否只使用一個檢測點進行判斷
        prev_status (str): 前一個狀態

    返回:
        str: 最終的檢測狀態，'none' 表示沒有火車要經過，'Approaching' 表示火車接近中，'Passing' 表示火車通過中，'departed' 表示火車已離開
    """
    result_state = ""
    if is_two_points:
        # 如果只有一個 cls=3 的目標，且 is_two_points 為 True，只使用一個 check_detection_position() 進行判斷
        if cls3_count == 1:
            result_state = cls3_positions[0]
            if prev_status == 'Approaching' and result_state == 'Passing':
                result_state = 'departed'
            elif prev_status == 'departed' and result_state == 'Approaching':
                result_state = 'departed'
            elif prev_status == 'departed' and result_state == 'none':
                result_state = 'none'
        else:
            result_state = cls3_positions[0]  # 保持前一個狀態
    else:
        # 如果有兩個 cls=3 的目標，且 is_two_points 為 False，需要使用兩個 check_detection_position() 進行判斷
        if cls3_count == 2:
            # 檢查位置狀態
            if prev_status == 'Passing':
                if all(pos == 'Passing' for pos in cls3_positions):
                    result_state = 'Passing'
                if 'Passing' in cls3_positions and 'Approaching' in cls3_positions:
                    result_state = 'departed'
            elif prev_status == 'departed':
                if all(pos == 'none' for pos in cls3_positions):
                    result_state = 'none'
                else:
                    result_state = 'departed'
            elif 'Approaching' in cls3_positions:
                if prev_status != 'Passing':
                    result_state = 'Approaching'
            elif 'Passing' in cls3_positions and 'none' in cls3_positions:
                if prev_status != 'Passing':
                    result_state = 'Approaching'
            elif 'Passing' in cls3_positions and 'Approaching' in cls3_positions:
                if prev_status != 'Passing':
                    result_state = 'Approaching'
            elif all(pos == 'Passing' for pos in cls3_positions):
                result_state = 'Passing'
            elif all(pos == 'none' for pos in cls3_positions):
                result_state = 'none'
        elif cls3_count == 1:
            # 如果只檢測到一個 cls=3 的目標
            result_state = cls3_positions[0]

            if prev_status == 'departed' and result_state != 'none':
                result_state = 'departed'
            elif prev_status == 'departed' and result_state == 'none':
                result_state = 'none'
            elif prev_status == 'Approaching' and 'none' in cls3_positions:
                result_state = 'Approaching'
            elif result_state == 'Passing':
                result_state = 'Approaching' if prev_status == 'Approaching' else 'Passing'
            elif prev_status == 'Passing' and result_state == 'Approaching':
                result_state = 'departed'
            else:
                result_state = cls3_positions[0]  # 保持前一個狀態
        else:
            result_state = prev_status  # 保持前一個狀態

    return result_state

#----------------------------------------------------------------------

#------判斷坐標點所在位置的範圍----------------------------------------------------------------

def is_in_yellow_range(x, y, yellow_img):
    """
    判斷坐標 (x, y) 是否在黃色區域內

    參數:
    - x (int): x 坐標
    - y (int): y 坐標
    - yellow_img (numpy.ndarray): 黃色區域圖像陣列

    返回:
    - bool: 如果坐標在黃色區域內，則返回 True，否則返回 False
    """
    # 獲取圖像的高度和寬度
    height, width, _ = yellow_img.shape
    # 如果坐標在圖像範圍內並且對應像素值是黃色，則在黃色區域內
    if 0 <= y < height and 0 <= x < width and np.array_equal(yellow_img[y, x], [0, 255, 255]):
        return True
    return False

def is_in_red_range(x, y, red_img):
    """
    判斷坐標 (x, y) 是否在紅色區域內

    參數:
    - x (int): x 坐標
    - y (int): y 坐標
    - red_img (numpy.ndarray): 紅色區域圖像陣列

    返回:
    - bool: 如果坐標在紅色區域內，則返回 True，否則返回 False
    """
    # 獲取圖像的高度和寬度
    height, width, _ = red_img.shape
    # 如果坐標在圖像範圍內並且對應像素值是紅色，則在紅色區域內
    if 0 <= y < height and 0 <= x < width and np.array_equal(red_img[y, x], [0, 0, 255]):
        return True
    return False

#----------------------------------------------------------------------

#------判斷目標在哪個範圍----------------------------------------------------------------

def check_and_print_detection(det, yellow_img, red_img, confirm_yellow_range, confirm_red_range):
    """
    檢查並打印檢測結果，以及目標框是否在黃色或紅色區域內。

    參數:
        det (list): 檢測結果，包含多個目標框的信息，每個目標框表示為 [x1, y1, x2, y2, confidence, class]
                    其中：
                    - x1, y1, x2, y2 是目標框的左上角和右下角座標
                    - confidence 是檢測的置信度
                    - class 是檢測到的目標類別的索引
        yellow_img (numpy.ndarray): 黃色區域的圖像
        red_img (numpy.ndarray): 紅色區域的圖像
        confirm_yellow_range (bool): 確認是否要檢查黃色區域
        confirm_red_range (bool): 確認是否要檢查紅色區域

    返回:
        tuple: 包含兩個布爾值，表示是否檢測到黃色區域和紅色區域
    """
    in_yellow_range = False
    in_red_range = False
    for *xyxy, conf, cls in reversed(det):
        if int(cls) != 3 and int(cls) != 6:
            class_num = int(cls)
            # 判斷目標框四邊是否在黃色區域內
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            # 判斷四個角點和中心點是否在 yellow_img 中
            points = [(x1, y1), (x2, y2), (x1, y2), (x2, y1), ((x1 + x2) // 2, (y1 + y2) // 2)]

            if confirm_yellow_range:
                for point in points:
                    if is_in_yellow_range(*point, yellow_img):
                        # 在黃色區域內
                        print(f'{class_num} class detected at x, y = {point[0]}, {point[1]} in yellow range')
                        in_yellow_range = True
                        break  # 離開迴圈

            if confirm_red_range:
                for point in points:
                    if is_in_red_range(*point, red_img):
                        # 在紅色區域內
                        print(f'{class_num} class detected at x, y = {point[0]}, {point[1]} in red range')
                        in_red_range = True
                        break  # 離開迴圈

    # 在迴圈結束後返回結果
    return in_yellow_range, in_red_range

#----------------------------------------------------------------------

#------判斷平交道目標是否有重疊----------------------------------------------------------------

def check_overlap(new_xywh, list_xywh):
    if not len(list_xywh):
        return False
    for old_xywh in list_xywh:
        if int(new_xywh[0]) in range(int(old_xywh[0]), int(old_xywh[0])+int(old_xywh[2])):
            if int(new_xywh[1]) in range(int(old_xywh[1]), int(old_xywh[1])+int(old_xywh[3])):
                return True
            if int(old_xywh[1]) in range(int(new_xywh[1]), int(new_xywh[1])+int(new_xywh[3])):
                return True

        if int(old_xywh[0]) in range(int(new_xywh[0]), int(new_xywh[0])+int(new_xywh[2])):
            if int(old_xywh[1]) in range(int(new_xywh[1]), int(new_xywh[1])+int(new_xywh[3])):
                return True
            if int(new_xywh[1]) in range(int(old_xywh[1]), int(old_xywh[1])+int(old_xywh[3])):
                return True

    return False         

#----------------------------------------------------------------------

#----------檢測程式------------------------------------------------------------

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    #------畫線參數----------------------------------------------------------------

    fence_height = True
    yellow_range = True
    red_range = True
    Create_mask = True
    high_value = 0
    low_value = 0
    yellow_point_list = []
    red_point_list = []
    top_points = []
    bottom_points = []
    result_state = ''
    level_crossing_state = "Safety"

    #----------------------------------------------------------------------

    for path, img, im0s, vid_cap in dataset:

        if Create_mask:
            yellow_img = np.zeros((im0s.shape[0], im0s.shape[1], 3), dtype='uint8')
            red_img = np.zeros((im0s.shape[0], im0s.shape[1], 3), dtype='uint8')
            Create_mask = False

        #-----畫線功能，並在檢測結果圖上畫線-----------------------------------------------------------------

        #select height
        if fence_height:
            horizontal_points, is_two_points = get_horizontal_points(im0s)
            if is_two_points:
                # 如果選擇了兩個點，做相應的處理
                high_value = horizontal_points[0][1]
                low_value = horizontal_points[1][1]
                print(high_value)
                print(low_value)
                top_points = [(0, high_value), (im0s.shape[1], high_value)]
                bottom_points = [(0, low_value), (im0s.shape[1], low_value)]

                # 繪製第一條水平線
                cv2.line(im0s, *top_points, (255, 0, 0), 2)
                # 繪製第二條水平線
                cv2.line(im0s, *bottom_points, (0, 255, 0), 2)
            else:
                # 如果選擇了四個點，做相應的處理
                # 將座標按照 y 值排序
                sorted_points = sorted(horizontal_points, key=lambda x: x[1])
                # 取得最上面的兩個點和最下面的兩個點
                top_points = sorted_points[:2]
                bottom_points = sorted_points[2:]

                # 計算斜率和截距
                if top_points[1][0] != top_points[0][0]:
                    k_top = (top_points[1][1] - top_points[0][1]) / (top_points[1][0] - top_points[0][0])
                    b_top = top_points[0][1] - k_top * top_points[0][0]
                else:
                    # 如果是水平線，設置一個為0的值
                    k_top = 0
                    b_top = top_points[0][0]  # 直線的 x 值

                if bottom_points[1][0] != bottom_points[0][0]:
                    k_bottom = (bottom_points[1][1] - bottom_points[0][1]) / (bottom_points[1][0] - bottom_points[0][0])
                    b_bottom = bottom_points[0][1] - k_bottom * bottom_points[0][0]
                else:
                    # 如果是水平線，設置一個為0的值
                    k_bottom = 0
                    b_bottom = bottom_points[0][0]  # 直線的 x 值
                    
                # 計算直線與圖像邊界的交點
                # 對於最上面的兩個點
                h, w, _ = im0s.shape  # 獲取圖像的高度和寬度
                if k_top != 0:
                    x_top_left = max(0, int(-b_top / k_top))  # 交點的 x 座標不能小於 0
                    x_top_right = min(w, int((h - b_top) / k_top))  # 交點的 x 座標不能大於圖像寬度
                    cv2.line(im0s, (x_top_left, int(b_top + k_top * x_top_left)), (x_top_right, int(b_top + k_top * x_top_right)), (255, 0, 0), 2)
                    top_points = [(x_top_left, int(b_top + k_top * x_top_left)), (x_top_right, int(b_top + k_top * x_top_right))]
                else:
                    high_value = horizontal_points[0][1]
                    top_points = [(0, high_value), (im0s.shape[1], high_value)]
                    cv2.line(im0s, *top_points, (255, 0, 0), 2)

                # 對於最下面的兩個點
                if k_bottom != 0:
                    x_bottom_left = max(0, int(-b_bottom / k_bottom))  # 交點的 x 座標不能小於 0
                    x_bottom_right = min(w, int((h - b_bottom) / k_bottom))  # 交點的 x 座標不能大於圖像寬度
                    cv2.line(im0s, (x_bottom_left, int(b_bottom + k_bottom * x_bottom_left)), (x_bottom_right, int(b_bottom + k_bottom * x_bottom_right)), (0, 255, 0), 2)
                    bottom_points = [(x_bottom_left, int(b_bottom + k_bottom * x_bottom_left)), (x_bottom_right, int(b_bottom + k_bottom * x_bottom_right))]
                else:
                    low_value = horizontal_points[1][1]
                    bottom_points = [(0, low_value), (im0s.shape[1], low_value)]
                    cv2.line(im0s, *bottom_points, (0, 255, 0), 2)

            print(top_points)
            print(bottom_points)
            fence_height = False

        #draw yellow range
        if yellow_range:
            yellow_points, yellow_point_list = get_range_points(im0s, yellow_point_list)
            yellow_points = np.array(yellow_point_list)
            print(yellow_points)
            cv2.fillPoly(yellow_img, [yellow_points], (0,255,255))
            yellow_range = False

        #draw red range
        if red_range:
            red_points, red_point_list = get_range_points(im0s, red_point_list)
            red_points = np.array(red_point_list)
            print(red_points)
            cv2.fillPoly(red_img, [red_points], (0,0,255))
            red_range = False

        #-----在檢測結果圖上畫出警示範圍-----------------------------------------------------------------
        if is_two_points:
            # 繪製第一條水平線
            cv2.line(im0s, *top_points, (255, 0, 0), 2)
            # 繪製第二條水平線
            cv2.line(im0s, *bottom_points, (0, 255, 0), 2)
        else:
            # 連線最上面的兩個點，繪製第一條水平線
            cv2.line(im0s, top_points[0], top_points[1], (255, 0, 0), 2)
            # 連線最下面的兩個點，繪製第二條水平線
            cv2.line(im0s, bottom_points[0], bottom_points[1], (0, 255, 0), 2)
        cv2.polylines(im0s, [yellow_points], True, (0,255,255), 2)
        cv2.polylines(im0s, [red_points], True, (0,0,255), 2)

        #----------------------------------------------------------------------

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                cls3_count = 0  # 計算 cls=3 的目標數量
                cls3_positions = []  # 存儲 cls=3 的位置
                cls3_xywh = []

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    if int(cls) == 3:
                        x_position = int(xyxy[0])
                        y_position = int(xyxy[1])
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                        if (check_overlap(xywh, cls3_xywh)):
                            continue
                        cls3_xywh.append(xywh)
                        position_result = check_detection_position(x_position, y_position, top_points, bottom_points)
                        # print(f'Class 3 detected at x, y = {x_position}, {y_position}, Position: {position_result}')
                        cls3_count += 1
                        cls3_positions.append(position_result)
                    
                # 确定平交道的檢測狀態
                result_state = determine_detection_status(cls3_count, cls3_positions, is_two_points, prev_status=result_state)

                """
                平交道警示判斷：
                'Safety'安全
                'Obstacles'有障礙物
                'Dangerous'危險的
                'none' 表示沒有火車要經過
                'Approaching' 表示火車接近中
                'Passing' 表示火車通過中
                'departed' 表示火車已離開
                in_yellow_range 是否在黃色範圍內
                in_red_range 是否在紅色範圍內
                """
                in_yellow_range = False
                in_red_range = False
                if result_state == "none" or result_state == "departed":
                    level_crossing_state = "Safety"
                    # 在右上角繪製結果
                    cv2.putText(im0, level_crossing_state, (im0.shape[1] - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif result_state == "Approaching":
                    in_yellow_range, in_red_range = check_and_print_detection(det, yellow_img, red_img, True, False)
                    if in_yellow_range:
                        level_crossing_state = "Obstacles"
                    if not in_yellow_range and not in_red_range:
                        level_crossing_state = "Safety"
                    # 在右上角繪製結果
                    cv2.putText(im0, level_crossing_state, (im0.shape[1] - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                elif result_state == "Passing":
                    in_yellow_range, in_red_range = check_and_print_detection(det, yellow_img, red_img, True, True)
                    if in_red_range:
                        level_crossing_state = "Dangerous"
                    elif in_yellow_range:
                        level_crossing_state = "Obstacles"
                    if not in_yellow_range and not in_red_range:
                        level_crossing_state = "Safety"
                    # 在右上角繪製結果
                    cv2.putText(im0, level_crossing_state, (im0.shape[1] - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # 在右上角繪製結果
            cv2.putText(im0, result_state, (im0.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
