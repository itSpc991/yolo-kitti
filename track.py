import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import torch
import sys
import os
from argparse import Namespace

# 添加ByteTrack路径
sys.path.append('./ByteTrack-main')
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

class ByteTrackWrapper:
    def __init__(self, model_path, tracker_args=None):
        # 加载YOLO模型
        self.model = YOLO(model_path)
        
        # 设置ByteTrack参数
        if tracker_args is None:
            tracker_args = {
                "track_thresh": 0.1,  # 降低追踪阈值
                "track_buffer": 30,
                "match_thresh": 0.5,  # 降低匹配阈值
                "min_box_area": 10,
                "mot20": False
            }
        
        # 将字典参数转换为Namespace对象
        args = Namespace(**tracker_args)
        self.tracker = BYTETracker(args)
        self.timer = Timer()
        self.frame_id = 0
        self.results = []
        
    def update(self, frame):
        # 使用YOLO进行目标检测
        print("开始检测...")
        detections = self.model(frame)[0]
        print(f"检测结果: {detections}")
        
        # 准备ByteTrack输入
        online_targets = []
        if len(detections.boxes) > 0:
            print(f"检测到 {len(detections.boxes)} 个目标")
            # 转换检测结果为ByteTrack格式
            dets = []
            for box in detections.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                print(f"目标: 坐标({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}), 置信度: {conf:.2f}, 类别: {cls}")
                dets.append([x1, y1, x2, y2, conf, cls])
            
            if len(dets) > 0:
                dets = np.array(dets)
                print(f"转换后的检测结果: {dets}")
                # 将NumPy数组转换为PyTorch张量
                dets = torch.from_numpy(dets).float()
                # 获取图像信息
                img_info = [frame.shape[0], frame.shape[1]]  # [height, width]
                # 获取图像尺寸
                img_size = (frame.shape[0], frame.shape[1])  # (height, width)
                # 更新追踪器
                online_targets = self.tracker.update(dets, img_info, img_size)
                print(f"追踪结果: {len(online_targets)} 个目标")
        
        # 绘制结果
        self.frame_id += 1
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > 10 and not vertical:
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'ID:{tid}', (int(x1), int(y1)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

def process_video(video_path, model_path, output_path):
    print(f"加载模型: {model_path}")
    # 初始化追踪器
    tracker = ByteTrackWrapper(model_path)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"视频信息: {width}x{height}, {fps}fps")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 处理每一帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"\n处理第 {frame_count} 帧")
        # 更新追踪器
        result_frame = tracker.update(frame)
        
        # 写入结果
        out.write(result_frame)
        
        # 显示处理进度
        cv2.imshow('Tracking', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 使用示例
    video_path = "./video/input.mp4"  # 输入视频路径
    output_path = "./video/output.mp4"  # 输出视频路径
    model_path = "yolo11n.pt"  # 使用训练好的模型
    
    process_video(video_path, model_path, output_path) 
