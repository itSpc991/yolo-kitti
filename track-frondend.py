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

def process_image(input_path, output_path, model_path="./yolo11n-kitti/train/weights/best.pt"):
    """
    处理图片文件，进行目标检测
    
    Args:
        input_path (str): 输入图片文件路径
        output_path (str): 输出图片文件路径
        model_path (str): YOLO模型路径
    """
    print(f"开始处理图片: {input_path}")
    print(f"输出路径: {output_path}")
    
    # 初始化追踪器
    tracker = ByteTrackWrapper(model_path)
    
    # 读取图片
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"错误: 无法读取图片 {input_path}")
        return False
    
    # 处理图片
    try:
        # 更新追踪器
        result_frame = tracker.update(frame)
        
        # 保存结果
        success = cv2.imwrite(output_path, result_frame)
        if not success:
            print(f"错误: 无法保存图片 {output_path}")
            return False
            
        # 检查输出文件是否存在且可读
        if not os.path.exists(output_path):
            print(f"错误: 输出文件 {output_path} 不存在")
            return False
            
        # 检查文件大小
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            print(f"错误: 输出文件 {output_path} 大小为 0")
            return False
            
        print(f"输出文件大小: {file_size} 字节")
        return True
        
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return False

def process_video(input_path, output_path, model_path="yolo11n.pt"):
    """
    处理视频文件，进行目标检测和追踪
    
    Args:
        input_path (str): 输入视频文件路径
        output_path (str): 输出视频文件路径
        model_path (str): YOLO模型路径
    """
    print(f"开始处理视频: {input_path}")
    print(f"输出路径: {output_path}")
    
    # 初始化追踪器
    tracker = ByteTrackWrapper(model_path)
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {input_path}")
        return False
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"视频信息: {width}x{height}, {fps}fps")
    
    # 创建视频写入器
    try:
        # 尝试使用 h264 编码器
        fourcc = cv2.VideoWriter_fourcc(*'h264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print("警告: h264 编码器不可用，尝试使用 mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print("警告: mp4v 编码器不可用，尝试使用 avc1")
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    raise Exception("无法创建视频写入器，请检查编码器支持")
    except Exception as e:
        print(f"创建视频写入器时出错: {str(e)}")
        return False
    
    print(f"使用编码器: {fourcc}")
    
    # 处理每一帧
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"总帧数: {total_frames}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            print(f"\n处理第 {frame_count}/{total_frames} 帧")
            
            # 更新追踪器
            result_frame = tracker.update(frame)
            
            # 写入结果
            out.write(result_frame)
            
            # 显示处理进度
            if frame_count % 10 == 0:
                print(f"进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.2f}%)")
        
        # 释放资源
        cap.release()
        out.release()
        print("视频处理完成")
        
        # 检查输出文件是否存在且可读
        if not os.path.exists(output_path):
            print(f"错误: 输出文件 {output_path} 不存在")
            return False
            
        # 检查文件大小
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            print(f"错误: 输出文件 {output_path} 大小为 0")
            return False
            
        print(f"输出文件大小: {file_size} 字节")
        return True
        
    except Exception as e:
        print(f"处理视频时出错: {str(e)}")
        # 确保资源被释放
        cap.release()
        out.release()
        return False

if __name__ == "__main__":
    # 从命令行参数获取输入和输出路径
    if len(sys.argv) != 3:
        print("用法: python track-frondend.py <input_file_path> <output_file_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件 {input_path} 不存在")
        sys.exit(1)
    
    # 根据文件类型选择处理方式
    success = False
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        success = process_image(input_path, output_path)
    else:
        success = process_video(input_path, output_path)
    
    if success:
        print(f"处理完成，输出文件保存在: {output_path}")
    else:
        print("处理失败")
        sys.exit(1)