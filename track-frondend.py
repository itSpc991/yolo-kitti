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
    def __init__(self, model_path, tracker_args=None, is_video=False):
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
        self.congestion_threshold = 0.3  # 拥挤度阈值，可以根据实际情况调整
        self.is_video = is_video  # 保存是否为视频的标志
        
        # 添加字体路径
        self.font_path = "/System/Library/Fonts/PingFang.ttc"  # macOS 系统中文字体
        
    def put_chinese_text(self, img, text, position, color):
        """
        在图片上显示中文
        
        Args:
            img: 图片
            text: 要显示的文本
            position: 位置，元组 (x, y)
            color: 颜色，元组 (B, G, R)
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # 将OpenCV图片转换为PIL图片
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 加载字体，大小为30
            try:
                font = ImageFont.truetype(self.font_path, 30)
            except Exception as e:
                print(f"无法加载字体 {self.font_path}，尝试使用默认字体")
                # 尝试使用其他常见的中文字体路径
                font_paths = [
                    "/System/Library/Fonts/STHeiti Light.ttc",  # macOS另一个中文字体
                    "/System/Library/Fonts/STHeiti Medium.ttc",
                    "/System/Library/Fonts/Hiragino Sans GB.ttc",
                    "/System/Library/Fonts/Apple LiGothic Medium.ttf"
                ]
                
                for path in font_paths:
                    try:
                        font = ImageFont.truetype(path, 30)
                        self.font_path = path  # 更新为可用的字体路径
                        break
                    except:
                        continue
                else:
                    # 如果所有字体都失败，使用默认字体
                    font = ImageFont.load_default()
            
            # 在PIL图片上绘制文字
            draw.text(position, text, font=font, fill=color[::-1])  # PIL使用RGB而CV2使用BGR，所以需要反转颜色
            
            # 将PIL图片转回OpenCV格式
            img_opencv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # 只复制文字区域
            rows, cols = img.shape[:2]
            roi = img_opencv[0:rows, 0:cols]
            img[0:rows, 0:cols] = roi
            
        except Exception as e:
            print(f"显示中文文本时出错: {str(e)}")
            # 如果显示中文失败，回退到英文显示
            # 将中文转换为英文显示
            english_text = text.replace("路段拥挤!", "Road Congested!").replace("路段正常", "Road Normal").replace("拥挤度", "Congestion")
            cv2.putText(img, english_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    def check_congestion(self, boxes, frame_shape):
        """
        检查路段拥挤程度
        
        Args:
            boxes: 检测框列表
            frame_shape: 图像尺寸
        
        Returns:
            is_congested: 是否拥挤
            congestion_ratio: 拥挤度
        """
        if len(boxes) == 0:
            return False, 0.0
            
        # 计算所有检测框的面积总和
        total_box_area = 0
        for box in boxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            total_box_area += w * h
            
        # 计算图像总面积
        frame_area = frame_shape[0] * frame_shape[1]
        
        # 计算拥挤度（检测框面积占比）
        congestion_ratio = total_box_area / frame_area
        
        # 判断是否拥挤
        is_congested = congestion_ratio > self.congestion_threshold
        
        return is_congested, congestion_ratio
        
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
        # 只在视频处理时添加拥挤度检测
        if self.is_video and len(online_targets) > 0:
            # 收集所有追踪目标的边界框
            boxes = []
            for t in online_targets:
                tlwh = t.tlwh
                x1, y1 = tlwh[0], tlwh[1]
                x2, y2 = x1 + tlwh[2], y1 + tlwh[3]
                boxes.append([x1, y1, x2, y2])
            
            # 检查拥挤度
            is_congested, congestion_ratio = self.check_congestion(boxes, frame.shape)
            
            # 在画面上显示拥挤状态
            status_text = "路段拥挤!" if is_congested else "路段正常"
            color = (0, 0, 255) if is_congested else (0, 255, 0)
            self.put_chinese_text(frame, f"{status_text} (拥挤度: {congestion_ratio:.2%})", 
                                (10, 30), color)
        
        # 绘制追踪结果
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
    tracker = ByteTrackWrapper(model_path, is_video=False)
    
    # 读取图片
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"错误: 无法读取图片 {input_path}")
        return False
    
    # 处理图片
    try:
        # 更新追踪器，不传入 is_video 参数
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
    tracker = ByteTrackWrapper(model_path, is_video=True)
    
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
            
            # 更新追踪器，显示拥挤状态
            result_frame = tracker.update(frame)  # 正确的调用方式
            
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