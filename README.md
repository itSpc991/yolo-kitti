# YOLO-KITTI：智能车辆感知系统

本项目是一个基于YOLO（You Only Look Once）进行目标检测、ByteTrack实现多目标跟踪的智能车辆感知和监控系统。该系统使用KITTI数据集进行训练和测试，能够在复杂环境中实现对车辆、行人等目标的高效检测与跟踪。

## 🚀 项目特点

* **实时目标检测**：采用YOLOv8和YOLOv11模型，提供高速高精度的目标检测。
* **多目标跟踪**：集成ByteTrack算法，实现高效、鲁棒的多目标跟踪。
* **模块化架构**：代码结构清晰，分为数据处理、目标检测、跟踪和前端可视化模块。
* **用户友好界面**：基于React的Web前端，可实时查看监控效果。

## 📂 项目结构

```
├── data/                # KITTI数据集及标签文件
├── models/              # YOLOv8和YOLOv11模型定义
├── tracking/            # ByteTrack多目标跟踪实现
├── utils/               # 数据处理与可视化工具脚本
├── frontend/            # 基于React的前端界面
├── README.md            # 项目文档（本文件）
```

## 📊 快速开始

### 1. 克隆项目

```bash
$ git clone https://github.com/itSpc991/yolo-kitti.git
$ cd yolo-kitti
```

### 2. 安装依赖

```bash
$ pip install -r requirements.txt
$ cd frontend
$ npm install
```

### 3. 准备数据集

* 从Kaggle下载KITTI数据集。
* 将数据集放置于 `data/` 目录。
* 运行数据预处理脚本：

```bash
$ python utils/preprocess_data.py
```

### 4. 训练模型

```bash
$ python models/train_yolo.py --model yolo_v8 --epochs 50
$ python models/train_yolo.py --model yolo_v11 --epochs 50
```

### 5. 运行多目标跟踪

```bash
$ python tracking/bytetrack.py --input data/test_video.mp4
```

### 6. 启动前端

```bash
$ cd frontend
$ npm start
```

## 📈 结果

* YOLOv8和YOLOv11模型实现了高精度的目标检测。
* ByteTrack在复杂场景中实现了稳健的多目标跟踪。

## 📌 未来改进

* 支持更多目标类别（如自行车、卡车等）。
* 集成Re-ID（再识别）以提升跟踪性能。
* 增加详细的跟踪统计信息。

## 🤝 贡献

欢迎提交问题或PR以改进项目。

## 📄 许可证

本项目基于MIT License许可。

##### 此README由GTP生成，仅供参考
