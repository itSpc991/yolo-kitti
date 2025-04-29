import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载原始模型
original_model = YOLO(os.path.join(current_dir, 'yolo11n.pt'))
# 加载你的模型
your_model = YOLO(os.path.join(current_dir, 'yolo11n-kitti/train/weights/best.pt'))

# 2. 在相同的测试集上进行推理
test_results_original = original_model.val(data=os.path.join(current_dir, 'kitti.yaml'))
test_results_yours = your_model.val(data=os.path.join(current_dir, 'kitti.yaml'))

# 3. 对比指标
print("\n=== 原始模型性能 ===")
print(f"mAP50: {test_results_original.box.map50:.3f}")
print(f"mAP50-95: {test_results_original.box.map:.3f}")
print(f"平均精确率: {test_results_original.box.mp:.3f}")
print(f"平均召回率: {test_results_original.box.mr:.3f}")

print("\n=== 你的模型性能 ===")
print(f"mAP50: {test_results_yours.box.map50:.3f}")
print(f"mAP50-95: {test_results_yours.box.map:.3f}")
print(f"平均精确率: {test_results_yours.box.mp:.3f}")
print(f"平均召回率: {test_results_yours.box.mr:.3f}")

# 计算性能提升百分比
map50_improvement = (test_results_yours.box.map50 - test_results_original.box.map50) / test_results_original.box.map50 * 100
map_improvement = (test_results_yours.box.map - test_results_original.box.map) / test_results_original.box.map * 100

print("\n=== 性能提升 ===")
print(f"mAP50 提升: {map50_improvement:.1f}%")
print(f"mAP50-95 提升: {map_improvement:.1f}%")

# 4. 可视化对比
def add_title_to_image(image, title):
    # 创建一个标题栏
    title_bar = np.ones((50, image.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(title_bar, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 垂直拼接标题和图片
    final_image = np.vstack([title_bar, image])
    return final_image

def process_single_image(img_name, output_dir):
    test_image = os.path.join(current_dir, 'valid', img_name)
    if os.path.exists(test_image):
        original_results = original_model(test_image)
        your_results = your_model(test_image)
        
        # 获取原始图片的检测结果
        orig_img = np.array(original_results[0].plot())
        orig_img = add_title_to_image(orig_img, f"Original Model - {img_name}")
        orig_path = os.path.join(output_dir, f'original_{img_name}')
        cv2.imwrite(orig_path, orig_img)
        
        # 获取你的模型的检测结果
        your_img = np.array(your_results[0].plot())
        your_img = add_title_to_image(your_img, f"Your Model - {img_name}")
        your_path = os.path.join(output_dir, f'your_{img_name}')
        cv2.imwrite(your_path, your_img)
        
        print(f"Saved comparison images for {img_name}")
    else:
        print(f"Warning: {img_name} not found")

# 设置输出目录
output_dir = os.path.join(current_dir, './contrast')

# 处理单张图片对比
# single_image = '000053.png'
# process_single_image(single_image, output_dir)

# 处理全量验证集对比
# print("\n开始处理全量验证集对比...")
# valid_dir = os.path.join(current_dir, './valid')
# all_images = [f for f in os.listdir(valid_dir) if f.endswith('.png')]
# total_images = len(all_images)

# for idx, img_name in enumerate(all_images, 1):
#     process_single_image(img_name, output_dir)
#     print(f"进度: {idx}/{total_images} ({idx/total_images*100:.1f}%)")

# print("\n全量验证集对比完成！")

os.makedirs(output_dir, exist_ok=True)

def generate_comparison_chart(test_results_original, test_results_yours, output_dir):
    # 准备数据
    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
    original_values = [
        test_results_original.box.map50,
        test_results_original.box.map,
        test_results_original.box.mp,
        test_results_original.box.mr
    ]
    your_values = [
        test_results_yours.box.map50,
        test_results_yours.box.map,
        test_results_yours.box.mp,
        test_results_yours.box.mr
    ]
    
    # 计算提升百分比
    improvements = [(y - o) / o * 100 for o, y in zip(original_values, your_values)]
    
    # 设置图表样式
    plt.style.use('default')  # 使用默认样式
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 设置背景色为白色
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    # 绘制性能对比柱状图
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, original_values, width, label='Original Model', color='#1f77b4')
    ax1.bar(x + width/2, your_values, width, label='Your Model', color='#ff7f0e')
    
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    
    # 添加数值标签
    for i, v in enumerate(original_values):
        ax1.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    for i, v in enumerate(your_values):
        ax1.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    # 绘制提升百分比折线图
    ax2.plot(metrics, improvements, marker='o', color='#2ca02c', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Performance Improvement')
    
    # 添加数值标签
    for i, v in enumerate(improvements):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    plt.close()

# 在脚本末尾添加调用
print("\n生成性能对比图表...")
generate_comparison_chart(test_results_original, test_results_yours, output_dir)
print("性能对比图表已保存到 contrast/performance_comparison.png")