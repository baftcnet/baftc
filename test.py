import os 
import sys
import torch
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix 
import shutil

def test_acc(x_test, y_test, save_path, device="cuda"):
    """
    测试多个模型版本的准确率并计算平均推理时间，计算计算量和混淆矩阵。

    参数:
    - model_name (str): 要加载的模型名称。
    - x_test (np.array): 测试输入数据。
    - y_test (np.array): 测试标签。
    - save_path (str): 保存模型权重和结果的目录。
    - device (str): 运行模型的设备 ('cuda')。

    返回:
    - tuple: (平均测试准确率, 平均推理时间)
    """
    # 确保保存路径存在
    if not os.path.exists(save_path):
        raise ValueError(f"目录 {save_path} 不存在。")
    model = EEGNet()
    folder_path = save_path
    file_extension = '.pth'
    file_list = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    if not file_list:
        raise ValueError(f"在 {folder_path} 中没有找到以 {file_extension} 结尾的模型文件")
    total_test_acc = 0
    total_inference_time = 0
    total_single_inference_time = 0
    total_predicted = []
    total_labels = []
    num_models = len(file_list)
    # 将数据转换为 PyTorch Tensor
    inputs_test = torch.from_numpy(x_test).float().to(device)
    labels_test = torch.from_numpy(y_test).long().to(device)
    # 假设标签是0和1，如果标签更多，请适当修改
    labels = np.unique(y_test)  # 自动获取标签顺序
    for model_file in file_list:
        model_path = os.path.join(folder_path, model_file)
        # 加载模型权重
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except FileNotFoundError:
            print(f"模型文件 {model_path} 未找到。")
            continue
        model.eval()
        with torch.no_grad():
            # 预热推理
            for _ in range(1):
                _ = model(inputs_test)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs_test = model(inputs_test)
            end_event.record()
            torch.cuda.synchronize()
            current_inference_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
            single_inference_time = current_inference_time / len(y_test)
            total_inference_time += current_inference_time
            total_single_inference_time += single_inference_time
            _, predicted_test = outputs_test.max(1)
            correct_test = predicted_test.eq(labels_test).sum().item()
            test_accuracy = correct_test / len(y_test)
            total_test_acc += test_accuracy
            total_predicted.extend(predicted_test.cpu().numpy())
            total_labels.extend(labels_test.cpu().numpy())
            print(f"测试准确率: {test_accuracy:.4f} 对于 {model_file}")
            print(f"单次操作平均推理时间: {single_inference_time:.6f}s 对于 {model_file}")
    avg_inference_time = total_inference_time / num_models
    avg_single_inference_time = total_single_inference_time / num_models
    avg_test_acc = total_test_acc / num_models  # 计算所有模型的平均准确率
    print(f"平均推理时间: {avg_inference_time:.4f}s")
    print(f"平均单次操作推理时间: {avg_single_inference_time:.6f}s")
    print(f"平均测试准确率: {avg_test_acc:.4f}")
    # 打印混淆矩阵和标签顺序
    cm = confusion_matrix(total_labels, total_predicted)
    print("标签顺序:")
    print(labels)
    print("混淆矩阵:")
    print(cm)
    # 保存结果
    with open(os.path.join(save_path, f'test_avg_acc:{avg_test_acc:.4f}.txt'), 'w') as file:
        file.write(f"平均测试准确率: {avg_test_acc:.4f}\n")
    with open(os.path.join(save_path, f'test_avg_inference_time:{avg_inference_time:.4f}.txt'), 'w') as file:
        file.write(f"平均推理时间: {avg_inference_time:.4f}s\n")
    with open(os.path.join(save_path, f'test_avg_single_inference_time:{avg_single_inference_time:.6f}.txt'), 'w') as file:
        file.write(f"平均单次操作推理时间: {avg_single_inference_time:.6f}s\n")
    return avg_test_acc, avg_inference_time
