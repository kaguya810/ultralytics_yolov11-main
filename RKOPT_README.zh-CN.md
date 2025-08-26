# YOLOv11 导出 RKNPU 适配模型说明

## Source

本仓库基于最新稳定版 https://github.com/ultralytics/ultralytics 进行修改，升级支持 YOLOv11 架构并提供先进的 RKNN 优化功能。

## 模型差异

相比过时的瑞芯微 YOLO 工具链，本现代化实现提供：

### 🚀 核心优化特性（输出结果不变）
- **修改输出结构**: 移除模型内后处理结构（量化不友好的操作迁移至 CPU）
- **NPU 性能增强**: 将 DFL 结构迁移至模型外部，提升 NPU 推理性能
- **加速后处理**: 新增置信度总和输出分支，加速阈值筛选过程
- **YOLOv11 架构支持**: 升级至最新稳定版 YOLO 架构，包含现代化改进

### 🔧 技术改进
- **更好的量化效果**: 后处理操作针对 INT8 部署优化
- **降低延迟**: DFL 操作迁移至 CPU 以实现最优 NPU 利用率
- **增强吞吐量**: 置信度求和分支支持更快的 NMS 替代方案

所有移除的操作均在 CPU 后处理中高效处理（参考实现可在 **RKNN_Model_Zoo** 中找到）。

## 导出 ONNX 模型

在满足 `./requirements.txt` 环境要求后，执行以下命令导出模型：

```bash
# 调整 ./ultralytics/cfg/default.yaml 中 model 文件路径（默认为 yolov8n.pt）
# 若自己训练模型，请调整至对应的路径
# 支持：检测、分割、姿态估计、OBB 模型

# 示例：
# yolov11n.pt - 检测模型
# yolov11n-seg.pt - 分割模型  
# yolov11n-pose.pt - 姿态估计模型
# yolov11n-obb.pt - 有向边界框模型

export PYTHONPATH=./
python ./ultralytics/engine/exporter.py

# 执行完毕后，会生成 ONNX 模型
# 原始模型 "yolov11n.pt" → 生成模型 "yolov11n.onnx"
```

## 转换 RKNN 模型、Python 演示、C 演示

请参考完整指南：https://github.com/airockchip/rknn_model_zoo

### 🎯 支持的模型类型
- ✅ 目标检测 (YOLOv11n/s/m/l/x)
- ✅ 实例分割 (YOLOv11n/s/m/l/x-seg)
- ✅ 姿态估计 (YOLOv11n/s/m/l/x-pose)  
- ✅ 有向边界框 (YOLOv11n/s/m/l/x-obb)

