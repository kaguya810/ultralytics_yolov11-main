# ultralytics_yolov11-main

为 YOLOv11 的 pt 模型转 rknn 工具链补足，支持最新版 YOLOv11 的 pt → onnx (rknn format) 转换。

## 项目简介

本项目旨在为 YOLOv11 提供一个完整且易用的 pt（PyTorch）到 rknn（Rockchip 神经网络）格式的转换工具链。网上现有的相关工具大多基于 Rockchip 官方的 yolov8-main 版本，已经严重过时，无法支持最新版 YOLO。为此，本项目基于 ultralytics 官方仓库，fork 下最新分支并做了如下改进：

- **支持 YOLOv11 最新稳定版本**：适配官方原版 YOLOv11，跟随社区最新进展。
- **重写 Head 方法与 Task/Exporter 等核心类**：保证底层结构和数据流与新版 YOLOv11 保持一致，便于后续升级和维护。
- **支持自定义模块扩展**：用户可自行添加新的网络结构或模块，满足各类定制需求。
- **pt → onnx(rknn format) 全流程**：实现 PyTorch 权重到 rknn 推理格式的自动化转换，极大简化嵌入式部署流程。

## 主要功能

- 支持 YOLOv11 最新模型结构的 pt 转 onnx 导出（兼容 rknn 工具链）
- 支持自定义 head 与任务类型
- 简单配置即可适配不同的训练权重和输入分辨率
- 完善的注释和分步骤指引，方便二次开发和集成

## 安装方法

1. 克隆本仓库

   ```bash
   git clone https://github.com/kaguya810/ultralytics_yolov11-main.git
   cd ultralytics_yolov11-main
   ```

2. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```

3. 配置环境

## 使用说明

1. 准备好训练好的 YOLOv11 `.pt` 权重文件。
2. 定位ultralytics_yolov8项目中的ultralytics/cfg/default.yaml;
2. 将其中的model参数路径改成你训练得到的的best.pt模型路径。
3. 执行导出：

   ```bash
   python ./ultralytics/engine/exporter.py
   ```
   导出得到的 `onnx` 文件即可用于 rknn-toolkit 进一步转换和部署。

## 适用场景

- 嵌入式端部署 YOLOv11，如瑞芯微（Rockchip）芯片
- 需要定制 head 或网络结构的场景
- 希望用最新 YOLOv11 算法在国产 SoC 上落地的开发者

## 贡献与交流

欢迎提交 Issue 与 PR，也欢迎在 Discussions 区提出问题与建议。

---

本项目为个人维护，代码仅供学习参考与开源交流，严禁用于任何违法用途。
