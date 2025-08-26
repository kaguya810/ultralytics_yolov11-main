# Ultralytics YOLOv11 with Advanced RKNN Optimization 🚀

[![GitHub stars](https://img.shields.io/github/stars/kaguya810/ultralytics_yolov11-main)](https://github.com/kaguya810/ultralytics_yolov11-main/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/kaguya810/ultralytics_yolov11-main)](https://github.com/kaguya810/ultralytics_yolov11-main/network)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**A modernized fork of Ultralytics with advanced RKNN optimization and YOLOv11 support**

This repository provides an up-to-date alternative to outdated Rockchip YOLO implementations, featuring:

## ✨ Key Features

- 🔥 **Latest YOLOv11 Architecture**: Based on the newest stable Ultralytics branch
- ⚡ **Advanced RKNN Optimization**: Optimized export pipeline for Rockchip NPU devices
- 🎯 **Comprehensive Model Support**: Detection, Segmentation, Pose Estimation, and OBB models
- 🔧 **Extensible Design**: Easy integration of custom modules and architectures
- 🚀 **Performance Focused**: Optimized inference pipeline for embedded deployment

## 🚀 RKNN Optimization Features

Unlike outdated alternatives, this implementation provides:

- **Modern Toolchain**: Streamlined PT → ONNX (RKNN format) → RKNN conversion
- **NPU-Optimized Architecture**: Removed post-processing and DFL operations for better quantization
- **Enhanced Performance**: Added confidence sum branch for faster post-processing
- **Maintained Accuracy**: All optimizations preserve inference results

For detailed RKNN export instructions:
- **English**: [RKOPT_README.md](RKOPT_README.md)
- **中文**: [RKOPT_README.zh-CN.md](RKOPT_README.zh-CN.md)

> 💡 **Note**: RKNN optimizations only affect model export. Training follows standard Ultralytics procedures.


