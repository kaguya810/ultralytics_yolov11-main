# YOLOv11 RKNN Optimization for Model Export

## Source
Based on the latest stable release from https://github.com/ultralytics/ultralytics, updated to support YOLOv11 architecture with advanced RKNN optimizations.

## What's Different

This modern implementation surpasses outdated Rockchip YOLO toolchains by providing:

### 🚀 Key Optimizations (Results Unchanged)
- **Modified Output Structure**: Removed post-processing from model (quantization-unfriendly operations moved to CPU)
- **NPU Performance Enhancement**: Relocated DFL structure outside model for better NPU inference performance  
- **Accelerated Post-Processing**: Added confidence sum output branch to speed up threshold filtering
- **YOLOv11 Architecture Support**: Updated to latest stable YOLO architecture with modern improvements

### 🔧 Technical Improvements
- **Better Quantization**: Post-processing operations optimized for INT8 deployment
- **Reduced Latency**: DFL operations moved to CPU for optimal NPU utilization
- **Enhanced Throughput**: Confidence sum branch enables faster NMS alternatives

All removed operations are handled efficiently in CPU post-processing (reference implementations available in **RKNN_Model_Zoo**).

## Export ONNX Model

After meeting the environment requirements in `./requirements.txt`, execute the following command to export models:

```bash
# Configure model path in ./ultralytics/cfg/default.yaml (default: yolov8n.pt)
# For custom trained models, provide the corresponding path
# Supports: Detection, Segmentation, Pose Estimation, OBB models

# Examples:
# yolov11n.pt - Detection model  
# yolov11n-seg.pt - Segmentation model
# yolov11n-pose.pt - Pose estimation model
# yolov11n-obb.pt - Oriented bounding box model

export PYTHONPATH=./
python ./ultralytics/engine/exporter.py

# Upon completion, ONNX model will be generated
# Original model "yolov11n.pt" → Generated model "yolov11n.onnx"
```

## Convert to RKNN Model, Python Demo, C Demo

Please refer to the comprehensive guide at: https://github.com/airockchip/rknn_model_zoo

### 🎯 Supported Model Types
- ✅ Object Detection (YOLOv11n/s/m/l/x)
- ✅ Instance Segmentation (YOLOv11n/s/m/l/x-seg)  
- ✅ Pose Estimation (YOLOv11n/s/m/l/x-pose)
- ✅ Oriented Bounding Boxes (YOLOv11n/s/m/l/x-obb)