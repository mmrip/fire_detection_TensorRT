# 基于YOLOV5的火焰识别部署



## 介绍



## 依赖

- CUDA
- CUDNN
- CUDA Toolkit
- TensorRT
- OpenCV
- protobuf（项目文件中包含 3rdparty/protobuf-3.11.4）

说明：在jetson平台上，JetPack中已经包含上述依赖，直接执行安装即可



## 安装

##### 安装protobuf

```shell
sudo apt-get install protobuf-compiler
cd 3rdparty/protobuf-3.11.4
./configure --prefix=/usr/local/protobuf
make
sudo make install
```

##### 编译源码

```shell
mkdir build
cd build
cmake ..
make -j4
```

## 运行

```shell
cd workspace
./fire_detection
```

## 测试结果

| Jetson Nano | 推理时间 |
| ----------- | -------- |
| FP32        | 110ms    |
| FP16        | 75ms     |
| INT8        | 110ms    |

## 感谢

> - 项目使用的火焰识别onnx文件来源：
>   - https://github.com/robmarkcole/fire-detection-from-images.git
>
> - 项目使用的封装参考了以下链接：
>   - https://github.com/shouxieai/tensorRT_Pro.git