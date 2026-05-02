#!/bin/bash

# 配置路径（根据你的环境修改）
CUDA_PATH="/usr/local/cuda-12.8"
NVCC="$CUDA_PATH/bin/nvcc"
NSYS="sudo $CUDA_PATH/bin/nsys"
NCU="sudo $CUDA_PATH/bin/ncu"
CUDNN_INC="/usr/include/x86_64-linux-gnu/"

# 检查参数
if [ -z "$1" ]; then
    echo "用法: $0 <source.cu> [report_name]"
    echo "示例: $0 matrixMultiply.cu my_experiment"
    exit 1
fi

SRC=$1
BASE_NAME=$(basename "$SRC" .cu)
FINAL_NAME=${2:-$BASE_NAME}
BUILD_DIR="./build"
TARGET="$BUILD_DIR/$BASE_NAME"

# 创建目录并编译
mkdir -p "$BUILD_DIR"
echo "------------------------------------------------"
echo "[1/3] 正在编译: $SRC -> $TARGET"

$NVCC -O3 -std=c++11 -gencode arch=compute_89,code=sm_89 \
    -I $CUDNN_INC \
    "$SRC" -o "$TARGET" -lcublas -lcudnn

if [ $? -ne 0 ]; then
    echo "编译失败，请检查代码。"
    exit 1
fi

# Nsight Systems
echo -e "\n[2/3] 正在生成 Nsight Systems 报告 (.nsys-rep)..."
$NSYS profile --stats=true --trace=cuda,cublas,osrt -o "$BUILD_DIR/$FINAL_NAME" --force-overwrite true "$TARGET"

# Nsight Compute
echo -e "\n[3/3] 正在生成 Nsight Compute 报告 (.ncu-res)..."
$NCU --set full -o "$BUILD_DIR/$FINAL_NAME" --force-overwrite "$TARGET"

echo "------------------------------------------------"
echo "所有报告已生成在 $BUILD_DIR/ 目录下："
echo "- $FINAL_NAME.nsys-rep"
echo "- $FINAL_NAME.ncu-res"