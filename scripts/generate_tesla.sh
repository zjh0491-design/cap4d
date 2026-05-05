#!/bin/bash

# =========================
# CAP4D Tesla Generate Script
# =========================

# -------------------------
# 0️⃣ 设置 CUDA / 多 GPU
# -------------------------
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/data1/zjh/cap4d/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc

QUALITY="high"   # 可以改成 "medium" 或 "low"# 选择生成质量：high / medium / low

# -------------------------
# 输出目录
# -------------------------
OUTPUT_DIR=examples/output/teslapro
AVATAR_DIR=$OUTPUT_DIR/avatar
ANIM_NUM=00    #动画编号控制
ANIM_DIR=$OUTPUT_DIR/animation_$ANIM_NUM

# 如果目录不存在，创建
mkdir -p $OUTPUT_DIR
mkdir -p $AVATAR_DIR
mkdir -p $ANIM_DIR

# -------------------------
# 1️⃣ 生成多视角图像 (MMDM)
# -------------------------
python cap4d/inference/generate_images.py \
    --config_path configs/generation/${QUALITY}_quality.yaml \
    --reference_data_path examples/input/tesla/ \
    --output_path $OUTPUT_DIR/ \

# -------------------------
# 2️⃣ 拟合 Gaussian Avatar (训练)
# -------------------------
python gaussianavatars/train.py \
    --config_path configs/avatar/default.yaml \
    --source_paths $OUTPUT_DIR/reference_images/ $OUTPUT_DIR/generated_images/ \
    --model_path $AVATAR_DIR/ \
    --load_existing_checkpoint 0 \


# -------------------------
# 3️⃣ 渲染动画并导出 3D 模型
# -------------------------
python gaussianavatars/animate.py \
    --model_path $AVATAR_DIR/ \
    --target_animation_path examples/input/animation/sequence_00/fit.npz \
    --target_cam_trajectory_path examples/input/animation/sequence_00/orbit.npz  \
    --output_path $ANIM_DIR/ \
    --export_ply 1 \
    --compress_ply 0 \