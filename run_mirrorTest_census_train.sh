#!/bin/bash

# 用于标识测试序号，与实验参数无关
TAG=1

# 实验参数
EPOCHS=(20)
DATASET="census"
FC_HIDDENS=(128) # 默认是128
LAYERS=(4)           # 默认是4
EMBED_SIZE=(32)        # 默认是32

# 获取当前时间戳函数
get_timestamp() {
    date +"%Y%m%d"
}

# 遍历数组并训练
for EPOCH in "${EPOCHS[@]}"; do
    for HIDDEN in "${FC_HIDDENS[@]}"; do
        for LAYER in "${LAYERS[@]}"; do
            for EMBED in "${EMBED_SIZE[@]}"; do
                TIMESTAMP=$(get_timestamp) # 获取当前时间戳
                LOG_DIR="./log/train/${TIMESTAMP}_tag_${TAG}_${DATASET}"
                LOG_FILE="${LOG_DIR}/fchiddens${HIDDEN}_layers${LAYER}_embed${EMBED}_epoch${EPOCH}.txt"
                
                # 检查并创建日志目录
                if [ ! -d "$LOG_DIR" ]; then
                    echo "Creating directory: $LOG_DIR"
                    mkdir -p "$LOG_DIR"
                fi

                echo "Training with TIMESTAMP=$TIMESTAMP TAG=$TAG DATASET=$DATASET FC_HIDDENS=$HIDDEN LAYERS=$LAYER EMBED_SIZE=$EMBED EPOCH=$EPOCH"
                echo "Logging output to $LOG_FILE"

                # 执行训练脚本并将输出重定向到日志文件
                CUDA_VISIBLE_DEVICES=0 python train_model.py --epochs=$EPOCH --fc-hiddens=$HIDDEN --layers=$LAYER --embed-size=$EMBED --residual --dataset=$DATASET > "$LOG_FILE" 2>&1
            done
        done
    done
done