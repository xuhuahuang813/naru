#!/bin/bash
'''
用于读取BASE_LOG_DIR下所有训练log，获取模型名称。在该模型上进行预测。
'''

# 用于标识测试序号，与实验参数无关
TAG=1
DATE="20250512"
BASE_LOG_DIR="log/train/${DATE}_tag_${TAG}_dmv"

# 测试参数
NUM_QUERIES_VALUES=(2000)          # 设置候选的NUM_QUERIES值
PSAMPLE_VALUES=(2000)   # 设置候选的PSAMPLE值 默认是2000
RESIDUAL="--residual"

# 测试数据和查询路径
DATASET="dmv"
QUERYSET="datasets/dmv_10w_test.json"

# 遍历日志目录中的所有文件
for LOG_FILE in "${BASE_LOG_DIR}"/*.txt; do
    if [ -f "$LOG_FILE" ]; then
        echo "Processing file: $LOG_FILE"

        # 从日志文件中读取最后一行的模型路径
        FULL_GLOB=$(tail -n 1 "$LOG_FILE")
        # 去掉 models/ 前缀，提取模型路径
        GLOB=$(echo "$FULL_GLOB" | sed 's|^models/||')
        
        # 从文件名中解析参数
        FILENAME=$(basename "$LOG_FILE")
        LAYERS=$(echo "$FILENAME" | grep -oP '(?<=layers)\d+')
        FC_HIDDENS=$(echo "$FILENAME" | grep -oP '(?<=fchiddens)\d+')
        EPOCH=$(echo "$FILENAME" | grep -oP '(?<=epoch)\d+')
        EMBED_SIZE=$(echo "$FILENAME" | grep -oP '(?<=embed)\d+')

        echo "Parsed parameters: DATASET=$DATASET LAYERS=$LAYERS FC_HIDDENS=$FC_HIDDENS EPOCH=$EPOCH EMBED_SIZE=$EMBED_SIZE"
        echo "Model path (GLOB): $GLOB"

        # 创建结果目录
        RESULT_DIR="./log/test/${DATE}_tag_${TAG}_dmv" 
        if [ ! -d "$RESULT_DIR" ]; then
            echo "Creating directory: $RESULT_DIR"
            mkdir -p "$RESULT_DIR"
        fi

        # 遍历 NUM_QUERIES_VALUES 和 PSAMPLE_VALUES
        for NUM_QUERIES in "${NUM_QUERIES_VALUES[@]}"; do
            for PSAMPLE in "${PSAMPLE_VALUES[@]}"; do
                RESULT_CSV="${RESULT_DIR}/fchiddens${FC_HIDDENS}_layers${LAYERS}_epoch${EPOCH}_emb${EMBED_SIZE}_QUERY${NUM_QUERIES}_PSAMPLE${PSAMPLE}.csv"
                RESULT_TXT="${RESULT_DIR}/fchiddens${FC_HIDDENS}_layers${LAYERS}_epoch${EPOCH}_emb${EMBED_SIZE}_QUERY${NUM_QUERIES}_PSAMPLE${PSAMPLE}.txt"
                
                echo "Evaluating: NUM_QUERIES=$NUM_QUERIES PSAMPLE=$PSAMPLE"
                echo "Results: $RESULT_CSV, $RESULT_TXT"

                # 运行评估脚本
                CUDA_VISIBLE_DEVICES=0 python eval_model.py \
                    --glob="$GLOB" \
                    $RESIDUAL \
                    --dataset="$DATASET" \
                    --queryset="$QUERYSET" \
                    --num-queries="$NUM_QUERIES" \
                    --psample="$PSAMPLE" \
                    --layers="$LAYERS" \
                    --fc-hiddens="$FC_HIDDENS" \
                    --embed-size="$EMBED_SIZE" \
                    --err-csv="$RESULT_CSV" > "$RESULT_TXT"
            done
        done
    else
        echo "No log files found in $BASE_LOG_DIR"
    fi
done