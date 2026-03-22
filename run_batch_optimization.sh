#!/bin/sh

# MCTS 批量标签优化脚本
# 依次优化多个标签的提示词
# 使用方法: sh run_batch_optimization.sh

# 配置参数
DATA_PATH="/ai/ltx/zhongda/datasets/与训练集同分布测试集-正负样本-已质检-纠正-干净版本-0105.xlsx"
MODEL_PATH="/ai/ltx/zhongda/weight_file/merged_global_step_1900-8922"
MODEL_TYPE="qwen3"
CONFUSION_FILE="/ai/ltx/zhongda/datasets/label_confusion_analysis.xlsx"
ITERATIONS=10
GPU_IDS="0,1,2,3"
API_URL="http://localhost:8000"

# 工作目录
SCRIPT_DIR="/ai/ltx/zhongda/mcts_prompt_gen_v4"
cd "$SCRIPT_DIR" || exit 1

# 需要优化的标签列表（使用换行分隔）
LABELS="无风险
意外受伤
行程延误
限制人身自由
性骚扰
交通事故"

# 计算总标签数
TOTAL_LABELS=$(echo "$LABELS" | wc -l | tr -d ' ')

# 记录开始时间
START_TIME=$(date +%s)
echo "======================================================================"
echo "🚀 MCTS 批量优化任务开始"
echo "======================================================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "总标签数: $TOTAL_LABELS"
echo "迭代次数: $ITERATIONS"
echo "GPU配置: $GPU_IDS"
echo "======================================================================"
echo ""

# 依次处理每个标签
SUCCESS_COUNT=0
FAIL_COUNT=0
INDEX=0

echo "$LABELS" | while IFS= read -r LABEL; do
    INDEX=$((INDEX + 1))
    
    echo ""
    echo "============TOTAL_LABELS=============================================="
    echo "📝 [$INDEX/${#LABELS[@]}] 正在优化标签: $LABEL"
    echo "======================================================================"
    echo "当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 运行优化
    python main.py \
        --target_label "$LABEL" \
        --data_path "$DATA_PATH" \
        --model_path "$MODEL_PATH" \
        --model_type "$MODEL_TYPE" \
        --confusion_file "$CONFUSION_FILE" \
        --iterations "$ITERATIONS" \
        --gpu_ids "$GPU_IDS" \
        --api_url "$API_URL"
    
    # 检查执行结果
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""TOTAL_LABELS] 标签 '$LABEL' 优化成功！"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "❌ [$INDEX/$TOTAL_LABELS
        echo "❌ [$INDEX/${#LABELS[@]}] 标签 '$LABEL' 优化失败！(Exit Code: $EXIT_CODE)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        
        # 询问是否继续（可选）
        # read -p "是否继续下一个标签？(y/n) " -n 1 -r
        # echo
        # if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        #     echo "用户中止任务"
        #     break
        # fi
    fi
    
    echo ""
    echo "----------------------------------------------------------------------"
    echo "进度: $INDEX/$TOTAL_LABELS 完成 | 成功: $SUCCESS_COUNT | 失败: $FAIL_COUNT"
    echo "----------------------------------------------------------------------"
    
    # 标签之间稍作休息（避免连续高负载）
    if [ $INDEX -lt $TOTAL_LABELS ]; then
        echo "⏸️  等待 5 秒后继续下一个标签..."
        sleep 5
    fi
done

# 注意：由于 while 循环在子shell中运行，计数器不会传回主shell
# 重新计算最终统计
SUCCESS_COUNT=0
FAIL_COUNT=0

# 计算总耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

# 输出最终统计
echo ""
echo "======================================================================"
echo "🎉 MCTS 批量优化任务完成"
echo "===========TOTAL_LABELS"
echo ""
echo "✅ 所有标签已处理完成"
echo "详细结果请查看上方输出日志"
# 计算成功率
if [ ${#LABELS[@]} -gt 0 ]; then
    SUCCESS_RATE=$(awk "BEGIN {printf \"%.1f\", $SUCCESS_COUNT/${#LABELS[@]}*100}")
    echo "  - 成功率: ${SUCCESS_RATE}%"
fi

echo ""
echo "✨ 优化结果已保存到 label_prompts_config.py"
echo "======================================================================"
