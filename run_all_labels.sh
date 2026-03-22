#!/bin/sh

# MCTS 批量标签优化脚本 - 直接调用版本
# 依次优化6个标签的提示词
# 使用方法: sh run_all_labels.sh

# 工作目录
cd /ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end || exit 1

echo "======================================================================"
echo "🚀 开始批量优化 1 个标签"
echo "======================================================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""


# 1. 无风险 
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 无风险"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "无风险" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 5 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 无风险 - 已完成"
echo "等待 5 秒..."
sleep 5

# 2.  肢体冲突-打架-轻微伤-门诊治疗-无需就医
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 肢体冲突-打架-轻微伤-门诊治疗-无需就医"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "肢体冲突-打架-轻微伤-门诊治疗-无需就医" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 5 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 肢体冲突-打架-轻微伤-门诊治疗-无需就医 - 已完成"
echo "等待 5 秒..."
sleep 5


# 4. 交通事故-轻微伤-门诊治疗-无需就医 
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 交通事故-轻微伤-门诊治疗-无需就医"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "交通事故-轻微伤-门诊治疗-无需就医" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 4 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 交通事故-轻微伤-门诊治疗-无需就医  - 已完成"
echo "等待 5 秒..."
sleep 5

# 5. 肢体冲突-打架-轻伤-住院治疗-观察 
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 肢体冲突-打架-轻伤-住院治疗-观察"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "肢体冲突-打架-轻伤-住院治疗-观察" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 3 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 肢体冲突-打架-轻伤-住院治疗-观察  - 已完成"
echo "等待 5 秒..."
sleep 5

# 7. 敏感身份进线-警察来电调取信息-非遗失物品
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 敏感身份进线-警察来电调取信息-非遗失物品"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "敏感身份进线-警察来电调取信息-非遗失物品" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 4 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 敏感身份进线-警察来电调取信息-非遗失物品  - 已完成"
echo "等待 5 秒..."
sleep 5

# 7. 敏感身份进线-特殊人群
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 敏感身份进线-特殊人群"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "敏感身份进线-特殊人群" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 5 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 敏感身份进线-特殊人群  - 已完成"
echo "等待 5 秒..."
sleep 5

# 8. 肢体冲突-打架-轻微拉扯-推搡-无需就医
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 肢体冲突-打架-轻微拉扯-推搡-无需就医"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "肢体冲突-打架-轻微拉扯-推搡-无需就医" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 5 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 肢体冲突-打架-轻微拉扯-推搡-无需就医  - 已完成"
echo "等待 5 秒..."
sleep 5

# 9. 意外受伤-轻微伤-门诊治疗-无需就医
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 意外受伤-轻微伤-门诊治疗-无需就医"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "意外受伤-轻微伤-门诊治疗-无需就医" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 5 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 意外受伤-轻微伤-门诊治疗-无需就医  - 已完成"
echo "等待 5 秒..."
sleep 5

# 9. 意外受伤-轻伤-住院治疗-观察
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 意外受伤-轻伤-住院治疗-观察"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "意外受伤-轻伤-住院治疗-观察" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 3 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 意外受伤-轻伤-住院治疗-观察  - 已完成"
echo "等待 5 秒..."
sleep 5

# 10. 已发生政府渠道外投
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 已发生政府渠道外投"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "已发生政府渠道外投" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 3 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 已发生政府渠道外投  - 已完成"
echo "等待 5 秒..."
sleep 5

# 11. 扬言政府渠道外投
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 扬言政府渠道外投"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "扬言政府渠道外投" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 5 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 扬言政府渠道外投  - 已完成"
echo "等待 5 秒..."
sleep 5


# 15. 交通事故-轻伤-住院治疗-观察 
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 交通事故-轻伤-住院治疗-观察"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "交通事故-轻伤-住院治疗-观察" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 3 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 交通事故-轻伤-住院治疗-观察  - 已完成"
echo "等待 5 秒..."
sleep 5

# 15. 偷拍-没有将信息发布到网上
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 偷拍-没有将信息发布到网上"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "偷拍-没有将信息发布到网上" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 4 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 偷拍-没有将信息发布到网上  - 已完成"
echo "等待 5 秒..."
sleep 5

# 15. 延误行程乘客索赔
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 延误行程乘客索赔"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "延误行程乘客索赔" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 3 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 延误行程乘客索赔  - 已完成"
echo "等待 5 秒..."
sleep 5

# 15. 限制人身自由-司乘已分离
echo ""
echo "======================================================================"
echo "📝 [1/1] 正在优化标签: 限制人身自由-司乘已分离"
echo "======================================================================"
/home/LD/miniconda3/envs/q3_1/bin/python main.py \
  --target_label "限制人身自由-司乘已分离" \
  --data_path "/ai/hlf/zhongda3_0/scc_B_sft-0311/valid—四版训练数据—无舆情多标签—校正2.xlsx" \
  --model_path "/ai/hlf/zhongda3_0/zhongda3_0_sft_B-第四版数据/checkpoint-400" \
  --model_type "qwen3" \
  --confusion_file "/ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end0303/datasets/label_confusion_analysis.xlsx" \
  --iterations 3 \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8085"

echo ""
echo "✓ [1/1] 限制人身自由-司乘已分离  - 已完成"
echo "等待 5 秒..."
sleep 5

# 完成
echo ""
echo "======================================================================"
echo "🎉 所有标签优化完成！"
echo "======================================================================"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "✨ 优化结果已保存到 label_prompts_config.py"
echo "======================================================================"

# conda activate q3_1 && cd /ai/hlf/zhongda3_0/mcts_prompt_gen_v4_b2_end
# sh run_all_labels.sh