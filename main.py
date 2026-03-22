"""
C-MCTS Main - 主入口 (v2)

用法:
    python main_v2.py \
      --target_label "伤害他人" \
      --data_path "/path/to/data.xlsx" \
      --model_path "/path/to/model" \
      --confusion_file "/path/to/confusion.xlsx" \
      --iterations 50
"""

import os
import sys
import argparse
import re
import json
from datetime import datetime

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

from mcts_core import MCTS, ConfusionPrior
from action_generator import ActionGenerator
from evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="C-MCTS: Confusion-Guided MCTS for Prompt Optimization")

    # 必选参数
    parser.add_argument("--target_label", type=str, required=True,
                        help="需要优化的目标标签名称")
    parser.add_argument("--data_path", type=str, required=True,
                        help="验证数据集路径 (Excel/CSV)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="用于评估的模型路径")

    # 可选参数
    parser.add_argument("--model_type", type=str, default="qwen3")
    parser.add_argument("--confusion_file", type=str, default=None,
                        help="混淆矩阵文件路径 (强烈推荐)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="MCTS 迭代次数")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000")

    # C-MCTS 超参数
    parser.add_argument("--c1", type=float, default=0.5,
                        help="PUCT 动态探索系数 c1")
    parser.add_argument("--c2", type=float, default=100.0,
                        help="PUCT 动态探索系数 c2")
    parser.add_argument("--pw_alpha", type=float, default=0.5,
                        help="Progressive Widening 指数")
    parser.add_argument("--pw_k0", type=int, default=4,
                        help="Progressive Widening 初始宽度")
    parser.add_argument("--coverage_ratio", type=float, default=0.5,
                        help="FN 覆盖率递减目标")
    parser.add_argument("--confusion_alpha", type=float, default=0.6,
                        help="混淆先验混合权重 (α * P_llm + (1-α) * P_confusion)")
    parser.add_argument("--confusion_temp", type=float, default=1.0,
                        help="混淆先验温度参数")

    return parser.parse_args()


def get_config_file_path():
    return os.path.join(current_dir, "label_prompts_config.py")


def extract_initial_prompt_from_config(target_label: str) -> str:
    """从 label_prompts_config.py 提取初始 prompt"""
    config_path = get_config_file_path()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = rf'"{re.escape(target_label)}"\s*:\s*"""(.*?)"""'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    raise ValueError(f"未在 label_prompts_config.py 中找到 '{target_label}'")


def write_back_optimized_prompt(target_label: str, optimized_prompt: str):
    """将优化后的 prompt 写回配置文件"""
    config_path = get_config_file_path()

    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 备份
    backup_path = config_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ 备份 → {backup_path}")

    # 替换
    pattern = rf'("{re.escape(target_label)}"\s*:\s*""").*?(""")'
    new_content = re.sub(
        pattern,
        lambda m: m.group(1) + optimized_prompt + m.group(2),
        content, flags=re.DOTALL)

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"✓ 已更新 '{target_label}' 的 prompt")


def main():
    args = parse_args()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 80)
    print(f"🚀 C-MCTS Prompt Optimization")
    print(f"   Target: {args.target_label}")
    print(f"   Time: {datetime.now()}")
    print(f"   Iterations: {args.iterations}")
    print("=" * 80)

    # ─── 1. 加载初始 prompt ───
    initial_prompt = extract_initial_prompt_from_config(args.target_label)
    print(f"\n📝 Initial Prompt ({len(initial_prompt)} chars):")
    print(f"   {initial_prompt[:200]}...")

    # ─── 2. 初始化组件 ───
    print("\n🔧 初始化组件...")

    # Action Generator
    DEEPSEEK_API_KEY = "sk-gzhFbRu8rikjl4N606EcB46974264f098246D1EeE59eCc20"
    action_gen = ActionGenerator(api_key=DEEPSEEK_API_KEY)

    # Evaluator
    evaluator = Evaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        target_label=args.target_label,
        confusion_file=args.confusion_file,
        gpu_ids=args.gpu_ids,
        model_type=args.model_type,
        api_url=args.api_url
    )

    # [P0] 混淆引导先验
    confusion_prior = None
    if evaluator.confused_labels:
        confusion_prior = ConfusionPrior(
            confused_labels=evaluator.confused_labels,
            confusion_rates=evaluator.confusion_rates,
            alpha=args.confusion_alpha,
            temperature=args.confusion_temp
        )
        print(f"\n📊 混淆引导先验:")
        print(f"   混淆标签 ({len(evaluator.confused_labels)}): "
              f"{', '.join(evaluator.confused_labels[:5])}...")
        print(f"   α={args.confusion_alpha}, τ={args.confusion_temp}")
    else:
        print(f"\n⚠️ 未提供混淆矩阵, 不使用混淆引导先验")

    # C-MCTS Engine
    mcts = MCTS(
        root_prompt=initial_prompt,
        evaluator=evaluator,
        action_generator=action_gen,
        c1=args.c1,
        c2=args.c2,
        coverage_ratio=args.coverage_ratio,
        pw_alpha=args.pw_alpha,
        pw_k0=args.pw_k0,
        confusion_prior=confusion_prior
    )

    # ─── 3. 运行搜索 ───
    print(f"\n🏁 开始 C-MCTS 搜索 ({args.iterations} 次迭代)...")
    best_path = mcts.search(iterations=args.iterations)

    # ─── 4. 输出结果 ───
    print("\n" + "=" * 80)
    print("🏆 优化完成!")
    print("=" * 80)

    if not best_path:
        print("❌ 未找到有效路径")
        return

    best_node = best_path[-1]
    print(f"\n最优节点:")
    print(f"  N={best_node.N}, Q={best_node.Q:.4f}")
    print(f"  深度={best_node.depth}")
    print(f"  冻结={best_node.is_frozen}")
    print(f"  动作={best_node.action_taken}")

    # 保存结果
    output_file = f"cmcts_optimized_{args.target_label}_{ts}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"C-MCTS Optimization Results\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Target: {args.target_label}\n")
        f.write(f"Score (Q): {best_node.Q:.4f}\n")
        f.write(f"Depth: {best_node.depth}\n")
        f.write(f"Frozen: {best_node.is_frozen}\n")
        f.write(f"Verified Nodes: {len(mcts.verified_nodes)}\n")
        f.write(f"Global Best F1: {mcts.global_best_f1:.4f}\n")
        f.write(f"\n{'=' * 50}\n")
        f.write(f"Optimized Prompt:\n\n")
        f.write(best_node.prompt_state)
    print(f"✓ 结果 → {output_file}")

    # 保存搜索树摘要
    summary = mcts.get_tree_summary()
    summary_file = f"cmcts_tree_summary_{args.target_label}_{ts}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ 树摘要 → {summary_file}")

    # 保存完整路径
    path_file = f"cmcts_path_{args.target_label}_{ts}.json"
    path_data = []
    for i, node in enumerate(best_path):
        path_data.append({
            "level": i,
            "node_id": node.id,
            "N": node.N,
            "Q": round(node.Q, 4),
            "depth": node.depth,
            "is_frozen": node.is_frozen,
            "action": node.action_taken,
            "prompt_preview": node.prompt_state[:200]
        })
    with open(path_file, 'w', encoding='utf-8') as f:
        json.dump(path_data, f, indent=2, ensure_ascii=False)
    print(f"✓ 路径 → {path_file}")

    # 写回配置
    print(f"\n📝 写回 label_prompts_config.py...")
    write_back_optimized_prompt(args.target_label, best_node.prompt_state)
    print(f"\n✅ C-MCTS 优化完成!")


if __name__ == "__main__":
    main()


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

conda activate q3_1 && cd /ai/ltx/zhongda/mcts_prompt_gen_v4

# 基本用法
python main_v2.py \
  --target_label "伤害他人" \
  --data_path "/ai/ltx/zhongda/datasets/与训练集同分布测试集.xlsx" \
  --model_path "/ai/ltx/zhongda/weight_file/merged_global_step_1900-8922" \
  --confusion_file "/ai/ltx/zhongda/datasets/label_confusion_analysis.xlsx" \
  --iterations 50 \
  --gpu_ids "0,1,2,3" \
  --api_url "http://localhost:8000"

# 自定义 C-MCTS 超参数
python main_v2.py \
  --target_label "交通事故" \
  --data_path "/ai/ltx/zhongda/datasets/与训练集同分布测试集.xlsx" \
  --model_path "/ai/ltx/zhongda/weight_file/merged_global_step_1900-8922" \
  --confusion_file "/ai/ltx/zhongda/datasets/label_confusion_analysis.xlsx" \
  --iterations 100 \
  --gpu_ids "0,1,2,3,4,5,6,7" \
  --c1 0.5 --c2 100.0 \
  --pw_alpha 0.5 --pw_k0 4 \
  --coverage_ratio 0.5 \
  --confusion_alpha 0.6 --confusion_temp 1.0 \
  --api_url "http://localhost:8000"

"""
