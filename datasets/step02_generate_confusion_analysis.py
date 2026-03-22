#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成标签混淆矩阵分析文件
分析预测结果中的标签混淆情况，生成与 label_confusion_analysis.xlsx 相同格式的文件
"""

import pandas as pd
import sys
from collections import defaultdict

def generate_confusion_analysis(input_file: str, output_file: str = "label_confusion_analysis.xlsx"):
    """
    分析预测结果，生成混淆对分析表
    
    Args:
        input_file: 输入的预测结果文件（需包含 '最终标签完成版本' 和 'extracted_prediction' 列）
        output_file: 输出文件路径，默认为 label_confusion_analysis.xlsx
    """
    
    print(f"正在读取文件: {input_file}")
    df = pd.read_excel(input_file)
    
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 检查必要的列
    if '最终标签完成版本' not in df.columns:
        print("错误: 文件中缺少 '最终标签完成版本' 列")
        return
    
    # 确定预测列名
    pred_col = None
    for col in ['extracted_prediction', 'qwen3-14b预测', '预测标签', 'prediction']:
        if col in df.columns:
            pred_col = col
            break
    
    if pred_col is None:
        print("错误: 找不到预测列，请确保文件中包含预测结果列")
        print(f"可用列: {list(df.columns)}")
        return
    
    print(f"使用预测列: {pred_col}")
    
    # 数据清洗：提取标签内容
    def clean_label(text):
        """清洗标签，去除空格等"""
        if pd.isna(text):
            return ""
        return str(text).strip()
    
    df['真实标签'] = df['最终标签完成版本'].apply(clean_label)
    df['预测标签'] = df[pred_col].apply(clean_label)
    
    # 过滤掉空标签
    df = df[(df['真实标签'] != "") & (df['预测标签'] != "")]
    
    print(f"\n有效数据: {len(df)} 条")
    
    # 统计混淆对
    # 只统计预测错误的情况（真实标签 != 预测标签）
    incorrect_df = df[df['真实标签'] != df['预测标签']]
    
    print(f"预测错误的数据: {len(incorrect_df)} 条")
    print(f"整体准确率: {(len(df) - len(incorrect_df)) / len(df) * 100:.2f}%")
    
    # 统计每个混淆对的出现次数
    confusion_pairs = defaultdict(int)
    
    for _, row in incorrect_df.iterrows():
        true_label = row['真实标签']
        pred_label = row['预测标签']
        confusion_pairs[(true_label, pred_label)] += 1
    
    # 构建混淆对分析表
    confusion_data = []
    for (true_label, pred_label), count in confusion_pairs.items():
        # 计算该真实标签的总数
        total_true = (df['真实标签'] == true_label).sum()
        # 计算混淆率
        confusion_rate = count / total_true if total_true > 0 else 0
        
        confusion_data.append({
            '真实标签': true_label,
            '错误预测为': pred_label,
            '错误次数': count,
            '该标签总数': total_true,
            '混淆率': f"{confusion_rate:.2%}"
        })
    
    # 转换为 DataFrame 并排序
    confusion_df = pd.DataFrame(confusion_data)
    
    if len(confusion_df) == 0:
        print("警告: 没有发现混淆对（所有预测都正确）")
        # 创建一个空的混淆对分析表
        confusion_df = pd.DataFrame(columns=['真实标签', '错误预测为', '错误次数', '该标签总数', '混淆率'])
    else:
        # 按错误次数降序排序
        confusion_df = confusion_df.sort_values('错误次数', ascending=False)
    
    # 生成统计摘要
    all_labels = sorted(set(df['真实标签'].unique()) | set(df['预测标签'].unique()))
    
    summary_data = []
    for label in all_labels:
        total = (df['真实标签'] == label).sum()
        correct = ((df['真实标签'] == label) & (df['预测标签'] == label)).sum()
        accuracy = correct / total if total > 0 else 0
        
        # 统计被混淆成其他标签的次数
        confused_to_others = ((df['真实标签'] == label) & (df['预测标签'] != label)).sum()
        # 统计其他标签被误判为该标签的次数
        others_confused_to_this = ((df['真实标签'] != label) & (df['预测标签'] == label)).sum()
        
        summary_data.append({
            '标签': label,
            '样本总数': total,
            '预测正确数': correct,
            '准确率': f"{accuracy:.2%}",
            '被误判为其他标签次数': confused_to_others,
            '其他标签误判为该标签次数': others_confused_to_this
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('样本总数', ascending=False)
    
    # 保存到 Excel，包含多个 sheet
    print(f"\n正在保存到: {output_file}")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: 混淆对分析（主要的，confusion_analyzer.py 会读取这个）
        confusion_df.to_excel(writer, sheet_name='混淆对分析', index=False)
        
        # Sheet 2: 按标签统计
        summary_df.to_excel(writer, sheet_name='标签统计', index=False)
        
        # Sheet 3: 详细错误案例（可选，显示前500个错误样本）
        if len(incorrect_df) > 0:
            error_samples = incorrect_df[['临时ID', '工单描述合并', '真实标签', '预测标签']].head(500).copy()
            error_samples.to_excel(writer, sheet_name='错误样本', index=False)
    
    print(f"✅ 混淆矩阵分析已保存!")
    print(f"\n📊 生成的 Excel 包含以下 sheet:")
    print(f"  1. 混淆对分析 - {len(confusion_df)} 个混淆对")
    print(f"  2. 标签统计 - {len(summary_df)} 个标签的详细统计")
    if len(incorrect_df) > 0:
        print(f"  3. 错误样本 - 前 {min(500, len(incorrect_df))} 个错误案例")
    
    # 显示最常见的混淆对
    print(f"\n🔍 最常见的混淆对 (Top 10):")
    print("=" * 80)
    for idx, row in confusion_df.head(10).iterrows():
        print(f"{row['真实标签']:12s} -> {row['错误预测为']:12s}  "
              f"错误{row['错误次数']:3d}次 / 总共{row['该标签总数']:4d}个 = {row['混淆率']}")
    
    # 显示准确率最低的标签
    print(f"\n⚠️  准确率最低的标签 (Top 10):")
    print("=" * 80)
    summary_df['准确率_数值'] = summary_df['准确率'].str.rstrip('%').astype(float) / 100
    worst_labels = summary_df.nsmallest(10, '准确率_数值')
    for idx, row in worst_labels.iterrows():
        print(f"{row['标签']:12s}: {row['准确率']:>7s}  "
              f"(正确{row['预测正确数']:3d}/{row['样本总数']:4d})")
    
    return output_file


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='生成标签混淆矩阵分析文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python generate_confusion_analysis.py                    # 使用默认文件 预测校正-1016a.xlsx
  python generate_confusion_analysis.py input.xlsx         # 指定输入文件
  python generate_confusion_analysis.py input.xlsx -o output.xlsx  # 指定输入和输出
        """
    )
    
    parser.add_argument('input_file', 
                       nargs='?',
                       default='/ai/hlf/zhongda/infer_qwen3/script2603/grpo-qwen3-14b预测-2026-03-04 10-50-13.xlsx',
                       help='输入的预测结果Excel文件 (默认: 预测校正-1016a.xlsx)')
    parser.add_argument('-o', '--output', 
                       default='label_confusion_analysis.xlsx',
                       help='输出文件名 (默认: label_confusion_analysis.xlsx)')
    
    args = parser.parse_args()
    
    try:
        generate_confusion_analysis(args.input_file, args.output)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# ltx/zhongda/mcts_prompt_gen/datasets/预测校正-20260105_205343.xlsx
