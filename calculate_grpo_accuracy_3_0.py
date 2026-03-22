#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算SFT模型预测准确率：支持整体及各标签的准确率、精确率、召回率、F1值
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime


def clean_chinese_text(text):
    """标准化标签文本：strip + 全角转半角数字，保留所有字符，不做删除"""
    if pd.isna(text):
        return ""
    s = str(text).strip()
    # 全角数字/字母转半角，统一比较基准
    result = []
    for ch in s:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:          # 全角可见字符 → 半角
            result.append(chr(cp - 0xFEE0))
        elif cp == 0x3000:                   # 全角空格 → 半角空格
            result.append(' ')
        else:
            result.append(ch)
    return ''.join(result).strip()


def extract_answer_tag(text):
    """从<answer>和第一个</answer>之间提取标签内容，兼容</answer></answer>重复闭合"""
    if pd.isna(text):
        return ""
    s = str(text)
    # 非贪婪匹配：取 <answer> 到第一个 </answer> 之间的内容
    match = re.search(r'<answer>(.*?)</answer>', s)
    if match:
        return match.group(1).strip()
    return s.strip()


def normalize_label(label):
    """标准化标签：去除前后空格"""
    if pd.isna(label):
        return ""
    return str(label).strip()


def compute_per_label_metrics(df, true_col, pred_col):
    """
    计算每个标签的 TP/FP/FN，以及精确率、召回率、F1值。
    返回 DataFrame，每行为一个标签。
    """
    all_labels = sorted(set(df[true_col].unique()) | set(df[pred_col].unique()))
    rows = []
    for label in all_labels:
        tp = int(((df[true_col] == label) & (df[pred_col] == label)).sum())
        fp = int(((df[true_col] != label) & (df[pred_col] == label)).sum())
        fn = int(((df[true_col] == label) & (df[pred_col] != label)).sum())
        support = int((df[true_col] == label).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            '标签':     label,
            '样本数(support)': support,
            'TP':       tp,
            'FP':       fp,
            'FN':       fn,
            '精确率(Precision)': round(precision, 4),
            '召回率(Recall)':    round(recall, 4),
            'F1值':              round(f1, 4),
            '精确率%': round(precision * 100, 2),
            '召回率%': round(recall * 100, 2),
            'F1%':     round(f1 * 100, 2),
        })
    result = pd.DataFrame(rows).set_index('标签')
    result = result.sort_values('样本数(support)', ascending=False)
    return result


def compute_overall_metrics(df, true_col, pred_col):
    """计算宏平均（macro）和加权平均（weighted）的精确率、召回率、F1"""
    per_label = compute_per_label_metrics(df, true_col, pred_col)
    # 只对有真实样本的标签计算平均
    valid = per_label[per_label['样本数(support)'] > 0]
    support = valid['样本数(support)']
    total = support.sum()

    macro_p  = valid['精确率(Precision)'].mean()
    macro_r  = valid['召回率(Recall)'].mean()
    macro_f1 = valid['F1值'].mean()

    weighted_p  = (valid['精确率(Precision)'] * support).sum() / total if total > 0 else 0
    weighted_r  = (valid['召回率(Recall)']    * support).sum() / total if total > 0 else 0
    weighted_f1 = (valid['F1值']              * support).sum() / total if total > 0 else 0

    total_count   = len(df)
    correct_count = int((df[true_col] == df[pred_col]).sum())
    overall_acc   = correct_count / total_count if total_count > 0 else 0

    return {
        'total_count':   total_count,
        'correct_count': correct_count,
        'overall_acc':   overall_acc,
        'macro_p':       round(macro_p, 4),
        'macro_r':       round(macro_r, 4),
        'macro_f1':      round(macro_f1, 4),
        'weighted_p':    round(weighted_p, 4),
        'weighted_r':    round(weighted_r, 4),
        'weighted_f1':   round(weighted_f1, 4),
    }


def calculate_accuracy(filename=None, prediction_col='qwen3-14b预测', human_label_col='最终标签'):
    """
    计算准确率、精确率、召回率、F1值，并保存到Excel。

    参数:
        filename:        Excel文件路径，None时使用默认路径
        prediction_col:  预测结果列名
        human_label_col: 人工标注列名

    返回:
        overall_acc:     整体准确率
        output_filename: 输出文件路径
    """
    if filename is None:
        filename = '/ai/hlf/zhongda/mcts_prompt_gen_v4_b/datasets/grpo-qwen3-14b预测-2026-02-27 15-43-24.xlsx'

    print("正在读取Excel文件...")
    df = pd.read_excel(filename, sheet_name=0)
    print(f"列名: {list(df.columns)}")
    print(f"总数据量: {len(df)}")

    # 1. 提取 <answer>…</answer> 内容
    print("正在提取预测结果...")
    df['extracted_prediction'] = df[prediction_col].apply(extract_answer_tag)

    # 2. 标准化（全半角统一，strip），不删除任何字符
    print("正在标准化标签文本...")
    df['cleaned_prediction']  = df['extracted_prediction'].apply(clean_chinese_text)
    df['cleaned_human_label'] = df[human_label_col].apply(normalize_label).apply(clean_chinese_text)

    # 3. 判断是否正确
    df['is_correct'] = df['cleaned_prediction'] == df['cleaned_human_label']

    # 4. 整体指标
    metrics = compute_overall_metrics(df, 'cleaned_human_label', 'cleaned_prediction')
    total_count   = metrics['total_count']
    correct_count = metrics['correct_count']
    overall_acc   = metrics['overall_acc']

    print(f"\n=== 整体指标 ===")
    print(f"总样本数:        {total_count}")
    print(f"预测正确数:      {correct_count}")
    print(f"预测错误数:      {total_count - correct_count}")
    print(f"整体准确率:      {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
    print(f"宏平均精确率:    {metrics['macro_p']:.4f}  ({metrics['macro_p']*100:.2f}%)")
    print(f"宏平均召回率:    {metrics['macro_r']:.4f}  ({metrics['macro_r']*100:.2f}%)")
    print(f"宏平均F1:        {metrics['macro_f1']:.4f}  ({metrics['macro_f1']*100:.2f}%)")
    print(f"加权平均精确率:  {metrics['weighted_p']:.4f}  ({metrics['weighted_p']*100:.2f}%)")
    print(f"加权平均召回率:  {metrics['weighted_r']:.4f}  ({metrics['weighted_r']*100:.2f}%)")
    print(f"加权平均F1:      {metrics['weighted_f1']:.4f}  ({metrics['weighted_f1']*100:.2f}%)")

    # 5. 各标签指标
    per_label_df = compute_per_label_metrics(df, 'cleaned_human_label', 'cleaned_prediction')
    print(f"\n=== 各标签指标 ===")
    print(per_label_df[['样本数(support)', '精确率%', '召回率%', 'F1%']].to_string())

    # 6. 错误样本
    incorrect_df = df[~df['is_correct']].copy()
    if len(incorrect_df) > 0:
        print(f"\n=== 错误分析 (共 {len(incorrect_df)} 条) ===")
        error_by_label = incorrect_df['cleaned_human_label'].value_counts()
        for label, count in error_by_label.head(15).items():
            total_for_label = (df['cleaned_human_label'] == label).sum()
            print(f"  '{label}': 错误 {count}/{total_for_label} ({count/total_for_label:.2%})")

    # 7. 构建汇总 sheet
    summary_data = {
        '指标': [
            '总样本数', '预测正确数', '预测错误数', '整体准确率',
            '宏平均精确率', '宏平均召回率', '宏平均F1',
            '加权平均精确率', '加权平均召回率', '加权平均F1',
        ],
        '值': [
            total_count, correct_count, total_count - correct_count,
            f"{overall_acc:.4f} ({overall_acc*100:.2f}%)",
            f"{metrics['macro_p']:.4f} ({metrics['macro_p']*100:.2f}%)",
            f"{metrics['macro_r']:.4f} ({metrics['macro_r']*100:.2f}%)",
            f"{metrics['macro_f1']:.4f} ({metrics['macro_f1']*100:.2f}%)",
            f"{metrics['weighted_p']:.4f} ({metrics['weighted_p']*100:.2f}%)",
            f"{metrics['weighted_r']:.4f} ({metrics['weighted_r']*100:.2f}%)",
            f"{metrics['weighted_f1']:.4f} ({metrics['weighted_f1']*100:.2f}%)",
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    # 8. 准备详细数据（动态列，不依赖 trace/query 是否存在）
    desired_detail = [prediction_col, 'extracted_prediction', 'cleaned_prediction',
                      human_label_col, 'cleaned_human_label', 'is_correct']
    available_detail = [c for c in desired_detail if c in df.columns]
    # 将原始列（非计算列）也带上
    original_cols = [c for c in df.columns if c not in available_detail]
    output_df = df[original_cols + available_detail].copy()

    desired_inc = [prediction_col, 'extracted_prediction', 'cleaned_prediction',
                   human_label_col, 'cleaned_human_label']
    available_inc = [c for c in desired_inc if c in incorrect_df.columns]
    inc_original  = [c for c in incorrect_df.columns if c not in available_inc]
    incorrect_output = incorrect_df[inc_original + available_inc].copy()

    # 9. 写入 Excel
    output_filename = f"accuracy_analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='整体指标汇总', index=False)
        per_label_df.to_excel(writer, sheet_name='各标签指标')
        output_df.to_excel(writer, sheet_name='详细数据', index=False)
        if len(incorrect_df) > 0:
            incorrect_output.to_excel(writer, sheet_name='错误样本', index=False)

    print(f"\n详细结果已保存到: {output_filename}")
    return overall_acc, output_filename


def calculate_accuracy_with_file(filename, prediction_col='qwen3-14b预测', human_label_col='最终标签'):
    """供外部调用的函数"""
    return calculate_accuracy(filename, prediction_col, human_label_col)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算 SFT 模型预测准确率、精确率、召回率、F1")
    parser.add_argument('file', nargs='?', help='输入的 Excel 文件路径（可选）')
    parser.add_argument('--prediction-col', default='qwen3-14b预测', help='预测结果列名')
    parser.add_argument('--human-col', default='最终标签', help='人工标注列名')
    args = parser.parse_args()

    calculate_accuracy(args.file, args.prediction_col, args.human_col)

# 运行示例:
# python calculate_sft_accuracy.py /ai/hlf/zhongda/mcts_prompt_gen_v4_b/datasets/grpo-qwen3-14b预测-2026-02-27\ 15-43-24.xlsx
