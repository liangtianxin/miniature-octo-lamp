#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算SFT模型预测准确率：去除非中文字符后与最终标签比较
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime

# 问题文本数组
PROBLEM_TEXTS = ["拉到外地", "不给我下车", "车门锁死", "车门锁住", "不让我下车", "下车的时候把车门给我们锁上", "拽住乘客不让走", "他也不让我下车", "不给费用不给走", "不给走", "被关在那了", "不让我下下车"]

def clean_chinese_text(text):
    """
    去除非中文字符，只保留中文字符、数字和常用标点符号
    """
    if pd.isna(text):
        return ""
    
    # 保留中文字符、数字、常用标点符号
    # \u4e00-\u9fff: 中文字符范围
    # 0-9: 数字
    # 常用标点：、。，；：！？（）【】「」
    pattern = r'[^\u4e00-\u9fff0-9、。，；：！？（）【】「」/]'
    cleaned = re.sub(pattern, '', str(text))
    return cleaned.strip()

def extract_answer_tag(text):
    """
    从<answer>标签中提取内容
    """
    if pd.isna(text):
        return ""
    
    match = re.search(r'<answer>(.*?)</answer>', str(text))
    if match:
        return match.group(1).strip()
    else:
        return str(text).strip()

def normalize_label(label):
    """
    标准化标签：去除前后空格，统一格式
    """
    if pd.isna(label):
        return ""
    return str(label).strip()

def calculate_accuracy(filename=None, prediction_col='qwen3-14b预测', human_label_col='最终标签'):
    """
    计算准确率
    
    参数:
        filename: Excel文件路径，如果为None则使用默认路径
        prediction_col: 预测结果列名
        human_label_col: 人工标注列名
    
    返回:
        accuracy: 准确率
        output_filename: 输出文件路径
    """
    # 默认文件名，可通过参数覆盖
    if filename is None:
        filename = '/mnt/workspace/ltx/zhongda/output_01/sft-merged_global_step_1900-8922-qwen3-14b预测-2025-10-10 12-14-34-与训练集同分布测试集-正负样本-已质检-纠正-1010.xlsx'

    print("正在读取Excel文件...")
    df = pd.read_excel(filename, sheet_name=0)
    
    print(f"总数据量: {len(df)}")
    
    # 1. 提取预测结果中的answer标签内容
    print("正在提取预测结果...")
    df['extracted_prediction'] = df[prediction_col].apply(extract_answer_tag)
    
    # 2. 去除预测结果中的非中文字符
    print("正在清理预测结果中的非中文字符...")
    df['cleaned_prediction'] = df['extracted_prediction'].apply(clean_chinese_text)
    
    # 3. 标准化最终标签
    print("正在标准化最终标签...")
    df['cleaned_human_label'] = df[human_label_col].apply(normalize_label).apply(clean_chinese_text)
    
    # 4. 计算准确率
    print("正在计算准确率...")
    df['is_correct'] = df['cleaned_prediction'] == df['cleaned_human_label']
    
    total_count = len(df)
    correct_count = df['is_correct'].sum()
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\n=== 准确率计算结果 ===")
    print(f"总样本数: {total_count}")
    print(f"预测正确数: {correct_count}")
    print(f"预测错误数: {total_count - correct_count}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 5. 分析错误情况
    print(f"\n=== 错误分析 ===")
    incorrect_df = df[~df['is_correct']].copy()
    
    if len(incorrect_df) > 0:
        print(f"错误样本数: {len(incorrect_df)}")
        
        # 按最终标签分组分析错误
        error_by_label = incorrect_df['cleaned_human_label'].value_counts()
        print(f"\n按最终标签分组的错误统计:")
        for label, count in error_by_label.head(10).items():
            total_for_label = (df['cleaned_human_label'] == label).sum()
            error_rate = count / total_for_label if total_for_label > 0 else 0
            print(f"  '{label}': {count}/{total_for_label} ({error_rate:.2%})")
        
        # 显示一些错误样本
        print(f"\n错误样本示例 (前10个):")
        for idx, row in incorrect_df.head(10).iterrows():
            print(f"  样本{idx}:")
            print(f"    原始预测: {row[prediction_col]}")
            print(f"    清理后预测: '{row['cleaned_prediction']}'")
            print(f"    人工标注: '{row['cleaned_human_label']}'")
            print(f"    用户问题: {row['query'][:100]}...")
            print()
    
    # 6. 按标签类别分析准确率
    print(f"\n=== 按标签类别分析准确率 ===")
    label_accuracy = df.groupby('cleaned_human_label').agg({
        'is_correct': ['count', 'sum']
    }).round(4)
    label_accuracy.columns = ['总数', '正确数']
    label_accuracy['准确率'] = (label_accuracy['正确数'] / label_accuracy['总数']).round(4)
    label_accuracy['准确率%'] = (label_accuracy['准确率'] * 100).round(2)
    
    # 按总数排序
    label_accuracy = label_accuracy.sort_values('总数', ascending=False)
    print(label_accuracy)
    
    # 7. 保存详细结果
    output_filename = f"accuracy_analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # 准备输出数据
    
    output_df = df[[
        'trace', 'query', prediction_col, 'extracted_prediction', 
        'cleaned_prediction', human_label_col, 'cleaned_human_label', 'is_correct'
    ]].copy()
    

    
    # 创建汇总sheet
    summary_data = {
        '指标': ['总样本数', '预测正确数', '预测错误数', '整体准确率'],
        '值': [total_count, correct_count, total_count - correct_count, f"{accuracy:.4f} ({accuracy*100:.2f}%)"]
    }
    summary_df = pd.DataFrame(summary_data)
    
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # 写入汇总数据
        summary_df.to_excel(writer, sheet_name='准确率汇总', index=False)
        
        # 写入按标签分类的准确率
        label_accuracy.to_excel(writer, sheet_name='按标签准确率')
        
        # 写入详细数据
        output_df.to_excel(writer, sheet_name='详细数据', index=False)
        
        # 写入错误样本
        if len(incorrect_df) > 0:
            incorrect_output = incorrect_df[[
                'trace', 'query', prediction_col, 'extracted_prediction',
                'cleaned_prediction', human_label_col, 'cleaned_human_label'
            ]].copy()
            incorrect_output.to_excel(writer, sheet_name='错误样本', index=False)
    
    print(f"\n详细结果已保存到: {output_filename}")
    
    return accuracy, output_filename


def calculate_accuracy_with_file(filename, prediction_col='qwen3-14b预测', human_label_col='最终标签'):
    """
    供外部调用的函数，计算指定文件的准确率
    
    参数:
        filename: Excel文件路径
        prediction_col: 预测结果列名
        human_label_col: 人工标注列名
    
    返回:
        accuracy: 准确率
        output_filename: 输出文件路径
    """
    return calculate_accuracy(filename, prediction_col, human_label_col)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算 SFT 模型预测准确率")
    parser.add_argument('file', nargs='?', help='输入的 Excel 文件路径（可选）')
    parser.add_argument('--prediction-col', default='qwen3-14b预测', help='预测结果列名')
    parser.add_argument('--human-col', default='最终标签', help='人工标注列名')
    args = parser.parse_args()

    # 调用计算函数
    if args.file:
        accuracy, output_file = calculate_accuracy(args.file, args.prediction_col, args.human_col)
    else:
        accuracy, output_file = calculate_accuracy()

# conda activate qwen3 && python /mnt/workspace/ltx/zhongda/tool/calculate_sft_accuracy.py
# python /mnt/workspace/ltx/zhongda/tool/calculate_sft_accuracy.py
