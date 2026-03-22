#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 01: 基准评估流程
1. 使用初始 prompt 对验证集进行预测，生成预测结果
2. 基于预测结果生成混淆矩阵分析
3. 为后续 MCTS 优化提供基准指标

使用方法:

conda activate q3_1 && cd /ai/ltx/zhongda/mcts_prompt_gen/ 


python step01_process.py \
  --model_path /ai/ltx/zhongda/weight_file/merged_global_step_1900-8922 \
  --model_type qwen3 \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --validation_file datasets/与训练集同分布测试集-正负样本-已质检-纠正-干净版本-0105.xlsx \
  --output_dir datasets

"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
import re
import tempfile
import multiprocessing as mp

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infer_14B_qwen3_solo_refactored import TextClassifier


def extract_label(text):
    """从预测结果中提取标签"""
    if pd.isna(text):
        return ""
    
    # 尝试提取 <answer>...</answer> 标签
    match = re.search(r'<answer>(.*?)</answer>', str(text))
    if match:
        return match.group(1).strip()
    
    # 如果没有标签，返回原文本（去除前后空格）
    return str(text).strip()


def process_data_chunk_for_baseline(gpu_id, model_path, model_type, data_chunk, start_idx, output_file, query_col):
    """
    在单个GPU上处理数据分片（用于基准评估）
    
    Args:
        gpu_id: GPU ID（单个整数）
        model_path: 模型路径
        model_type: 模型类型
        data_chunk: 数据分片
        start_idx: 起始索引
        output_file: 输出文件路径
        query_col: query列名
    """
    print(f"\n{'='*60}")
    print(f"进程启动 - GPU {gpu_id} | 处理样本 {start_idx} 到 {start_idx + len(data_chunk) - 1}")
    print(f"{'='*60}\n")
    
    classifier = None
    results = []
    
    try:
        # 初始化分类器（单GPU）
        classifier = TextClassifier(model_path, gpu_id, model_type=model_type)
        
        # 处理数据
        for idx, (original_idx, row) in enumerate(data_chunk.iterrows()):
            try:
                if (idx + 1) % 100 == 0:
                    print(f"GPU {gpu_id}: 已处理 {idx + 1}/{len(data_chunk)} 样本 (原始索引: {original_idx})")
                
                row_dict = row.to_dict()
                row_dict['_original_index'] = original_idx  # 保存原始索引
                
                # 只保留需要的字段
                trace = row_dict.get('trace', '')
                query_text = str(row_dict.get(query_col, ''))
                
                # 确保每一行都被处理并添加到结果中
                if query_text.strip():
                    try:
                        result = classifier.classify_text(query_text)
                        predicted_label = result['initial_classification']
                    except Exception as e:
                        print(f"GPU {gpu_id} 样本 {original_idx} 分类失败: {str(e)}")
                        predicted_label = f'<error>{str(e)}</error>'
                else:
                    predicted_label = '<empty_query>'
                
                # 构建输出格式：只保留5列
                result_row = {
                    '_original_index': original_idx,
                    'trace': trace,
                    '工单描述合并': query_text,
                    'extracted_prediction': extract_label(predicted_label)
                }
                results.append(result_row)
                
            except Exception as e:
                # 即使单个样本处理失败，也要保留原始数据
                print(f"GPU {gpu_id} 样本 {original_idx} 处理异常: {str(e)}")
                try:
                    row_dict = row.to_dict()
                    result_row = {
                        '_original_index': original_idx,
                        'trace': row_dict.get('trace', ''),
                        '工单描述合并': str(row_dict.get(query_col, '')),
                        'extracted_prediction': ''
                    }
                    results.append(result_row)
                except:
                    print(f"GPU {gpu_id} 样本 {original_idx} 无法保存异常记录")
        
        # 验证处理数量
        expected_count = len(data_chunk)
        actual_count = len(results)
        print(f"\nGPU {gpu_id}: 预期处理 {expected_count} 个样本, 实际处理 {actual_count} 个样本")
        
        if actual_count != expected_count:
            print(f"⚠️ GPU {gpu_id} 警告: 处理数量不匹配! 缺失 {expected_count - actual_count} 个样本")
        
        # 保存结果到临时文件
        df_result = pd.DataFrame(results)
        df_result.to_pickle(output_file)
        print(f"GPU {gpu_id}: 结果已保存到 {output_file}")
        
        return len(df_result)  # 返回处理的样本数
        
    except Exception as e:
        print(f"❌ GPU {gpu_id} 进程级错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 即使出错也尝试保存已处理的数据
        if results:
            try:
                df_result = pd.DataFrame(results)
                df_result.to_pickle(output_file)
                print(f"GPU {gpu_id}: 已保存 {len(results)} 个部分结果")
                return len(results)
            except Exception as save_error:
                print(f"❌ GPU {gpu_id} 保存失败: {str(save_error)}")
        
        return 0
        
    finally:
        # 清理资源
        if classifier is not None:
            del classifier
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


def run_prediction(
    model_path: str,
    validation_file: str,
    output_file: str,
    model_type: str = "qwen3",
    gpu_ids: str = "0,1,2,3,4,5",
    query_col: str = "工单描述合并",
    label_col: str = "最终标签"
):
    """
    使用多GPU并行对验证集进行预测
    
    Args:
        model_path: 模型权重路径
        validation_file: 验证集文件路径
        output_file: 输出预测结果文件路径
        model_type: 模型类型
        gpu_ids: GPU ID列表（逗号分隔，如 "0,1,2,3"）
        query_col: query 列名
        label_col: 真实标签列名
    """
    
    print(f"\n{'='*60}")
    print(f"Step 1: 运行基准预测（多GPU并行）")
    print(f"{'='*60}")
    print(f"模型路径: {model_path}")
    print(f"模型类型: {model_type}")
    print(f"验证集: {validation_file}")
    
    # 解析GPU ID列表
    gpu_id_list = [int(x.strip()) for x in gpu_ids.split(',')]
    num_gpus = len(gpu_id_list)
    print(f"使用 {num_gpus} 个GPU: {gpu_id_list}")
    print(f"{'='*60}\n")
    
    # 1. 加载验证集
    print(f"正在加载验证集...")
    df = pd.read_excel(validation_file)
    df = df.reset_index(drop=True)
    total_samples = len(df)
    print(f"✓ 加载完成: {total_samples} 条数据")
    
    # 检查必要的列
    if query_col not in df.columns:
        raise ValueError(f"验证集缺少 '{query_col}' 列")
    if label_col not in df.columns:
        raise ValueError(f"验证集缺少 '{label_col}' 列")
    
    print(f"列名: {list(df.columns)}\n")
    
    # 2. 数据分片
    chunk_size = (total_samples + num_gpus - 1) // num_gpus
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='baseline_inference_')
    print(f"临时目录: {temp_dir}\n")
    
    # 准备多进程任务
    processes = []
    temp_files = []
    
    for i, gpu_id in enumerate(gpu_id_list):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        
        if start_idx >= total_samples:
            break
        
        data_chunk = df.iloc[start_idx:end_idx].copy()
        temp_file = os.path.join(temp_dir, f'result_gpu_{gpu_id}.pkl')
        temp_files.append(temp_file)
        
        print(f"GPU {gpu_id}: 分配样本 {start_idx} 到 {end_idx-1} (共 {len(data_chunk)} 个)")
        
        # 创建进程
        p = mp.Process(
            target=process_data_chunk_for_baseline,
            args=(gpu_id, model_path, model_type, data_chunk, start_idx, temp_file, query_col)
        )
        processes.append(p)
    
    # 3. 启动所有进程
    print(f"\n启动 {len(processes)} 个并行进程...\n")
    for p in processes:
        p.start()
    
    # 4. 等待所有进程完成
    failed_processes = []
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            print(f"⚠️  进程 {i+1}/{len(processes)} 异常退出 (退出码: {p.exitcode})")
            failed_processes.append(i)
        else:
            print(f"✓ 进程 {i+1}/{len(processes)} 正常完成")
    
    if failed_processes:
        print(f"\n警告: {len(failed_processes)} 个进程异常退出: {failed_processes}")
    
    print("\n所有GPU进程已完成,开始合并结果...\n")
    
    # 5. 合并结果并验证
    all_results = []
    processed_samples = 0
    missing_files = []
    chunk_stats = {}
    
    for i, temp_file in enumerate(temp_files):
        gpu_id = gpu_id_list[i]
        if os.path.exists(temp_file):
            try:
                df_chunk = pd.read_pickle(temp_file)
                samples_in_chunk = len(df_chunk)
                processed_samples += samples_in_chunk
                
                # 计算该GPU应该处理的样本数
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_samples)
                expected_samples = end_idx - start_idx
                
                chunk_stats[gpu_id] = {
                    'expected': expected_samples,
                    'actual': samples_in_chunk,
                    'missing': expected_samples - samples_in_chunk
                }
                
                status = "✓" if samples_in_chunk == expected_samples else "⚠️"
                print(f"{status} GPU {gpu_id}: 读取 {samples_in_chunk}/{expected_samples} 个样本")
                
                all_results.append(df_chunk)
                os.remove(temp_file)  # 删除临时文件
            except Exception as e:
                print(f"❌ 读取 {temp_file} 失败: {str(e)}")
                import traceback
                traceback.print_exc()
                missing_files.append(i)
        else:
            print(f"❌ GPU {gpu_id} 的临时文件不存在: {temp_file}")
            
            # 计算该GPU应该处理的样本数
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            expected_samples = end_idx - start_idx
            
            chunk_stats[gpu_id] = {
                'expected': expected_samples,
                'actual': 0,
                'missing': expected_samples
            }
            missing_files.append(i)
    
    # 删除临时目录
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    # 6. 验证数据完整性
    print(f"\n{'='*60}")
    print(f"数据完整性验证:")
    print(f"{'='*60}")
    print(f"原始样本数: {total_samples}")
    print(f"已处理样本数: {processed_samples}")
    print(f"丢失样本数: {total_samples - processed_samples}")
    
    # 显示每个GPU的处理情况
    print(f"\n每个GPU的处理统计:")
    for gpu_id, stats in chunk_stats.items():
        status = "✓" if stats['missing'] == 0 else "⚠️"
        print(f"  {status} GPU {gpu_id}: 预期={stats['expected']}, 实际={stats['actual']}, 缺失={stats['missing']}")
    
    if missing_files:
        print(f"\n⚠️  缺失GPU结果: {[gpu_id_list[i] for i in missing_files]}")
    
    if processed_samples < total_samples:
        print(f"\n❌ 警告: 有 {total_samples - processed_samples} 个样本未被处理!")
        print(f"   数据完整性: {processed_samples/total_samples*100:.2f}%")
    else:
        print(f"\n✓ 所有样本都已成功处理!")
    
    print(f"{'='*60}\n")
    
    # 7. 合并所有结果
    if all_results:
        # 按原始索引排序合并,保持数据顺序
        df_result = pd.concat(all_results, ignore_index=False)
        if '_original_index' in df_result.columns:
            df_result = df_result.sort_values('_original_index').reset_index(drop=True)
            
            # 检查是否有重复或缺失的索引
            processed_indices = set(df_result['_original_index'])
            original_indices = set(df.index)
            missing_indices = original_indices - processed_indices
            
            if missing_indices:
                print(f"⚠️ 缺失的样本索引: {sorted(list(missing_indices))[:10]}... (共 {len(missing_indices)} 个)")
            else:
                print(f"✓ 所有索引都已处理")
            
            # 删除临时的原始索引列
            df_result = df_result.drop(columns=['_original_index'])
        else:
            df_result = df_result.reset_index(drop=True)
        
        # 添加最终标签和is_correct列
        # 从原始验证集中获取真实标签
        df_result['最终标签'] = df[label_col].values[:len(df_result)]
        
        # 计算is_correct：预测是否正确
        df_result['is_correct'] = (df_result['extracted_prediction'] == df_result['最终标签'])
        
        # 确保列顺序与目标格式一致：trace, 工单描述合并, extracted_prediction, 最终标签, is_correct
        df_result = df_result[['trace', '工单描述合并', 'extracted_prediction', '最终标签', 'is_correct']]
        
        # 8. 保存结果
        print(f"\n保存预测结果到: {output_file}")
        df_result.to_excel(output_file, index=False)
        print(f"✓ 预测结果已保存 ({len(df_result)} 条)")
        
        return output_file
    else:
        raise RuntimeError("所有GPU进程都失败，没有生成任何结果")





def main():
    parser = argparse.ArgumentParser(description="Step 01: 基准评估流程")
    
    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型权重路径")
    
    # 可选参数
    parser.add_argument("--model_type", type=str, default="qwen3",
                        help="模型类型 (默认: qwen3)")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7",
                        help="使用的 GPU ID 列表，逗号分隔 (默认: 0,1,2,3,4,5,6,7)")
    parser.add_argument("--validation_file", type=str,
                        default="datasets/与训练集同分布测试集-正负样本-已质检-纠正-干净版本-0105.xlsx",
                        help="验证集文件路径")
    parser.add_argument("--output_dir", type=str, default="datasets",
                        help="输出目录 (默认: datasets)")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_file = os.path.join(args.output_dir, f"预测校正-{timestamp}.xlsx")
    
    print(f"\n{'#'*60}")
    print(f"#  MCTS 提示词优化 - Step 01: 基准评估")
    print(f"{'#'*60}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 运行预测（多GPU并行）
        pred_file = run_prediction(
            model_path=args.model_path,
            validation_file=args.validation_file,
            output_file=prediction_file,
            model_type=args.model_type,
            gpu_ids=args.gpu_ids
        )
        
        print(f"\n{'#'*60}")
        print(f"#  预测完成！")
        print(f"{'#'*60}")
        print(f"\n输出文件: {pred_file}\n")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


