#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri Feb 7 15:18:38 2025

@author: liangtianxin
"""


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# 不再硬编码GPU设置，由外部控制
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'



import re  
import csv  
import json
import copy
import torch
import pandas as pd
from swift import Swift
from datetime import datetime  
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
from swift.plugin import InferStats
from swift.llm import VllmEngine
from swift.llm.model import LLMModelType

import subprocess
import multiprocessing as mp
import tempfile
import time

# GPU检测和选择函数
def get_available_gpus(min_memory_gb=15):
    """
    检测可用的GPU并返回显存足够的GPU列表
    
    Args:
        min_memory_gb: 最小显存要求(GB)
        
    Returns:
        可用GPU ID列表
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        available_gpus = []
        for line in result.stdout.strip().split('\n'):
            gpu_id, free_memory = line.split(', ')
            free_memory_gb = int(free_memory) / 1024
            
            print(f"GPU {gpu_id}: 可用显存 {free_memory_gb:.2f} GB")
            
            if free_memory_gb >= min_memory_gb:
                available_gpus.append(gpu_id)
                print(f"  ✓ GPU {gpu_id} 显存充足,可以使用")
            else:
                print(f"  ✗ GPU {gpu_id} 显存不足(需要至少 {min_memory_gb} GB)")
        
        return available_gpus
    except Exception as e:
        print(f"GPU检测失败: {str(e)}")
        return []

def select_gpus_for_inference(num_gpus=None, min_memory_gb=15):
    """
    自动选择可用的GPU
    
    Args:
        num_gpus: 需要的GPU数量,如果为None则使用所有可用GPU
        min_memory_gb: 最小显存要求(GB)
        
    Returns:
        选中的GPU ID列表
    """
    available_gpus = get_available_gpus(min_memory_gb)
    
    if not available_gpus:
        raise RuntimeError("没有找到符合要求的可用GPU")
    
    if num_gpus is None:
        selected_gpus = available_gpus
    else:
        selected_gpus = available_gpus[:num_gpus]
    
    print(f"\n选择使用的GPU: {','.join(selected_gpus)}")
    return selected_gpus

# 导入准确率计算模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from calculate_sft_accuracy import calculate_accuracy_with_file

# 导入共享的标签提示词配置
from label_prompts_config import LABEL_PROMPTS

# 导入混淆分析器
try:
    from confusion_analyzer import ConfusionAnalyzer
    CONFUSION_ANALYZER_AVAILABLE = True
except ImportError:
    CONFUSION_ANALYZER_AVAILABLE = False
    print("警告: 混淆分析器不可用，将使用传统模式")

def keep_chinese_label(text):
    if '\n' in text:
        text01 = text.split('\n')[0]
    elif ' ' in text:
        text01 = text.split(' ')[0]
    else:
        text01 = text
    return re.sub(r'[^\u4e00-\u9fff]+', '', str(text01))

def remove_non_chinese_characters(text):  
    # 使用正则表达式匹配非中文字符，并将它们替换为空  
    result = re.sub(r'[^\u4e00-\u9fff]', '', text)  
    result = result.replace('"', '')   
    result = result.replace('"', '')  
    result = result.replace('"', '') 
    result = result.replace("'", '') 
    return result  

class TextClassifier:
    def __init__(self, ckpt_dir, gpu_id, model_type='qwen3', api_url='http://localhost:8000'):
        """
        初始化文本分类器(单GPU或远程API)
        
        Args:
            ckpt_dir: 模型检查点目录
            gpu_id: 单个GPU ID字符串 (如果使用远程API，可为None)
            model_type: 模型类型，默认'qwen3'
            api_url: 远程推理服务的URL (例如 "http://localhost:8000")，如果提供则优先使用远程服务
        """
        self.ckpt_dir = ckpt_dir
        self.model_type = model_type
        self.api_url = api_url
        
        if self.api_url:
            print(f"🚀 初始化 TextClassifier (远程模式): {self.api_url}")
            self.llm_engine = None
        else:
            # 设置CUDA_VISIBLE_DEVICES为单个GPU（必须是字符串）
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"进程 {os.getpid()} 使用GPU: {gpu_id}")
            
            # 配置VllmEngine,单GPU（带缓存优化）
            self.llm_engine = VllmEngine(
                self.ckpt_dir,
                model_type=self.model_type,
                gpu_memory_utilization=0.9,
                enable_prefix_caching=True,  # ✅ KV 缓存复用（前缀编码）已开启
            )


        # 从共享配置加载所有标签的提示词字典
        # 修改提示词请编辑 label_prompts_config.py 文件
        # 从共享配置加载所有标签的提示词字典
        # 修改提示词请编辑 label_prompts_config.py 文件
        from label_prompts_config import get_label_prompts
        self.label_prompts = get_label_prompts()
  

  
    def build_system_prompt(self, selected_labels=None):
        """
        构建系统提示词
        
        Args:
            selected_labels: 要包含的标签列表，如果为None则包含所有标签
            
        Returns:
            完整的系统提示词字符串
        """
        # 如果未指定标签，使用所有标签
        if selected_labels is None:
            selected_labels = list(self.label_prompts.keys())
        
        # 构建标签部分的提示词
        labels_content = "\n            ".join([
            self.label_prompts[label] 
            for label in selected_labels 
            if label in self.label_prompts
        ])
        
        # 构建完整的系统提示词
        system_prompt = f"""
            首先你是一个人，同时也是一名优秀的网约车客服人员，主要负责乘客端重大风险业务，你能够依据如下规则{labels_content}
            判断当前乘客输入信息属于哪一种重大风险类别。
            
            
            # 输出要求
            1. "Answer: <your answer>.  /no_think"
            2. 仅仅输出预测的结果，并放到  <answer> </answer> 之中，如 <answer>无风险 </answer>
            
            """
        
        return system_prompt

    def _call_remote_api(self, text, selected_labels=None):
        """调用远程推理服务"""
        import requests
        if not self.api_url:
            raise ValueError("API URL not configured")
            
        try:
            payload = {
                "text": text,
                "selected_labels": selected_labels
            }
            # 增加超时时间
            response = requests.post(f"{self.api_url}/infer", json=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                # 远程服务返回的已经包含了 <answer> 标签
                return data['prediction'] 
            else:
                print(f"API Error. Status: {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return f"<error>API Error: {response.status_code}</error>"
        except Exception as e:
            print(f"Remote inference failed: {e}")
            return f"<error>{str(e)}</error>"
        
    def classify_text(self, input_text, selected_labels=None):
        """
        单文本分类函数
        
        Args:
            input_text: 输入文本
            selected_labels: 要使用的标签列表，如果为None则使用所有标签
            
        Returns:
            包含分类结果的字典
        """
        # 如果配置了 API URL，优先使用远程服务
        if self.api_url:
            prediction = self._call_remote_api(input_text, selected_labels)
            return {
                'initial_classification': prediction,
            }

        query = f"""请判断以下乘客输入信息的风险类别：
        输入信息: < {input_text} > 
        """
        
        # 使用新的提示词构建方法
        system_content = self.build_system_prompt(selected_labels)
        
        message = [
            {'role': 'system', 'content': system_content},
            {"role": "user", 'content': query}
        ]

        data = dict()
        data['messages'] = message
        # 配置推理参数（KV 缓存会自动复用系统提示词前缀）
        request_config = RequestConfig(
            max_tokens=500,
            temperature=0.7,
        )
        metric = InferStats()
        infer_requests = [InferRequest(**data)]
        
        try:
            resp_list = self.llm_engine.infer(infer_requests, request_config) 
            query0 = infer_requests[0].messages[0]['content']
            response = resp_list[0].choices[0].message.content
            print(response)
        finally:
            # 显式销毁进程组
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        
        init_label = copy.deepcopy(response)
        
        return {
            'initial_classification': f'<answer>{init_label}</answer>',
        }

def process_data_chunk(gpu_id, ckpt_dir, data_chunk, start_idx, output_file, target_label, confused_labels):
    """
    在单个GPU上处理数据分片
    
    Args:
        gpu_id: GPU ID
        ckpt_dir: 模型检查点目录
        data_chunk: 数据分片
        start_idx: 起始索引
        output_file: 输出文件路径
        target_label: 目标标签列表
        confused_labels: 混淆标签列表
    """
    print(f"\n{'='*60}")
    print(f"进程启动 - GPU {gpu_id} | 处理样本 {start_idx} 到 {start_idx + len(data_chunk) - 1}")
    print(f"{'='*60}\n")
    
    classifier = None
    results = []
    
    try:
        # 初始化分类器
        classifier = TextClassifier(ckpt_dir, gpu_id)
        
        # 处理数据
        for idx, (original_idx, row) in enumerate(data_chunk.iterrows()):
            try:
                global_idx = start_idx + idx
                if (idx + 1) % 100 == 0:
                    print(f"GPU {gpu_id}: 已处理 {idx + 1}/{len(data_chunk)} 样本 (原始索引: {original_idx})")
                
                row_dict = row.to_dict()
                row_dict['_original_index'] = original_idx  # 保存原始索引
                
                query = str(row_dict.get('工单描述合并', ''))
                extracted_prediction = str(row_dict.get('extracted_prediction', ''))
                human_annotation_label = str(row_dict.get('最终标签', ''))
                
                should_process = False
                if len(query) > 0:
                    if extracted_prediction in confused_labels and human_annotation_label in target_label:
                        should_process = True
                    elif extracted_prediction in target_label and human_annotation_label in confused_labels:
                        should_process = True
                
                if should_process:
                    try:
                        result = classifier.classify_text(query)
                        predicted_label = result['initial_classification']
                    except Exception as e:
                        print(f"GPU {gpu_id} 样本 {original_idx} 分类失败: {str(e)}")
                        predicted_label = f'<error>{str(e)}</error>'
                    
                    row_dict['qwen3-14b预测'] = predicted_label
                    results.append(row_dict)
                
            except Exception as e:
                # 即使单个样本处理失败，也要保留原始数据
                print(f"GPU {gpu_id} 样本 {original_idx} 处理异常: {str(e)}")
        
        # 保存结果到临时文件
        if results:
            df_result = pd.DataFrame(results)
            df_result.to_pickle(output_file)
            print(f"GPU {gpu_id}: 结果已保存到 {output_file} (共 {len(results)} 条)")
            return len(df_result)
        else:
            print(f"GPU {gpu_id}: 没有符合条件的数据需要保存")
            return 0
        
    except Exception as e:
        print(f"❌ GPU {gpu_id} 进程级错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0
        
    finally:
        # 清理资源
        if classifier is not None:
            del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main(target_label=None, gpu_ids=None, num_gpus=None, min_memory_gb=15):
    """
    主函数
    
    Args:
        target_label: 目标优化标签，如果为None则使用默认值
    """
    
    # 读取Excel数据
    path_filename = '/ai/hlf/zhongda/datasets/valid——重大-清理后.xlsx'
    version = path_filename.split('/')[-1].rsplit('.', 1)[0]
    # 注释掉模型相关代码，因为只需要改进读写
    ckpt_dir ='/ai/ltx/zhongda3_0/0222_grpo'
    # 文件路径，调用函数进行检
    v_ckpt_dir = ckpt_dir.split('/')[-1].rsplit('.', 1)[0].split('标注')[-1]
    
    # TARGET_LABEL 从参数传入，如果未提供则使用默认值
    if target_label is None:
        TARGET_LABEL = ["诈骗"]  # 默认值
    else:
        TARGET_LABEL = [target_label] if isinstance(target_label, str) else target_label
    
    CONFUSION_FILE ='/ai/hlf/zhongda/mcts_prompt_gen_v4_a/label_confusion_analysis_scc.xlsx'

    if CONFUSION_ANALYZER_AVAILABLE:
        try:
            analyzer = ConfusionAnalyzer(CONFUSION_FILE)
            confused_labels = analyzer.get_confused_labels(TARGET_LABEL[0])
        except Exception as e:
            print(f"警告: 初始化混淆分析器失败: {e}")
            confused_labels = []
    else:
        confused_labels = []
        print("警告: 无法加载混淆分析器，仅处理目标标签")
    
    # 关闭，暂时只提升一个标签。
    confused_labels = confused_labels+TARGET_LABEL

    # 如果没有指定GPU,自动选择
    if gpu_ids is None:
        gpu_ids = select_gpus_for_inference(num_gpus=num_gpus, min_memory_gb=min_memory_gb)
    
    print(f"\n{'='*60}")
    print(f"数据并行模式: 使用 {len(gpu_ids)} 个GPU进行并行处理")
    print(f"GPU列表: {gpu_ids}")
    print(f"目标标签: {TARGET_LABEL}")
    print(f"混淆标签: {confused_labels}")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()  

    # 读取数据
    data = pd.read_excel(path_filename, sheet_name='Sheet1')
    data = data.reset_index(drop=True)
    total_samples = len(data)
    print(f"总样本数: {total_samples}")
    
    # 数据分片
    num_gpus_available = len(gpu_ids)
    chunk_size = (total_samples + num_gpus_available - 1) // num_gpus_available
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='gpu_inference_')
    print(f"临时目录: {temp_dir}\n")
    
    # 准备多进程任务
    processes = []
    temp_files = []
    
    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        
        if start_idx >= total_samples:
            break
        
        data_chunk = data.iloc[start_idx:end_idx].copy()
        temp_file = os.path.join(temp_dir, f'result_gpu_{gpu_id}.pkl')
        temp_files.append(temp_file)
        
        print(f"GPU {gpu_id}: 分配样本 {start_idx} 到 {end_idx-1} (共 {len(data_chunk)} 个)")
        
        # 创建进程
        p = mp.Process(
            target=process_data_chunk,
            args=(gpu_id, ckpt_dir, data_chunk, start_idx, temp_file, TARGET_LABEL, confused_labels)
        )
        processes.append(p)
    
    # 启动所有进程
    print(f"\n启动 {len(processes)} 个并行进程...\n")
    for p in processes:
        p.start()
    
    # 等待所有进程完成并检查退出状态
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
    
    # 合并结果并验证
    all_results = []
    processed_samples = 0
    missing_files = []
    
    for i, temp_file in enumerate(temp_files):
        gpu_id = gpu_ids[i]
        if os.path.exists(temp_file):
            try:
                df_chunk = pd.read_pickle(temp_file)
                samples_in_chunk = len(df_chunk)
                processed_samples += samples_in_chunk
                
                print(f"✓ GPU {gpu_id}: 读取 {samples_in_chunk} 个结果")
                
                all_results.append(df_chunk)
                os.remove(temp_file)  # 删除临时文件
            except Exception as e:
                print(f"❌ 读取 {temp_file} 失败: {str(e)}")
                import traceback
                traceback.print_exc()
                missing_files.append(i)
        else:
            print(f"⚠️ GPU {gpu_id} 没有生成结果文件 (可能没有符合条件的数据)")
            missing_files.append(i)
    
    # 删除临时目录
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    # 合并所有结果
    if all_results:
        # 按原始索引排序合并,保持数据顺序
        dict_result03 = pd.concat(all_results, ignore_index=False)
        if '_original_index' in dict_result03.columns:
            dict_result03 = dict_result03.sort_values('_original_index').reset_index(drop=True)
            dict_result03 = dict_result03.drop('_original_index', axis=1)
        else:
            dict_result03 = dict_result03.reset_index(drop=True)
        
        print(f"合并后数据行数: {len(dict_result03)}")
    else:
        print("⚠️ 警告: 没有生成任何预测结果!")
        return None

    # 获取当前时间
    current_time = datetime.now()
    end_time = datetime.now()  
    elapsed_time = (end_time - start_time).total_seconds() 
    # 格式化时间为用"-"连接的字符串
    formatted_time = current_time.strftime("%Y-%m-%d %H-%M-%S")
    # 输出格式化后的时间
    print(formatted_time)
    print(f"生成运行时间: {elapsed_time:.6f} 秒")
    # 保存结果到Excel

    path = f'/ai/hlf/zhongda/infer_qwen3/output/grpo-{v_ckpt_dir}-qwen3-14b预测-{formatted_time}-{version}.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        dict_result03.to_excel(writer, sheet_name='Sheet1', index=False)
    
    print(f"\n预测结果已保存到: {path}")
    
    # 自动调用准确率计算
    print("\n" + "="*50)
    print("开始计算准确率...")
    print("="*50 + "\n")
    try:
        accuracy, accuracy_file = calculate_accuracy_with_file(
            filename=path,
            prediction_col='qwen3-14b预测',
            human_label_col='最终标签'
        )
        print("\n" + "="*50)
        print("准确率计算完成！")
        print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"详细分析结果已保存到: {accuracy_file}")
        print("="*50 + "\n")
    except Exception as e:
        print(f"\n准确率计算出错: {str(e)}")
        print("但预测结果已成功保存。")
    
    return path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='文本分类推理脚本 - 支持多GPU自动选择')
    parser.add_argument('--target_label', type=str, default=None, 
                        help='目标优化标签，例如：行程延误、诈骗等')
    parser.add_argument('--gpus', type=str, default=None,
                        help='指定GPU ID,用逗号分隔,例如: 0,1,2。如不指定则自动选择')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='需要使用的GPU数量。如不指定则使用所有可用GPU')
    parser.add_argument('--min_memory', type=float, default=15,
                        help='最小显存要求(GB),默认15GB')
    
    args = parser.parse_args()
    
    # 处理GPU参数
    gpu_ids = None
    if args.gpus:
        gpu_ids = args.gpus.split(',')
        print(f"使用指定的GPU: {gpu_ids}")
    
    main(target_label=args.target_label, 
         gpu_ids=gpu_ids, 
         num_gpus=args.num_gpus,
         min_memory_gb=args.min_memory)


# conda activate q3_1 && cd /ai/hlf/zhongda/mcts_prompt_gen_v4_b && python3 infer_14B_qwen3_solo_refactored.py


"""

使用说明：
1. 现在所有29个标签的提示词都存储在 self.label_prompts 字典中
2. 每个标签可以单独修改，只需要修改字典中对应的键值
3. 运行时默认使用所有标签，也可以通过 selected_labels 参数只使用部分标签
4. 例如只优化"诈骗"标签的提示词：
   - 直接修改 self.label_prompts["诈骗"] 的内容
   - 运行时传入 --target_label 诈骗
   
示例：修改单个标签的提示词
只需要找到对应标签，修改其内容即可：
self.label_prompts["诈骗"] = '''诈骗:你修改后的新提示词...'''
按最终标签分组的错误统计:
  '诈骗': 24/49 (48.98%)
  '无风险': 7/7 (100.00%)
=== 按标签类别分析准确率 ===
                     总数  正确数     准确率   准确率%
cleaned_human_label                        
诈骗                   36   26  0.7222  72.22
无风险                   7    0  0.0000   0.00

=== 按标签类别分析准确率 ===
                     总数  正确数     准确率   准确率%
cleaned_human_label                        
限制人身自由               65   25  0.3846  38.46
无风险                   6    0  0.0000   0.00

=== 按标签类别分析准确率 ===
                     总数  正确数     准确率   准确率%
cleaned_human_label                        
限制人身自由               65   25  0.3846  38.46
无风险                   6    0  0.0000   0.00


按最终标签分组的错误统计:
  '骚扰': 40/115 (34.78%)
  '无风险': 2/8 (25.00%)
=== 按标签类别分析准确率 ===
                      总数  正确数     准确率   准确率%
cleaned_human_label                         
骚扰                   115   75  0.6522  65.22
无风险                    8    6  0.7500  75.00

详细结果已保存到: accuracy_analysis_result_20251208_172216.xlsx
==================================================
准确率计算完成！
准确率: 0.6585 (65.85%)


  python3 infer_14B_qwen3_solo_refactored.py --target_label "诈骗"

  python3 infer_14B_qwen3_solo_refactored.py --gpus 0,1

 conda activate q3_1 && cd /ai/ltx/zhongda/infer_qwen3/script && python3 infer_14B_qwen3_solo_refactored.py --target_label "诈骗" --gpus 2,3,4,5 --num_gpus 4

限制人身自由
  
 conda activate q3_1 && cd /ai/ltx/zhongda/infer_qwen3/script && python3 infer_14B_qwen3_solo_refactored.py --target_label "限制人身自由" --gpus 2,3,4,5 --num_gpus 4

骚扰

 conda activate q3_1 && cd /ai/ltx/zhongda/infer_qwen3/script && python3 infer_14B_qwen3_solo_refactored.py --target_label "骚扰" --gpus 2,3,4,5 --num_gpus 4

盗窃

conda activate q3_1 && cd /ai/ltx/zhongda/infer_qwen3/script && python3 infer_14B_qwen3_solo_refactored.py --target_label "盗窃" --gpus 2,3,4,5 --num_gpus 4

猝死
conda activate q3_1 && cd /ai/ltx/zhongda/infer_qwen3/script && python3 infer_14B_qwen3_solo_refactored.py --target_label "猝死" --gpus 2,3,4,5 --num_gpus 4



"""













