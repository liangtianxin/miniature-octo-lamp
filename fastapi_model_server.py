# fastapi_model_server.py
# 独立的 FastAPI 服务，启动时加载模型权重，支持多卡推理，带有系统提示词

import os
import re
import json
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from swift.llm import VllmEngine, InferRequest, RequestConfig
from typing import List, Dict, Any, Optional

# 导入共享的标签提示词配置
from label_prompts_config import get_label_prompts

# 配置参数
MODEL_CKPT_DIR = '/ai/ltx/zhongda3_0/0222_grpo'  # 替换为你的模型路径
#GPU_IDS = ['0', '1', '2', '3']  # 指定使用的 GPU ID 列表
GPU_IDS = ['4', '5', '6', '7']  # 指定使用的 GPU ID 列表

TENSOR_PARALLEL_SIZE = len(GPU_IDS)  # 多卡并行度

# 从共享配置加载所有标签的提示词字典
# 修改提示词请编辑 label_prompts_config.py 文件
label_prompts = get_label_prompts()

def build_system_prompt(selected_labels: Optional[List[str]] = None, label_overrides: Optional[Dict[str, str]] = None) -> str:
    """
    构建系统提示词

    Args:
        selected_labels: 要包含的标签列表，如果为None则包含所有标签
        label_overrides: 要覆盖的标签提示词字典

    Returns:
        完整的系统提示词字符串
    """
    # 复制基础提示词字典
    current_prompts = label_prompts.copy()
    
    # 如果有覆盖，更新字典
    if label_overrides:
        current_prompts.update(label_overrides)
        
    # 如果未指定标签，使用所有标签
    if selected_labels is None:
        selected_labels = list(current_prompts.keys())

    # 构建标签部分的提示词
    labels_content = "\n            ".join([
        current_prompts[label]
        for label in selected_labels
        if label in current_prompts
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

# FastAPI 应用
app = FastAPI(title="文本分类推理服务", description="基于 Qwen3-14B 的网约车风险分类服务")

# 全局模型变量
model_engine = None

class InferenceRequest(BaseModel):
    text: str
    selected_labels: Optional[List[str]] = None  # 可选，指定使用的标签
    label_overrides: Optional[Dict[str, str]] = None # 可选，覆盖特定标签的提示词

class InferenceResponse(BaseModel):
    prediction: str
    system_prompt_used: str

@app.on_event("startup")
async def load_model():
    """启动时加载模型权重到指定 GPU"""
    global model_engine
    try:
        print(f"正在加载模型权重到 GPU: {GPU_IDS} ...")
        # 设置 CUDA_VISIBLE_DEVICES
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(GPU_IDS)

        # 初始化 VllmEngine，支持多卡和缓存优化
        model_engine = VllmEngine(
            model_id_or_path=MODEL_CKPT_DIR,
            model_type='qwen3',
            gpu_memory_utilization=0.85,
            enable_prefix_caching=True,  # ✅ KV 缓存复用（前缀编码）已开启
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,  # 多卡并行
        )
        print("✅ 模型权重加载完成！")
        print(f"   ✓ KV 缓存复用（前缀编码）: 已开启")
        print(f"   ✓ 张量并行度: {TENSOR_PARALLEL_SIZE}")
        print(f"   ✓ GPU 显存利用率: 85%")
        print("✅ 模型权重加载完成！")
        print(f"   ✓ KV 缓存复用（前缀编码）: 已开启")
        print(f"   ✓ 张量并行度: {TENSOR_PARALLEL_SIZE}")
        print(f"   ✓ GPU 显存利用率: 85%")
        print(f"   ✓ Block Size: 16 (高效缓存管理)")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        raise

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """推理接口"""
    if model_engine is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    try:
        # 构建查询
        query = f"""请判断以下乘客输入信息的风险类别：
        输入信息: < {request.text} >
        """

        # 构建系统提示词
        system_prompt = build_system_prompt(request.selected_labels, request.label_overrides)

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ]

        # 推理配置（分类任务用低温，确保稳定输出）
        request_config = RequestConfig(
            max_tokens=500,
            temperature=0.01,  # 低温度确保分类结果一致
        )
        infer_req = InferRequest(messages=messages)

        # 执行推理
        resp_list = model_engine.infer([infer_req], request_config)
        response = resp_list[0].choices[0].message.content

        # 提取预测结果
        match = re.search(r'<answer>(.*?)</answer>', response)
        if match:
            prediction = match.group(1).strip()
        else:
            prediction = response.strip()

        return InferenceResponse(
            prediction=f'<answer>{prediction}</answer>',
            system_prompt_used=system_prompt
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

@app.get("/")
async def root():
    """根路径欢迎信息"""
    return {
        "message": "FastAPI Model Server is Running", 
        "endpoints": {
            "health_check": "/health",
            "inference": "/infer",
            "documentation": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model_loaded": model_engine is not None}

if __name__ == "__main__":
    import uvicorn
    print("启动 FastAPI 服务...")
    uvicorn.run(app, host="0.0.0.0", port=8081)

# cd /ai/hlf/fastapi && conda activate q3_1 && python /ai/hlf/zhongda/mcts_prompt_gen_v4_b2/fastapi_model_server.py