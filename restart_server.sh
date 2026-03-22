#!/bin/bash
# 查找正在使用端口 8081 的进程 PID
PID=$(lsof -ti :8081)

if [ -n "$PID" ]; then
  echo "发现进程 $PID 正在占用端口 8081，正在终止..."
  kill -9 $PID
  echo "进程 $PID 已终止。"
else
  echo "端口 8081 当前空闲。"
fi

echo "正在启动 fastapi_model_server.py ..."
# 确保在当前环境下运行
python fastapi_model_server.py

# conda activate q3_1 && cd /ai/ltx/zhongda/mcts_prompt_gen_v4 && sh restart_server.sh