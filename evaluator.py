"""
C-MCTS Evaluator v2 - 改进版评估器

改进点:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[P0] 1. 双层评估独立 baseline - Level A/B 各自维护基线, 避免抽样集与全量集互相误判
[P0] 2. 复合奖励函数 - 替代简单 F1, 综合考虑 F1增量 + 混淆安全 + 覆盖率
[P0] 3. 混淆矩阵深度利用 - 提取混淆率数值传递给 ConfusionPrior
[P1] 4. SRM 缓存优化 - 按 level 分离缓存, 避免串扰
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import os
import sys
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import logging
import hashlib
import pickle
import traceback

# ─── 日志 ─────────────────────────────────────────────────────────────────────

logger = logging.getLogger("cmcts.evaluator")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('cmcts_log.txt', mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# 导入推理环境
try:
    from infer_14B_qwen3_solo_refactored import TextClassifier
except ImportError:
    logger.warning("infer_14B_qwen3_solo_refactored 未找到, 使用 mock 模式")
    TextClassifier = None

try:
    from label_prompts_config import get_label_prompts
    label_prompts = get_label_prompts()
except ImportError:
    logger.warning("label_prompts_config 未找到, 使用空字典")
    label_prompts = {}


# ─── Worker ───────────────────────────────────────────────────────────────────

def worker_gpu(queries: list[str], model_path: str, gpu_id: str, model_type: str,
               prompt_state: str, target_label: str,
               api_url: str = "http://localhost:8000") -> dict:
    """GPU worker: 调用 FastAPI 推理服务"""
    try:
        import requests

        session = requests.Session()

        if not isinstance(prompt_state, str):
            msg = f"worker_gpu: prompt_state 类型错误 {type(prompt_state)}"
            return {"success": False, "predictions": None, "error": msg}

        predictions = []
        for query in queries:
            payload = {
                "text": str(query),
                "label_overrides": {target_label: prompt_state}
            }
            try:
                response = session.post(f"{api_url}/infer", json=payload, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    match = re.search(r'<answer>(.*?)</answer>', prediction)
                    pred_label = match.group(1).strip() if match else prediction.strip()
                    predictions.append(pred_label)
                else:
                    predictions.append("<ERROR>")
            except Exception as e:
                predictions.append("<ERROR>")

        session.close()
        return {"success": True, "predictions": predictions, "error": None}
    except Exception as e:
        return {"success": False, "predictions": None, "error": str(e)}


# ─── Baseline ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BaselineMetrics:
    other_acc: float
    target_precision: float
    target_recall: float
    target_f1: float
    per_confused_acc: Dict[str, float]


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluator v2
# ═══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    """C-MCTS 评估器 v2

    双层评估架构:
    - Level A: 快速抽样评估 (N=1000), 用于早期淘汰明显差的候选
    - Level B: 精确混淆标签评估, 用于最终验证

    复合奖励函数:
    R = w1 * ΔF1 + w2 * safety_bonus + w3 * base_f1
    其中:
    - ΔF1 = (f1 - baseline_f1) / baseline_f1  (相对提升)
    - safety_bonus = 1.0 if no violations else -penalty
    - base_f1 = 当前绝对 F1 值
    """

    # 奖励函数权重
    W_DELTA_F1 = 0.4     # F1 相对提升的权重
    W_SAFETY = 0.2       # 铁律安全性的权重
    W_BASE_F1 = 0.4      # 绝对 F1 值的权重
    DROP_TOLERANCE = 0.01  # 混淆标签准确率下降容忍度

    def __init__(
        self,
        model_path: str,
        data_path: str,
        target_label: str,
        confusion_file: Optional[str] = None,
        frozen_labels_prompts: Optional[Dict[str, str]] = None,
        gpu_ids: str = "0,1,2,3,4,5,6,7",
        model_type: str = "qwen3",
        api_url: str = "http://localhost:8000",
        *,
        query_col: str = "cleaned_text",
        true_label_col: str = "终极合并标签",
        level_a_sample_size: int = 1000,
        seed: int = 42,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.target_label = target_label
        self.confusion_file = confusion_file
        self.frozen_labels_prompts = frozen_labels_prompts or {}
        self.gpu_ids = gpu_ids
        self.model_type = model_type
        self.api_url = api_url
        self.query_col = query_col
        self.true_label_col = true_label_col
        self.level_a_sample_size = level_a_sample_size
        self.seed = seed

        # 加载数据
        self.df = pd.read_excel(data_path) if data_path.endswith(".xlsx") else pd.read_csv(data_path)
        self._validate_columns(self.df)

        # [P0] 加载混淆矩阵 + 提取混淆率
        self.confused_labels: List[str] = []
        self.confusion_rates: Dict[str, float] = {}  # {标签: 混淆率}
        if confusion_file:
            self._load_confusion_matrix(confusion_file)

        # [P0] 双层 baseline
        self._baseline_by_level: Dict[str, BaselineMetrics] = {}

        # [P1] SRM 缓存 (按 level 分离)
        self.cache_dir = "cmcts_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"{self.target_label}_cache.pkl")
        self.result_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._load_cache()

    def _load_confusion_matrix(self, confusion_file: str):
        """[P0] 加载混淆矩阵并提取混淆率数值"""
        try:
            _df = pd.read_excel(confusion_file) if confusion_file.endswith(".xlsx") \
                else pd.read_csv(confusion_file)
            _df.columns = [str(c).strip() for c in _df.columns]

            for _col in ("混淆率", "错误次数"):
                if _col in _df.columns:
                    _df[_col] = pd.to_numeric(_df[_col], errors="coerce").fillna(0.0)

            _sub = _df[_df["真实标签"] == self.target_label].copy()

            # 按混淆率/错误次数降序排列
            _sort_col = next((c for c in ("混淆率", "错误次数") if c in _sub.columns), None)
            if _sort_col and len(_sub) > 0:
                _sub = _sub.sort_values(_sort_col, ascending=False).reset_index(drop=True)

            _uniq, _seen = [], set()
            for _, row in _sub.iterrows():
                _lbl = str(row["错误预测为"]).strip()
                if _lbl.lower() in {"nan", "none", "", "null"}:
                    continue
                if _lbl != self.target_label and _lbl not in _seen:
                    _uniq.append(_lbl)
                    _seen.add(_lbl)
                    # 提取混淆率数值
                    if "混淆率" in _sub.columns:
                        self.confusion_rates[_lbl] = float(row.get("混淆率", 0.0))
                    elif "错误次数" in _sub.columns:
                        self.confusion_rates[_lbl] = float(row.get("错误次数", 0.0))

            self.confused_labels = _uniq
            logger.info(f"✓ 混淆矩阵加载: {len(self.confused_labels)} 个混淆标签")
            logger.info(f"  Top-5 混淆: {list(self.confusion_rates.items())[:5]}")

        except Exception as e:
            logger.warning(f"混淆矩阵加载失败: {e}，将对所有非目标标签约束检查")
            traceback.print_exc()

    def _validate_columns(self, df: pd.DataFrame):
        missing = [c for c in [self.query_col, self.true_label_col] if c not in df.columns]
        if missing:
            raise ValueError(f"数据缺少列: {missing}. 当前列: {df.columns.tolist()}")

    # ─── 缓存 ─────────────────────────────────────────────────────────────────

    def _get_memo_key(self, prompt_state: str, level: str) -> str:
        key_str = f"{prompt_state}_{level}_{self.target_label}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.result_cache = pickle.load(f)
                logger.info(f"✓ 缓存加载: {len(self.result_cache)} 条")
            except Exception:
                logger.warning("缓存加载失败, 使用空缓存")

    def _save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.result_cache, f)
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    # ─── Baseline ─────────────────────────────────────────────────────────────

    def set_baseline(self, prompt_state: str, *, level: str) -> BaselineMetrics:
        score, analysis = self._evaluate_raw(prompt_state, level=level)
        baseline = BaselineMetrics(
            other_acc=analysis["other_labels_acc"],
            target_precision=analysis["precision"],
            target_recall=analysis["recall"],
            target_f1=analysis["f1"],
            per_confused_acc=analysis.get("per_confused_acc", {}),
        )
        self._baseline_by_level[level] = baseline
        logger.info(f"✓ Baseline[{level}] P={baseline.target_precision:.4f} "
                     f"R={baseline.target_recall:.4f} F1={baseline.target_f1:.4f} "
                     f"OtherAcc={baseline.other_acc:.4f}")
        return baseline

    # ─── 主评估接口 ───────────────────────────────────────────────────────────

    def evaluate(self, prompt_state: str, level: str = "A") -> Tuple[float, Dict[str, Any]]:
        """评估提示词, 返回 (复合奖励, 分析报告)"""
        # 缓存检查
        cache_key = self._get_memo_key(prompt_state, level)
        if cache_key in self.result_cache:
            cached = self.result_cache[cache_key]
            logger.info(f"💾 缓存命中 [{level}] F1={cached[0]:.4f}")
            return cached

        # 自动设置 baseline
        if level not in self._baseline_by_level:
            self.set_baseline(prompt_state, level=level)
            score, analysis = self._evaluate_raw(prompt_state, level=level)
            analysis["is_baseline"] = True
            self.result_cache[cache_key] = (score, analysis)
            self._save_cache()
            return score, analysis

        # 评估
        raw_f1, analysis = self._evaluate_raw(prompt_state, level=level)
        baseline = self._baseline_by_level[level]

        # [P0] 复合奖励函数
        final_score, violations = self._compute_reward(raw_f1, analysis, baseline)

        # 记录增量信息
        analysis["f1_gain"] = raw_f1 - baseline.target_f1
        analysis["precision_gain"] = analysis["precision"] - baseline.target_precision
        analysis["recall_gain"] = analysis["recall"] - baseline.target_recall
        analysis["baseline_f1"] = baseline.target_f1
        analysis["baseline_precision"] = baseline.target_precision
        analysis["baseline_recall"] = baseline.target_recall
        analysis["violations"] = violations
        analysis["is_safe"] = len(violations) == 0

        self.result_cache[cache_key] = (final_score, analysis)
        self._save_cache()

        return final_score, analysis

    def _compute_reward(self, f1: float, analysis: Dict, baseline: BaselineMetrics
                        ) -> Tuple[float, List[str]]:
        """[P0] 复合奖励函数

        R = w1 * norm_delta_f1 + w2 * safety + w3 * f1

        Returns:
            (reward, violations_list)
        """
        violations = []

        # 1. F1 相对提升
        if baseline.target_f1 > 0:
            delta_f1 = (f1 - baseline.target_f1) / baseline.target_f1
        else:
            delta_f1 = f1  # baseline F1 = 0 时, 任何正值都是提升

        # 归一化到 [-1, 1] 范围
        norm_delta = max(-1.0, min(1.0, delta_f1))

        # 2. 安全性检查
        per_confused_acc = analysis.get("per_confused_acc", {})
        safety = 1.0  # 默认安全

        if baseline.per_confused_acc:
            for label, base_acc in baseline.per_confused_acc.items():
                cur_acc = per_confused_acc.get(label)
                if cur_acc is None:
                    continue
                drop = base_acc - cur_acc
                if drop > self.DROP_TOLERANCE:
                    violations.append(f"confused_{label}_drop_{drop:.4f}")
                    safety -= drop * 2  # 每 1% 下降扣 2% 安全分
        else:
            other_drop = baseline.other_acc - analysis.get("other_labels_acc", 1.0)
            if other_drop > self.DROP_TOLERANCE:
                violations.append(f"other_acc_drop_{other_drop:.4f}")
                safety -= other_drop * 2

        safety = max(0.0, min(1.0, safety))

        # 3. 综合奖励
        reward = (self.W_DELTA_F1 * norm_delta +
                  self.W_SAFETY * safety +
                  self.W_BASE_F1 * f1)

        # 确保奖励为正
        reward = max(0.0001, reward)

        if violations:
            logger.info(f"⚠️ 违规: {violations}, safety={safety:.3f}, reward={reward:.4f}")

        return reward, violations

    # ─── 原始评估 ─────────────────────────────────────────────────────────────

    def _evaluate_raw(self, prompt_state: str, level: str) -> Tuple[float, Dict[str, Any]]:
        eval_df = self._select_eval_df(level)

        logger.info(f"🔬 推理 {len(eval_df)} 条样本 (Level {level})...")

        # GPU 并行推理
        gpu_list = self.gpu_ids.split(",")
        num_gpus = len(gpu_list)
        queries = eval_df[self.query_col].tolist()
        batch_size = (len(queries) + num_gpus - 1) // num_gpus
        query_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        while len(query_batches) < num_gpus:
            query_batches.append([])

        predictions = [None] * len(queries)
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i, gpu_id in enumerate(gpu_list):
                if query_batches[i]:
                    future = executor.submit(
                        worker_gpu, query_batches[i], self.model_path,
                        gpu_id, self.model_type, prompt_state,
                        self.target_label, self.api_url)
                    futures.append((future, i))

            for future, batch_idx in futures:
                try:
                    result = future.result()
                    if result["success"] and result["predictions"]:
                        start_idx = batch_idx * batch_size
                        for j, pred in enumerate(result["predictions"]):
                            if start_idx + j < len(predictions):
                                predictions[start_idx + j] = pred
                    else:
                        logger.warning(f"Batch {batch_idx} 失败: {result['error']}")
                except Exception as e:
                    logger.error(f"Batch {batch_idx} 异常: {e}")

        # 过滤有效预测
        valid_predictions, valid_indices = [], []
        for idx, pred in enumerate(predictions):
            if pred is not None and pred != "<ERROR>":
                valid_predictions.append(pred)
                valid_indices.append(idx)

        if not valid_predictions:
            raise RuntimeError("所有预测失败! 请检查推理服务.")

        if len(valid_predictions) < len(queries):
            logger.warning(f"部分预测失败: {len(valid_predictions)}/{len(queries)}")
            eval_df = eval_df.iloc[valid_indices].copy()
        else:
            eval_df = eval_df.copy()

        eval_df["prediction"] = valid_predictions
        eval_df["pred_label"] = eval_df["prediction"].apply(self._extract_label)

        # 计算指标
        tp = len(eval_df[(eval_df[self.true_label_col] == self.target_label) &
                         (eval_df["pred_label"] == self.target_label)])
        fn = len(eval_df[(eval_df[self.true_label_col] == self.target_label) &
                         (eval_df["pred_label"] != self.target_label)])
        fp = len(eval_df[(eval_df[self.true_label_col] != self.target_label) &
                         (eval_df["pred_label"] == self.target_label)])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_confused_acc = self._per_label_accuracy(eval_df, self.confused_labels) \
            if self.confused_labels else {}
        other_labels_acc = self._overall_accuracy_for_labels(
            eval_df, self.confused_labels if self.confused_labels else
            [l for l in eval_df[self.true_label_col].unique() if l != self.target_label])

        analysis = {
            "target_label": self.target_label,
            "total_samples": len(eval_df),
            "false_negatives": fn,
            "false_positives": fp,
            "fn_examples": eval_df[
                (eval_df[self.true_label_col] == self.target_label) &
                (eval_df["pred_label"] != self.target_label)
            ][self.query_col].head(10).tolist(),
            "fp_examples": eval_df[
                (eval_df[self.true_label_col] != self.target_label) &
                (eval_df["pred_label"] == self.target_label)
            ][self.query_col].head(10).tolist(),
            "all_fn_df": eval_df[
                (eval_df[self.true_label_col] == self.target_label) &
                (eval_df["pred_label"] != self.target_label)],
            "all_fp_df": eval_df[
                (eval_df[self.true_label_col] != self.target_label) &
                (eval_df["pred_label"] == self.target_label)],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "other_labels_acc": other_labels_acc,
            "per_confused_acc": per_confused_acc,
        }

        return f1, analysis

    def _select_eval_df(self, level: str) -> pd.DataFrame:
        """选择评估数据集

        Level A: 随机抽样 (快速评估)
        Level B: 目标 + 混淆标签 (精确验证)
        """
        if level == "A":
            n = min(self.level_a_sample_size, len(self.df))
            return self.df.sample(n=n, random_state=self.seed)

        if self.confused_labels:
            relevant = [self.target_label] + self.confused_labels
            filtered = self.df[self.df[self.true_label_col].isin(relevant)]
            logger.info(f"Level B: {len(filtered)}/{len(self.df)} 条 "
                         f"({self.target_label} + {len(self.confused_labels)} 混淆标签)")
            return filtered
        else:
            logger.info(f"Level B: 全量 {len(self.df)} 条")
            return self.df

    def _extract_label(self, text) -> str:
        if pd.isna(text):
            return ""
        match = re.search(r"<answer>(.*?)</answer>", str(text))
        return match.group(1).strip() if match else str(text).strip()

    def _per_label_accuracy(self, df: pd.DataFrame, labels: List[str]) -> Dict[str, float]:
        result = {}
        for label in labels:
            sub = df[df[self.true_label_col] == label]
            if len(sub) > 0:
                result[label] = float((sub["pred_label"] == sub[self.true_label_col]).mean())
        return result

    def _overall_accuracy_for_labels(self, df: pd.DataFrame, labels: List[str]) -> float:
        sub = df[df[self.true_label_col].isin(labels)]
        return float((sub["pred_label"] == sub[self.true_label_col]).mean()) if len(sub) > 0 else 1.0

    def save_errors_to_excel(self, analysis: Dict, iteration: int, node_id: str):
        """保存 FN/FP 到 Excel 和 TXT"""
        fn_df = analysis.get("all_fn_df")
        fp_df = analysis.get("all_fp_df")

        if (fn_df is None or fn_df.empty) and (fp_df is None or fp_df.empty):
            return

        output_dir = "cmcts_error_logs"
        os.makedirs(output_dir, exist_ok=True)

        excel_path = os.path.join(output_dir, f"iter_{iteration}_{node_id}_errors.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            if fn_df is not None and not fn_df.empty:
                fn_df.to_excel(writer, sheet_name='FN', index=False)
            if fp_df is not None and not fp_df.empty:
                fp_df.to_excel(writer, sheet_name='FP', index=False)
        logger.info(f"📊 错误样本 → {excel_path}")

        txt_path = os.path.join(output_dir, f"iter_{iteration}_{node_id}_errors.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"=== 迭代 {iteration}, 节点 {node_id} ===\n\n")
            for tag, df_part in [("FN (漏检)", fn_df), ("FP (误检)", fp_df)]:
                if df_part is not None and not df_part.empty:
                    f.write(f"{tag}: {len(df_part)} 条\n")
                    f.write("-" * 40 + "\n")
                    for idx, row in df_part.head(20).iterrows():
                        query = str(row.get(self.query_col, ''))
                        true_l = str(row.get(self.true_label_col, ''))
                        pred_l = str(row.get('pred_label', ''))
                        f.write(f"  真实={true_l}, 预测={pred_l}\n")
                        f.write(f"  文本: {query[:200]}\n\n")
