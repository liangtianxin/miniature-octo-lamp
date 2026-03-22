from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConfusionAnalyzer:
    path: str

    def __post_init__(self) -> None:
        self.df = pd.read_excel(self.path) if self.path.endswith(".xlsx") else pd.read_csv(self.path)
        self.df.columns = [str(c).strip() for c in self.df.columns]
        required = {"真实标签", "错误预测为"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"confusion_file 缺少列: {sorted(missing)}")
        for col in ("混淆率", "错误次数"):
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).astype(float)
        logger.info(f"混淆矩阵加载成功，共 {len(self.df)} 行")

    def get_confused_labels(self, target_label: str, top_k=None):
        sub = self.df[self.df["真实标签"] == target_label].copy()
        if len(sub) == 0:
            return []
        sort_col = None
        for candidate in ("混淆率", "错误次数"):
            if candidate in sub.columns:
                sort_col = candidate
                break
        if sort_col is not None:
            sort_vals = pd.to_numeric(sub[sort_col], errors="coerce").fillna(0.0).astype(float).values
            order = np.argsort(sort_vals)[::-1]
            sub = sub.iloc[order].reset_index(drop=True)
        uniq, seen = [], set()
        limit = int(top_k) if top_k is not None else None
        for raw in sub["错误预测为"].astype(str):
            lbl = str(raw).strip()
            if lbl.lower() in {"nan", "none", "", "null"}:
                continue
            if lbl != target_label and lbl not in seen:
                uniq.append(lbl)
                seen.add(lbl)
            if limit is not None and len(uniq) >= limit:
                break
        return uniq


def normalize_label(value: str) -> str:
    v = str(value).strip()
    if v.lower() in {"nan", "none", "", "null"}:
        return ""
    return v
