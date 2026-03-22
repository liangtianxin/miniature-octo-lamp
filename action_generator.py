"""
C-MCTS Action Generator v2 - 改进版动作生成器

改进点:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[P0] 1. 混淆引导生成 - 将混淆矩阵信息深度嵌入 system prompt, 
       引导 LLM 针对高混淆标签对生成有区分力的规则
[P0] 2. 多样性控制 - 强制不同动作覆盖不同策略类型
       (ADD_POSITIVE / ADD_NEGATIVE / MODIFY_BOUNDARY / REFINE_CONTEXT)
[P1] 3. 动作质量预评估 - 基于规则文本相似度去重
[P1] 4. 失败修复模式 (Refine) - 从 core3 移植, 针对评估失败的节点
       生成修正动作
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import requests
import json
import re
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("cmcts.action_gen")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)

# 策略类型定义 - 强制多样性
ACTION_STRATEGIES = [
    {
        "type": "ADD_POSITIVE",
        "description": "增加正例覆盖规则: 针对漏检(FN)案例, 添加新规则以覆盖被遗漏的正样本",
        "focus": "fn_examples"
    },
    {
        "type": "ADD_NEGATIVE",
        "description": "增加负例排除规则: 针对误检(FP)案例, 添加排除规则以过滤被错误匹配的负样本",
        "focus": "fp_examples"
    },
    {
        "type": "MODIFY_BOUNDARY",
        "description": "修改边界条件: 调整现有规则的适用范围, 使其更精确地区分目标标签与混淆标签",
        "focus": "confused_labels"
    },
    {
        "type": "REFINE_CONTEXT",
        "description": "细化上下文条件: 为现有规则添加前置条件或上下文约束, 减少歧义匹配",
        "focus": "both"
    },
]


class ActionGenerator:
    """C-MCTS 动作生成器 v2

    核心改进:
    1. 每个动作必须属于一个策略类型, 确保多样性
    2. 混淆矩阵信息直接嵌入 prompt, 引导 LLM 生成有针对性的规则
    3. 支持 refine 模式: 针对失败节点生成修正动作
    """

    def __init__(self, api_key: str,
                 base_url: str = "http://172.20.168.207/v1/chat/completions",
                 model: str = "deepseek-v3-0324-ygcx",
                 max_rules: int = 20,
                 max_chars_per_rule: int = 200):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_rules = max_rules
        self.max_chars_per_rule = max_chars_per_rule
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_actions(self, current_prompt: str, error_analysis: Dict,
                         confused_labels: Optional[List[str]] = None,
                         k: int = 6,
                         refine: bool = False) -> List[Dict]:
        """生成 k 个候选动作

        [改进] 强制多样性: k 个动作按策略类型均匀分配
        [改进] 混淆引导: 高混淆标签获得更多关注
        [改进] 支持 refine 模式

        Returns:
            List[Dict] -> [{'new_prompt': str, 'confidence': float,
                            'description': str, 'strategy': str}]
        """
        current_rules = self._count_rules(current_prompt)
        can_add = current_rules < self.max_rules

        # [P0] 分配策略类型
        strategies = self._assign_strategies(k, can_add)

        # [P0-fix] 逐个生成: 每个策略独立调用一次 LLM, 而非一次生成 k 个
        # 根因: LLM 无法在一次 JSON 响应中可靠生成 k 个包含完整 prompt 的方案
        target_label = error_analysis.get('target_label', '')
        temperature = 0.9 if not refine else 0.6
        all_actions = []

        for idx, strategy in enumerate(strategies):
            system_prompt = self._build_system_prompt_single(
                strategy=strategy, can_add=can_add,
                confused_labels=confused_labels, refine=refine)

            user_prompt = self._build_user_prompt_single(
                current_prompt=current_prompt,
                error_analysis=error_analysis,
                confused_labels=confused_labels,
                current_rules=current_rules,
                strategy=strategy, refine=refine)

            response_content = self._call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=temperature)

            parsed = self._parse_response(response_content)
            if parsed:
                action = parsed[0]  # 取第一个
                action['strategy'] = strategy['type']
                if target_label:
                    action = self._postprocess_actions([action], current_prompt, target_label)[0]
                all_actions.append(action)
                logger.info(f"  ✓ [{idx+1}/{k}] {strategy['type']}: {action.get('description', '?')[:50]}")
            else:
                logger.warning(f"  ✗ [{idx+1}/{k}] {strategy['type']}: 生成失败")

        # [P1] 去重
        all_actions = self._deduplicate_actions(all_actions)

        # 归一化置信度
        if all_actions:
            confidences = [a.get('confidence', 0.5) for a in all_actions]
            probs = self._softmax(confidences)
            for i, action in enumerate(all_actions):
                action['confidence'] = probs[i]

        logger.info(f"✓ 生成 {len(all_actions)} 个动作 (去重后, 共请求 {k} 个)")
        return all_actions

    def _assign_strategies(self, k: int, can_add: bool) -> List[Dict]:
        """[P0] 为 k 个动作分配策略类型, 确保多样性"""
        available = ACTION_STRATEGIES.copy()
        if not can_add:
            # 不能新增规则时, 移除 ADD 类型
            available = [s for s in available if not s["type"].startswith("ADD")]

        strategies = []
        for i in range(k):
            strategies.append(available[i % len(available)])
        return strategies

    def _build_system_prompt_single(self, strategy: Dict, can_add: bool,
                                     confused_labels: Optional[List[str]],
                                     refine: bool) -> str:
        """构建单动作系统提示词 (一次只请求1个方案, 大幅提高成功率)"""

        confusion_section = ""
        if confused_labels:
            top_confused = confused_labels[:5]
            confusion_section = f"""\n高混淆标签 (修改时必须特别注意区分度): {', '.join(top_confused)}"""

        mode = "修复" if refine else "优化"

        return f"""你是提示词{mode}工程师。根据错误分析, 用 [{strategy['type']}] 策略修改提示词。

策略说明: {strategy['description']}

铁律: 混淆标签准确率不能下降。仅修改目标标签的规则。每条规则不超过{self.max_chars_per_rule}字。{confusion_section}

请输出一个JSON对象 (不是数组), 格式:
{{
  "description": "简短描述",
  "new_prompt": "完整的修改后提示词",
  "confidence": 0.85
}}"""

    def _build_system_prompt(self, k: int, can_add: bool,
                              strategies: List[Dict],
                              confused_labels: Optional[List[str]],
                              refine: bool) -> str:
        """[兼容旧接口] 构建多动作系统提示词"""

        strategy_desc = "\n".join([
            f"  {i+1}. [{s['type']}] {s['description']}"
            for i, s in enumerate(strategies)
        ])

        confusion_section = ""
        if confused_labels:
            top_confused = confused_labels[:5]
            confusion_section = f"""\n高混淆标签: {', '.join(top_confused)}"""

        mode = "修复" if refine else "优化"

        return f"""你是提示词{mode}工程师。根据错误分析修改提示词。

铁律: 混淆标签准确率不能下降。{confusion_section}

生成 {k} 个方案, 策略类型:\n{strategy_desc}

JSON数组格式: [{{"description":"...","new_prompt":"...","confidence":0.85}}, ...]
"""

    def _build_user_prompt_single(self, current_prompt: str, error_analysis: Dict,
                                    confused_labels: Optional[List[str]],
                                    current_rules: int,
                                    strategy: Dict, refine: bool) -> str:
        """构建单动作用户提示词 (精简版, 只请求1个方案)"""

        target_label = error_analysis.get('target_label', 'Unknown')

        # 只展示 3 个案例, 减少 token 消耗
        fn_examples = self._format_examples(error_analysis.get('fn_examples', [])[:3])
        fp_examples = self._format_examples(error_analysis.get('fp_examples', [])[:3])

        return f"""当前提示词:
{current_prompt}

目标标签: {target_label} | FN={error_analysis.get('false_negatives', 0)} FP={error_analysis.get('false_positives', 0)} | 规则数: {current_rules}/{self.max_rules}

漏检案例:
{fn_examples}

误检案例:
{fp_examples}

请用 [{strategy['type']}] 策略生成1个修改方案。new_prompt 必须是从"{target_label}:"开始的完整提示词。"""

    def _build_user_prompt(self, current_prompt: str, error_analysis: Dict,
                            confused_labels: Optional[List[str]],
                            current_rules: int, k: int,
                            strategies: List[Dict], refine: bool) -> str:
        """[兼容旧接口] 构建多动作用户提示词"""

        target_label = error_analysis.get('target_label', 'Unknown')
        fn_examples = self._format_examples(error_analysis.get('fn_examples', [])[:3])
        fp_examples = self._format_examples(error_analysis.get('fp_examples', [])[:3])

        strategy_assignments = "\n".join([
            f"  方案{i+1}: [{s['type']}] {s['description']}"
            for i, s in enumerate(strategies)
        ])

        return f"""当前提示词:\n{current_prompt}\n\n目标标签: {target_label} | FN={error_analysis.get('false_negatives', 0)} FP={error_analysis.get('false_positives', 0)}\n\n漏检: {fn_examples}\n误检: {fp_examples}\n\n生成{k}个方案: {strategy_assignments}\n\nJSON数组: [{{"description":"...","new_prompt":"完整提示词","confidence":0.85}}]"""

    def _postprocess_actions(self, actions: List[Dict], current_prompt: str,
                              target_label: str) -> List[Dict]:
        """后处理: 确保 new_prompt 完整性"""
        for action in actions:
            new_prompt = action.get('new_prompt', '').strip()
            if not new_prompt:
                continue

            # 如果不以标签名开头, 拼接
            if not new_prompt.startswith(f'{target_label}:'):
                if new_prompt[0:1].isdigit() or new_prompt.startswith(';'):
                    sep = '' if current_prompt.endswith(';') else ';'
                    action['new_prompt'] = current_prompt + sep + new_prompt
                    logger.info(f"⚠️ 自动拼接不完整prompt")
                else:
                    sep = '' if current_prompt.endswith(';') else ';'
                    action['new_prompt'] = current_prompt + sep + new_prompt
            else:
                # 验证已有规则保留
                start = current_prompt[:min(50, len(current_prompt))]
                if start not in new_prompt:
                    logger.warning(f"⚠️ LLM 可能修改了已有规则")

        return actions

    def _deduplicate_actions(self, actions: List[Dict]) -> List[Dict]:
        """[P1-fix] 基于规则差异去重 (而非全文字符Jaccard)
        
        旧版问题: 字符级 Jaccard 对共享长前缀的 prompt 几乎总是 >0.95,
        导致 4 个有意义的不同方案被错误判定为重复.
        
        新版: 提取每个 prompt 的规则集合, 比较规则集合的差异.
        """
        if not actions:
            return actions

        unique = [actions[0]]
        for action in actions[1:]:
            is_dup = False
            new_p = action.get('new_prompt', '')
            new_rules = self._extract_rules(new_p)
            for existing in unique:
                existing_p = existing.get('new_prompt', '')
                existing_rules = self._extract_rules(existing_p)
                if self._rule_set_similarity(new_rules, existing_rules) > 0.85:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(action)

        if len(unique) < len(actions):
            logger.info(f"🔄 去重: {len(actions)} → {len(unique)}")
        return unique

    def _extract_rules(self, prompt: str) -> List[str]:
        """从 prompt 中提取规则列表 (按行分割, 去除空行和标题行)"""
        lines = [l.strip() for l in prompt.split('\n') if l.strip()]
        rules = []
        for line in lines:
            # 跳过标签名行 (如 "无风险:")
            if line.endswith(':') and len(line) < 30:
                continue
            rules.append(line)
        return rules

    def _rule_set_similarity(self, rules_a: List[str], rules_b: List[str]) -> float:
        """比较两个规则集合的相似度: 完全相同的规则数 / 总规则数"""
        if not rules_a and not rules_b:
            return 1.0
        if not rules_a or not rules_b:
            return 0.0
        set_a = set(rules_a)
        set_b = set(rules_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _text_similarity(self, a: str, b: str) -> float:
        """简单的字符级 Jaccard 相似度 (保留用于其他场景)"""
        if not a or not b:
            return 0.0
        set_a = set(a)
        set_b = set(b)
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    def _format_examples(self, examples: list) -> str:
        return "\n".join([f"- {ex}" for ex in examples[:5]])

    def _call_llm(self, messages: list, temperature: float = 0.8) -> str:
        try:
            data = {
                "messages": messages,
                "model": self.model,
                "temperature": temperature,
                "response_format": {"type": "json_object"}
            }
            response = requests.post(
                self.base_url, headers=self.headers,
                json=data, timeout=120)

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"API Error: {response.status_code}")
                return ""
        except Exception as e:
            logger.error(f"Request Failed: {e}")
            return ""

    def _parse_response(self, content: str) -> List[Dict]:
        """健壮的 JSON 解析 (增强日志)"""
        if not content:
            logger.warning("⚠️ LLM 返回空内容")
            return []

        logger.info(f"📝 LLM 原始响应长度: {len(content)} 字符")

        try:
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)

            if isinstance(data, list):
                valid = [d for d in data if isinstance(d, dict) and d.get('new_prompt')]
                logger.info(f"✓ 解析为 list: {len(data)} 项, 有效(含new_prompt): {len(valid)} 项")
                return valid
            elif isinstance(data, dict):
                # 1. 直接包含 new_prompt
                if 'new_prompt' in data:
                    return [data]

                # 2. 标准 list 包装键
                for key in ('actions', 'data', 'results'):
                    if key in data and isinstance(data[key], list):
                        valid = [d for d in data[key] if isinstance(d, dict) and d.get('new_prompt')]
                        if valid:
                            logger.info(f"✓ 解析为 dict['{key}']: {len(valid)} 项")
                            return valid

                # 3. [fix] 处理中文键名如 '方案1'、'solution' 等嵌套结构
                for key, val in data.items():
                    if isinstance(val, dict) and val.get('new_prompt'):
                        logger.info(f"✓ 从嵌套键 '{key}' 提取到 new_prompt")
                        return [val]
                    if isinstance(val, dict):
                        # 再深入一层
                        for sub_key, sub_val in val.items():
                            if isinstance(sub_val, dict) and sub_val.get('new_prompt'):
                                logger.info(f"✓ 从嵌套键 '{key}.{sub_key}' 提取到 new_prompt")
                                return [sub_val]

                # 4. [fix] 如果 dict 中有任何看起来像 prompt 的长文本值, 尝试作为 new_prompt
                for key, val in data.items():
                    if isinstance(val, str) and len(val) > 100 and ':' in val:
                        logger.info(f"✓ 将键 '{key}' 的长文本值作为 new_prompt")
                        return [{'new_prompt': val, 'description': key, 'confidence': 0.5}]

                logger.warning(f"⚠️ JSON dict 无法提取 new_prompt. 键: {list(data.keys())}")
            return []

        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON 解析异常: {e}")
            logger.warning(f"   内容前200字: {content[:200]}")
            # 尝试提取 JSON
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    valid = [d for d in data if isinstance(d, dict) and d.get('new_prompt')]
                    logger.info(f"✓ 正则提取 JSON: {len(valid)} 项")
                    return valid
                except:
                    pass
            logger.error(f"❌ JSON 完全解析失败: {content[:300]}")
            return []

    def _count_rules(self, prompt: str) -> int:
        lines = [l.strip() for l in prompt.split('\n') if l.strip()]
        numbered = [l for l in lines if re.match(r'^\d+[\.、]', l)]
        return len(numbered) if numbered else len(lines)

    def _softmax(self, x: list) -> list:
        arr = np.array(x, dtype=float)
        exp_x = np.exp(arr - np.max(arr))
        return (exp_x / exp_x.sum()).tolist()
