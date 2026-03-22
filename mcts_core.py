"""
C-MCTS: Confusion-Guided Monte Carlo Tree Search for Automated Prompt Optimization

核心改进（相对于 mcts_core5）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[P0] 1. 修复 FBPS 理论缺陷 - 预评估结果存为 prior_reward, 不设 N=1, 
       _simulate 中重新评估后正确回传
[P0] 2. 混淆引导先验 (Confusion-Guided Prior) - 将混淆矩阵转化为动作空间的
       先验概率分布, 高混淆对获得更高探索权重
[P0] 3. 双层评估 (Level A/B) - 从 core3 移植, Level A 快速筛选 + Level B 精确验证
[P1] 4. 冻结链机制 (Chain Freezing) - 从 core3 移植, 通过验证则冻结为锚点
[P1] 5. Progressive Widening - 子节点数随 N^α 增长, 避免过早扩展
[P1] 6. 动态探索系数 (Dynamic PUCT) - 保留 core5 的 log-growth 公式
[P2] 7. 搜索树统计与可视化支持
[P2] 8. 收敛性追踪
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import copy
import json
import numpy as np
import traceback
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

# ─── 日志配置 ────────────────────────────────────────────────────────────────

logger = logging.getLogger("cmcts")
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


# ═══════════════════════════════════════════════════════════════════════════════
#  TreeNode - MCTS 搜索树节点
# ═══════════════════════════════════════════════════════════════════════════════

class TreeNode:
    """MCTS 搜索树节点

    核心改进:
    - prior_reward: FBPS 预评估的奖励值 (不再污染 N/Q)
    - confusion_prior: 混淆引导的先验概率 (基于混淆矩阵)
    - is_frozen: 通过双层验证后冻结
    - depth: 节点深度追踪
    """

    _id_counter = 0

    def __init__(self, prompt_state: str, parent: Optional['TreeNode'] = None,
                 prior_prob: float = 0.0, action_taken: Optional[str] = None):
        TreeNode._id_counter += 1
        self.id = TreeNode._id_counter

        self.prompt_state = prompt_state
        self.parent = parent
        self.children: List['TreeNode'] = []
        self.action_taken = action_taken
        self.depth = (parent.depth + 1) if parent else 0

        # MCTS 统计量
        self.N = 0              # 访问次数 (只通过 backpropagate 增加)
        self.Q = 0.0            # 平均价值
        self.W = 0.0            # 累积价值 (Q = W / N, 避免浮点累积误差)
        self.P = prior_prob     # LLM 给出的先验置信度

        # [P0-fix] FBPS 预评估结果 - 不再直接写入 N/Q
        self.prior_reward: Optional[float] = None  # 预评估得分, 用于 best_child 排序
        self.prior_analysis: Optional[Dict] = None  # 预评估分析报告

        # 节点状态
        self.is_dead = False
        self.is_frozen = False  # [P1] 通过双层验证后冻结

        # 评估结果
        self.error_analysis: Optional[Dict] = None  # Level B 分析
        self.local_analysis: Optional[Dict] = None   # Level A 分析
        self.global_analysis: Optional[Dict] = None   # Level B 全局分析

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def best_child(self, c1: float = 0.5, c2: float = 100.0) -> Optional['TreeNode']:
        """动态探索系数 PUCT 公式

        Score = Q_effective + P * E / (1 + N)

        其中 E = √N_parent × (c1 + log((N_parent + c2 + 1) / c2))

        [P0-fix] 对于 N=0 但有 prior_reward 的节点:
          Q_effective = prior_reward (预评估值作为初始估计)
        对于 N>0 的节点:
          Q_effective = W / N (真实回传的平均值)
        对于 N=0 且无 prior_reward 的节点:
          Q_effective = 0 (完全依赖探索项)
        """
        best_score = -float('inf')
        best_node = None

        N_parent = max(self.N, 1)
        sqrt_n = math.sqrt(N_parent)
        log_term = math.log((N_parent + c2 + 1) / c2)
        explore_base = sqrt_n * (c1 + log_term)

        for child in self.children:
            if child.is_dead:
                continue

            # [P0-fix] Q_effective: 区分已回传 vs 仅预评估 vs 未评估
            if child.N > 0:
                q_eff = child.W / child.N  # 真实统计值
            elif child.prior_reward is not None:
                # 预评估值打折: 降低确定性, 让探索项有空间发挥
                # 如果不打折, prior_reward 差距会完全压制探索项,
                # 导致 PUCT 退化为贪心排序
                q_eff = child.prior_reward * 0.5
            else:
                q_eff = 0.0

            # 探索项
            explore = child.P * explore_base / (1 + child.N)

            score = q_eff + explore

            if score > best_score:
                best_score = score
                best_node = child

        return best_node

    def __repr__(self):
        status = "🧊" if self.is_frozen else ("💀" if self.is_dead else "🟢")
        return (f"Node[{self.id}]{status} d={self.depth} N={self.N} "
                f"Q={self.Q:.4f} P={self.P:.3f} "
                f"act={self.action_taken[:30] if self.action_taken else 'Root'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ConfusionPrior - 混淆引导先验模块
# ═══════════════════════════════════════════════════════════════════════════════

class ConfusionPrior:
    """[P0] 将混淆矩阵转化为动作空间的先验概率分布

    理论基础:
    - 混淆矩阵 C[i][j] 表示标签 i 被误判为标签 j 的频率
    - 高混淆对 (i, j) 意味着当前 prompt 对 i/j 的区分度不够
    - 针对高混淆对生成的修改动作应获得更高的先验概率

    数学建模:
    - 设目标标签 t 的混淆向量为 c_t = [C[t][j] for j ≠ t]
    - 归一化为概率分布: p_confusion = softmax(c_t / τ), τ 为温度参数
    - 动作的混淆引导先验: P_action = α * P_llm + (1-α) * P_confusion_match
      其中 P_confusion_match 衡量该动作对高混淆标签的针对性
    """

    def __init__(self, confused_labels: List[str], confusion_rates: Optional[Dict[str, float]] = None,
                 alpha: float = 0.6, temperature: float = 1.0):
        """
        :param confused_labels: 按混淆率排序的混淆标签列表
        :param confusion_rates: {标签名: 混淆率} 字典
        :param alpha: LLM先验 vs 混淆先验的混合权重 (α * P_llm + (1-α) * P_confusion)
        :param temperature: 混淆分布的温度参数 (越小越集中于高混淆对)
        """
        self.confused_labels = confused_labels
        self.alpha = alpha
        self.temperature = temperature

        # 构建混淆率分布
        # [fix] 检查 confusion_rates 是否有非零值, 全零等同于无数据
        has_nonzero_rates = confusion_rates and any(v > 0 for v in confusion_rates.values())
        if has_nonzero_rates:
            self.confusion_dist = self._build_confusion_distribution(confusion_rates)
        elif confused_labels:
            # 没有有效混淆率数据 → 按排名位置构建递减权重
            # 排在前面的标签获得更高的先验 (假设混淆矩阵已按重要性排序)
            n = len(confused_labels)
            rates = {label: (n - i) / n for i, label in enumerate(confused_labels)}
            self.confusion_dist = self._build_confusion_distribution(rates)
            logger.warning(f"[ConfusionPrior] 混淆率全为0, 退化为基于排名的递减权重")
        else:
            self.confusion_dist = {}

        logger.info(f"[ConfusionPrior] 初始化完成: {len(self.confused_labels)} 个混淆标签, "
                     f"α={alpha}, τ={temperature}")
        if self.confusion_dist:
            top3 = sorted(self.confusion_dist.items(), key=lambda x: -x[1])[:3]
            logger.info(f"  Top-3 混淆分布: {top3}")

    def _build_confusion_distribution(self, rates: Dict[str, float]) -> Dict[str, float]:
        """将混淆率转化为概率分布 (softmax with temperature)"""
        if not rates:
            return {}
        labels = list(rates.keys())
        values = np.array([rates[l] for l in labels])
        # Softmax with temperature
        scaled = values / self.temperature
        exp_vals = np.exp(scaled - np.max(scaled))
        probs = exp_vals / exp_vals.sum()
        return {labels[i]: float(probs[i]) for i in range(len(labels))}

    def adjust_prior(self, action_description: str, llm_confidence: float) -> float:
        """计算混淆引导后的先验概率

        P_adjusted = α * P_llm + (1-α) * confusion_relevance

        confusion_relevance = Σ p_confusion[label] * I(label mentioned in action)
        """
        if not self.confusion_dist:
            return llm_confidence

        # 计算动作描述与混淆标签的相关度
        confusion_relevance = 0.0
        for label, prob in self.confusion_dist.items():
            if label in action_description:
                confusion_relevance += prob

        # 如果动作没有提及任何混淆标签，用均匀值
        if confusion_relevance == 0.0:
            confusion_relevance = 1.0 / max(len(self.confusion_dist), 1)

        # 混合
        adjusted = self.alpha * llm_confidence + (1 - self.alpha) * confusion_relevance
        return max(0.01, min(0.99, adjusted))


# ═══════════════════════════════════════════════════════════════════════════════
#  SearchStats - 搜索统计追踪器
# ═══════════════════════════════════════════════════════════════════════════════

class SearchStats:
    """[P2] 追踪搜索过程的统计信息, 用于论文实验分析"""

    def __init__(self):
        self.iteration_rewards: List[float] = []
        self.iteration_f1s: List[float] = []
        self.best_f1_history: List[float] = []  # 每次迭代后的全局最优 F1
        self.tree_depth_history: List[int] = []
        self.tree_size_history: List[int] = []
        self.dead_rate_history: List[float] = []
        self.frozen_count_history: List[int] = []
        self.level_a_pass_rate: List[float] = []
        self.level_b_pass_rate: List[float] = []
        self.timing: List[float] = []  # 每次迭代耗时

        self._level_a_total = 0
        self._level_a_pass = 0
        self._level_b_total = 0
        self._level_b_pass = 0

    def record_iteration(self, reward: float, f1: float, best_f1: float,
                         tree_depth: int, tree_size: int, dead_count: int,
                         total_count: int, frozen_count: int, elapsed: float):
        self.iteration_rewards.append(reward)
        self.iteration_f1s.append(f1)
        self.best_f1_history.append(best_f1)
        self.tree_depth_history.append(tree_depth)
        self.tree_size_history.append(tree_size)
        self.dead_rate_history.append(dead_count / max(total_count, 1))
        self.frozen_count_history.append(frozen_count)
        self.timing.append(elapsed)

    def record_level_a(self, passed: bool):
        self._level_a_total += 1
        if passed:
            self._level_a_pass += 1
        rate = self._level_a_pass / self._level_a_total
        self.level_a_pass_rate.append(rate)

    def record_level_b(self, passed: bool):
        self._level_b_total += 1
        if passed:
            self._level_b_pass += 1
        rate = self._level_b_pass / self._level_b_total
        self.level_b_pass_rate.append(rate)

    def save_to_json(self, filepath: str):
        data = {
            "iteration_rewards": self.iteration_rewards,
            "iteration_f1s": self.iteration_f1s,
            "best_f1_history": self.best_f1_history,
            "tree_depth_history": self.tree_depth_history,
            "tree_size_history": self.tree_size_history,
            "dead_rate_history": self.dead_rate_history,
            "frozen_count_history": self.frozen_count_history,
            "level_a_pass_rate": self.level_a_pass_rate,
            "level_b_pass_rate": self.level_b_pass_rate,
            "timing": self.timing,
            "total_time": sum(self.timing),
            "mean_time_per_iter": np.mean(self.timing) if self.timing else 0,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"📊 搜索统计已保存至: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MCTS - C-MCTS 搜索引擎 (完全重写)
# ═══════════════════════════════════════════════════════════════════════════════

class MCTS:
    """C-MCTS: Confusion-Guided MCTS for Prompt Optimization

    算法流程:
    1. Selection: 始终从ROOT出发, 用动态 PUCT 选择路径 (允许回溯探索兄弟分支)
    2. Expansion: Progressive Widening 控制扩展宽度 + 混淆引导先验
    3. Simulation: Level A 快速筛选 → Level B 精确验证 (双层评估)
    4. Backpropagation: 仅通过标准回传更新 N/W/Q (不再在扩展时污染)
    5. Best Path: 搜索结束后, 沿 Q 值最高路径提取最优 prompt
    """

    def __init__(self, root_prompt: str, evaluator, action_generator,
                 c1: float = 0.5, c2: float = 100.0,
                 coverage_ratio: float = 0.5,
                 pw_alpha: float = 0.5, pw_k0: int = 4,
                 confusion_prior: Optional[ConfusionPrior] = None):
        """
        :param root_prompt: 初始提示词
        :param evaluator: 评估器实例
        :param action_generator: 动作生成器实例
        :param c1, c2: 动态探索系数超参数
        :param coverage_ratio: FN 覆盖率递减目标 (从 core3 移植)
        :param pw_alpha: Progressive Widening 指数 (子节点数 ≤ k0 * N^α)
        :param pw_k0: Progressive Widening 初始宽度
        :param confusion_prior: 混淆引导先验实例
        """
        self.root = TreeNode(prompt_state=root_prompt, prior_prob=1.0)
        self.evaluator = evaluator
        self.action_generator = action_generator
        self.c1 = c1
        self.c2 = c2
        self.coverage_ratio = coverage_ratio
        self.pw_alpha = pw_alpha
        self.pw_k0 = pw_k0
        self.confusion_prior = confusion_prior

        # 已验证节点记录 (不再用于限制搜索起点, 仅用于输出)
        self.verified_nodes: List[TreeNode] = []  # 通过双层验证的节点
        self.remaining_fn_count: Optional[int] = None

        # [P2] 搜索统计
        self.stats = SearchStats()
        self.global_best_f1 = 0.0
        self.global_best_node: Optional[TreeNode] = None

        # 树统计
        self._total_nodes = 1
        self._dead_nodes = 0
        self._frozen_nodes = 0

    def search(self, iterations: int = 50) -> List[TreeNode]:
        """执行 C-MCTS 搜索"""
        logger.info("=" * 80)
        logger.info("🚀 C-MCTS 搜索启动")
        logger.info(f"   参数: iterations={iterations}, c1={self.c1}, c2={self.c2}")
        logger.info(f"   Progressive Widening: α={self.pw_alpha}, k0={self.pw_k0}")
        logger.info(f"   Coverage Ratio: {self.coverage_ratio}")
        logger.info("=" * 80)

        # ─── 初始化 baseline ───
        try:
            self.evaluator.set_baseline(self.root.prompt_state, level='A')
            self.evaluator.set_baseline(self.root.prompt_state, level='B')
            _, root_local = self.evaluator.evaluate(self.root.prompt_state, level='A')
            self.root.local_analysis = root_local
            self.remaining_fn_count = root_local.get("false_negatives", None)
            self.global_best_f1 = root_local.get("f1", 0.0)
            self.global_best_node = self.root
            logger.info(f"✓ Baseline 设置完成. Root F1={self.global_best_f1:.4f}, "
                         f"FN={self.remaining_fn_count}")
        except Exception as e:
            logger.error(f"❌ Baseline 设置失败: {e}")
            traceback.print_exc()
            return self.get_best_path()

        dead_count = 0
        successful_count = 0

        for i in range(iterations):
            iter_start = time.time()
            logger.info(f"\n{'=' * 80}")
            logger.info(f"🚀 C-MCTS Iteration {i + 1}/{iterations}")
            logger.info(f"{'=' * 80}")

            # ─── 1. Selection (始终从root开始, 允许回溯) ───
            node = self._select(self.root)
            logger.info(f"📍 [Selection] 选中: {node}")

            if node.is_dead:
                reward = 0.0001
                self._backpropagate(node, reward)
                dead_count += 1
                elapsed = time.time() - iter_start
                self.stats.record_iteration(
                    reward=reward, f1=0.0, best_f1=self.global_best_f1,
                    tree_depth=self._get_max_depth(), tree_size=self._total_nodes,
                    dead_count=dead_count, total_count=i + 1,
                    frozen_count=self._frozen_nodes, elapsed=elapsed)
                logger.info(f"💀 选中死亡节点, 跳过. ({elapsed:.1f}s)")
                continue

            # ─── 2. Expansion (Progressive Widening) ───
            if node.is_leaf() or self._should_widen(node):
                self._expand(node, iteration=i + 1)

            # 从扩展后的节点选择一个子节点
            if not node.is_leaf():
                node = self._select_child(node)
                logger.info(f"🌿 [Expansion→Selection] 进入子节点: {node}")

            # ─── 3. Simulation (双层评估) ───
            # [fix] 如果节点已被充分评估且无法扩展, 标记为叶子瓶颈
            if node.N > 0 and node.is_leaf() and not self._should_widen(node):
                logger.info(f"⚠️ 节点 {node.id} 已评估 {node.N} 次且无法扩展, "
                            f"尝试回退到父节点的兄弟分支")
                # 不再浪费评估预算, 直接用已有 Q 值回传
                reward = node.Q if node.Q > 0 else 0.0001
                f1 = node.local_analysis.get('f1', 0.0) if node.local_analysis else 0.0
            else:
                reward, f1 = self._simulate(node, iteration=i + 1)

            # ─── 4. Backpropagation ───
            self._backpropagate(node, reward)

            # ─── 5. Record verified nodes (不推进搜索起点, 保留回溯能力) ───
            if node.is_frozen and node not in self.verified_nodes:
                self.verified_nodes.append(node)
                if node.local_analysis:
                    self.remaining_fn_count = node.local_analysis.get(
                        "false_negatives", self.remaining_fn_count)
                self._frozen_nodes += 1
                logger.info(f"✅ [Verified] 新验证节点! 已验证={len(self.verified_nodes)}")

            # 更新全局最优
            if f1 > self.global_best_f1:
                self.global_best_f1 = f1
                self.global_best_node = node
                logger.info(f"🏆 [New Best] F1={f1:.4f}, Node={node}")

            if reward <= 0.0001:
                dead_count += 1
            else:
                successful_count += 1

            elapsed = time.time() - iter_start
            self.stats.record_iteration(
                reward=reward, f1=f1, best_f1=self.global_best_f1,
                tree_depth=self._get_max_depth(), tree_size=self._total_nodes,
                dead_count=dead_count, total_count=i + 1,
                frozen_count=self._frozen_nodes, elapsed=elapsed)

            logger.info(f"\n✨ Iteration {i + 1} 完成 ({elapsed:.1f}s). "
                         f"Reward={reward:.4f}, F1={f1:.4f}, Best_F1={self.global_best_f1:.4f}")
            logger.info(f"📊 {successful_count}✓ / {dead_count}✗ / {i + 1}总 "
                         f"(失败率: {dead_count / (i + 1) * 100:.1f}%)")

        # ─── 搜索完成总结 ───
        logger.info(f"\n{'#' * 80}")
        logger.info(f"🏁 C-MCTS 搜索完成")
        logger.info(f"   总迭代: {iterations}, 成功: {successful_count}, 失败: {dead_count}")
        logger.info(f"   成功率: {successful_count / max(iterations, 1) * 100:.1f}%")
        logger.info(f"   全局最优 F1: {self.global_best_f1:.4f}")
        logger.info(f"   已验证节点: {len(self.verified_nodes)}")
        logger.info(f"   搜索树大小: {self._total_nodes} 节点")
        logger.info(f"   总耗时: {sum(self.stats.timing):.1f}s")

        # 保存统计
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.stats.save_to_json(f"cmcts_stats_{ts}.json")

        return self.get_best_path()

    # ─── Selection ────────────────────────────────────────────────────────────

    def _select(self, start_node: TreeNode) -> TreeNode:
        """从 start_node 开始沿树下潜到叶子节点"""
        node = start_node
        while not node.is_leaf():
            next_node = node.best_child(self.c1, self.c2)
            if next_node is None:
                logger.info(f"⚠️ 所有子节点已死亡, 标记当前节点为死亡")
                node.is_dead = True
                self._dead_nodes += 1
                return node
            node = next_node
        return node

    def _select_child(self, node: TreeNode) -> TreeNode:
        """从非叶子节点选择一个子节点 (单步)"""
        best = node.best_child(self.c1, self.c2)
        if best is None:
            node.is_dead = True
            self._dead_nodes += 1
            return node
        return best

    # ─── Expansion ────────────────────────────────────────────────────────────

    def _should_widen(self, node: TreeNode) -> bool:
        """[P1] Progressive Widening: 判断是否应该继续扩展子节点

        允许扩展条件: len(children) < k0 * N^α
        """
        if node.is_dead or node.N == 0:
            return False
        max_children = self.pw_k0 * (node.N ** self.pw_alpha)
        return len(node.children) < max_children

    def _expand(self, node: TreeNode, iteration: int = 0):
        """扩展阶段: 生成候选动作 + FBPS 预评估 (修复版)"""
        logger.info(f"🌱 [Expand] 节点 {node.id} (N={node.N}, Q={node.Q:.4f}, "
                     f"children={len(node.children)})")

        # 1. 确保有错误分析
        parent_analysis = node.error_analysis
        if parent_analysis is None:
            logger.info("🧐 正在为节点生成分析报告...")
            try:
                _, parent_analysis = self.evaluator.evaluate(node.prompt_state, level='B')
                node.error_analysis = parent_analysis
                self._print_analysis(parent_analysis, f"Node_{node.id}")
                self.evaluator.save_errors_to_excel(
                    parent_analysis, iteration,
                    "Root" if node.parent is None else f"Node_{node.id}")
            except Exception as e:
                logger.error(f"❌ 分析报告生成失败: {e}")
                node.is_dead = True
                self._dead_nodes += 1
                return

        # 2. 决定生成数量
        existing = len(node.children)
        if existing == 0:
            k = self.pw_k0  # 首次扩展: 生成 k0 个
        else:
            k = 2  # 后续扩展 (Progressive Widening): 少量补充

        # 3. 生成候选动作 (generate_actions 内部已逐策略独立调用LLM)
        confused_labels = getattr(self.evaluator, 'confused_labels', [])
        try:
            logger.info(f"📞 调用 ActionGenerator 逐策略生成 {k} 个候选动作...")
            actions = self.action_generator.generate_actions(
                current_prompt=node.prompt_state,
                error_analysis=parent_analysis,
                confused_labels=confused_labels,
                k=k
            )
            logger.info(f"✓ 返回 {len(actions) if actions else 0} 个动作")
        except Exception as e:
            logger.error(f"❌ 动作生成失败: {e}\n{traceback.format_exc()}")
            actions = None

        if not actions:
            logger.warning("⚠️ 无动作生成, 跳过扩展")
            return

        # 4. 创建子节点 + 混淆引导先验调整
        new_children = []
        for i, action in enumerate(actions):
            # [P0] 混淆引导先验
            raw_confidence = action.get('confidence', 0.5)
            description = action.get('description', f'Action_{i}')

            if self.confusion_prior:
                adjusted_prior = self.confusion_prior.adjust_prior(description, raw_confidence)
            else:
                adjusted_prior = raw_confidence

            child = TreeNode(
                prompt_state=action.get('new_prompt', ''),
                parent=node,
                prior_prob=adjusted_prior,
                action_taken=description
            )
            node.children.append(child)
            new_children.append(child)
            self._total_nodes += 1

        # 5. [P0-fix] FBPS 预评估: 存为 prior_reward, 不设 N=1
        logger.info(f"🚀 [FBPS] 预评估 {len(new_children)} 个新子节点 (不污染N/Q)...")
        for child in new_children:
            try:
                reward, analysis = self.evaluator.evaluate(child.prompt_state, level='B')
                child.prior_reward = reward      # 预评估值, 不写入 N/Q
                child.prior_analysis = analysis   # 保存分析供后续使用
                child.error_analysis = analysis
                logger.info(f"  ✓ {child.action_taken[:30]}... prior={child.prior_reward:.4f}")
            except Exception as e:
                logger.error(f"  ❌ {child.action_taken[:30]}... {e}")
                child.prior_reward = 0.0001
                child.is_dead = True
                self._dead_nodes += 1

        logger.info(f"✓ [FBPS] 预评估完成, {len(new_children)} 个子节点已设置 prior_reward")

    # ─── Simulation (双层评估) ────────────────────────────────────────────────

    def _simulate(self, node: TreeNode, iteration: int = 0) -> Tuple[float, float]:
        """[P0+P1] 双层评估: Level A 快速筛选 → Level B 精确验证

        Returns:
            (reward, f1) - 奖励值和F1分数
        """
        node_display = node.action_taken if node.action_taken else 'Root'
        logger.info(f"🧪 [Simulate] {node_display}")

        try:
            # ─── Stage 1: Level A 局部快速评估 ───
            score_a, analysis_a = self.evaluator.evaluate(node.prompt_state, level='A')
            node.local_analysis = analysis_a
            f1_a = analysis_a.get('f1', 0.0)

            precision_a = analysis_a.get('precision', 0.0)
            recall_a = analysis_a.get('recall', 0.0)

            level_a_pass = precision_a >= 0.3 and recall_a >= 0.3  # 宽松阈值
            self.stats.record_level_a(level_a_pass)

            if not level_a_pass:
                logger.info(f"❌ Level A 失败: P={precision_a:.4f}, R={recall_a:.4f}")
                node.is_dead = True
                self._dead_nodes += 1
                return 0.0001, f1_a

            # 覆盖率检查
            if self.remaining_fn_count is not None and self.remaining_fn_count > 0:
                target_fn = self.remaining_fn_count * self.coverage_ratio
                current_fn = analysis_a.get("false_negatives", 0)
                if current_fn > target_fn:
                    logger.info(f"⚠️ 覆盖率不足: FN={current_fn} > target={target_fn:.0f}")
                    # 不标记为死亡, 但给予较低奖励
                    self.stats.record_level_b(False)
                    return max(0.01, score_a * 0.5), f1_a

            logger.info(f"✓ Level A 通过: P={precision_a:.4f}, R={recall_a:.4f}, F1={f1_a:.4f}")

            # ─── Stage 2: Level B 全局精确验证 ───
            score_b, analysis_b = self.evaluator.evaluate(node.prompt_state, level='B')
            node.global_analysis = analysis_b
            f1_b = analysis_b.get('f1', 0.0)

            violations = analysis_b.get('violations', [])
            is_safe = len(violations) == 0

            self.stats.record_level_b(is_safe)

            if not is_safe:
                logger.info(f"❌ Level B 失败: {violations}")
                # 不标记死亡, 但惩罚
                return max(0.0001, score_b - 0.3), f1_b

            # 通过双层验证 → 冻结
            node.error_analysis = analysis_b
            node.is_frozen = True

            logger.info(f"✅ 双层验证通过! Score={score_b:.4f}, F1={f1_b:.4f}")
            return score_b, f1_b

        except Exception as e:
            logger.error(f"❌ Simulation 异常: {e}\n{traceback.format_exc()}")
            node.is_dead = True
            self._dead_nodes += 1
            return 0.0001, 0.0

    # ─── Backpropagation ──────────────────────────────────────────────────────

    def _backpropagate(self, node: TreeNode, reward: float):
        """[P0-fix] 标准回传: 只通过此路径更新 N/W/Q

        Q = W / N, 避免旧版 Q = (Q*(N-1)+R)/N 的浮点累积误差
        """
        current = node
        while current is not None:
            current.N += 1
            current.W += reward
            current.Q = current.W / current.N
            current = current.parent

    # ─── Best Path ────────────────────────────────────────────────────────────

    def get_best_path(self) -> List[TreeNode]:
        """获取最佳路径: 始终沿 Q 值最大路径追踪

        MCTS 的核心优势: 经过多次迭代后, Q 值自然收敛到最优路径,
        无需人为冻结。N 值高的节点被充分评估, Q 值更可靠。
        """
        node = self.root
        path = [node]
        logger.info(f"\n🎯 追踪 Q 值最大路径 (经过 {self.root.N} 次迭代收敛)...")

        while not node.is_leaf():
            if not node.children:
                break
            # 选 Q 值最高的存活子节点
            alive = [c for c in node.children if not c.is_dead]
            if not alive:
                break
            best_child = max(alive, key=lambda x: x.Q if x.N > 0 else (x.prior_reward or 0))
            node = best_child
            path.append(node)
            logger.info(f"  Level {len(path) - 1}: {node}")

        logger.info(f"\n✅ 最佳路径 (深度={len(path) - 1})")
        if len(path) > 1:
            logger.info(f"  Root → Best: Q={path[-1].Q:.4f}")
        return path

    # ─── 辅助方法 ─────────────────────────────────────────────────────────────

    def _get_max_depth(self) -> int:
        """计算搜索树最大深度"""
        def _depth(node):
            if node.is_leaf():
                return node.depth
            return max(_depth(c) for c in node.children)
        return _depth(self.root)

    def _print_analysis(self, analysis: Dict, node_name: str):
        """打印评估分析详情"""
        logger.info(f"\n--- 🧠 {node_name} 分析 ---")
        logger.info(f"| 标签: {analysis.get('target_label')}")
        logger.info(f"| 样本: {analysis.get('total_samples')}")
        logger.info(f"| P={analysis.get('precision', 0):.4f} "
                     f"R={analysis.get('recall', 0):.4f} "
                     f"F1={analysis.get('f1', 0):.4f}")
        logger.info(f"| FN={analysis.get('false_negatives', 0)} "
                     f"FP={analysis.get('false_positives', 0)}")

        for key in ('fn_examples', 'fp_examples'):
            examples = analysis.get(key, [])
            if examples:
                label = "漏检" if "fn" in key else "误检"
                logger.info(f"| {label}案例:")
                for i, ex in enumerate(examples[:3]):
                    logger.info(f"|   [{i + 1}] {str(ex)[:100]}...")

        violations = analysis.get('violations', [])
        if violations:
            logger.info(f"| ⚠️ 违规: {', '.join(violations)}")
        else:
            logger.info(f"| ✓ 铁律检查通过")
        logger.info("-" * 40)

    def get_tree_summary(self) -> Dict:
        """获取搜索树的摘要统计 (用于论文)"""
        all_nodes = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            all_nodes.append(node)
            queue.extend(node.children)

        alive = [n for n in all_nodes if not n.is_dead]
        frozen = [n for n in all_nodes if n.is_frozen]

        return {
            "total_nodes": len(all_nodes),
            "alive_nodes": len(alive),
            "dead_nodes": len(all_nodes) - len(alive),
            "frozen_nodes": len(frozen),
            "max_depth": max(n.depth for n in all_nodes),
            "avg_branching_factor": np.mean([len(n.children) for n in all_nodes if n.children]),
            "verified_nodes": len(self.verified_nodes),
            "global_best_f1": self.global_best_f1,
            "best_node_depth": self.global_best_node.depth if self.global_best_node else 0,
        }
