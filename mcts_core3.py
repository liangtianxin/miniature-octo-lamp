import math
import copy
import numpy as np

class TreeNode:
    def __init__(self, prompt_state, parent=None, prior_prob=0.0, action_taken=None):
        """
        MCTS 节点
        :param prompt_state: 当前节点的提示词状态 (List[str] 或 str)
        :param parent: 父节点
        :param prior_prob: P(s, a) - DeepSeek 给出的置信度
        :param action_taken: 到达此节点所采取的动作 (描述字符串)
        """
        self.prompt_state = prompt_state
        self.parent = parent
        self.children = []
        self.action_taken = action_taken
        
        # MCTS 统计量
        self.N = 0  # 访问次数
        self.Q = 0.0  # 平均价值 (Reward)
        self.P = prior_prob  # 先验概率
        
        self.is_dead = False  # 如果 Level A 评估失败，标记为死亡
        self.is_frozen = False  # 是否已冻结（通过全局验证）
        self.error_analysis = None  # 存储该节点的错误分析报告，供子节点扩展使用
        self.local_analysis = None  # Level A 分析
        self.global_analysis = None  # Level B 分析

    def is_leaf(self):
        return len(self.children) == 0

    def best_child(self, c_puct=1.0):
        """
        根据 PUCT 公式选择最佳子节点
        PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        """
        best_score = -float('inf')
        best_node = None
        
        # 父节点访问次数的平方根
        sqrt_n_parent = math.sqrt(self.N)
        
        for child in self.children:
            if child.is_dead:
                continue
                
            # PUCT 公式
            exploit = child.Q
            explore = c_puct * child.P * sqrt_n_parent / (1 + child.N)
            score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_node = child
                
        return best_node

class MCTS:
    def __init__(self, root_prompt, evaluator, action_generator, c_puct=1.0, coverage_ratio=0.5):
        """
        MCTS 搜索引擎
        :param root_prompt: 初始提示词
        :param evaluator: 评估器实例 (Evaluator)
        :param action_generator: 动作生成器实例 (ActionGenerator)
        :param c_puct: 探索系数
        """
        self.root = TreeNode(prompt_state=root_prompt, prior_prob=1.0)
        self.evaluator = evaluator
        self.action_generator = action_generator
        self.c_puct = c_puct
        self.coverage_ratio = coverage_ratio
        self.frozen_node = self.root
        self.chain = [self.root]
        self.remaining_fn_count = None
        
    def search(self, iterations=50):
        """
        执行 MCTS 搜索
        :param iterations: 迭代次数
        """
        # 初始化 Level A / Level B 基线（以根提示词为基准）
        try:
            self.evaluator.set_baseline(self.root.prompt_state, level='A')
            self.evaluator.set_baseline(self.root.prompt_state, level='B')
            _, root_local = self.evaluator.evaluate(self.root.prompt_state, level='A')
            self.root.local_analysis = root_local
            self.remaining_fn_count = root_local.get("false_negatives", None)
        except Exception as e:
            print(f"❌ Failed to set baselines: {e}")
            return self.get_best_path()

        dead_count = 0
        successful_count = 0
        
        for i in range(iterations):
            print(f"\n--- MCTS Iteration {i+1}/{iterations} ---")
            
            # 1. Selection
            node = self._select(self.frozen_node)
            
            # 2. Expansion
            # 如果节点未被评估过且不是死节点，则扩展
            if not node.is_dead:
                # 如果是叶子节点，尝试扩展
                if node.is_leaf():
                    self._expand(node)
                
                # 如果扩展后仍是叶子（无法扩展），则回溯当前值
                # 否则选择一个新子节点进行模拟
                if not node.is_leaf():
                    node = self._select(node) # 移动到新扩展的子节点
            
            # 3. Simulation (Evaluation)
            reward = self._simulate(node)
            
            # 4. Backpropagation
            self._backpropagate(node, reward)

            # 冻结提示词链：通过全局验证则冻结为当前层级
            if node.is_frozen and node is not self.frozen_node:
                self.frozen_node = node
                self.chain.append(node)
                if node.local_analysis:
                    self.remaining_fn_count = node.local_analysis.get("false_negatives", self.remaining_fn_count)
                print(f"🧊 Freeze success. Chain length={len(self.chain)-1}")
            
            if reward < 0:
                dead_count += 1
            else:
                successful_count += 1
            
            print(f"Iteration {i+1} finished. Reward: {reward:.4f}")
            print(f"📊 Progress: {successful_count} successful / {dead_count} dead / {i+1} total (Dead rate: {dead_count/(i+1)*100:.1f}%)")

        print(f"\n=== Search Summary ===")
        print(f"Total iterations: {iterations}")
        print(f"Successful nodes: {successful_count}")
        print(f"Dead nodes: {dead_count}")
        print(f"Success rate: {successful_count/iterations*100:.1f}%")
        
        return self.get_best_path()

    def _select(self, node):
        """
        选择阶段：一直下潜直到叶子节点
        """
        while not node.is_leaf():
            next_node = node.best_child(self.c_puct)
            if next_node is None:
                # 所有子节点都 Dead 了，返回当前节点并标记
                print(f"⚠️ All children are dead at this node. Marking as dead.")
                node.is_dead = True
                return node
            node = next_node
        return node

    def _expand(self, node, retry_count=0):
        """
        扩展阶段：调用 DeepSeek 生成候选动作
        :param retry_count: 重试次数（当生成的动作质量不高时）
        """
        print(f"Expanding node (N={node.N}, Q={node.Q:.4f}, retry={retry_count})...")
        
        # 1. 获取错误分析（必须有才能生成动作）
        parent_analysis = node.error_analysis
        if parent_analysis is None:
            print("🔍 Generating initial analysis for expansion...")
            try:
                _, parent_analysis = self.evaluator.evaluate(node.prompt_state, level='B')
                node.error_analysis = parent_analysis
                print(f"✓ Analysis generated. P={parent_analysis.get('precision', 0):.4f}, R={parent_analysis.get('recall', 0):.4f}")
            except Exception as e:
                print(f"❌ Failed to generate analysis: {e}")
                node.is_dead = True
                return

        # 2. 调用 ActionGenerator 生成候选动作
        confused_labels = getattr(self.evaluator, 'confused_labels', [])
        
        try:
            actions = self.action_generator.generate_actions(
                current_prompt=node.prompt_state,
                error_analysis=parent_analysis,
                confused_labels=confused_labels,
                k=6
            )
        except Exception as e:
            print(f"❌ Failed to generate actions: {e}")
            actions = None
            
        if not actions:
            print("⚠️ No actions generated. Skipping expansion.")
            return

        # 3. 创建子节点
        print(f"📝 Creating {len(actions)} child nodes...")
        for i, action in enumerate(actions):
            child = TreeNode(
                prompt_state=action.get('new_prompt', ''),
                parent=node,
                prior_prob=action.get('confidence', 0.5),
                action_taken=action.get('description', f'Action_{i}')
            )
            node.children.append(child)
        
        print(f"✓ Successfully expanded {len(node.children)} children.")

    def _simulate(self, node):
        """
        模拟阶段：先进行 Level A 局部评估，再进行 Level B 全局验证
        """
        print(f"🔬 Simulating node: {node.action_taken if node.action_taken else 'Root'}")
        
        try:
            # 3.1 Level A 局部评估：P/R >= 0.5
            score_a, analysis_a = self.evaluator.evaluate(node.prompt_state, level='A')
            node.local_analysis = analysis_a
            if analysis_a.get('precision', 0.0) < 0.5 or analysis_a.get('recall', 0.0) < 0.5:
                print(f"❌ Level A failed. P={analysis_a.get('precision', 0):.4f}, R={analysis_a.get('recall', 0):.4f}")
                return self._refine_on_failure(node, analysis_a, stage="local")

            # 覆盖率层级推进：剩余 FN 至少减少 50%
            if self.remaining_fn_count is not None:
                target_fn = self.remaining_fn_count * self.coverage_ratio
                if analysis_a.get("false_negatives", 0) > target_fn:
                    print(f"❌ Coverage failed. FN={analysis_a.get('false_negatives', 0)} > target_fn={target_fn:.2f}")
                    return self._refine_on_failure(node, analysis_a, stage="local")

            # 3.2 Level B 全局验证：混淆标签准确率不下降
            score_b, analysis_b = self.evaluator.evaluate(node.prompt_state, level='B')
            node.global_analysis = analysis_b
            if not analysis_b.get('is_safe', True):
                print(f"❌ Level B failed. Violations: {analysis_b.get('violations', [])}")
                return self._refine_on_failure(node, analysis_b, stage="global")
            
            # 保存错误分析供下次扩展使用
            node.error_analysis = analysis_b
            node.is_frozen = True
            
            print(f"✓ Score={score_b:.4f}, P={analysis_b.get('precision', 0):.4f}, R={analysis_b.get('recall', 0):.4f}")
            return score_b
            
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            node.is_dead = True
            return -1.0

    def _refine_on_failure(self, node, analysis, stage: str):
        """
        局部失败/全局失败时进行智能微调回溯，参考 FP/FN 生成修正提示词
        """
        print(f"🔁 Refining on {stage} failure...")
        try:
            actions = self.action_generator.generate_actions(
                current_prompt=node.prompt_state,
                error_analysis=analysis,
                confused_labels=getattr(self.evaluator, 'confused_labels', []),
                k=2,
                refine=True
            )
        except Exception as e:
            print(f"❌ Failed to refine: {e}")
            node.is_dead = True
            return -1.0

        for action in actions or []:
            refined_prompt = action.get('new_prompt', '')
            if not refined_prompt:
                continue

            # 重新执行 Level A
            score_a, analysis_a = self.evaluator.evaluate(refined_prompt, level='A')
            if analysis_a.get('precision', 0.0) < 0.5 or analysis_a.get('recall', 0.0) < 0.5:
                continue
            if self.remaining_fn_count is not None:
                target_fn = self.remaining_fn_count * self.coverage_ratio
                if analysis_a.get("false_negatives", 0) > target_fn:
                    continue

            # 重新执行 Level B
            score_b, analysis_b = self.evaluator.evaluate(refined_prompt, level='B')
            if not analysis_b.get('is_safe', True):
                continue

            # 通过验证，更新节点为修正后的提示词
            node.prompt_state = refined_prompt
            node.action_taken = f"Refined: {action.get('description', '')}"
            node.error_analysis = analysis_b
            node.local_analysis = analysis_a
            node.global_analysis = analysis_b
            node.is_frozen = True
            print(f"✅ Refinement success. Score={score_b:.4f}")
            return score_b

        node.is_dead = True
        return -1.0

    def _backpropagate(self, node, reward):
        """
        回溯阶段：更新路径上的 N 和 Q
        """
        while node is not None:
            node.N += 1
            # Q 值更新公式: Q_new = (Q_old * (N-1) + Reward) / N
            node.Q = (node.Q * (node.N - 1) + reward) / node.N
            node = node.parent

    def get_best_path(self):
        """
        获取最佳路径：选择 Q 值最高的路径（这代表最优的提示词）
        不再使用访问次数 N 作为判断标准，因为我们已保证所有存活的节点都会被评估。
        """
        if hasattr(self, "chain") and len(self.chain) > 1:
            return self.chain
        node = self.root
        path = [node]
        
        print(f"\n🎯 Tracing best path from root...")
        
        while not node.is_leaf():
            # 选择 Q 值最高的子节点（这代表最优的提示词）
            if not node.children:
                print(f"  Leaf reached (no children)")
                break
                
            # 按 Q 值排序，选择最好的
            best_child = max(node.children, key=lambda x: x.Q)
            
            node = best_child
            path.append(node)
            print(f"  Level {len(path)-1}: Q={node.Q:.4f}, N={node.N}, Action: {node.action_taken[:50] if node.action_taken else 'Root'}")
        
        print(f"\n✅ Best path found (depth={len(path)-1}):")
        print(f"  Root: Q={path[0].Q:.4f}")
        if len(path) > 1:
            print(f"  Best child: Q={path[-1].Q:.4f}, Action: {path[-1].action_taken[:100] if path[-1].action_taken else 'N/A'}")
        else:
            print(f"  ⚠️ No children found! Returning root node.")
        
        return path
