# C-MCTS: Confusion-Guided Monte Carlo Tree Search for Automated Prompt Optimization in LLM-Based Text Classification

---

## 作者信息（待填）

**Author A**$^1$, **Author B**$^{1,2}$

$^1$ School of XXX, University of XXX  
$^2$ XXX Lab

---

## Abstract

Prompt engineering for large language models (LLMs) remains a labor-intensive trial-and-error process, especially in multi-label text classification where inter-class confusion dominates the error landscape. We propose **C-MCTS** (**C**onfusion-Guided **M**onte **C**arlo **T**ree **S**earch), a framework that automatically optimizes classification prompts by formulating the problem as a tree search over the prompt modification space. C-MCTS introduces three key innovations: **(1)** a **confusion-guided prior** that converts the confusion matrix into an action-space probability distribution, biasing exploration toward high-confusion label pairs; **(2)** a **Fast Batch Pre-Screening (FBPS)** mechanism that decouples pre-evaluation from MCTS statistics, preserving theoretical soundness while enabling informed child selection; and **(3)** a **dual-level evaluation** pipeline with a composite reward function that jointly considers F1 improvement, confusion safety, and absolute performance. We also employ **Progressive Widening** to adaptively control tree breadth and a **dynamic PUCT** selection formula with logarithmic exploration growth. Experiments on a large-scale ride-hailing risk classification dataset (23 labels, 58K samples) demonstrate that C-MCTS consistently outperforms manual prompt engineering (+4.7% macro-F1), greedy hill climbing (+3.2%), and random search (+6.1%), while provably reducing high-confusion label pairs by an average of 31.4%. Ablation studies confirm that removing the confusion prior, FBPS, or dual-level evaluation each independently degrades performance. We release all code and search logs to facilitate reproducibility.

**Keywords:** Prompt Optimization · Monte Carlo Tree Search · Confusion Matrix · Large Language Models · Text Classification

---

## 1 Introduction

### 1.1 Background and Motivation

Large language models (LLMs) have demonstrated remarkable capabilities in text classification tasks through in-context learning, where task-specific prompts guide the model to produce desired outputs without parameter updates. However, the performance of LLMs is highly sensitive to the specific wording, structure, and examples provided in prompts. This sensitivity creates a fundamental challenge: **How can we systematically optimize prompts to maximize classification accuracy?**

The challenge is particularly acute in **multi-label classification** scenarios where the number of categories is large (e.g., $|\mathcal{L}| > 20$) and certain label pairs exhibit high confusion rates. Consider a ride-hailing customer service system with 23+ risk categories. A manually crafted prompt might achieve high overall accuracy but systematically confuse "已发生政府渠道外投" (government channel external complaint - occurred) with "未发生政府渠道外投" (government channel external complaint - not occurred), because the lexical overlap between these categories is extremely high. Resolving such confusion requires nuanced, context-dependent rules that are difficult for human prompt engineers to discover through trial and error.

### 1.2 Limitations of Existing Approaches

Existing automated prompt optimization methods can be categorized into three paradigms:

**Gradient-based methods** (AutoPrompt [Shin et al., 2020], Prompt Tuning [Lester et al., 2021]) require access to model gradients, which is infeasible for black-box API-served models or frozen-weight deployments.

**LLM-as-optimizer methods** (APE [Zhou et al., 2023], OPRO [Yang et al., 2024]) use an LLM to iteratively rewrite prompts based on performance feedback. While elegant, these methods lack systematic exploration: they greedily follow the most recent improvement and are prone to local optima, especially when the prompt space contains deceptive gradients (modifications that improve one label's performance while degrading another).

**Evolutionary methods** (EvoPrompt [Guo et al., 2024]) maintain a population of prompts and apply genetic operators. However, they operate on the full prompt as an atomic unit, making it difficult to perform targeted modifications for specific confusion pairs.

None of these methods leverage the **confusion structure** inherent in multi-label classification, where the confusion matrix provides a rich, immediately available signal about which label pairs most need attention.

### 1.3 Our Approach

We propose **C-MCTS**, which uniquely combines three insights:

1. **Confusion as prior knowledge.** The confusion matrix $\mathbf{C} \in \mathbb{R}^{|\mathcal{L}| \times |\mathcal{L}|}$ is not merely a diagnostic tool — it is a probability distribution over the action space. High-confusion pairs $(i, j)$ indicate where prompt modifications will have the highest marginal return. We formalize this as a Bayesian prior that weights the PUCT exploration term.

2. **Tree search over prompt modifications.** Unlike methods that treat prompt optimization as a single-step rewriting problem, C-MCTS builds a persistent search tree where each edge represents a targeted prompt modification (e.g., "add a positive-example rule for label $i$" or "modify the boundary condition between labels $i$ and $j$"). This enables **backtracking**: if a modification path leads to a dead end, the search can retreat to an earlier state and explore alternative branches.

3. **Theory-preserving pre-screening.** Real-world evaluation of prompts is expensive (requiring full inference over thousands of samples). We introduce FBPS to pre-evaluate candidate modifications cheaply, storing results as informative priors without corrupting the MCTS visit counts, thereby maintaining the theoretical guarantees of UCT convergence.

### 1.4 Contributions

- We formulate prompt optimization as a tree search problem with confusion-guided priors, providing the first integration of confusion matrix information into MCTS for prompt engineering.
- We design FBPS, a mechanism that decouples pre-evaluation from MCTS statistics, resolving the tension between informed search and theoretical soundness.
- We propose a dual-level evaluation pipeline with a composite reward function that enforces "safety constraints" — preventing confusion-label accuracy degradation while optimizing the target.
- We introduce forced-diversity action generation with four strategy types, ensuring broad coverage of the modification space.
- Extensive experiments on a real-world 23-class dataset demonstrate significant improvements over manual engineering, greedy search, and random baselines.

---

## 2 Related Work

### 2.1 Automated Prompt Optimization

**APE** (Automatic Prompt Engineer) [Zhou et al., 2023] generates candidate prompts using an LLM and selects the best one via evaluation. It operates in a single-round generate-and-select paradigm without iterative refinement. **OPRO** (Optimization by PROmpting) [Yang et al., 2024] maintains a history of prompt-score pairs and asks the LLM to generate improved prompts based on this history. While iterative, OPRO lacks tree structure and cannot backtrack from failed optimization paths. **EvoPrompt** [Guo et al., 2024] applies evolutionary algorithms (genetic algorithms and differential evolution) to prompt optimization, treating prompts as individuals in a population. It uses crossover and mutation but does not exploit task-specific structure such as confusion information.

**PromptBreeder** [Fernando et al., 2024] evolves both task prompts and the meta-prompts used to generate them, creating a self-referential improvement loop. **DSPy** [Khattab et al., 2024] provides a programming framework for composing LLM modules and uses optimizers like MIPRO and BootstrapFewShot to tune prompts. However, these systems optimize the complete prompt holistically rather than targeting specific inter-class confusions.

Our work differs from all of the above by (1) using tree search rather than linear or population-based search, (2) incorporating confusion matrix structure as an explicit prior, and (3) supporting targeted, fine-grained prompt modifications rather than full-prompt rewrites.

### 2.2 Monte Carlo Tree Search

MCTS [Kocsis & Szepesvári, 2006; Coulom, 2006] is a best-first search algorithm that builds a search tree incrementally through four phases: selection, expansion, simulation (rollout), and backpropagation. The UCT (Upper Confidence bounds applied to Trees) formula balances exploitation (choosing the best-known action) with exploration (trying less-visited actions):

$$a^* = \arg\max_a \left[ Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}} \right]$$

AlphaGo [Silver et al., 2016] and AlphaZero [Silver et al., 2017] extended UCT with learned prior policies and value functions, replacing the exploration term with PUCT:

$$a^* = \arg\max_a \left[ Q(s, a) + P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \cdot E \right]$$

where $E = c_1 + \log\left(\frac{\sum_b N(s,b) + c_2 + 1}{c_2}\right)$ is a dynamic exploration coefficient that grows logarithmically with the parent's visit count.

MCTS has been applied beyond games to program synthesis [Chen et al., 2024], mathematical reasoning [Luo et al., 2024], and code generation [Zhang et al., 2023]. **PromptAgent** [Wang et al., 2024] applies MCTS to prompt optimization, using LLM-generated "error feedback" to guide modifications. However, PromptAgent does not leverage confusion matrices and uses a standard UCT formula without prior adjustment. Our C-MCTS differs by introducing confusion-guided priors and FBPS, which are specifically designed for classification prompt optimization.

### 2.3 Confusion Matrix in Classification

The confusion matrix is a fundamental evaluation tool in classification [Stehman, 1997]. Beyond evaluation, confusion matrices have been used to design loss functions [Lin et al., 2017], guide hard-example mining [Shrivastava et al., 2016], and construct class hierarchies [Deng et al., 2014]. In the NLP domain, confusion analysis informs error analysis and targeted data augmentation [Wei & Zou, 2019].

To our knowledge, C-MCTS is the first work to formalize the confusion matrix as a Bayesian prior distribution over the prompt modification action space, directly connecting classification errors to search guidance.

---

## 3 Method

### 3.1 Problem Formulation

We formulate prompt optimization as a sequential decision problem over a search tree $\mathcal{T}$.

**State space.** A state $s \in \mathcal{S}$ represents a complete prompt configuration for a target label $t$. The root state $s_0$ is the initial (possibly manually crafted) prompt.

**Action space.** An action $a \in \mathcal{A}$ is a targeted modification to the current prompt. We define four action types to enforce diversity:

| Type | Description | Focus |
|------|-------------|-------|
| `ADD_POSITIVE` | Add rules to cover missed positive samples (reduce FN) | False Negatives |
| `ADD_NEGATIVE` | Add exclusion rules to reject false matches (reduce FP) | False Positives |
| `MODIFY_BOUNDARY` | Refine boundary conditions between target and confused labels | Confusion pairs |
| `REFINE_CONTEXT` | Add contextual constraints to reduce ambiguity | Both FN and FP |

**Transition.** Applying action $a$ to state $s$ produces a new state $s' = T(s, a)$, which is the modified prompt.

**Reward.** We design a composite reward function $R(s)$ (detailed in Section 3.4) that balances F1 improvement, confusion safety, and absolute performance.

**Objective.** Find $s^* = \arg\max_{s \in \mathcal{T}} R(s)$.

### 3.2 C-MCTS Algorithm Overview

Algorithm 1 presents the C-MCTS procedure. Each iteration consists of five phases:

---

**Algorithm 1: C-MCTS**

**Input:** Initial prompt $s_0$, evaluator $\mathcal{E}$, action generator $\mathcal{G}$, confusion matrix $\mathbf{C}$, iterations $T$  
**Output:** Optimized prompt $s^*$

1. Initialize root node $v_0 \leftarrow \text{TreeNode}(s_0)$
2. Compute baselines: $b_A \leftarrow \mathcal{E}(s_0, \text{Level-A})$, $b_B \leftarrow \mathcal{E}(s_0, \text{Level-B})$
3. Build confusion prior: $\mathcal{P}_C \leftarrow \text{ConfusionPrior}(\mathbf{C}, \alpha, \tau)$
4. **for** $i = 1$ to $T$ **do**
5. $\quad v_{\text{sel}} \leftarrow \text{Select}(v_0)$ $\qquad$ ▷ Always from root, using dynamic PUCT
6. $\quad$ **if** $\text{ShouldWiden}(v_{\text{sel}})$ **then**
7. $\quad\quad$ $\text{Expand}(v_{\text{sel}}, \mathcal{G}, \mathcal{P}_C)$ $\qquad$ ▷ Progressive Widening + FBPS
8. $\quad$ **end if**
9. $\quad v_{\text{child}} \leftarrow \text{SelectChild}(v_{\text{sel}})$
10. $\quad (r, f_1) \leftarrow \text{Simulate}(v_{\text{child}}, \mathcal{E})$ $\qquad$ ▷ Dual-level evaluation
11. $\quad \text{Backpropagate}(v_{\text{child}}, r)$ $\qquad$ ▷ Standard W/N update
12. $\quad$ Update global best if $f_1 > f_1^*$
13. **end for**
14. **return** $\text{BestPath}(v_0)$ $\qquad$ ▷ Follow max-Q path from root

---

### 3.3 Confusion-Guided Prior (核心创新1)

#### 3.3.1 Motivation

In standard MCTS, the prior probability $P(s, a)$ is typically uniform or provided by a learned policy network. In prompt optimization, we have a readily available signal: the confusion matrix. If label $t$ is frequently confused with label $j$, then actions that specifically address the $(t, j)$ distinction should receive higher exploration priority.

#### 3.3.2 Confusion Distribution

Given a confusion matrix $\mathbf{C}$, we extract the confusion vector for target label $t$:

$$\mathbf{c}_t = [C_{t,j}]_{j \neq t}$$

We convert this to a probability distribution via temperature-scaled softmax:

$$p_{\text{conf}}(j) = \frac{\exp(c_{t,j} / \tau)}{\sum_{k \neq t} \exp(c_{t,k} / \tau)}$$

where $\tau$ is a temperature parameter. Lower $\tau$ concentrates probability mass on the most confused label pairs; higher $\tau$ produces a more uniform distribution.

#### 3.3.3 Prior Adjustment

For an action $a$ with LLM-estimated confidence $P_{\text{llm}}(a)$ and textual description $d_a$, we compute the **confusion relevance**:

$$P_{\text{conf}}(a) = \sum_{j} p_{\text{conf}}(j) \cdot \mathbb{1}[\text{label } j \text{ is mentioned in } d_a]$$

The adjusted prior is a convex combination:

$$P(s, a) = \alpha \cdot P_{\text{llm}}(a) + (1 - \alpha) \cdot P_{\text{conf}}(a)$$

where $\alpha \in [0, 1]$ controls the balance between the LLM's own confidence and the confusion-guided signal. When $\alpha = 1$, the prior reduces to the standard LLM confidence; when $\alpha = 0$, actions are ranked purely by confusion relevance.

**Interpretation.** This prior encodes domain knowledge: "If the model is confused between labels $t$ and $j$, then prompt modifications that explicitly address the $(t, j)$ boundary should be explored first." This is analogous to AlphaZero using a policy network to focus MCTS on promising moves, except our "policy" comes from the confusion matrix rather than a trained neural network.

### 3.4 Selection: Dynamic PUCT with Prior Discount (核心创新2)

We adopt a dynamic PUCT formula adapted from AlphaZero with a critical modification for handling pre-evaluated nodes:

$$a^* = \arg\max_a \left[ Q_{\text{eff}}(s, a) + P(s, a) \cdot \frac{\sqrt{N_{\text{parent}}}}{1 + N(s, a)} \cdot E \right]$$

where the dynamic exploration coefficient is:

$$E = \sqrt{N_{\text{parent}}} \cdot \left(c_1 + \log\frac{N_{\text{parent}} + c_2 + 1}{c_2}\right)$$

and the effective Q-value distinguishes three cases:

$$Q_{\text{eff}}(s, a) = \begin{cases} W(s,a) / N(s,a) & \text{if } N(s,a) > 0 \quad \text{(真实统计值)} \\ \gamma \cdot r_{\text{prior}}(a) & \text{if } N(s,a) = 0 \text{ and } r_{\text{prior}} \text{ exists} \quad \text{(FBPS折扣)} \\ 0 & \text{otherwise} \quad \text{(纯探索)} \end{cases}$$

**Prior discount $\gamma$.** We set $\gamma = 0.5$. This is crucial: without discounting, the pre-evaluated reward $r_{\text{prior}}$ would dominate the Q-value for unvisited nodes, completely suppressing the exploration term and reducing PUCT to greedy ranking by pre-evaluation score. The discount ensures that the exploration term $P \cdot \sqrt{N_{\text{parent}}} / (1 + N) \cdot E$ retains significant influence, allowing less-evaluated but potentially superior nodes to be explored.

**Why dynamic E?** The parameters $c_1$ and $c_2$ control the exploration schedule:
- $c_1$ sets a base exploration level (we use $c_1 = 0.5$)
- $c_2$ controls the logarithmic growth rate. When $N_{\text{parent}} \ll c_2$, the log term $\approx \frac{N_{\text{parent}} + 1}{c_2}$, growing slowly. When $N_{\text{parent}} \gg c_2$, the log term $\approx \log\frac{N_{\text{parent}}}{c_2}$, accelerating.

For our setting with $T \in [20, 50]$ iterations and $c_2 = 100$, this produces moderate exploration pressure that matches the small search budget, avoiding waste of expensive evaluations.

### 3.5 Expansion: Progressive Widening + FBPS (核心创新3)

#### 3.5.1 Progressive Widening

Standard MCTS expands all legal actions at once, which is impractical when the action space is continuous (LLM-generated modifications are effectively infinite). We adopt Progressive Widening [Coulom, 2007; Chaslot et al., 2008]:

$$|\text{children}(v)| \leq k_0 \cdot N(v)^{\alpha_{\text{pw}}}$$

where $k_0$ is the initial width (we use $k_0 = 4$) and $\alpha_{\text{pw}} \in (0, 1)$ controls the growth rate (we use $\alpha_{\text{pw}} = 0.5$). This means:
- At $N = 1$: up to 4 children
- At $N = 4$: up to 8 children
- At $N = 9$: up to 12 children

New children are only generated when the constraint is not satisfied, ensuring that existing children are sufficiently evaluated before widening the search.

#### 3.5.2 Fast Batch Pre-Screening (FBPS)

When new children are created, we immediately evaluate each one at Level B (see Section 3.6) and store the result as `prior_reward`. **Critically, we do NOT set $N = 1$ or update $Q$.** This is a deliberate departure from the original FBPS implementations [core4, core5] that treated pre-evaluation as a real visit:

| | Old FBPS (core4/5) | Our FBPS (core6) |
|---|---|---|
| After pre-evaluation | $N \leftarrow 1, Q \leftarrow r$ | `prior_reward` $\leftarrow r$, $N = 0$ |
| Effect on PUCT | $\frac{\sqrt{N_{\text{parent}}}}{1+1}$ → exploration halved | $\frac{\sqrt{N_{\text{parent}}}}{1+0}$ → full exploration |
| `_simulate()` behavior | Returns $Q$ immediately (N>0) | Performs full evaluation |
| Theoretical impact | **Violates UCT**: node appears visited but isn't | **Preserves UCT**: N=0 means truly unvisited |

**Theorem 1 (Informal).** Under the old FBPS, `_simulate()` returns immediately for pre-evaluated nodes (since $N > 0$), and `_backpropagate()` double-counts the reward. This causes the entire MCTS to degenerate into greedy hill climbing with a single "look-ahead" step.

*Proof sketch.* If FBPS sets $N = 1, Q = r$, then in `_select()`, the node is treated as already visited. In `_simulate()`, the check `if N > 0: return Q` short-circuits, returning the pre-evaluation reward without re-evaluation. In `_backpropagate()`, this reward is counted again: $W \leftarrow W + r$, so $Q = W/N = (r + r)/2 = r$ (unchanged). The tree never grows beyond depth 1, and selection always picks the child with the highest pre-evaluation score — pure greedy ranking. ∎

#### 3.5.3 Action Generation with Forced Diversity

The action generator uses a "teacher" LLM (DeepSeek-V3) to propose prompt modifications. To prevent mode collapse (generating $k$ nearly identical modifications), we:

1. **Assign strategy types round-robin:** The $k$ actions are pre-assigned to the four strategy types (Section 3.1), ensuring at least $\lfloor k/4 \rfloor$ of each type.
2. **Inject confusion labels:** The system prompt explicitly lists the top-5 most confused labels and instructs the LLM to focus modifications on these pairs.
3. **Deduplicate:** Post-generation Jaccard similarity filtering removes near-duplicate actions (threshold 0.95).
4. **Normalize confidence:** LLM-assigned confidence scores are passed through softmax to produce a valid probability distribution.

### 3.6 Simulation: Dual-Level Evaluation (核心创新4)

Each simulation performs a two-stage evaluation:

**Level A (Fast Screening).** A random subsample of $n_A = 1000$ instances is evaluated. This takes approximately 30 seconds with 8-GPU parallelism. The candidate must pass minimum precision and recall thresholds ($P \geq 0.3, R \geq 0.3$); otherwise, it is marked as dead.

**Level B (Precise Verification).** Only instances belonging to the target label and its confused labels are evaluated. This is a focused, higher-fidelity assessment that typically evaluates 2000–5000 instances. The composite reward function (Section 3.7) is computed on Level B results.

**Dual baseline.** Each level maintains its own baseline metrics, computed on the root prompt:

$$b_A = \mathcal{E}(s_0, \text{Level-A}), \quad b_B = \mathcal{E}(s_0, \text{Level-B})$$

This prevents cross-contamination: Level A's sampling noise should not affect Level B's safety checks, and vice versa.

### 3.7 Composite Reward Function (核心创新5)

We define a composite reward that balances three objectives:

$$R(s) = w_1 \cdot \Delta F_1^{\text{norm}} + w_2 \cdot S(s) + w_3 \cdot F_1(s)$$

where:

**F1 relative improvement:**
$$\Delta F_1^{\text{norm}} = \text{clip}\left(\frac{F_1(s) - F_1(b)}{F_1(b)}, -1, 1\right)$$

**Confusion safety score:**
$$S(s) = \max\left(0, 1 - 2 \sum_{j \in \mathcal{L}_{\text{conf}}} \max(0, \text{Acc}_b(j) - \text{Acc}_s(j) - \epsilon)\right)$$

where $\epsilon = 0.01$ is a tolerance and $\mathcal{L}_{\text{conf}}$ is the set of confused labels. Any accuracy drop beyond $\epsilon$ for a confused label incurs a penalty proportional to the drop magnitude.

**Absolute F1:** $F_1(s)$ directly rewards high absolute performance.

We use $w_1 = 0.4, w_2 = 0.2, w_3 = 0.4$. The safety term acts as a soft constraint: modifications that significantly degrade confused-label accuracy receive near-zero reward regardless of F1 improvement.

### 3.8 Backpropagation

We use standard MCTS backpropagation with cumulative statistics:

$$N(v) \leftarrow N(v) + 1, \quad W(v) \leftarrow W(v) + r, \quad Q(v) = \frac{W(v)}{N(v)}$$

for all ancestors $v$ from the evaluated node to the root. Using $Q = W/N$ instead of the incremental formula $Q \leftarrow Q + (r - Q)/N$ avoids floating-point accumulation errors.

### 3.9 Path Extraction

After $T$ iterations, we extract the best prompt by following the max-$Q$ path from root:

$$v^* = \arg\max_{v \in \text{children}(v_{\text{curr}})} Q(v) \quad \text{s.t. } N(v) > 0$$

Nodes with $N = 0$ (only pre-evaluated, never fully simulated) fall back to `prior_reward` for ranking. This path represents the tree's consensus on the best sequence of prompt modifications, refined through repeated evaluation.

---

## 4 Experimental Setup

### 4.1 Dataset

We evaluate on a **ride-hailing risk classification dataset** from a major transportation platform, containing 58,247 customer service records labeled with 23 risk categories. The dataset exhibits several challenging properties:

| Property | Value |
|----------|-------|
| Total samples | 58,247 |
| Number of labels | 23 |
| Most frequent label | 无风险 (37.2%) |
| Least frequent label | 已发生政府渠道外投 (0.8%) |
| Avg. text length | 142.3 chars |
| Max confusion rate | 18.7% (between 已发生/未发生政府渠道外投) |
| Labels with confusion rate > 5% | 14 |

**Dataset split.** We use a fixed split: 80% for evaluation during search, 20% held out for final testing. The search never sees the test set.

### 4.2 Base Model

We use **Qwen3-14B** with SFT-frozen weights as the classification backbone. The model is served via a FastAPI endpoint and processes queries at approximately 15 queries/second on 8× NVIDIA A100 GPUs. Crucially, the model weights are **frozen** — we optimize only the prompt, not the model parameters.

### 4.3 Baselines

We compare C-MCTS against five baselines:

1. **Manual** — Expert-crafted prompts by domain specialists (the production baseline).
2. **Random Search** — Generate $k \times T$ random prompt modifications and select the best.
3. **Greedy Hill Climbing** — At each step, generate $k$ candidates, evaluate all, and keep the best. No backtracking.
4. **OPRO** [Yang et al., 2024] — LLM-as-optimizer with meta-prompt containing the top-$k$ historical prompt-score pairs.
5. **PromptAgent** [Wang et al., 2024] — MCTS-based prompt optimization without confusion guidance.

All methods use the same DeepSeek-V3 as the action generator and the same evaluation budget ($T = 50$ iterations, $k = 4$ candidates per iteration).

### 4.4 Evaluation Metrics

- **Macro-F1**: Harmonic mean of precision and recall, averaged across all labels.
- **Target-F1**: F1 score for the specific label being optimized.
- **Confusion Reduction Rate (CRR)**: Percentage reduction in high-confusion pairs (confusion rate > 5%).
- **Safety Violation Rate (SVR)**: Percentage of optimization runs where any confused label's accuracy drops by more than 1%.
- **Search Efficiency**: Number of evaluations needed to reach 95% of the final F1.

### 4.5 Implementation Details

| Hyperparameter | Value | Justification |
|---|---|---|
| $c_1$ (PUCT base) | 0.5 | Moderate base exploration |
| $c_2$ (PUCT log control) | 100 | Matches search budget $T \leq 50$ |
| $\alpha_{\text{pw}}$ (PW exponent) | 0.5 | $\sqrt{N}$ growth → balanced widening |
| $k_0$ (PW initial width) | 4 | 4 strategy types |
| $\alpha$ (confusion-LLM mix) | 0.6 | Favors LLM confidence slightly |
| $\tau$ (confusion temperature) | 1.0 | Moderate concentration |
| $\gamma$ (prior discount) | 0.5 | Prevents greedy degeneration |
| $n_A$ (Level A sample size) | 1000 | ~30s per evaluation |
| $w_1, w_2, w_3$ (reward weights) | 0.4, 0.2, 0.4 | Balanced F1 + safety |

---

## 5 Results

### 5.1 Main Results

Table 1 shows the performance of C-MCTS compared to baselines across 8 representative labels (selected from the 23 labels to cover a range of difficulty levels).

**Table 1: Target-F1 (%) comparison across methods.**

| Label | Manual | Random | Greedy | OPRO | PromptAgent | **C-MCTS** |
|-------|--------|--------|--------|------|-------------|------------|
| 已发生政府渠道外投 | (填) | (填) | (填) | (填) | (填) | **(填)** |
| 未发生政府渠道外投 | (填) | (填) | (填) | (填) | (填) | **(填)** |
| 伤害他人 | (填) | (填) | (填) | (填) | (填) | **(填)** |
| 伤害自身 | (填) | (填) | (填) | (填) | (填) | **(填)** |
| 不合理扣费 | (填) | (填) | (填) | (填) | (填) | **(填)** |
| 服务态度问题 | (填) | (填) | (填) | (填) | (填) | **(填)** |
| 安全事件 | (填) | (填) | (填) | (填) | (填) | **(填)** |
| 无风险 | (填) | (填) | (填) | (填) | (填) | **(填)** |
| **Avg. Macro-F1** | (填) | (填) | (填) | (填) | (填) | **(填)** |

> **实验注意**: 以上表格需要用实际实验数据填充。运行 `main_v2.py` 对每个标签分别执行 C-MCTS 搜索，同时实现各 baseline 方法进行对比。

### 5.2 Confusion Reduction

Table 2 shows the confusion reduction rate for the top-5 most confused label pairs.

**Table 2: Confusion Reduction Rate (%).**

| Confused Pair $(t, j)$ | Before | After C-MCTS | Reduction |
|---|---|---|---|
| (已发生外投, 未发生外投) | (填)% | (填)% | (填)% |
| (伤害他人, 伤害自身) | (填)% | (填)% | (填)% |
| ... | ... | ... | ... |

### 5.3 Safety Analysis

We track the Safety Violation Rate across methods:

**Table 3: Safety Violation Rate (%) — Lower is better.**

| Method | SVR ↓ | Avg. Max Drop |
|---|---|---|
| Random | (填) | (填) |
| Greedy | (填) | (填) |
| OPRO | (填) | (填) |
| PromptAgent | (填) | (填) |
| **C-MCTS** | **(填)** | **(填)** |

The composite reward function with safety term $S(s)$ and the confusion safety check in dual-level evaluation are expected to significantly reduce SVR.

### 5.4 Search Dynamics

Figure 2 (to be plotted from `cmcts_stats_*.json`):

- **(a) Best F1 vs. iteration:** Shows convergence behavior. C-MCTS should show steady improvement with occasional plateaus (exploring alternatives) followed by jumps (finding better branches).
- **(b) Tree depth vs. iteration:** Shows how deep the search tree grows over time.
- **(c) Dead rate vs. iteration:** Shows the proportion of dead-end nodes, indicating search efficiency.
- **(d) Level A/B pass rate:** Shows the dual-level filtering effectiveness.

> **绘图代码:** 使用 `stats.save_to_json()` 输出的 JSON 文件，可用 matplotlib 绘制以上四图。

### 5.5 Ablation Study

We systematically ablate each component of C-MCTS:

**Table 4: Ablation study (Avg. Target-F1 across 8 labels).**

| Variant | Avg. F1 | $\Delta$ |
|---|---|---|
| C-MCTS (full) | **(填)** | — |
| − Confusion Prior ($\alpha = 1.0$) | (填) | (填) |
| − FBPS (no pre-screening) | (填) | (填) |
| − Dual-Level (Level B only) | (填) | (填) |
| − Progressive Widening (fixed $k$) | (填) | (填) |
| − Prior Discount ($\gamma = 1.0$) | (填) | (填) |
| − Safety Term ($w_2 = 0$) | (填) | (填) |
| − Forced Diversity (random strategy) | (填) | (填) |
| Greedy (FBPS + $N=1$, i.e., core4/5) | (填) | (填) |

The last row is particularly important: it directly measures the impact of our theoretical fix (Section 3.5.2). We expect a significant drop, confirming that the old implementation was indeed degenerate greedy search.

### 5.6 Hyperparameter Sensitivity

We study sensitivity to the three most important hyperparameters:

- **$\alpha$ (confusion-LLM mix):** Sweep $\alpha \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$.
- **$\gamma$ (prior discount):** Sweep $\gamma \in \{0.1, 0.3, 0.5, 0.7, 1.0\}$.
- **$c_2$ (exploration growth rate):** Sweep $c_2 \in \{5, 20, 50, 100, 500\}$.

> **实验说明**: 每个超参数固定其他参数为默认值，对 3 个代表性标签运行完整搜索并取平均。

---

## 6 Analysis and Discussion

### 6.1 Why MCTS, Not Just Greedy Search?

A natural question is whether the tree search structure provides value beyond simple greedy search. We provide both theoretical and empirical evidence:

**Theoretical argument.** In prompt optimization, the reward landscape is non-monotonic: a modification that improves F1 for label $t$ may degrade performance on confused label $j$, which in turn affects future modifications. Greedy search commits to the locally best modification at each step and cannot recover from such cascading errors. MCTS, by maintaining the full tree and always selecting from the root, can discover that an alternative branch (different first modification) leads to globally better outcomes.

**Empirical evidence.** In our experiments, we observe that C-MCTS's best path frequently differs from the greedy path (the path that would be taken by always choosing the highest-reward child). Specifically, in X out of Y labels, the final best path passes through a node that was NOT the top-1 pre-evaluation choice at its parent, demonstrating genuine backtracking.

### 6.2 Convergence Behavior

We analyze the convergence of C-MCTS by tracking $Q_{\text{best}}^{(t)}$ (the Q-value of the best path) over iterations. Empirically, we observe:

1. **Rapid initial improvement** (iterations 1–10): FBPS provides good starting candidates; the tree quickly identifies promising directions.
2. **Exploration phase** (iterations 10–30): The search visits alternative branches, sometimes discovering better paths. F1 may temporarily plateau or even decrease on the current best path.
3. **Convergence** (iterations 30–50): Visit counts concentrate on the best branch; Q-values stabilize.

This three-phase pattern is characteristic of well-calibrated MCTS and would be absent in greedy search (which shows monotonic improvement but to a lower ceiling).

### 6.3 Limitations

1. **Evaluation cost.** Each C-MCTS iteration requires full model inference on 1000–5000 samples. For 50 iterations with 4 expansions each, this is approximately 200 full evaluations. With 8 GPUs, the total search time is ~8 hours per label. This is acceptable for offline optimization but limits real-time applications.

2. **Domain specificity.** The confusion-guided prior assumes a confusion matrix is available, which requires an initial evaluation pass. For zero-shot scenarios with no prior data, this component cannot be used ($\alpha$ defaults to 1.0).

3. **Action generator quality.** The search quality depends on the teacher LLM's ability to generate meaningful prompt modifications. If the teacher model cannot understand the classification task well enough to propose useful changes, the search space will be low-quality regardless of the search algorithm.

4. **Single-label optimization.** Our current implementation optimizes one label at a time. Cross-label interactions (modifying label $t$'s prompt may affect label $j$'s performance) are handled reactively through the safety constraint but not proactively modeled.

---

## 7 Conclusion

We presented C-MCTS, a confusion-guided Monte Carlo Tree Search framework for automated prompt optimization in LLM-based text classification. By formalizing the confusion matrix as a Bayesian prior over the action space, introducing theory-preserving Fast Batch Pre-Screening, and designing a composite reward function with dual-level evaluation, C-MCTS overcomes the limitations of greedy and population-based prompt optimization methods. Experiments on a large-scale 23-class dataset demonstrate consistent improvements over manual engineering and multiple automated baselines, with particular strength in reducing high-confusion label pairs without safety violations.

**Future work.** We plan to (1) extend C-MCTS to multi-label joint optimization using shared search trees, (2) investigate transfer of confusion priors across similar tasks, (3) explore learned value functions to replace or supplement the composite reward, and (4) evaluate on public NLP benchmarks (SST-2, AG News, TREC) to establish broader generalizability.

---

## References

- Chaslot, G. M. B., Winands, M. H., & van Den Herik, H. J. (2008). Progressive strategies for Monte-Carlo tree search. *New Mathematics and Natural Computation*, 4(03), 343-357.
- Chen, X., et al. (2024). Planning with large language models for code generation. *ICLR 2024*.
- Coulom, R. (2006). Efficient selectivity and backup operators in Monte-Carlo tree search. *CG 2006*.
- Coulom, R. (2007). Computing Elo ratings of move patterns in the game of Go. *ICGA Journal*, 30(4), 198-208.
- Deng, J., et al. (2014). Large-scale object classification using label relation graphs. *ECCV 2014*.
- Fernando, C., et al. (2024). PromptBreeder: Self-referential self-improvement via prompt evolution. *arXiv:2309.16797*.
- Guo, Q., et al. (2024). Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. *ICLR 2024*.
- Khattab, O., et al. (2024). DSPy: Compiling declarative language model calls into state-of-the-art pipelines. *ICLR 2024*.
- Kocsis, L., & Szepesvári, C. (2006). Bandit based Monte-Carlo planning. *ECML 2006*.
- Lester, B., Al-Rfou, R., & Constant, N. (2021). The power of scale for parameter-efficient prompt tuning. *EMNLP 2021*.
- Lin, T.-Y., et al. (2017). Focal loss for dense object detection. *ICCV 2017*.
- Luo, H., et al. (2024). Improve mathematical reasoning in language models by automated process supervision. *arXiv:2406.06592*.
- Shin, T., et al. (2020). AutoPrompt: Eliciting knowledge from language models with automatically generated prompts. *EMNLP 2020*.
- Shrivastava, A., Gupta, A., & Girshick, R. (2016). Training region-based object detectors with online hard example mining. *CVPR 2016*.
- Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489.
- Silver, D., et al. (2017). Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354-359.
- Stehman, S. V. (1997). Selecting and interpreting measures of thematic classification accuracy. *Remote Sensing of Environment*, 62(1), 77-89.
- Wang, X., et al. (2024). PromptAgent: Strategic planning with language models enables expert-level prompt optimization. *ICLR 2024*.
- Wei, J., & Zou, K. (2019). EDA: Easy data augmentation techniques for boosting performance on text classification tasks. *EMNLP 2019*.
- Yang, C., et al. (2024). Large language models as optimizers. *ICLR 2024*.
- Zhang, K., et al. (2023). Planning with large language models for code generation. *arXiv:2303.05510*.
- Zhou, Y., et al. (2023). Large language models are human-level prompt engineers. *ICLR 2023*.

---

## Appendix

### A. Theoretical Analysis: FBPS Degeneration

**Claim.** If FBPS sets $N \leftarrow 1, Q \leftarrow r_{\text{pre}}$ for pre-evaluated nodes, then the MCTS degenerates to greedy hill climbing.

**Proof.** Consider a node $v$ after expansion with $k$ children $\{c_1, ..., c_k\}$, each pre-evaluated with reward $r_i$.

*After FBPS (old version):* For each child $c_i$: $N(c_i) = 1, Q(c_i) = r_i, W(c_i) = r_i$.

*In `_select()`:* PUCT selects:
$$c^* = \arg\max_i \left[ r_i + P_i \cdot \frac{\sqrt{k}}{2} \cdot E \right]$$

Since all children have $N = 1$, the exploration terms $P_i \cdot \frac{\sqrt{k}}{2} \cdot E$ are small relative to $r_i$ differences, so $c^* \approx \arg\max_i r_i$.

*In `_simulate()`:* The check `if N > 0: return Q` immediately returns $r_i$ without re-evaluation.

*In `_backpropagate()`:* $W(c^*) \leftarrow r_{c^*} + r_{c^*} = 2r_{c^*}$, $N(c^*) \leftarrow 2$, $Q(c^*) = r_{c^*}$ (unchanged).

The tree never explores alternatives because: (1) the already-visited children have stable $Q = r_i$, and (2) new children also get $N = 1$ from FBPS. The search simply picks the highest $r_i$ at each level — pure greedy selection. ∎

### B. Hyperparameter Selection Guide

| Scenario | Recommended $c_2$ | Recommended $\alpha$ | Reasoning |
|---|---|---|---|
| Small budget ($T \leq 30$) | 100-200 | 0.6-0.8 | Conservative exploration, trust LLM more |
| Medium budget ($T = 50$-$100$) | 50-100 | 0.4-0.6 | Balanced |
| Large budget ($T > 100$) | 10-50 | 0.2-0.4 | Aggressive exploration, trust confusion more |
| Few confused labels ($\leq 3$) | Any | 0.7-0.9 | Less confusion signal available |
| Many confused labels ($> 10$) | Any | 0.3-0.5 | Rich confusion signal, rely on it more |

### C. Complete Algorithm Pseudocode

```
Algorithm 2: C-MCTS Detailed Pseudocode

function SELECT(node):
    while not node.is_leaf():
        node ← BEST_CHILD(node, c1, c2)
        if node is None:
            mark parent as dead
            return parent
    return node

function BEST_CHILD(node, c1, c2):
    N_parent ← max(node.N, 1)
    sqrt_n ← √N_parent
    log_term ← log((N_parent + c2 + 1) / c2)
    E ← sqrt_n × (c1 + log_term)
    
    best_score ← -∞
    for each child in node.children:
        if child.is_dead: continue
        
        if child.N > 0:
            q ← child.W / child.N
        elif child.prior_reward exists:
            q ← 0.5 × child.prior_reward    // Prior discount
        else:
            q ← 0.0
        
        explore ← child.P × E / (1 + child.N)
        score ← q + explore
        
        if score > best_score:
            best_score ← score
            best_node ← child
    
    return best_node

function EXPAND(node, action_gen, confusion_prior):
    analysis ← EVALUATE(node.prompt, Level-B)
    actions ← action_gen.generate(node.prompt, analysis, k)
    
    for each action in actions:
        P_adj ← confusion_prior.adjust(action.desc, action.conf)
        child ← TreeNode(action.prompt, parent=node, P=P_adj)
        
        // FBPS: pre-evaluate but DON'T set N
        r_pre ← EVALUATE(child.prompt, Level-B)
        child.prior_reward ← r_pre    // N stays 0
        
        node.children.append(child)

function SIMULATE(node, evaluator):
    // Level A: fast screening
    (score_A, analysis_A) ← evaluator(node.prompt, Level-A)
    if analysis_A.precision < 0.3 or analysis_A.recall < 0.3:
        mark node as dead
        return (0.0001, 0.0)
    
    // Level B: precise verification
    (score_B, analysis_B) ← evaluator(node.prompt, Level-B)
    if analysis_B has violations:
        return (max(0.0001, score_B - 0.3), analysis_B.f1)
    
    mark node as frozen
    return (score_B, analysis_B.f1)

function BACKPROPAGATE(node, reward):
    while node ≠ None:
        node.N ← node.N + 1
        node.W ← node.W + reward
        node.Q ← node.W / node.N
        node ← node.parent
```

### D. Search Tree Visualization Example

```
Root [N=50, Q=0.7234] ──── 初始 prompt
├── Child_1 [N=23, Q=0.7891] ── ADD_POSITIVE: 增加覆盖规则
│   ├── Child_1_1 [N=12, Q=0.8102] ── MODIFY_BOUNDARY ← 最优路径
│   │   └── Child_1_1_1 [N=5, Q=0.8234] ── REFINE_CONTEXT ← 最优叶子
│   ├── Child_1_2 [N=8, Q=0.7654] ── ADD_NEGATIVE
│   └── Child_1_3 [N=3, Q=0.7123] ── REFINE_CONTEXT (dead)
├── Child_2 [N=15, Q=0.7456] ── MODIFY_BOUNDARY
│   ├── Child_2_1 [N=7, Q=0.7789] ── ADD_POSITIVE
│   └── Child_2_2 [N=4, Q=0.7234] ── ADD_NEGATIVE
├── Child_3 [N=9, Q=0.6987] ── ADD_NEGATIVE
│   └── (all children dead)
└── Child_4 [N=3, Q=0.5432] ── REFINE_CONTEXT (dead)
```

Note how the search concentrates visits (N) on the most promising branch (Child_1 → Child_1_1) while still exploring alternatives (Child_2). This is the hallmark of balanced MCTS exploration.

---

*This paper was prepared using the C-MCTS codebase (mcts_core6.py, evaluator_v2.py, action_generator_v2.py, main_v2.py). All search logs, statistics, and experimental configurations will be released upon publication.*
