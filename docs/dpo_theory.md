# DPO 理论推导与代码对应详解

> 本文档对 DPO（Direct Preference Optimization，直接偏好优化）的数学原理进行完整推导，并与 PPO（Proximal Policy Optimization）的核心机制逐一对比，同时将所有理论概念与 `dpo_training.py` 中的代码变量一一对应。所有符号**首次出现时均有定义**。

---

## 目录

1. [DPO 的出发点：PPO 的痛点](#1-dpo-的出发点ppo-的痛点)
2. [偏好学习的数学框架：Bradley-Terry 模型](#2-偏好学习的数学框架bradley-terry-模型)
3. [RLHF 的完整目标函数](#3-rlhf-的完整目标函数)
4. [DPO 核心推导：从 RLHF 到闭合解](#4-dpo-核心推导从-rlhf-到闭合解)
5. [DPO 损失函数详解](#5-dpo-损失函数详解)
6. [参考策略的作用与隐式 KL 约束](#6-参考策略的作用与隐式-kl-约束)
7. [DPO 训练数据格式与代码对应](#7-dpo-训练数据格式与代码对应)
8. [DPO vs PPO 全面对比](#8-dpo-vs-ppo-全面对比)
9. [完整数据流总结](#9-完整数据流总结)

---

## 1. DPO 的出发点：PPO 的痛点

### 1.1 PPO-RLHF 流水线回顾

在基于人类反馈的强化学习（RLHF，Reinforcement Learning from Human Feedback）的经典范式中，PPO 的完整训练流水线需要如下四个模型同时在线运行：

| 模型 | 角色 | PPO 代码对应 |
|------|------|-------------|
| **Actor**（策略模型） | 被优化的 LLM，生成回答 | `policy` |
| **Critic**（价值模型） | 估计当前状态的价值 $V(s_t)$，与 Actor 并行训练 | `critic` |
| **Reward Model（RM）** | 对生成的完整回答打分 | `reward_model` |
| **Reference Model（参考模型）** | SFT 阶段得到的原始模型，提供 KL 惩罚基准，参数冻结 | `ref_policy` |

这一流水线存在以下已知问题：

1. **显存开销巨大**：四个模型同时加载，7B 参数模型通常需要 4×16GB 显存。
2. **训练流程复杂**：需要 Rollout（在线生成）→ Reward 计算 → GAE 优势估计 → Clip 更新，多个步骤耦合，超参众多（$\gamma$、$\lambda$、$\epsilon$、KL 系数等）。
3. **样本效率低**：每批数据必须通过 Actor 实时生成（on-policy 或 near-on-policy），生成速度慢。
4. **训练不稳定**：Actor 和 Critic 同时更新，容易出现"奖励黑客"（reward hacking）或策略崩溃。

> **核心矛盾**：PPO 需要一个明确的奖励信号 $r(x, y)$ 来计算 TD Error 和 GAE，而这个奖励信号由独立的 RM 提供，整个流程天然要求在线采样。

### 1.2 DPO 的核心洞见

DPO（Rafailov 等，2023）指出：**我们并不需要显式地训练奖励模型，也不需要在线 Rollout。只要利用一个数学恒等式，就能把 RLHF 的优化目标直接转化为一个对（chosen, rejected）响应对的分类损失。**

| 对比维度 | PPO-RLHF | DPO |
|---------|---------|-----|
| 所需模型数量 | 4 个（Actor + Critic + RM + Reference） | 2 个（Policy + Reference） |
| 是否需要在线采样 | 是（Rollout 生成） | 否（离线偏好数据集） |
| 是否需要显式奖励模型 | 是 | 否（隐式编码在 Policy 中） |
| 训练范式 | 强化学习（RL Loop） | 监督学习（类 SFT） |
| 超参复杂度 | 高（多个 RL 超参） | 低（主要是 $\beta$） |

---

## 2. 偏好学习的数学框架：Bradley-Terry 模型

### 2.1 人类偏好数据

DPO 的训练数据是三元组：

$$
\mathcal{D} = \left\{ \left( x^{(i)},\, y_w^{(i)},\, y_l^{(i)} \right) \right\}_{i=1}^{N}
$$

**符号说明**：
- $x$：**提示词（Prompt）**，即用户的问题或指令
- $y_w$：**chosen 回答**（preferred/won），人类标注者认为更好的响应
- $y_l$：**rejected 回答**（less preferred/lost），人类认为较差的响应
- $N$：数据集中的样本对总数

**对应代码数据格式**（`dpo_training.py` 中的 `return_prompt_and_responses`）：

```python
# 每条数据包含三个字段
{
    "prompt":   "问题/指令文本",
    "chosen":   "人类标注的优质回答",
    "rejected": "人类标注的较差回答"
}
```

### 2.2 Bradley-Terry 偏好概率模型

**Bradley-Terry 模型** 假设存在一个潜在的奖励函数 $r^*(x, y)$，人类的偏好概率由奖励差决定：

$$
\boxed{
P\!\left(y_w \succ y_l \mid x\right) = \sigma\!\left(r^*(x, y_w) - r^*(x, y_l)\right) = \frac{1}{1 + e^{-(r^*(x, y_w) - r^*(x, y_l))}}
}
$$

**逐词解读**：
- $y_w \succ y_l$：读作"$y_w$ 被人类偏好于 $y_l$"
- $\sigma(\cdot)$：Sigmoid 函数，$\sigma(z) = \frac{1}{1+e^{-z}}$，把实数映射到 $(0,1)$ 的概率值
- $r^*(x, y_w) - r^*(x, y_l)$：两个回答奖励的差值——差越大，人类越喜欢 $y_w$
- 当 $r^*(x, y_w) = r^*(x, y_l)$ 时，$\sigma(0) = 0.5$，即两个回答同等好（无偏好）

**与 PPO 的对比**：PPO 中的奖励 $r_t$ 是 Reward Model 对每个 token 的逐步打分，并通过 TD Error 传播到整条轨迹；BT 模型中的 $r^*(x, y)$ 是对整条完整响应的全局评分，只关心相对排序，不关心绝对值。

### 2.3 奖励模型的监督学习目标

在经典 RLHF 的第二阶段，需要先用偏好数据训练一个参数化的奖励模型 $r_\phi(x, y)$，最小化负对数似然：

$$
\mathcal{L}_\text{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \right]
$$

> **DPO 的关键突破**：跳过这一步，直接从偏好数据中优化策略，避免了独立训练 RM 的额外开销和误差传播。

---

## 3. RLHF 的完整目标函数

### 3.1 带 KL 约束的 RLHF 目标

RLHF 的核心优化问题是：**在让模型回答尽可能获得高奖励的同时，不让模型偏离原始 SFT 模型太远**。

$$
\boxed{
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ r(x, y) \right] - \beta\, D_{\mathrm{KL}}\!\left[\pi_\theta(\cdot \mid x) \,\|\, \pi_\text{ref}(\cdot \mid x)\right]
}
$$

**符号说明**：

| 符号 | 含义 | 代码对应 |
|------|------|---------|
| $\pi_\theta(y \mid x)$ | **待优化策略（Policy）**：参数为 $\theta$ 的 LLM，给定 $x$ 生成 $y$ 的概率 | `model`（DPO 训练的目标模型） |
| $\pi_\text{ref}(y \mid x)$ | **参考策略（Reference Policy）**：SFT 模型，参数固定，不参与梯度更新 | `ref_model`（冻结的 SFT 模型） |
| $r(x, y)$ | **奖励函数**：对完整响应 $y$ 的评分（PPO 中由 RM 提供，DPO 中隐式表示） | PPO：`reward_model(x, y)`；DPO：隐式 |
| $\beta$ | **KL 惩罚系数**，$\beta > 0$，控制新策略偏离参考策略的程度 | `args.beta`（通常取 0.1～0.5） |
| $D_{\mathrm{KL}}[\pi_\theta \| \pi_\text{ref}]$ | **KL 散度**，衡量两个分布的差异，$\ge 0$，等于 0 当且仅当两分布完全相同 | PPO 中显式计算并加入奖励；DPO 中隐式约束 |

### 3.2 KL 散度的展开形式

$$
D_{\mathrm{KL}}\!\left[\pi_\theta \| \pi_\text{ref}\right] = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)} \right]
$$

将目标函数完整展开：

$$
\max_{\pi_\theta} \mathbb{E}_{x,\, y \sim \pi_\theta} \left[ r(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)} \right]
$$

**与 PPO 的对比**：

在 PPO 的代码中，KL 惩罚被**显式地加到每个 token 的即时奖励** $r_t$ 中：

```python
# PPO 中 KL 惩罚的实现（ppo_training.py）
kl = new_logprob - ref_logprob          # 逐 token 的 log ratio
non_score_reward = -kl_coef * kl        # KL 惩罚作为即时奖励的一部分
rewards[t] = non_score_reward[t]        # 加入 GAE 计算
```

而在 DPO 中，KL 惩罚**不需要显式计算**，它被自然地编码进了损失函数的数学结构中（见第 4 节）。

---

## 4. DPO 核心推导：从 RLHF 到闭合解

这是 DPO 的数学核心，也是它与 PPO 最根本的区别所在。

### 4.1 最优策略的解析解

**定理**：对于第 3.1 节的带 KL 约束的 RLHF 目标，存在以下**唯一最优策略**的解析解：

$$
\boxed{
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_\text{ref}(y \mid x)\, \exp\!\left(\frac{r(x, y)}{\beta}\right)
}
$$

**符号说明**：
- $Z(x)$：**配分函数（Partition Function）**，归一化常数，使 $\sum_y \pi^*(y|x) = 1$
$$Z(x) = \sum_{y} \pi_\text{ref}(y \mid x)\, \exp\!\left(\frac{r(x, y)}{\beta}\right)$$
- $\pi^*(y \mid x)$：最优策略，即对每个 $y$，以参考策略为基础，对高奖励回答指数级放大概率

**推导过程**：

展开目标函数，将期望中的 $y \sim \pi_\theta$ 改写：

$$
\mathbb{E}_{y \sim \pi_\theta} \left[ r(x,y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} \right]
= -\beta \sum_y \pi_\theta(y|x) \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x) \exp(r(x,y)/\beta)}
$$

注意到括号内正是 $\pi_\theta$ 相对于 $\pi_\text{ref}(y|x) \exp(r/\beta)$ 的 KL 散度（相差一个与 $\pi_\theta$ 无关的常数 $\beta \log Z(x)$）。

**KL 散度非负**，当且仅当两个分布完全相同时取到最小值 0，因此目标函数在以下时取最大值：

$$
\pi_\theta(y|x) \propto \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x,y)}{\beta}\right)
$$

归一化后即得 $\pi^*$。

> **直觉理解**：最优策略在参考策略 $\pi_\text{ref}$ 的基础上，对高奖励响应指数级"上调"概率，对低奖励响应"下调"概率，$\beta$ 控制调整的幅度——$\beta$ 越小，奖励信号越强，策略越激进地靠向高奖励区域；$\beta$ 越大，越保守地靠近参考策略。

**与 PPO 的对比**：PPO 没有这个解析解，它只能通过一步一步的梯度更新（Rollout → GAE → Clip）来**数值逼近**最优策略，而 DPO 直接给出了最优解的形式，并以此为基础构造损失函数。

### 4.2 奖励函数的重参数化

从最优策略公式反解奖励函数：

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)
$$

这一恒等式是 DPO 的**关键桥梁**：它把一个不可见的奖励函数 $r(x,y)$ 表达成了两个策略（目标策略 $\pi^*$ 和参考策略 $\pi_\text{ref}$）的对数比——这是可以计算的。

**符号解读**：
- $\beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)}$：目标策略与参考策略的对数比，乘以 $\beta$，可以看作"隐式奖励"
- $\beta \log Z(x)$：只依赖于 $x$ 的常数，不依赖于 $y$，在偏好概率中会被消去

### 4.3 代入 Bradley-Terry 模型

将第 4.2 节的重参数化代入第 2.2 节的 Bradley-Terry 偏好概率：

$$
P(y_w \succ y_l \mid x) = \sigma\!\left(r(x, y_w) - r(x, y_l)\right)
$$

代入重参数化的 $r$：

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} + \cancel{\beta \log Z(x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)} - \cancel{\beta \log Z(x)}
$$

注意：$\beta \log Z(x)$ 在相减时**完全消去**，因为它只依赖于 $x$，与 $y$ 无关。

得到：

$$
\boxed{
P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)
}
$$

其中用 $\pi_\theta$ 替代了 $\pi^*$（因为我们要用 $\pi_\theta$ 来逼近 $\pi^*$）。

> **这就是 DPO 最关键的一步**：偏好概率现在完全可以用两个语言模型（$\pi_\theta$ 和 $\pi_\text{ref}$）计算，**不再需要显式的奖励模型 $r(x,y)$**。配分函数 $Z(x)$ 的消去使得整个表达式不依赖于难以计算的归一化常数。

### 4.4 DPO 最终损失函数

对上述偏好概率取负对数期望，得到 DPO 的训练损失：

$$
\boxed{
\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[
  \log \sigma\!\left(
    \underbrace{\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)}}_{\text{chosen 的隐式奖励}} -
    \underbrace{\beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}}_{\text{rejected 的隐式奖励}}
  \right)
\right]
}
$$

---

## 5. DPO 损失函数详解

### 5.1 逐项解析

定义**隐式奖励（Implicit Reward）**：

$$
\hat{r}_\theta(x, y) \triangleq \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}
$$

则 DPO 损失可以写成：

$$
\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E} \left[ \log \sigma\!\left(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)\right) \right]
$$

**逐词解读**：
- $\hat{r}_\theta(x, y_w) > \hat{r}_\theta(x, y_l)$：chosen 的隐式奖励 > rejected 的隐式奖励，则 $\sigma(\cdot) > 0.5$，损失减小
- $\log \sigma(\cdot)$：对数似然，当预测概率 → 1 时，$\log \sigma \to 0$（损失最小）；当概率 → 0 时，$\log \sigma \to -\infty$（损失极大）
- 负号：把最大化对数似然转化为**最小化**负对数似然（标准监督学习范式）

### 5.2 梯度方向分析

对 $\mathcal{L}_\text{DPO}(\theta)$ 关于 $\theta$ 求梯度（略去常数因子）：

$$
\nabla_\theta \mathcal{L}_\text{DPO} \propto -\underbrace{\sigma\!\left(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w)\right)}_{\text{权重：当前排序越"错"，权重越大}} \left[
  \underbrace{\beta \nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{提高 chosen 的对数概率}} -
  \underbrace{\beta \nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{降低 rejected 的对数概率}}
\right]
$$

**直觉解读**：
- 梯度的作用方向：**增大** $\pi_\theta(y_w|x)$（使模型更倾向于生成 chosen 回答），**减小** $\pi_\theta(y_l|x)$（使模型远离 rejected 回答）
- 权重 $\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))$：当当前策略已经正确排序（chosen 奖励 > rejected 奖励）时，权重变小，梯度平缓（类似 hard example mining）；当排序错误时，权重大，梯度强，纠错更激进

**与 PPO 梯度的对比**：

PPO 中 Policy Gradient 的梯度（见 `ppo_theory.md` 第 4.4 节）：
$$\nabla_\theta L^{\text{PPO}} = r_t(\theta) \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t$$

| 维度 | PPO | DPO |
|------|-----|-----|
| 梯度权重 | ratio $r_t(\theta)$（新旧策略比值） | $\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))$（当前排序的错误程度） |
| 优势信号 | GAE 估计的 $\hat{A}_t$（需要 Critic） | 偏好数据直接给出（chosen vs rejected） |
| 更新粒度 | 逐 token 级别 | 整条响应级别 |
| 是否需要在线数据 | 是（每轮 Rollout） | 否（离线数据集） |

### 5.3 代码对应

在 `trl` 库的 `DPOTrainer` 中，DPO 损失的核心计算步骤：

```python
# 1. 计算 policy 模型对 chosen/rejected 的对数概率（整条响应求和）
policy_chosen_logps  = log_probs_sum(model, x, y_w)   # log π_θ(y_w | x)
policy_rejected_logps = log_probs_sum(model, x, y_l)  # log π_θ(y_l | x)

# 2. 计算 reference 模型对 chosen/rejected 的对数概率（冻结，不反向传播）
with torch.no_grad():
    ref_chosen_logps  = log_probs_sum(ref_model, x, y_w)  # log π_ref(y_w | x)
    ref_rejected_logps = log_probs_sum(ref_model, x, y_l) # log π_ref(y_l | x)

# 3. 计算隐式奖励差值（β × log ratio）
chosen_rewards  = beta * (policy_chosen_logps  - ref_chosen_logps)   # r̂_θ(x, y_w)
rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps) # r̂_θ(x, y_l)

# 4. DPO 损失
loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

> **关键细节**：`log_probs_sum` 计算的是整条响应所有 token 的 log probability 之和，即 $\log \pi_\theta(y|x) = \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t})$，这与 PPO 中逐 token 计算 `logprobs[t]` 的方式本质相同，但 DPO 在整句级别做聚合。

---

## 6. 参考策略的作用与隐式 KL 约束

### 6.1 参考策略（Reference Policy）

DPO 训练中保留了一个冻结的参考策略 $\pi_\text{ref}$，通常是 SFT 阶段得到的模型。

**$\pi_\text{ref}$ 的三重作用**：

1. **数学锚点**：在推导中它提供了最优策略的解析形式，使配分函数可以被消去
2. **隐式 KL 约束**：损失函数中的 $\log \frac{\pi_\theta}{\pi_\text{ref}}$ 项天然地惩罚 $\pi_\theta$ 偏离 $\pi_\text{ref}$ 过远
3. **防止退化**：若无参考策略，模型会通过无限提高 $y_w$ 的 log prob 或无限降低 $y_l$ 的 log prob 来将损失压低到 0，导致训练不稳定

**与 PPO 的对比**：

| 维度 | PPO | DPO |
|------|-----|-----|
| Reference Model 的使用方式 | 每个 token 显式计算 KL，加入 $r_t$ | 每条响应计算 log ratio，嵌入损失函数 |
| KL 约束的体现 | 显式：`rewards[t] -= kl_coef * kl_t` | 隐式：`log(π_θ/π_ref)` 在损失中 |
| Reference Model 是否需要前向传播 | 是（每个 rollout step 都算） | 是（但只做一次，无梯度） |

### 6.2 隐式 KL 约束的数学形式

可以证明，DPO 损失的梯度等价于在以下约束下最大化期望奖励：

$$
D_{\mathrm{KL}}[\pi_\theta \| \pi_\text{ref}] \le \delta
$$

其中约束强度由 $\beta$ 控制（$\beta$ 越小，KL 约束越松，策略变化越激进）。

**这就是为什么 DPO 不需要显式 KL 惩罚**：$\beta$ 参数已经隐式地起到了 PPO 中 `kl_coef` 的作用。

### 6.3 β 参数的直觉理解

$$
\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}
$$

- **$\beta \to 0$**：隐式奖励趋于零，策略几乎不更新（过于保守）
- **$\beta \to \infty$**：KL 约束消失，策略可以任意偏离参考策略（类似无约束监督学习：让 chosen 的 logprob 趋于 1，rejected 趋于 0）
- **实践中**：$\beta \in [0.1, 0.5]$，代码中对应 `args.beta`

---

## 7. DPO 训练数据格式与代码对应

### 7.1 数据格式

DPO 使用**离线偏好对数据集**，无需在线生成，这与 PPO 的 Rollout 阶段形成鲜明对比：

```json
{
    "system":   "你是一位医学助手。",
    "history":  [],
    "question": "高血压患者应如何调整饮食？",
    "chosen":   "高血压患者建议低盐饮食（每日钠摄入 < 2g），多食富含钾的蔬果...",
    "rejected": "可以随便吃，只要按时吃药就行。"
}
```

对应 `dpo_training.py` 的处理代码：

```python
def return_prompt_and_responses(examples) -> Dict[str, str]:
    # 构造 prompt（system + history + question）
    prompts, chosens, rejecteds = [], [], []
    for system, history, question, chosen, rejected in zip(...):
        prompt = build_prompt(system, history, question)  # 对话模板拼接
        prompts.append(prompt)
        chosens.append(chosen)    # y_w
        rejecteds.append(rejected) # y_l
    return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}
```

**与 PPO 数据准备的对比**：

| 阶段 | PPO | DPO |
|------|-----|-----|
| 数据来源 | 只需 Prompt（回答实时生成） | 需要 (Prompt, Chosen, Rejected) 三元组 |
| 是否需要 RM 打分 | 是（在线 Rollout 后打分） | 否（人类标注已隐含偏好） |
| 数据准备成本 | Prompt 数据集（较易获取） | 偏好对数据集（标注成本更高） |

### 7.2 模型初始化

```python
# DPO 需要两个模型：Policy + Reference
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)   # π_θ（被训练）

# Reference model：深拷贝或重新加载，参数完全冻结
ref_model = deepcopy(model)  # π_ref（参数固定）
for param in ref_model.parameters():
    param.requires_grad = False
```

**与 PPO 的对比**：PPO 还需要额外初始化 Critic 模型和 Reward Model：

```python
# PPO 需要四个模型
actor   = AutoModelForCausalLM.from_pretrained(...)   # π_θ（Actor）
critic  = AutoModelForSequenceClassification.from_pretrained(...)  # V(s)（Critic）
ref_model = deepcopy(actor)   # π_ref（参考策略，冻结）
reward_model = AutoModelForSequenceClassification.from_pretrained(...)  # r(x,y)（奖励）
```

---

## 8. DPO vs PPO 全面对比

### 8.1 算法原理对比

| 维度 | PPO | DPO |
|------|-----|-----|
| **理论基础** | 策略梯度 + 重要性采样 + Clip 约束 | RLHF 最优策略的解析解 + Bradley-Terry 模型 |
| **优化目标** | 最大化期望累计折扣回报 $J(\theta)$ | 最大化偏好对数据的对数似然 $-\mathcal{L}_\text{DPO}$ |
| **KL 约束** | 显式：KL 惩罚加入每 token 奖励 | 隐式：嵌入损失函数的 $\log(\pi_\theta/\pi_\text{ref})$ 项 |
| **奖励信号** | 外部 Reward Model 显式打分 | 偏好数据隐式提供（chosen > rejected） |
| **优势函数** | 需要 GAE 估计（Critic + TD Error） | 不需要（偏好信号直接替代） |
| **核心公式** | $r_t(\theta) \cdot \nabla_\theta \log \pi_\theta(a_t\|s_t) \cdot \hat{A}_t$ | $\nabla_\theta \log \sigma(\hat{r}_\theta(y_w) - \hat{r}_\theta(y_l))$ |

### 8.2 工程实现对比

| 维度 | PPO | DPO |
|------|-----|-----|
| **模型数量** | 4（Actor + Critic + RM + Reference） | 2（Policy + Reference） |
| **显存占用** | 高（4 × 模型大小） | 低（2 × 模型大小） |
| **是否需要在线采样** | 是（每轮 Rollout 生成回答） | 否（使用离线偏好对） |
| **训练速度** | 慢（自回归采样是瓶颈） | 快（类 SFT 并行前向传播） |
| **超参数量** | 多（$\gamma, \lambda, \epsilon, \text{kl\_coef}, \text{ppo\_epochs}$ 等） | 少（主要是 $\beta$） |
| **训练稳定性** | 较复杂（需要 Clip 防止更新过大） | 较稳定（无 Clip，有自然约束） |
| **数据需求** | 只需 Prompt 数据 | 需要 (Prompt, Chosen, Rejected) 三元组 |

### 8.3 优缺点对比

**PPO 的优势**：
- 支持在线学习，模型可以主动探索新的高奖励回答
- 奖励信号灵活，可以设计多维度奖励（如安全性 + 质量 + 事实性）
- 理论上可以发现 SFT 数据中从未出现的更优回答

**PPO 的劣势**：
- 训练流水线复杂，工程难度大
- 需要高质量 RM，RM 的误差会传播到最终策略
- 在线生成速度慢，训练成本高
- 多个模型并行加载，显存压力大

**DPO 的优势**：
- 训练简单，类似监督学习（SFT），易于实现和调试
- 不需要独立的 RM，避免 RM 过拟合问题
- 显存和计算效率高，训练速度快
- 理论保证：直接对 RLHF 最优解进行参数化

**DPO 的劣势**：
- 依赖离线偏好对数据，标注成本高
- 无法在线探索（distribution shift 问题：训练分布 ≠ 部署分布）
- $\log \pi_\theta(y_w|x)$ 可能下降（已有研究指出 DPO 不总是提升 chosen 的概率）
- 对长回答的归一化敏感（不同长度的响应 log prob 量级不同）

### 8.4 何时选择 DPO vs PPO

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| 有高质量偏好对数据，计算资源有限 | **DPO** | 训练简单，资源消耗低 |
| 需要动态探索，偏好难以提前标注 | **PPO** | 支持在线学习，可自我发现更优策略 |
| 快速 PoC，验证偏好对齐效果 | **DPO** | 实现简单，迭代快 |
| 生产级 RLHF，追求最终效果上限 | **PPO** | 在线采样可突破离线数据分布限制 |
| 多维奖励（安全 + 质量 + 事实） | **PPO** | 多个奖励信号可灵活组合进 $r_t$ |

---

## 9. 完整数据流总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DPO 完整数据流                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  输入数据（离线偏好对，无需在线生成）                                   │
│     (x, y_w, y_l) ← 来自 data/reward/ 下的 jsonl 文件               │
│     ↓                                                               │
│  1. 数据预处理                                                        │
│     构造 prompt（system + history + question）                        │
│     分别对 y_w、y_l 进行 tokenize 并截断                              │
│     ↓                                                               │
│  2. Policy 模型前向传播（有梯度）                                      │
│     log π_θ(y_w|x) = Σ_t log π_θ(y_w,t | x, y_w,<t)               │
│     log π_θ(y_l|x) = Σ_t log π_θ(y_l,t | x, y_l,<t)               │
│     ↓                                                               │
│  3. Reference 模型前向传播（无梯度，torch.no_grad()）                  │
│     log π_ref(y_w|x)，log π_ref(y_l|x)                             │
│     ↓                                                               │
│  4. 计算隐式奖励                                                      │
│     r̂_θ(y_w) = β × [log π_θ(y_w|x) - log π_ref(y_w|x)]            │
│     r̂_θ(y_l) = β × [log π_θ(y_l|x) - log π_ref(y_l|x)]            │
│     ↓                                                               │
│  5. DPO 损失计算                                                      │
│     loss = -log σ(r̂_θ(y_w) - r̂_θ(y_l))                            │
│     ↓                                                               │
│  6. 反向传播（仅 Policy 模型）                                         │
│     ∇_θ loss → 更新 θ（π_ref 参数全程冻结）                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### DPO vs PPO 数据流对比

```
PPO 数据流：                          DPO 数据流：
────────────────────                  ────────────────────
Prompt 数据集                          偏好对数据集
    ↓                                      ↓
Actor 自回归采样（慢）                 Policy + Reference 并行前向（快）
    ↓                                      ↓
RM 打分 + KL 惩罚                      log ratio 计算隐式奖励
    ↓                                      ↓
Critic 估计 V(s)                       （无需 Critic）
    ↓                                      ↓
GAE 计算优势 Â                         偏好对直接提供监督信号
    ↓                                      ↓
Clip PPO 多轮更新                      一轮监督学习更新
```

### 各理论要素对应总结

| 理论概念 | 在 DPO 中的体现 | 对应 PPO 中的概念 |
|---------|--------------|----------------|
| **Bradley-Terry 模型** | 偏好概率 $P(y_w \succ y_l) = \sigma(r_w - r_l)$ | PPO 无显式偏好概率（用累计回报代替） |
| **RLHF 最优策略** | $\pi^*(y\|x) \propto \pi_\text{ref}(y\|x) e^{r(x,y)/\beta}$ | PPO 数值逼近, 无闭合解 |
| **配分函数消去** | $Z(x)$ 在 $r_w - r_l$ 中抵消，不需计算 | PPO 无此步骤 |
| **隐式奖励** | $\hat{r}_\theta = \beta \log(\pi_\theta / \pi_\text{ref})$ | PPO：RM 显式输出的 $r(x,y)$ |
| **KL 约束** | 隐式：$\log(\pi_\theta/\pi_\text{ref})$ 项自然约束 | PPO：显式 `kl_coef × KL` 加入奖励 |
| **参考策略** | 提供 log ratio 基准，防止策略退化 | PPO：Reference 仅用于计算 KL 惩罚 |
| **$\beta$ 参数** | 控制 KL 约束强度（隐式） | PPO：`kl_coef` 控制 KL 约束（显式） |
| **偏好监督信号** | chosen/rejected 对，离线数据集 | PPO：RM 在线打分 |
| **优势估计** | 不需要（偏好对直接提供方向） | PPO：GAE（需要 Critic + TD Error） |

---

*本文档对应 MedicalGPT DPO 训练流程，代码见 `dpo_training.py`。理论推导参考 Rafailov 等 2023 年论文《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》。*
