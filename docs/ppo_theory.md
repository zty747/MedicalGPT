# PPO 理论推导与代码对应详解

> 本文档对应整理内容 a（PPO 实践笔记）与文档 b（强化学习理论图片），将两者逐句对齐，并补全数学推导细节。所有符号**首次出现时均有定义**，公式力求与代码变量名一一对应。

---

## 目录

1. [强化学习基本符号定义](#1-强化学习基本符号定义)
2. [目标函数：我们到底在最大化什么](#2-目标函数我们到底在最大化什么)
3. [蒙特卡洛采样：如何把期望变成可算的数](#3-蒙特卡洛采样如何把期望变成可算的数)
4. [策略梯度定理：从 J(θ) 到 ∇log π](#4-策略梯度定理从-jθ-到-log-π)
5. [对数求导技巧详解](#5-对数求导技巧详解)
6. [贝尔曼方程与 TD Error（δ）](#6-贝尔曼方程与-td-errorδ)
7. [Critic、Value Function 与 GAE](#7-criticvalue-function-与-gae)
8. [PPO 的 Clip 目标函数](#8-ppo-的-clip-目标函数)
9. [Critic 的更新：Value Loss 与 Returns](#9-critic-的更新value-loss-与-returns)
10. [反向传播梯度的完整路径](#10-反向传播梯度的完整路径)
11. [完整数据流总结](#11-完整数据流总结)

---

## 1. 强化学习基本符号定义

在正式推导前，先把所有后续会用到的符号统一定义清楚。

| 符号 | 含义 | 代码对应变量 |
|------|------|-------------|
| $s_t$ | **状态（State）**：第 $t$ 步时模型"看到"的全部文本前缀（query + 已生成的 token） | `query_response[:, :t]` |
| $a_t$ | **动作（Action）**：第 $t$ 步时模型采样出的那一个 token | `response[t]` |
| $\pi_\theta(a_t \mid s_t)$ | **策略（Policy）**：参数为 $\theta$ 的模型，在状态 $s_t$ 下选择动作 $a_t$ 的概率 | Actor（CausalLM）的 softmax 输出 |
| $r_t$ | **即时奖励（Reward）**：第 $t$ 步得到的分数，包含 KL 惩罚和 RM 分数 | `rewards[t]` |
| $V(s_t)$ | **状态价值（Value）**：从 $s_t$ 出发直到序列结束，能拿到的累计折扣奖励的期望 | Critic 的输出 `values[t]` |
| $\gamma$ | **折扣因子（Discount Factor）**，$0 < \gamma \le 1$，控制对"未来奖励"的重视程度 | `args.gamma`（通常取 1.0） |
| $\lambda$ | **GAE 平滑参数**，$0 \le \lambda \le 1$，平衡偏差与方差 | `args.lam` |
| $\delta_t$ | **TD Error**：对 $s_t$ 处动作好坏的即时评估 | 代码里中间变量 `delta` |
| $A_t$ | **优势（Advantage）**：动作 $a_t$ 比"平均水平"好多少（GAE 版） | `advantages[t]` |
| $\hat{A}_t$ | **白化后优势**：对 $A_t$ 进行零均值、单位方差归一化 | `mb_advantage`（白化后） |
| $\theta$ | Actor（策略模型）的全部参数 | `policy.parameters()` |
| $T$ | 一条轨迹（一次完整生成）的总 token 数 | `response_length` |
| $\tau$ | **轨迹（Trajectory）**：$(s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$ 的完整序列 | 一次 rollout |

> **关于 $\sim$ 符号**：在概率论中，$x \sim P$ 表示"$x$ 是从分布 $P$ 中采样出来的"。例如 $a_t \sim \pi_\theta(\cdot \mid s_t)$ 就是说"动作 $a_t$ 是按照当前策略的概率分布采样得到的"——这正对应了整理内容 a 中提到的 top_k=0、top_p=1 的采样方式，而不是贪心取最大 logit。

---

## 2. 目标函数：我们到底在最大化什么

### 2.1 累计折扣回报

给定一条轨迹，第 $t$ 步的**折扣累计回报（Discounted Return）**定义为：

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-t} r_T = \sum_{k=0}^{T-t} \gamma^k r_{t+k}
$$

**符号说明**：
- $r_t$：第 $t$ 步的即时奖励（对应代码中 `rewards[t]`，包含 KL 惩罚和 RM 分数）
- $\gamma^k$：对第 $t+k$ 步的未来奖励打折扣，$k$ 越大折扣越重
- $G_t$：从第 $t$ 步出发，整条轨迹剩余部分能拿到的"折扣总分"

### 2.2 策略目标函数

强化学习要最大化的目标是**期望累计回报**：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ G_0 \right] = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

**符号说明**：
- $\mathbb{E}[\cdot]$：数学期望符号，表示对所有可能轨迹取平均
- $\tau \sim \pi_\theta$：轨迹 $\tau$ 是按照当前策略 $\pi_\theta$ 采样出来的（即让 Actor 自由生成）
- $J(\theta)$：策略的"总分"，我们希望通过调整 $\theta$ 让它尽可能大
我:直观上是要让分高的路径概率尽量大（由于是概率采样rollout）
---

## 3. 蒙特卡洛采样：如何把期望变成可算的数

### 3.1 问题所在

$J(\theta)$ 的定义里有一个期望 $\mathbb{E}_{\tau \sim \pi_\theta}[\cdot]$，理论上要遍历所有可能轨迹。对于 LLM 来说，词表有 151936 个 token，序列长度可能数百，穷举根本不可能。

### 3.2 蒙特卡洛近似

**蒙特卡洛（Monte Carlo）方法的核心思想**：用有限次随机采样的平均值来近似期望值。

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0] \approx \frac{1}{N} \sum_{i=1}^{N} G_0^{(i)}
$$

**符号说明**：
- $N$：采样的轨迹条数（对应代码中 `batch_size` × `rollout` 次数）
- $G_0^{(i)}$：第 $i$ 条采样轨迹的总回报
- $\approx$：近似等于，样本数 $N$ 越大近似越精确

**对应代码中的 Rollout 阶段**（整理内容 a 中提到的）：

```python
# Actor 按概率分布采样，而非贪心（top_k=0, top_p=1 等于完整分布采样）
# 这就是蒙特卡洛采样：让模型"跑"N条轨迹，记录每个token的logprob
response, logprobs = generate_with_logprobs(actor, query)
```

> **蒙特卡洛 vs 贪心**：贪心解码每步取最大 logit，永远得到同一条轨迹，无法估计期望。蒙特卡洛采样每次生成不同的回答，这 N 条回答的平均分才能近似 $J(\theta)$。整理内容 a 提到的 top_k=0、top_p=1 正是为了保证采样的随机性，从而进行蒙特卡洛估计。

---

## 4. 策略梯度定理：从 J(θ) 到 ∇log π

这是整理内容 a 中"Policy Gradient 的由来"这一疑问的完整答案。

### 4.1 目标：对 J(θ) 求梯度

为了用梯度上升最大化 $J(\theta)$，需要计算 $\nabla_\theta J(\theta)$。

展开期望：

$$
J(\theta) = \sum_{\tau} P(\tau; \theta) \cdot G(\tau)
$$

**符号说明**：
- $\sum_\tau$：对所有可能轨迹求和
- $P(\tau; \theta)$：轨迹 $\tau$ 在策略 $\pi_\theta$ 下出现的概率
- $G(\tau)$：轨迹 $\tau$ 的总回报（一个确定的数，不依赖 $\theta$）

对 $\theta$ 求导：

$$
\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau; \theta) \cdot G(\tau)
$$

### 4.2 对数求导技巧（Log Derivative Trick）

这里有一个妙招。注意到：

$$
\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta)
$$

**推导**（对数的微分规则）：

$$
\nabla_\theta \log f = \frac{\nabla_\theta f}{f} \implies \nabla_\theta f = f \cdot \nabla_\theta \log f
$$

代入之前的式子：

$$
\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \cdot \nabla_\theta \log P(\tau; \theta) \cdot G(\tau)
= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau; \theta) \cdot G(\tau) \right]
$$

### 4.3 轨迹概率的分解

一条轨迹的概率为：

$$
P(\tau; \theta) = P(s_0) \prod_{t=0}^{T} \pi_\theta(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)
$$

**符号说明**：
- $P(s_0)$：初始状态（query）的概率，与 $\theta$ 无关
- $\pi_\theta(a_t \mid s_t)$：策略在 $s_t$ 下选 $a_t$ 的概率（这才依赖 $\theta$）
- $P(s_{t+1} \mid s_t, a_t)$：环境转移概率（在 LLM 中，选定 token 后下一状态确定，故此项为 1 或与 $\theta$ 无关）

取对数（乘积变求和）：

$$
\log P(\tau; \theta) = \underbrace{\log P(s_0)}_{\text{与}\theta\text{无关}} + \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) + \underbrace{\sum_{t=0}^{T} \log P(s_{t+1} \mid s_t, a_t)}_{\text{与}\theta\text{无关}}
$$

对 $\theta$ 求梯度，与 $\theta$ 无关的项消失：

$$
\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

### 4.4 策略梯度定理最终形式

代入并利用蒙特卡洛近似（单条轨迹），得到**策略梯度定理**：

$$
\boxed{
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]
}
$$

用优势函数 $\hat{A}_t$（见第 7 节）替换 $G_t$（降低方差），得到整理内容 a 中的 Policy Gradient 公式：

$$
\nabla_\theta L = \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t
$$

> **这就是整理内容 a 的疑问"这是由什么改写过来的"的答案**：Policy Gradient 公式是从期望总回报 $J(\theta)$ 出发，用对数求导技巧 + 轨迹分解 + 蒙特卡洛近似，一步步推导出来的。$G_t$ 被 $\hat{A}_t$ 替换是为了降低方差（见第 7 节）。

---

## 5. 对数求导技巧详解

整理内容 a 提问："为什么是 $\nabla \log \pi$ 而不是 $\nabla \pi$？$\pi_\theta(a|s)$ 在哪？"

### 5.1 为什么用 log

**直接梯度 $\nabla_\theta \pi_\theta(a|s)$ 的问题**：

对于 softmax 输出，$\pi_\theta(a|s) = \text{softmax}(\text{logits})_a$，其梯度涉及整个 softmax 的 Jacobian，每个 token 的梯度都与其他 token 耦合，计算复杂。

**$\nabla_\theta \log \pi_\theta(a|s)$ 的优势**：

$$
\log \pi_\theta(a|s) = \log \text{softmax}(\text{logits})_a = \text{logits}_a - \log \sum_{a'} e^{\text{logits}_{a'}}
$$

这就是 **log-softmax**，PyTorch 中直接有 `F.log_softmax(logits, dim=-1)`，梯度简洁稳定。

### 5.2 π 在哪里

整理内容 a 中问"$\pi_\theta(a|s)$ 在哪"。答案是：它藏在 ratio $r_t(\theta)$ 里：

$$
r_t(\theta) = \frac{\pi_{\theta_\text{new}}(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)} = \exp(\text{new\_logprob}_t - \text{old\_logprob}_t)
$$

因为 $\log \pi(a|s) = \text{logprob}$，所以 $\pi(a|s) = e^{\text{logprob}}$。两个策略的 $\pi$ 之比用 exp(差) 表示，这正是代码中的计算方式：

```python
ratio = torch.exp(new_logprobs - mb_logprobs)  # mb_logprobs 是 old policy 的 logprob
```

---

## 6. 贝尔曼方程与 TD Error（δ）

### 6.1 贝尔曼方程

**贝尔曼方程（Bellman Equation）** 描述的是价值函数的递推关系：

$$
\boxed{
V(s_t) = \mathbb{E}_{a_t \sim \pi} \left[ r_t + \gamma V(s_{t+1}) \right]
}
$$

**逐词解读**：
- 左边 $V(s_t)$：处于状态 $s_t$ 时，整条轨迹剩余部分的期望总回报（由 Critic 估计）
- 右边 $r_t$：在 $s_t$ 采取动作后立即获得的奖励（KL 惩罚 + RM 分数）
- 右边 $\gamma V(s_{t+1})$：从下一状态 $s_{t+1}$ 出发，后续轨迹的期望总回报打折后的值
- 合在一起就是说：**现在的价值 = 立即奖励 + 打折的未来价值**

**贝尔曼方程在代码中的体现**：整理内容 a 中的"马尔可夫性"假设就来源于此——下一状态的价值只依赖于当前状态，不依赖更早的历史。Critic 网络学习的就是这个 $V(s_t)$。

### 6.2 TD Error（δ_t）

当 Critic 估计不完美时，贝尔曼方程两边会有误差，这个误差就是 **TD Error（Temporal Difference Error，时序差分误差）**：

$$
\boxed{
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
}
$$

**逐词解读（对应整理内容 a）**：
- $r_t$：即时奖励，对应代码中 `rewards[t]`（= KL 惩罚 + RM 分数），整理内容 a 说"此时的 delta 已经汇总了 KL 散度 + RM reward 和 Critic 的 V"，正是因为 $r_t$ 里包含了 KL 和 RM，而 $V(s_t)$、$V(s_{t+1})$ 来自 Critic
- $V(s_{t+1})$：代码中 `values[t+1]`，Critic 对下一状态的估计
- $V(s_t)$：代码中 `values[t]`，Critic 对当前状态的估计
- $\delta_t > 0$：说明实际得到的（$r_t + \gamma V(s_{t+1})$）比预期的（$V(s_t)$）更好，动作 $a_t$ 优于平均水平
- $\delta_t < 0$：说明实际不如预期，动作 $a_t$ 差于平均水平
- $\delta_t = 0$：恰好符合预期

**代码对应**（整理内容 a 中"先算差分"的部分）：

```python
# 文档 b 中的公式在代码中的体现：
delta = rewards[t] + args.gamma * values[t + 1] * not_done - values[t]
# 其中 not_done 处理序列结束后不再有后续价值的情况
```

### 6.3 为什么说贝尔曼方程体现在 TD Error 里

贝尔曼方程告诉我们**理想状态下** $V(s_t) = r_t + \gamma V(s_{t+1})$，即 $\delta_t = 0$。

当 Critic 还没学好时，$\delta_t \ne 0$。TD Error 就是"贝尔曼方程的残差"——用它来更新 Critic，让 Critic 的 $V$ 估计越来越满足贝尔曼方程。

---

## 7. Critic、Value Function 与 GAE

### 7.1 为什么需要优势函数 A

用 $G_t$（实际总回报）直接估计策略梯度方差太大——不同轨迹的 $G_t$ 可能差距很大。

引入**基线（Baseline）** $V(s_t)$，用**优势（Advantage）** $A_t = G_t - V(s_t)$ 替代 $G_t$：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

$A_t$ 表示动作 $a_t$ 相对于"该状态的平均水平"好多少，方差更小，训练更稳定。

### 7.2 GAE：广义优势估计

**GAE（Generalized Advantage Estimation，广义优势估计）** 用多步 TD Error 的指数加权和来估计 $A_t$。

递推公式（**必须从后往前计算**）：

$$
\boxed{
A_t = \delta_t + \gamma \lambda A_{t+1}
}
$$

展开得：

$$
A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2 \delta_{t+2} + \cdots
$$

**逐词解读（对应整理内容 a）**：
- $\delta_t$：当前时刻的 TD Error（对 $s_t$ 的即时好坏评估）
- $\gamma \lambda A_{t+1}$：下一时刻优势的折扣累加
- $\lambda = 0$：退化为纯 TD（单步 $\delta_t$），低方差但高偏差
- $\lambda = 1$：退化为蒙特卡洛（整条轨迹的实际回报减 baseline），高方差但低偏差
- 实践中 $\lambda \approx 0.95$ 取中间值

**为什么必须从后往前算**（整理内容 a 的疑问）：

因为 $A_t$ 依赖 $A_{t+1}$，而 $A_{t+1}$ 又依赖 $A_{t+2}$，……所以要先算 $A_T$（最后一步），再算 $A_{T-1}$，以此类推。整理内容 a 中说"倒序把后面 token 的初步贡献按一定比例划归到前面"，正是这个意思。

**代码对应**：

```python
lastgaelam = 0
for t in reversed(range(response_length)):
    delta = rewards[t] + args.gamma * next_values - values[t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam
    advantages[t] = lastgaelam
```

---

## 8. PPO 的 Clip 目标函数

### 8.1 为什么需要 Clip

普通策略梯度每次 rollout 只能用一次（on-policy），样本利用率低。PPO 想用同一批数据更新多次（`ppo_epochs` 轮），但如果新旧策略差距太大，训练会不稳定。

**Clip 机制**：限制新旧策略之比 $r_t(\theta)$ 的范围，保证新策略不会偏离旧策略太远。

### 8.2 ratio 的定义

$$
r_t(\theta) = \frac{\pi_{\theta_\text{new}}(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)} = \exp\left(\text{new\_logprob}_t - \text{old\_logprob}_t\right)
$$

**逐词解读（对应整理内容 a）**：
- $\pi_{\theta_\text{old}}$（old policy）：rollout 时记录的 logprobs，即代码中 `mb_logprobs`
- $\pi_{\theta_\text{new}}$（new policy）：当前参数下重新对同一 token 序列计算的 logprobs，即 `new_logprobs`
- old policy **不是**额外复制一份模型，而是缓存的 `mb_logprobs`

**代码对应**：

```python
new_logprobs = get_logprobs(policy, query_response)  # 当前参数
ratio = torch.exp(new_logprobs - mb_logprobs)        # mb_logprobs 是 rollout 时缓存的
```

### 8.3 PPO Clip 损失函数

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

**符号说明**：
- $\epsilon$：clip 范围超参数（代码中 `args.cliprange`，通常取 0.2），限制 $r_t \in [0.8, 1.2]$
- $\hat{A}_t$：白化后的优势，即整理内容 a 中的 `mb_advantage`
- $\min(\cdot, \cdot)$：取两者的较小值，目的是"悲观估计"——阻止策略过度利用优势

**对应整理内容 a 中的具体计算**：

```python
# pg_losses  = -A_hat * ratio         （未 clip 的 loss）
# pg_losses2 = -A_hat * clip(ratio)   （clip 后的 loss）
pg_loss = torch.max(pg_losses, pg_losses2).mean()  # 取逐 token 最大（最悲观），再取均值
```

> **注意符号**：代码中用的是**损失函数**（要最小化），所以 $pg\_loss = -L^{\text{CLIP}}$，前面有负号。

**数值举例（来自整理内容 a）**：

| token | $\hat{A}_t$ | ratio | clip(ratio) | pg_loss | pg_loss2 | max |
|-------|-------------|-------|-------------|---------|----------|-----|
| 0 | -1.6527 | 0.7408 | **0.8** | +1.2243 | **+1.3221** | +1.3221 |
| 1 | +0.6578 | 0.8607 | 0.8607 | -0.5661 | -0.5661 | -0.5661 |
| 2 | +0.9104 | **1.3499** | **1.2** | -1.2289 | **-1.0925** | -1.0925 |
| 3 | +0.0845 | 0.8607 | 0.8607 | -0.0727 | -0.0727 | -0.0727 |

均值 = $(1.3221 - 0.5661 - 1.0925 - 0.0727)/4 = -0.4092/4 = -0.1023$

---

## 9. Critic 的更新：Value Loss 与 Returns

整理内容 a 中提到 `returns = advantages + values`，这是 Critic 的更新目标。

### 9.1 Returns 的计算

**Return（回报）** 是 Critic 的训练目标（应该预测的"正确答案"）：

$$
\text{returns}_t = A_t + V(s_t) = \delta_t + \gamma V(s_{t+1}) - V(s_t) + V(s_t) = r_t + \gamma V(s_{t+1})
$$

化简后就是**贝尔曼目标**：$\text{returns}_t = r_t + \gamma V(s_{t+1})$。

**代码对应**：

```python
returns = advantages + values  # 即 A_t + V(s_t) = r_t + γV(s_{t+1})
```

### 9.2 Value Loss

Critic 用**均方误差**学习这个目标：

$$
L^{\text{value}} = \mathbb{E}_t \left[ \left( V_\theta(s_t) - \text{returns}_t \right)^2 \right]
$$

Critic 的更新就是让其输出 $V_\theta(s_t)$ 尽量接近 $\text{returns}_t$（即贝尔曼目标）。

---

## 10. 反向传播梯度的完整路径

整理内容 a 中给出的梯度公式：

$$
\nabla_\theta pg\_loss_t = \exp(\text{new\_logprob}_t - \text{old\_logprob}_t) \cdot \nabla_\theta \text{new\_logprob}_t \cdot \hat{A}_t
$$

这正是第 4.4 节推导结果的代码实现形式。

**对应关系**：

| 数学符号 | 代码含义 |
|---------|---------|
| $\exp(\text{new\_logprob}_t - \text{old\_logprob}_t)$ | `ratio`（新旧策略之比 $r_t(\theta)$） |
| $\nabla_\theta \text{new\_logprob}_t$ | PyTorch autograd 对 `new_logprobs` 的梯度 |
| $\hat{A}_t$ | `mb_advantage`（白化后优势） |
| $r_t(\theta) \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)$ | $\approx \nabla_\theta \log \pi_{\theta_\text{new}}(a_t|s_t)$ 当 $r_t \approx 1$ 时 |

**与标准 Policy Gradient 的关系**：

$$
\nabla_\theta L^{\text{PG}} = \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t
$$

PPO 中多了一个 $r_t(\theta)$ 系数（ratio），整理内容 a 称之为"对 Policy Gradient 的魔改"，本质是 **Importance Sampling（重要性采样）**——允许用 old policy 采样的数据来估计 new policy 的梯度。

---

## 11. 完整数据流总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PPO 完整数据流                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Rollout（蒙特卡洛采样）                                           │
│     Actor 按 π_θ(a|s) 采样 token，记录 logprobs                      │
│     ↓                                                               │
│  2. Reward 计算                                                      │
│     RM Model → score（只在最后 token 给分）                          │
│     KL 惩罚 = -kl_coef × (logprob - ref_logprob)                   │
│     rewards[t] = kl_penalty[t]（+ score，仅在最后 token）            │
│     ↓                                                               │
│  3. TD Error & GAE（贝尔曼方程体现）                                  │
│     Critic → values[t]                                             │
│     δ_t = rewards[t] + γ·values[t+1] - values[t]                  │
│     A_t = δ_t + γλ·A_{t+1}（从后往前）                              │
│     returns[t] = A_t + values[t]                                   │
│     ↓                                                               │
│  4. 白化 Advantage                                                   │
│     Â_t = (A_t - mean(A)) / std(A)                                 │
│     ↓                                                               │
│  5. PPO Update（多个 epoch）                                          │
│     ratio = exp(new_logprob - old_logprob)                         │
│     pg_loss = max(-Â·ratio, -Â·clip(ratio, 1-ε, 1+ε))             │
│     value_loss = (V_θ(s) - returns)²                               │
│     total_loss = pg_loss + vf_coef × value_loss                    │
│     反向传播更新 Actor(θ) 和 Critic                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 各理论要素对应总结

| 理论概念 | 在 PPO 中的体现 | 代码变量 |
|---------|--------------|---------|
| **蒙特卡洛采样** | Rollout 阶段按概率分布采样 token，生成轨迹 | `generate()` 使用采样而非贪心 |
| **贝尔曼方程** | $V(s_t) = r_t + \gamma V(s_{t+1})$，即 returns 的定义 | `returns = advantages + values` |
| **TD Error (δ)** | 即时评估动作好坏的差值信号，Critic 训练的依据 | `delta = rewards[t] + γ·values[t+1] - values[t]` |
| **GAE** | 多步 TD Error 的指数加权，从后往前累积 | `advantages[t] = delta + γλ·advantages[t+1]` |
| **Policy Gradient** | $\nabla \log \pi \cdot \hat{A}$ | pg_loss 反向传播 |
| **对数求导技巧** | 把 $\nabla \pi$ 变成 $\pi \cdot \nabla \log \pi$，化简计算 | `log_softmax` + autograd |
| **Importance Sampling** | ratio 允许 off-policy 多轮更新 | `ratio = exp(new_logprob - old_logprob)` |
| **KL 惩罚** | 防止 Actor 偏离 Reference 太远，作为 $r_t$ 的一部分 | `non_score_reward = -kl_coef * kl` |

---

*本文档对应 MedicalGPT PPO 训练流程，代码见 `ppo_training.py`。*
