# 任务：实现 Conditional Constrained Diffusion（按附带 note 的方法）

你要从零搭一个项目，把多资产（10 只科技股）日频 logret 的**联合横截面分布**用 diffusion 建模，并实现一个 h-function 分类器，最终用 Doob 变换做"条件在利率冲击事件上"的条件生成，研究极端利率冲击下这 10 只股票的相关性结构如何变化。方法依据是 Renyuan 的 note `Conditional_constrained_diffusion_model.pdf`（Bayes 倾斜 + Doob h-transform + 直接训练时间相关分类器）。

请严格按本规格实现。涉及 SDE 闭式、损失、引导项的公式我已写好且自洽，直接照抄，不要自行改符号约定。

---

## 0. 数据语义（先读懂再写）

- `X ∈ R^10`：**某一个交易日**这 10 只股票的 logret 向量。每个样本 = 一天的横截面。diffusion 生成的就是一个 10 维向量。**不建模跨日时间动态**，只建模 10 只股票当日 logret 的联合分布。
- `Z = Δy_t`：10 年期 TIPS 实际收益率的**每日变化量**（标量）。
- 事件 `S = {Δy_t 落在 top 10%}`：即"利率急升"日。
- `B = 1{Z ∈ S} ∈ {0,1}`：二元事件标签。
- **关键：Z 不作为特征进入任何网络。** Z 只用来算阈值、生成标签 B。模型输入只有 X（和扩散时间 t）。

---

## 1. 环境与项目结构

依赖：`torch`, `numpy`, `pandas`, `matplotlib`。（不要用 sklearn 做标准化，手写均值/方差即可，方便保存与反变换。）

```
rate_shock_diffusion/
├── config.py          # 所有超参与常量
├── data.py            # 加载、清洗、logret、Δy、标准化、切分、打标签
├── sde.py             # VP SDE：beta(t)、alpha/sigma 闭式、前向加噪、反向 drift
├── models.py          # TransformerScore（backbone）、TransformerClassifier（h）
├── train_pretrain.py  # 预训练无条件 diffusion
├── train_hfunction.py # 训练 h-function
├── sample.py          # 无条件采样 + Doob 引导条件采样
├── viz.py             # 直方图、相关性热力图
└── main.py            # 串起全流程，按顺序跑
```

全程固定随机种子（`seed=0`，numpy + torch）。设备：`device = "cuda" if torch.cuda.is_available() else "cpu"`，所有张量与模型放到 device。所有图保存到 `figures/`，checkpoint 保存到 `ckpt/`。

---

## 2. 数据处理（data.py）

输入文件路径用 `os.path.expanduser("~/Desktop/tech_stocks_tips.csv")`（Mac/Linux 通用；若 Windows 找不到，回退到当前目录同名文件）。

**先打印 `df.columns` 和 `df.head()` 自检。** 预期结构：第一列是日期（设为索引），随后 10 列是股票价格（列名为 ticker：IBM, CSCO, AAPL, MSFT, ORCL, INTC, TXN, QCOM, AMAT, ADBE，顺序以文件为准），最后一列是实际收益率水平 `y10_real`。把这 10 个 ticker 列识别为 `price_cols`，收益率列识别为 `y10_real`（若列名不同，取唯一那个非 ticker 的数值列）。

处理顺序，严格如下：

1. **删 NaN 行**：`df = df.dropna(how="any")`（任何一列有 NaN 的整行删掉）。先做这一步。
2. **价格 → logret**：对每个 price_col，`logret = log(P_t) - log(P_{t-1})`（`np.log(df[price_cols]).diff()`）。
3. **y → Δy**：`dy = df["y10_real"].diff()`。
4. 上面两个 diff 各产生一行 NaN（首行），合并后再 `dropna()` 对齐。得到一个干净表：10 列 logret + 1 列 `dy`。
5. **时序切分（不打乱）**：按时间顺序，前 80% 为训练集，后 20% 为测试集。`n_train = int(0.8 * len(data))`，`train = data.iloc[:n_train]`，`test = data.iloc[n_train:]`。**绝不 shuffle**（避免前视泄漏）。
6. **标准化 10 列 logret（只用训练集统计量）**：
   - `mu = train[logret_cols].mean()`，`sd = train[logret_cols].std(ddof=0)`，逐列。
   - `X_train = (train[logret_cols] - mu) / sd`，`X_test = (test[logret_cols] - mu) / sd`。
   - **保存 `mu, sd`**（每只股票一个），后面要把生成样本反变换回原始 logret 画直方图：`x_raw = x_std * sd + mu`。
   - 注：逐列标准化会抹掉各股波动率差异，但**保留相关性结构**（相关系数与尺度无关），正是我们要研究的对象，所以没问题。
7. **事件标签 B（阈值只用训练集）**：
   - `thr = train["dy"].quantile(0.90)`（训练集 Δy 的 90 分位）。
   - `B_train = (train["dy"] >= thr).astype(float)`，`B_test = (test["dy"] >= thr).astype(float)`（测试集**沿用训练集阈值**，不重新算，避免泄漏）。
   - 打印训练集正例数量与占比（应 ≈ 10%）。

输出：`X_train (N_tr,10) float32 tensor`、`B_train (N_tr,) float32`、`X_test`、`B_test`、`mu`、`sd`、`thr`、`logret_cols`。

> **尺度问题的明确结论（重要）**：logret 与 Δy 尺度不同**不影响模型**。Δy 不进任何网络，只用于上面第 7 步打二元标签。进模型的只有 10 列 logret，已逐列标准化到同一尺度（零均值单位方差），既满足 VP SDE 对 ~N(0,I) 的要求，又保留相关结构。不要把 Δy 拼进 X，也不要对 Δy 做标准化后喂模型。

---

## 3. VP SDE（sde.py）

VP SDE，linear schedule，常量：`BETA_MIN = 0.01`，`BETA_MAX = 10.0`，时间 `t ∈ [0,1]`（t=0 为数据，t=1 为噪声）。

前向 SDE：`dx = -0.5 β(t) x dt + sqrt(β(t)) dW`，`β(t) = BETA_MIN + t*(BETA_MAX - BETA_MIN)`。

实现以下闭式（全部支持 batch 的 t 向量）：

```
beta(t)        = BETA_MIN + t*(BETA_MAX - BETA_MIN)
beta_integral  B(t) = BETA_MIN*t + 0.5*(BETA_MAX - BETA_MIN)*t**2
alpha(t)       = exp(-0.5 * B(t))                 # 前向均值系数
sigma(t)       = sqrt(1 - exp(-B(t)))             # 前向标准差   (注意 sigma^2 = 1 - alpha^2)
```

前向加噪（闭式，训练用）：给定干净 `x0` 与 `t`，采 `z ~ N(0,I)`，
```
x_t = alpha(t) * x0 + sigma(t) * z
```

score 与 ε 的关系（ε-prediction 参数化）：`score(x_t,t) = -eps_theta(x_t,t) / sigma(t)`。

**无条件反向采样**（Euler–Maruyama，t 从 1 走到 0，步长 dt<0）：
```
drift = -0.5*beta(t)*x - beta(t)*score(x,t)      # = -0.5*beta*x + beta*eps_theta/sigma
x_{t+dt} = x + drift*dt + sqrt(beta(t))*sqrt(-dt)*N(0,I)   # dt<0
```

**条件反向采样（Doob 引导，最终研究步用）**：条件 score = 无条件 score + ∇_x log h：
```
cond_score = score(x,t) + grad_x log( h_phi(x,t) + DELTA )
drift      = -0.5*beta(t)*x - beta(t)*cond_score
```
其中 `DELTA = 1e-3` 是数值下限（对应 note eq (14) 的 δ）。`grad_x log(h+δ)` 用 autograd 对 x 求导得到（采样时 `x.requires_grad_(True)`，对 `torch.log(h+DELTA).sum()` 反传取 `x.grad`）。可选地给引导项乘一个强度 `GAMMA`（默认 1.0 = 精确 Doob，留出来方便做敏感性实验）。

> 说明：note 里 eq (14) 用的是 noise→data 递增时间约定（加 `+a∇log h`），本规格用 Song 等的 score-SDE 反向时间约定（数据在 t=0），符号我已对齐，照本规格实现即可，**不要**两套混用。

---

## 4. 模型架构（models.py）—— backbone 与 h 都用 Transformer

两个网络共享同一套"tokenize 10 只股票 → 互相 attention"的设计。

**公共 tokenizer**：把输入 `x ∈ R^{B×10}` 变成 10 个 token：
- 每个标量 `x_j`（第 j 只股票的（带噪）logret）经一个 `Linear(1 → d_model)` 升维。
- 加一个**可学习的逐股票身份 embedding** `nn.Embedding(10, d_model)`（让模型区分哪个 token 是哪只股票；横截面无自然顺序，所以用 learnable id 而非正弦位置编码）。
- 得到 `tokens ∈ R^{B×10×d_model}`。

**时间/噪声条件**：对扩散时间 `t` 做 sinusoidal embedding（维度 `d_model`）再过一个 2 层 MLP，得到 `t_emb ∈ R^{B×d_model}`，**广播加到每个 token 上**（或用 FiLM；加法即可）。

**Transformer encoder**：`nn.TransformerEncoder`，`L` 层，每层多头自注意力（在 10 个 token 之间做 attention，捕捉横截面依赖）+ FFN。`batch_first=True`。

两个 head：

- **TransformerScore（diffusion backbone，ε 预测）**：encoder 输出 `B×10×d_model` → 逐 token 过 `Linear(d_model → 1)` → squeeze 成 `B×10`，即预测的噪声 `eps_theta`。
- **TransformerClassifier（h-function）**：encoder 输出后做 mean-pool（对 10 个 token 求均值，或加一个 learnable CLS token 取其输出）→ `MLP(d_model → d_model → 1)` 得到 **logit**（不要在网络里加 sigmoid，训练用 BCEWithLogits 更稳）。推理时 `h = sigmoid(logit)`。

默认超参（写进 config，可调）：`d_model=128, n_heads=4, n_layers=4, dim_ff=256, dropout=0.0`。两个网络结构相同，只是 head 不同。

---

## 5. 预训练无条件 diffusion（train_pretrain.py）

去噪 score matching / ε-prediction：

```
对每个 batch x0 (来自 X_train):
    t ~ Uniform(eps0, 1)        # eps0=1e-3，避开 t=0 奇异
    z ~ N(0, I)
    x_t = alpha(t)*x0 + sigma(t)*z
    loss = MSE( eps_theta(x_t, t), z )      # 逐元素均方
```

优化器 Adam，`lr=2e-4`，`batch_size=256`，训练到 loss 平台（数据集约几千行，建议 ~300 epoch 或固定 ~30k steps，打印 loss 曲线并保存到 figures/）。建议维护一份 **EMA 权重**（decay=0.999），采样时用 EMA。保存 checkpoint 到 `ckpt/pretrain.pt`。

---

## 6. 可视化 1：生成 vs 实际 logret 直方图（viz.py）

预训练后，用无条件反向采样（§3）生成 `M=20000` 个样本（10 维，标准化空间），**反标准化回原始 logret**（`x_raw = x_std*sd + mu`，逐列）。

画一张 `2×5` 子图，每只股票一格：把**生成的 logret 分布**与**测试集实际 logret 分布**叠加直方图（`density=True`，半透明，不同颜色，图例标注 generated / actual）。子图标题为 ticker。保存 `figures/hist_pretrain_vs_actual.png`。这是检验 backbone 是否学到了无条件边际分布。

---

## 7. 训练 h-function（train_hfunction.py）

按 note eq (9) 直接训练时间相关二元分类器。**用全体训练集**（正例=top10% 的 B=1，负例=其余 B=0），不是只用 top10%：

```
对每个 batch (x0, b) (来自 X_train, B_train):
    t ~ Uniform(eps0, 1)
    z ~ N(0, I)
    x_t = alpha(t)*x0 + sigma(t)*z          # 与 backbone 同一前向加噪
    logit = h_phi(x_t, t)                    # 标量
    loss = BCEWithLogitsLoss(pos_weight=w)(logit, b)
```

**类别不平衡**：正例仅 ~10%，用 `pos_weight ≈ (#neg/#pos) ≈ 9`（按训练集实际比例算）传入 `BCEWithLogitsLoss`，或用平衡采样。否则分类器会退化成全预测 0。

Adam `lr=2e-4`，`batch_size=256`，训练到验证 BCE 收敛。监控正类的 AUC 或 average precision 以确认确实学到信号（若 AUC≈0.5 说明 X 与事件几乎独立、引导会退化，需在日志里明确警告）。保存 `ckpt/hfunction.pt`，并保存 loss 曲线。

---

## 8. 条件生成 + 可视化 2（研究结论，sample.py + viz.py）

这是项目的最终目的：对比"利率冲击事件下"的相关性结构。

1. **无条件生成**：用预训练 backbone 采 `M=20000` 个样本 → `X_uncond`（反标准化后算相关）。
2. **条件生成**：用 §3 的 Doob 引导反向 SDE（backbone + h-function），采 `M=20000` 个样本 → `X_cond`，即近似 `X | Z∈S`。
3. **实际对照**：
   - `X_actual_all`：测试集（或全样本）全部实际 logret。
   - `X_actual_S`：实际数据中 `B=1`（Δy 在 top10%）那些天的 logret。
4. **相关性对比**：对上面四组各算 `10×10` 皮尔逊相关矩阵，画四张热力图并排（同色标 `vmin=-1,vmax=1`，标注 ticker），保存 `figures/corr_compare.png`。核心检验：**条件生成 `X_cond` 的相关结构应当向实际事件日 `X_actual_S` 靠拢，并明显区别于无条件 `X_uncond`**（典型表现：利率冲击日横截面相关性整体抬升/抱团）。
5. 量化差距：打印 `||corr(X_cond) - corr(X_actual_S)||_F` 与 `||corr(X_uncond) - corr(X_actual_S)||_F`，前者应更小，说明条件化抓到了事件相关结构。
6. （可选）再画一张条件 vs 无条件生成的 logret 直方图对比，看事件下边际是否变宽/偏移。

---

## 9. 运行顺序（main.py）

按序执行并打印每步关键信息（样本量、阈值、正例占比、各 loss、相关矩阵 Frobenius 距离）：

1. `data.py` 处理数据，打印自检信息。
2. `train_pretrain.py` 训练 backbone（+EMA），存 ckpt。
3. `viz` 生成图 1（hist_pretrain_vs_actual）。
4. `train_hfunction.py` 训练 h，存 ckpt，打印 AUC。
5. `sample.py` 做无条件 + Doob 条件采样。
6. `viz` 生成图 2（corr_compare）+ 打印量化指标。

写一个简短 `README.md` 说明如何 `pip install` 依赖、把 csv 放到桌面、`python main.py` 一键跑通，以及各产出文件含义。

---

## 关键正确性检查清单（实现时务必满足）

- [ ] 标准化的 `mu/sd` 与事件阈值 `thr` **只用训练集**统计，测试集沿用，无前视泄漏。
- [ ] 时序切分不 shuffle。
- [ ] Δy 不进任何网络，只用于打标签 B。
- [ ] h-function 用**全体**训练集（正负例都要），并处理 10/90 类别不平衡。
- [ ] backbone 与 h 用**同一套** VP SDE 前向加噪（同 alpha/sigma）。
- [ ] backbone 与 h 都是 Transformer：先把 10 只股票各 token 化，再做横截面 self-attention。
- [ ] Doob 引导项 `grad_x log(h+δ)` 用 autograd 对输入 x 求导，δ=1e-3。
- [ ] 反向采样符号严格按 §3 的本规格（不要混入 note 的相反时间约定）。
- [ ] 生成样本画图前**反标准化**回原始 logret。
