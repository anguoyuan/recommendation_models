import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Tuple
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.transformers import RMSNorm_npu as RMSNormNPU
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const


# =============================================================================
# 3.1 Input Tokenization
# =============================================================================

@ModelRegistry.register()
class TokenMixerLargeInput(BaseModel):
    """
    TokenMixer-Large 输入 Tokenization 模块 (Section 3.1)

    将原始稀疏 one-hot 特征 → 稠密 embedding → 维度对齐的语义 token。
    各语义组通过独立的 MLP 映射到统一的 inner_dim 维度。
    可选地引入一个全局 token (类似 BERT [CLS]) 聚合全局信息。

    输入:  (B, D_raw)  — 拼接后的特征向量
    输出:  (B, T, inner_dim) — T 个语义 token
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        self.T = model_cfg[Const.HP].get("T")
        self.token_dim = model_cfg[Const.HP].get("token_dim")
        self.inner_dim = model_cfg[Const.HP].get("inner_dim")
        self.usable_dim = model_cfg[Const.HP].get("usable_dim")
        self.use_global_token = model_cfg[Const.HP].get("use_global_token", False)

        self.num_local_tokens = self.T - 1 if self.use_global_token else self.T

        self.proj = nn.Linear(self.token_dim, self.inner_dim, bias=False)

        if self.use_global_token:
            self.global_mlp = nn.Sequential(
                nn.Linear(self.num_local_tokens * self.inner_dim, self.inner_dim),
                nn.GELU(),
                nn.Linear(self.inner_dim, self.inner_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.size()
        x = x[:, :self.usable_dim]

        local_x = x.view(B, self.num_local_tokens, self.token_dim)
        local_tokens = self.proj(local_x)  # (B, num_local, inner_dim)

        if not self.use_global_token:
            return local_tokens

        global_in = local_tokens.reshape(B, self.num_local_tokens * self.inner_dim)
        global_token = self.global_mlp(global_in).unsqueeze(1)  # (B, 1, inner_dim)
        return torch.cat([global_token, local_tokens], dim=1)   # (B, T, inner_dim)


# =============================================================================
# 3.2 Token Mixing & Reverting
# =============================================================================

@ModelRegistry.register()
class TokenMixerMixing(BaseModel):
    """
    TokenMixer-Large 的 Multi-Head Token Mixing 模块 (Section 3.2)

    将每个 token 的 embedding 均匀拆分为 H 个 head，再将所有 token 的同一个 head
    拼接在一起，形成 H 个"混合 token"，每个混合 token 的维度为 T*D/H。

    公式: s_h = Concat(x_1^h, x_2^h, ..., x_T^h),  h = 1..H
    其中 x_t^h 是第 t 个 token 的第 h 个 head 切片 (维度 D/H)

    操作步骤:
        (B, T, D) → view(B, T, H, D/H) → permute(0,2,1,3)
                  → (B, H, T, D/H)    → view(B, H, T*D/H)

    默认 H = T，此时输出 (B, T, T*D/T) = (B, T, D)，形状与输入一致。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.T = model_cfg[Const.HP].get("T")
        self.inner_dim = model_cfg[Const.HP].get("inner_dim")
        self.H = model_cfg[Const.HP].get("H")
        assert self.inner_dim % self.H == 0, \
            f"inner_dim ({self.inner_dim}) must be divisible by H ({self.H})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, H, T*D/H)"""
        B, T, D = x.shape
        H = self.H
        d = D // H  # per-head dim
        return (
            x.view(B, T, H, d)          # (B, T, H, D/H)
             .permute(0, 2, 1, 3)        # (B, H, T, D/H)
             .contiguous()
             .view(B, H, T * d)          # (B, H, T*D/H)
        )


@ModelRegistry.register()
class TokenMixerReverting(BaseModel):
    """
    TokenMixer-Large 的 Token Reverting 模块 (Section 3.2)

    TokenMixerMixing 的严格逆操作。将混合后的 (B, H, T*D/H) 还原为 (B, T, D)，
    保证残差连接的维度一致性，使残差信号在任意深度的网络中稳定传播。

    操作步骤 (Mixing 的逆):
        (B, H, T*D/H) → view(B, H, T, D/H) → permute(0,2,1,3)
                      → (B, T, H, D/H)     → view(B, T, D)
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.T = model_cfg[Const.HP].get("T")
        self.inner_dim = model_cfg[Const.HP].get("inner_dim")
        self.H = model_cfg[Const.HP].get("H")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, H, T*D/H) -> (B, T, D)"""
        B, H, _ = h.shape
        T = self.T
        d = self.inner_dim // H  # = D/H
        return (
            h.view(B, H, T, d)          # (B, H, T, D/H)
             .permute(0, 2, 1, 3)        # (B, T, H, D/H)
             .contiguous()
             .view(B, T, H * d)          # (B, T, D)
        )


# =============================================================================
# 3.3 Per-token SwiGLU & Sparse-PerToken MoE (S-P MoE)
# =============================================================================

@ModelRegistry.register()
class PerTokenSwiGLU(BaseModel):
    """
    Per-token SwiGLU (Section 3.3)

    每个 token 位置 t 拥有独立参数，建模不同 token 的异构特征子空间。

    公式:
        pSwiGLU(x_t) = W_down^t · (Swish(W_gate^t · x_t) ⊙ (W_up^t · x_t)) + b^t

    其中:
        W_gate^t ∈ R^{D × kD}   — gate 分支
        W_up^t   ∈ R^{D × kD}   — value 分支
        W_down^t ∈ R^{kD × D}   — 输出投影
        ⊙ 表示逐元素乘法, Swish = SiLU

    输入/输出: (bs, T, D) → (bs, T, D)
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        D = model_cfg[Const.HP].get("D")
        T = model_cfg[Const.HP].get("T")
        k = model_cfg[Const.HP].get("k")
        bias = model_cfg[Const.HP].get("bias")
        dropout_p = model_cfg[Const.HP].get("dropout_p")
        down_init_scale = model_cfg[Const.HP].get("down_init_scale", 1.0)

        self.D = D
        self.T = T
        self.kD = int(round(k * D))
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # Gate branch: D -> kD
        self.W_gate = nn.Parameter(torch.empty(T, D, self.kD))
        # Up (value) branch: D -> kD
        self.W_up = nn.Parameter(torch.empty(T, D, self.kD))
        # Down (output) branch: kD -> D
        self.W_down = nn.Parameter(torch.empty(T, self.kD, D))

        if bias:
            self.b_gate = nn.Parameter(torch.empty(T, self.kD))
            self.b_up = nn.Parameter(torch.empty(T, self.kD))
            self.b_down = nn.Parameter(torch.empty(T, D))
        else:
            self.b_gate = None
            self.b_up = None
            self.b_down = None

        self.down_init_scale = down_init_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W_gate)
        nn.init.xavier_normal_(self.W_up)
        nn.init.xavier_normal_(self.W_down)
        # 论文建议降低 W_down 的初始化方差以稳定训练
        if self.down_init_scale != 1.0:
            self.W_down.data.mul_(self.down_init_scale)

        if self.b_gate is not None:
            nn.init.zeros_(self.b_gate)
            nn.init.zeros_(self.b_up)
            nn.init.zeros_(self.b_down)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (bs, T, D) -> (bs, T, D)"""
        bs, T, D = s.shape

        # Gate branch
        g = torch.einsum("bti,tik->btk", s, self.W_gate)
        if self.b_gate is not None:
            g = g + self.b_gate

        # Up branch
        v = torch.einsum("bti,tik->btk", s, self.W_up)
        if self.b_up is not None:
            v = v + self.b_up

        # SwiGLU: silu(gate) ⊙ up
        h = F.silu(g) * v
        h = self.dropout(h)

        # Down projection
        out = torch.einsum("btk,tkd->btd", h, self.W_down)
        if self.b_down is not None:
            out = out + self.b_down

        out = self.dropout(out)
        return out


@ModelRegistry.register(req_subs={"PerTokenSwiGLU"})
class SparsePerTokenMoE(BaseModel):
    """
    Sparse-PerToken MoE (S-P MoE) (Section 3.3)

    在原始 RankMixer 的 ReLU-MoE 基础上升级为 "Sparse Train, Sparse Infer" 范式。
    核心改进:
      1. 专家架构从 PerTokenFFN 升级为 PerTokenSwiGLU
      2. 引入 per-token 共享专家 (SharedExpert)，始终激活，保证训练稳定性
      3. TopK 稀疏路由 + Gate Scaling (α) 补偿稀疏激活导致的梯度衰减

    公式:
        y_t = α · Σ_{j∈TopK(g(x_t))} g_j(x_t) · Expert_j(x_t) + SharedExpert(x_t)

    其中:
        g(x_t) = ReLU(W_r · x_t)          — 路由得分 (ReLU保持动态稀疏性)
        α = num_routed_experts / top_k      — 门控缩放因子
        SharedExpert 为 per-token SwiGLU    — 始终激活的共享专家

    输入:  (bs, T, D)
    输出:  (bs, T, D), (reg_loss, active_ratio)
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        num_routed_experts = model_cfg[Const.HP].get("num_routed_experts")
        inner_dim = model_cfg[Const.HP].get("inner_dim")
        num_tokens = model_cfg[Const.HP].get("num_tokens")
        k = model_cfg[Const.HP].get("k")
        top_k = model_cfg[Const.HP].get("top_k")
        bias = model_cfg[Const.HP].get("bias")
        dropout_p = model_cfg[Const.HP].get("dropout_p")
        down_init_scale = model_cfg[Const.HP].get("down_init_scale", 1.0)

        self.D = inner_dim
        self.T = num_tokens
        self.num_routed = num_routed_experts
        self.top_k = top_k
        self.alpha = num_routed_experts / top_k  # gate scaling factor

        # Router: D -> num_routed_experts
        self.router = nn.Linear(self.D, self.num_routed, bias=False)

        # Per-token SwiGLU expert config
        expert_hp = {
            "D": inner_dim,
            "T": num_tokens,
            "k": k,
            "bias": bias,
            "dropout_p": dropout_p,
            "down_init_scale": down_init_scale,
        }
        self.model_cfg[Const.SUB_MODELS]["PerTokenSwiGLU"][Const.HP] = expert_hp

        # Routed experts
        self.routed_experts = nn.ModuleList(
            [self.init_sub_model("PerTokenSwiGLU") for _ in range(num_routed_experts)]
        )

        # Shared expert (per-token, always activated)
        self.shared_expert = self.init_sub_model("PerTokenSwiGLU")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.router.weight)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            s: (bs, T, D)
        Returns:
            y: (bs, T, D)
            (reg_loss, active_ratio): 辅助损失和稀疏度统计
        """
        B, T, D = s.shape

        # ReLU routing — 保持动态稀疏性
        gates = F.relu(self.router(s))  # (B, T, num_routed)

        # TopK sparse selection
        topk_vals, topk_idx = gates.topk(self.top_k, dim=-1)  # (B, T, top_k)

        # 构建稀疏 gate mask，只保留 TopK 位置的值
        mask = torch.zeros_like(gates)
        mask.scatter_(-1, topk_idx, 1.0)
        sparse_gates = gates * mask  # 非 TopK 位置清零

        # Gate scaling: α 补偿稀疏激活导致的梯度衰减
        scaled_gates = self.alpha * sparse_gates  # (B, T, num_routed)

        # 统计
        reg_loss = scaled_gates.sum(-1).sum(-1)        # (B,) 用于辅助正则化
        active_ratio = (sparse_gates > 0).float().mean()  # 稀疏率

        # 计算所有 routed expert 的输出
        expert_outs = torch.stack(
            [exp(s) for exp in self.routed_experts], dim=2
        )  # (B, T, num_routed, D)

        # 加权组合 routed experts
        routed_out = torch.einsum("btjd,btj->btd", expert_outs, scaled_gates)

        # Shared expert (始终激活)
        shared_out = self.shared_expert(s)  # (B, T, D)

        y = routed_out + shared_out
        return y, (reg_loss, active_ratio)


# =============================================================================
# 3.4 TokenMixer-Large Block & Overall Model
# =============================================================================

@ModelRegistry.register(req_subs={"TokenMixerMixing", "TokenMixerReverting", "SparsePerTokenMoE"})
class TokenMixerLargeBlock(BaseModel):
    """
    TokenMixer-Large Block (Section 3.4)

    每个 Block 的结构为:
        (RMSNorm → Mixing → S-P MoE → Reverting) + Residual
        (RMSNorm → S-P MoE) + Residual

    采用 Pre-Norm 风格 (RMSNorm)，经消融实验验证优于 Post-Norm。

    第一个 S-P MoE 在 mixed token 空间中操作 (channel mixing)；
    第二个 S-P MoE 在恢复后的 original token 空间中操作。
    Reverting 操作保证了残差连接的维度一致性。

    伪代码:
        h = Reverting(S_P_MoE_1(Mixing(RMSNorm(x))))
        x = x + h
        y = S_P_MoE_2(RMSNorm(x))
        x = x + y
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        inner_dim = model_cfg[Const.HP].get("inner_dim")
        T = model_cfg[Const.HP].get("T")
        # H 默认等于 T，此时 mixing 输出 (B, T, D) 与输入同形
        H = model_cfg[Const.HP].get("H", T)
        num_routed_experts = model_cfg[Const.HP].get("num_routed_experts")
        top_k = model_cfg[Const.HP].get("top_k")
        k = model_cfg[Const.HP].get("k")
        dropout_p = model_cfg[Const.HP].get("dropout_p")
        down_init_scale = model_cfg[Const.HP].get("down_init_scale", 1.0)

        # mixing 输出的 token 数和维度
        # (B, T, D) → Mixing → (B, H, T*D/H)
        mixed_tokens = H
        mixed_dim = T * inner_dim // H

        # Pre-Norm: 两个 RMSNorm 分别作用于 (B,T,D) 空间
        self.norm1 = RMSNormNPU(inner_dim, Const.EPS)
        self.norm2 = RMSNormNPU(inner_dim, Const.EPS)

        # Mixing (B, T, D) → (B, H, T*D/H)
        self.model_cfg[Const.SUB_MODELS]["TokenMixerMixing"][Const.HP] = {
            "T": T,
            "inner_dim": inner_dim,
            "H": H,
        }
        self.mixing = self.init_sub_model("TokenMixerMixing")

        # Reverting (B, H, T*D/H) → (B, T, D)
        self.model_cfg[Const.SUB_MODELS]["TokenMixerReverting"][Const.HP] = {
            "T": T,
            "inner_dim": inner_dim,
            "H": H,
        }
        self.reverting = self.init_sub_model("TokenMixerReverting")

        # S-P MoE 1: 在 mixed token 空间 (B, H, T*D/H) 上操作
        # num_tokens = H, inner_dim = T*D/H
        self.model_cfg[Const.SUB_MODELS]["SparsePerTokenMoE"][Const.HP] = {
            "num_routed_experts": num_routed_experts,
            "inner_dim": mixed_dim,
            "num_tokens": mixed_tokens,
            "k": k,
            "top_k": top_k,
            "bias": True,
            "dropout_p": dropout_p,
            "down_init_scale": down_init_scale,
        }
        self.sp_moe_1 = self.init_sub_model("SparsePerTokenMoE")

        # S-P MoE 2: 在 original token 空间 (B, T, D) 上操作
        # num_tokens = T, inner_dim = D
        self.model_cfg[Const.SUB_MODELS]["SparsePerTokenMoE"][Const.HP] = {
            "num_routed_experts": num_routed_experts,
            "inner_dim": inner_dim,
            "num_tokens": T,
            "k": k,
            "top_k": top_k,
            "bias": True,
            "dropout_p": dropout_p,
            "down_init_scale": down_init_scale,
        }
        self.sp_moe_2 = self.init_sub_model("SparsePerTokenMoE")

    def forward(self, x: torch.Tensor, loss_old: torch.Tensor, sparsity_old: torch.Tensor):
        # ---- 第一路: Mixing path ----
        h = self.norm1(x)
        h = self.mixing(h)
        h, (loss1, sp1) = self.sp_moe_1(h)
        h = self.reverting(h)
        x = x + h  # residual

        # ---- 第二路: Channel path ----
        y = self.norm2(x)
        y, (loss2, sp2) = self.sp_moe_2(y)
        x = x + y  # residual

        return x, (loss_old + loss1 + loss2, sparsity_old + sp1 + sp2)


@ModelRegistry.register(req_subs={"TokenMixerLargeInput", "TokenMixerLargeBlock"})
class TokenMixerLarge(BaseModel):
    """
    TokenMixer-Large 模型 (arxiv 2602.06563)

    由 Tokenization → N 层 TokenMixerLargeBlock → 输出投影 构成。

    支持 Interval Residual: 每 interval_residual_every 个 Block
    额外添加一个跳跃连接，缓解深层模型的梯度消失问题。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        model_conf = common_hp["model_conf"]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)

        x_input_dim = model_cfg[Const.HP].get("x_input_dim", 1024)
        T = model_cfg[Const.HP].get("T")
        use_global_token = model_cfg[Const.HP].get("use_global_token", False)

        num_routed_experts = model_cfg[Const.HP].get("num_routed_experts", 4)
        top_k = model_cfg[Const.HP].get("top_k", 2)
        k = model_cfg[Const.HP].get("k", 4)
        t_multiplier = model_cfg[Const.HP].get("t_multiplier", 1)
        n_layers = model_cfg[Const.HP].get("n_layers", 2)
        dropout_p = model_cfg[Const.HP].get("dropout", 0.05)
        down_init_scale = model_cfg[Const.HP].get("down_init_scale", 0.01)
        interval_residual_every = model_cfg[Const.HP].get("interval_residual_every", 0)

        num_local_tokens = T - 1 if use_global_token else T
        token_dim = x_input_dim // num_local_tokens
        usable_dim = token_dim * num_local_tokens

        if usable_dim < x_input_dim:
            print(
                f"[TokenMixerLarge] Truncate input dim from {x_input_dim} "
                f"to {usable_dim} (T={T}, local_tokens={num_local_tokens})"
            )

        inner_dim = int(num_local_tokens * t_multiplier)
        # H 默认等于 T，此时 S-P MoE 1/2 的参数量相同
        H = model_cfg[Const.HP].get("H", T)
        assert inner_dim % H == 0, \
            f"inner_dim ({inner_dim}) must be divisible by H ({H})"

        self.interval_residual_every = interval_residual_every

        # 3.1 Tokenization
        self.model_cfg[Const.SUB_MODELS]["TokenMixerLargeInput"][Const.HP] = {
            "T": T,
            "token_dim": token_dim,
            "inner_dim": inner_dim,
            "use_global_token": use_global_token,
            "usable_dim": usable_dim,
        }
        self.input_model = self.init_sub_model("TokenMixerLargeInput")

        # 3.4 Blocks
        self.model_cfg[Const.SUB_MODELS]["TokenMixerLargeBlock"][Const.HP] = {
            "inner_dim": inner_dim,
            "T": T,
            "H": H,
            "num_routed_experts": num_routed_experts,
            "top_k": top_k,
            "k": k,
            "dropout_p": dropout_p,
            "down_init_scale": down_init_scale,
        }
        self.blocks = nn.ModuleList([
            self.init_sub_model("TokenMixerLargeBlock")
            for _ in range(n_layers)
        ])

        # Output
        self.output_proj = nn.Linear(inner_dim, self._embedding_dim, bias=False)
        self.output_norm = RMSNormNPU(self._embedding_dim, Const.EPS)

    def forward(self, x: torch.Tensor):
        B, N, _ = x.size()
        x_batch = x.view(B * N, -1)
        h = self.input_model(x_batch)  # (B*N, T, inner_dim)

        l1_loss = 0.0
        sparsity = 0.0
        interval_checkpoint = h  # interval residual anchor

        for i, block in enumerate(self.blocks):
            h, (l1_loss, sparsity) = block(h, l1_loss, sparsity)

            # Interval Residual: 每 N 层额外加一个跳跃连接
            if self.interval_residual_every > 0 and (i + 1) % self.interval_residual_every == 0:
                h = h + interval_checkpoint
                interval_checkpoint = h

        # Pool over tokens
        h_out = h.mean(dim=1)  # (B*N, inner_dim)

        # 每个 block 包含 2 个 S-P MoE，归一化 loss 和 sparsity
        num_moe = 2 * len(self.blocks)
        if num_moe > 0:
            l1_loss = l1_loss / num_moe
            sparsity = sparsity / num_moe
        l1_loss = l1_loss.mean()

        output = self.output_proj(h_out)
        y = output.view(B, N, -1)
        y = self.output_norm(y)
        return y, l1_loss, sparsity
