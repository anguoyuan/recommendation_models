import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import OrderedDict
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.transformers import RMSNorm_npu as RMSNormNPU
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
import logging


@ModelRegistry.register()
class RankMixingInput(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        dim_per_token = model_cfg[Const.HP].get("dim_per_token")
        x_dim = model_cfg[Const.HP].get("x_dim")
        t_multiplier = model_cfg[Const.HP].get("t_multiplier", 1)
        assert (x_dim % dim_per_token) == 0
        T = x_dim // dim_per_token
        num_heads = T
        inner_dim = int(num_heads * t_multiplier)
        assert (inner_dim % num_heads) == 0
        self.x_dim = x_dim
        print('self.x_dim', self.x_dim) # 12288
        self.num_heads = num_heads # 16
        self.dim_per_token = dim_per_token
        print('self.dim_per_token', self.dim_per_token) # 768
        self.inner_dim = inner_dim
        print('self.inner_dim', self.inner_dim) # 32
        self.proj = torch.nn.Linear(dim_per_token, inner_dim, bias=False)

    def forward(self, x):
        # 输入x是所有用户、商品、序列特征拼接而成的特征向量，形状为(B, \sum_{D_e}), D_e是不同特征的长度
        B, D = x.size()
        assert self.x_dim <= D
        x = x[:, :self.num_heads * self.dim_per_token]
        # 形状（B, T, inner_dim）
        proj_x = self.proj(x.view(B, self.num_heads, self.dim_per_token)) # 计算量 (256*50) * 16 * 768 * 32
        return proj_x # 256 *50, 16, 32


@ModelRegistry.register()
class TokenMxing(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        RankMixer中的TokenMixing模块,
        对应原论文公式2,3,4,5
        dim_per_token：公式2里的d，即重新划分的token的维度
        x_dim: 所有特征拼接起来的维度总和
        为避免信息损失，要求x_dim可以被dim_per_token整除。
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        dim_per_token = model_cfg[Const.HP].get("dim_per_token")
        x_dim = model_cfg[Const.HP].get("x_dim")
        t_multiplier = model_cfg[Const.HP].get("t_multiplier")
        assert (x_dim % dim_per_token) == 0
        T = x_dim // dim_per_token
        num_heads = T
        inner_dim = int(num_heads * t_multiplier)
        assert (inner_dim % num_heads) == 0
        self.num_heads = num_heads
        self.ln = torch.nn.LayerNorm((inner_dim,), eps=1e-7)

    def forward(self, x):
        B, D = x.size(0), x.size(2)
        tm_x = x.transpose(1, 2).contiguous().view(B, self.num_heads, D)
        # 形状（B, num_heads, inner_dim）
        return self.ln(x + tm_x)


@ModelRegistry.register()
class PerTokenFFN(BaseModel):
    """
    Per-token position-wise MLP with configurable depth L.
    Each token position t has its own stack of Linear layers.

    For L layers:
      layer 1: D -> kD
      layer 2..L-1: kD -> kD
      layer L: kD -> D
    Activations: GELU after layers 1..L-1 (no activation on last).
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        D = model_cfg[Const.HP].get("D")
        T = model_cfg[Const.HP].get("T")
        k = model_cfg[Const.HP].get("k")
        num_layers = model_cfg[Const.HP].get("num_layers")
        bias = model_cfg[Const.HP].get("bias")
        dropout_p = model_cfg[Const.HP].get("dropout_p")
        assert num_layers >= 2, "num_layers must be >= 2"

        self.D = D
        self.T = T
        self.kD = int(round(k * D)) # 4*32
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # Build per-position weights for each layer
        in_dims = [D] + [self.kD] * (num_layers - 1)
        out_dims = [self.kD] * (num_layers - 1) + [D]
        # out_dims length is num_layers; first num_layers-1 are kD, last is D

        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(T, din, dout))
            for din, dout in zip(in_dims, out_dims)
        ])
        if bias:
            self.b = nn.ParameterList([
                nn.Parameter(torch.empty(T, dout))
                for dout in out_dims
            ])
        else:
            self.b = None

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_layers):
            nn.init.xavier_normal_(self.W[i])
            if self.b is not None:
                nn.init.zeros_(self.b[i])

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (bs, T, D)  ->  v: (bs, T, D)
        """
        bs, T, D = s.shape
        assert T == self.T and D == self.D, f"expected (bs,{self.T},{self.D}), got {tuple(s.shape)}"

        x = s
        # Apply layers 0..L-2 with GELU, final layer without activation
        for i in range(self.num_layers):
            x = torch.einsum("bti,tid->btd", x, self.W[i])  # 一层计算量 256*50 * 16* 32 * (32*4)
            if self.b is not None:
                x = x + self.b[i]
            if i < self.num_layers - 1:
                x = F.gelu(x)
                x = self.dropout(x)
        # optional dropout on output (comment out if you want *exact* equations)
        x = self.dropout(x)
        return x


@ModelRegistry.register(req_subs={"PerTokenFFN"})
class SparseMoE(BaseModel):
    """
    ReLU-Routed Sparse MoE with per-token experts.

    Given router h(·) and experts e_j(·):
        G_{i,j} = ReLU(h(s_i))
        v_i     = sum_{j=1..N_e} G_{i,j} * e_j(s_i)

    Args:
        num_experts_per_token (int): N_e, number of experts per token.
        inner_dim (int): D, hidden size per token.
        num_tokens (int): T, number of token positions.
        k, bias, dropout_p: forwarded to PerTokenFFN.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        num_experts_per_token = model_cfg[Const.HP].get("num_experts_per_token")
        inner_dim = model_cfg[Const.HP].get("inner_dim")
        num_tokens = model_cfg[Const.HP].get("num_tokens")
        k = model_cfg[Const.HP].get("k")
        num_layers_per_expert = model_cfg[Const.HP].get("num_layers_per_expert")
        bias = model_cfg[Const.HP].get("bias")
        dropout_p = model_cfg[Const.HP].get("dropout_p")

        self.D = inner_dim #32
        self.T = num_tokens
        self.Ne = num_experts_per_token
        self.relu_threshold = 1e-4

        # Router h(·): shared across positions, maps R^D -> R^{N_e}
        self.router = nn.Linear(self.D, self.Ne, bias=False)

        # Experts: N_e copies of PerTokenFFN (each is position-specific over T)
        self.model_cfg[Const.SUB_MODELS]["PerTokenFFN"][Const.HP] = {"D": inner_dim,  #32
                                                                     "T": num_tokens, # 16
                                                                     "k": k,
                                                                     "num_layers": num_layers_per_expert,
                                                                     "bias": bias,
                                                                     "dropout_p": dropout_p}
        self.token_experts = nn.ModuleList(
            [self.init_sub_model("PerTokenFFN") for _ in range(self.Ne)]
        )
        self.ln = torch.nn.LayerNorm((inner_dim,), eps=1e-7)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.router.weight)

    def forward(self, s: torch.Tensor):
        """
        Args:
            s: (bs, T, D)

        Returns:
            v: (bs, T, D)  -- routed mixture of experts output
        """
        assert s.dim() == 3 and s.size(1) == self.T and s.size(2) == self.D, \
            f"expected (bs,{self.T},{self.D}), got {tuple(s.shape)}"

        T = s.size(1)

        # Router logits -> ReLU gates (no softmax, no top-k)
        # shape: (bs, T, N_e)
        print('sssss', s.shape)
        gates = F.relu(self.router(s))
        float_mask = (gates > 0).float().detach()
        reg_loss = gates.sum(-1).sum(-1)
        sparsity = float_mask.sum() / (T * self.Ne)

        expert_outputs = torch.stack([exp(s) for exp in self.token_experts], dim=2) # s = 256 *50, 16, 32

        # Weighted sum over experts: v[b,t,d] = sum_j gates[b,t,j] * out[b,t,j,d]
        v = torch.einsum("btjd,btj->btd", expert_outputs, gates)  # 计算量 2*（50*256)×16×1×32

        return self.ln(s + v), (reg_loss, sparsity)


@ModelRegistry.register(req_subs={"TokenMxing", "SparseMoE"})
class RankMixerBlock(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        dim_per_token = model_cfg[Const.HP].get("dim_per_token")
        all_dim = model_cfg[Const.HP].get("all_dim")
        num_experts = model_cfg[Const.HP].get("num_experts")
        inner_dim = model_cfg[Const.HP].get("inner_dim")
        ffn_layers = model_cfg[Const.HP].get("ffn_layers")
        T = model_cfg[Const.HP].get("T")
        k = model_cfg[Const.HP].get("k")
        t_multiplier = model_cfg[Const.HP].get("t_multiplier")
        dropout_p = model_cfg[Const.HP].get("dropout_p")

        self.model_cfg[Const.SUB_MODELS]["TokenMxing"][Const.HP] = {"dim_per_token": dim_per_token,
                                                                    "x_dim": all_dim,
                                                                    "k": k,
                                                                    "t_multiplier": t_multiplier}
        self.tokenmixing = self.init_sub_model("TokenMxing")

        self.model_cfg[Const.SUB_MODELS]["SparseMoE"][Const.HP] = {"num_experts_per_token": num_experts,
                                                                   "inner_dim": inner_dim,
                                                                   "num_tokens": T,
                                                                   "k": k,
                                                                   "num_layers_per_expert": ffn_layers,
                                                                   "bias": True,
                                                                   "dropout_p": dropout_p
                                                                   }

        self.moe = self.init_sub_model("SparseMoE")

    def forward(self, x: torch.Tensor, loss_old: torch.Tensor, sparsity_old: torch.Tensor):
        x = self.tokenmixing(x)
        y, (loss, sparsity) = self.moe(x)
        return y, (loss + loss_old, sparsity + sparsity_old)


@ModelRegistry.register(req_subs={"RankMixingInput", "RankMixerBlock"})
class RankMixer(BaseModel):
    """
    某节跳动的RankMixer模型，根据论文复现。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        feat_conf = common_hp["feature_conf"]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)

        x_input_dim = model_cfg[Const.HP].get('x_input_dim', 1024)
        all_dim = model_cfg[Const.HP].get('all_dim', 1024)

        dim_per_token = model_cfg[Const.HP].get("dim_per_token", 128)

        num_experts = model_cfg[Const.HP].get("num_experts", 10)
        k = model_cfg[Const.HP].get("k", 4)
        print('k', k) # 4
        t_multiplier = model_cfg[Const.HP].get("t_multiplier", 1)
        ffn_layers = model_cfg[Const.HP].get("ffn_layers", 2)
        print('ffn_layers', ffn_layers) # 2
        n_layers = model_cfg[Const.HP].get("n_layers", 2)
        print('n_layers', n_layers)
        dropout_p = model_cfg[Const.HP].get("dropout", 0.05)
        T = all_dim // dim_per_token
        inner_dim = int(T * t_multiplier) # 32

        self.model_cfg[Const.SUB_MODELS]["RankMixingInput"][Const.HP] = {"dim_per_token": dim_per_token,
                                                                         "x_dim": all_dim,
                                                                         "t_multiplier": t_multiplier}
        self.input_model = self.init_sub_model("RankMixingInput")

        self.model_cfg[Const.SUB_MODELS]["RankMixerBlock"][Const.HP] = {"dim_per_token": dim_per_token,
                                                                        "all_dim": all_dim,
                                                                        "num_experts": num_experts,
                                                                        "inner_dim": inner_dim,
                                                                        "ffn_layers": ffn_layers,
                                                                        "T": T,
                                                                        "k": k,
                                                                        "t_multiplier": t_multiplier,
                                                                        "dropout_p": dropout_p
                                                                        }
        self.rm_blocks = nn.Sequential(*[
            self.init_sub_model("RankMixerBlock")
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(inner_dim, self._embedding_dim, bias=False)
        self.output_norm = RMSNormNPU(self._embedding_dim, Const.EPS)

    def adjust_dim_fast(self, all_dim, dim_per_token):
        divisors = [d for d in range(1, all_dim + 1) if all_dim % d == 0]
        return min(divisors, key=lambda x: abs(x - dim_per_token))

    def forward(self, x):
        results = []
        B, N, _ = x.size()
        x_batch = x.view(B * N, -1) # (256*50,12288)
        rm_in = self.input_model(x_batch) # 256*50，16，32
        l1_loss = 0.0
        sparsity = 0.0
        for i in range(len(self.rm_blocks)):
            block_i = self.rm_blocks[i]
            rm_in, (l1_loss, sparsity) = block_i(rm_in, l1_loss, sparsity)
        rm_out = rm_in.mean(dim=1)
        l1_loss = l1_loss / len(self.rm_blocks)
        l1_loss = l1_loss.mean()
        sparsity = sparsity / len(self.rm_blocks)
        output = self.output_proj(rm_out) # 256*50,256  计算量 256*50*32*256
        y = output.view(B, N, -1)
        y = self.output_norm(y)
        return y, l1_loss, sparsity
