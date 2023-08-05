import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Conv2d, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint,
                         load_state_dict)
from torch.nn.modules.utils import _pair as to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw, pvt_convert

from deepshift.modules import LinearShift, Conv2dShift

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

act_integer_bits=16 
act_fraction_bits=16
weight_bits=5


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        linear=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.act_layer = act_layer
        self.drop = drop
        self.linear = linear

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)
        self.B = 0
        self.N = 0 
        self.C = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        self.shape = []
        self.H = H
        self.W = W
        self.shape.append(x.shape)
        x = self.fc1(x)
        # print(torch.isnan(x).any())
        if self.linear:
            x = self.relu(x)
        self.shape.append(x.shape)
        x = self.dwconv(x, H, W)
        # print(torch.isnan(x).any())
        x = self.act(x)
        # print('ok')
        x = self.drop(x)
        self.shape.append(x.shape)
        x = self.fc2(x)
        # print(torch.isnan(x).any())
        x = self.drop(x)
            
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        linear=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # print(torch.isnan(x).any())
        # print(torch.isnan(self.qkv.weight).any())
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = torch.matmul(q , k.transpose(-2, -1)) * self.scale
        # FIXME: visual_distr
        # normalize attn
        # attn = attn / attn.norm(dim=-1, keepdim=True)
        # print(torch.isnan(q).any())
        # print(torch.isnan(k).any())
        # print(torch.isnan(attn).any())
        # visual_distr(attn)
        attn = attn.softmax(dim=-1)
        # print(torch.isnan(attn).any())
        # if torch.isnan(attn).any():
        #     exit()
        attn = self.attn_drop(attn)

        x = torch.matmul(attn , v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(torch.isnan(x).any())
        return x
    
class SRAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        linear=False,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio

        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = (
                    self.kv(x_)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
            else:
                kv = (
                    self.kv(x)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = torch.matmul(q , k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn , v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



# ===========================================================================
# ======================  shiftadd attention  ===============================
# ===========================================================================

# shift kernel


def Shift_Linear(in_features, out_features, convert_weights=True, freeze_sign=False, use_kernel=False, use_cuda=True, rounding='deterministic', weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits, SP2=False, bias=True):
    
    return LinearShift(in_features, out_features, bias, freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits, SP2=SP2)


def Shift_Conv(in_channels, out_channels, kernel_size, stride, padding, bias, groups, 
               convert_weights=True, freeze_sign=False, use_kernel=False, use_cuda=True, rounding='deterministic', weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits, SP2=False, padding_mode='zeros', dilation=1):
    
    return Conv2dShift(in_channels, out_channels, kernel_size, stride,
                                                    padding, dilation, groups=groups,
                                                    bias=bias, padding_mode=padding_mode,
                                                    freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, 
                                                    weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits,
                                                    SP2=SP2)

# Moe, Mixture of experts

class BaseGate(nn.Module):
    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None


class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        # self.w_gate = nn.Parameter(torch.zeros(d_model, num_expert), requires_grad=True)

    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        # print(torch.isnan(inp).any())
        gate = self.gate(inp)
        # print(torch.isnan(gate).any())
        # print("")
        # gate = F.softmax(gate, dim=-1)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val[:, : self.top_k]
        gate_top_k_idx = gate_top_k_idx[:, : self.top_k]
        top_k_logits = gate_top_k_val[:, : self.top_k]
        top_k_indices = gate_top_k_idx[:, : self.top_k]
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        zeros = torch.zeros_like(gate, requires_grad=True).type(torch.float32)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        return gates



class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # print(gates.shape)
        # print(torch.isnan(gates).any())
        # print(self._part_sizes)
        # print("")
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
        # index = torch.arange(gates.shape[0],dtype=torch.long,device=gates.device)
        # sorted_experts, index_sorted_experts = gates[:,1].type(torch.long).sort()

        # expert_index = sorted_experts.unsqueeze(1)
        # self._batch_index = index[index_sorted_experts]

        # sorted_experts = torch.stack((index, sorted_experts), dim=1)
        # index_sorted_experts = torch.stack((index, index_sorted_experts), dim=1)
        # # calculate num samples that each expert gets
        # self._part_sizes = (self._gates > 0).sum(0).tolist()
        # # expand gates to match with self._batch_index
        # gates_exp = self._gates[self._batch_index]
        # self._nonzero_gates = torch.gather(gates_exp, 1, expert_index)


    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].flatten(1)
        # print(inp_exp.shape[0])
        # print(self._part_sizes)
        # print("")
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class PyTorchMoE_FC(nn.Module):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        d_model=None,
        d_hidden=None,
        bias=True,
        num_expert=2,
        world_size=1,
        top_k=1,
        gate = NaiveGate, 
        **kwargs
    ):
        super(PyTorchMoE_FC, self).__init__()
        self.d_hidden = d_hidden
        self.num_expert = num_expert
        self.d_model = d_model

        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_hidden, bias=bias),
            Shift_Linear(d_model, d_hidden, bias=bias)
        ])
        self.gate = gate(d_model, num_expert, world_size, top_k)
        # self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        B, N, C = inp.shape
        inp = inp.reshape(-1, self.d_model)
        gates = self.gate(inp)
        dispatcher = SparseDispatcher(self.num_expert, gates)
        expert_inputs = dispatcher.dispatch(inp)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_expert)]
        y = dispatcher.combine(expert_outputs)
        return y.reshape(B, N, self.d_hidden)


# Linear Attention

class LinAngularAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.sparse_reg = sparse_reg

        # self.q_quant = FastQuantQK(
        #     head_dim, None, generalized_attention=False,
        #     kernel_fn=nn.ReLU(), no_projection=False
        # )
        # self.k_quant = FastQuantQK(
        #     head_dim, None, generalized_attention=False,
        #     kernel_fn=nn.ReLU(), no_projection=False
        # )

        self.qkv = PyTorchMoE_FC(dim, dim * 3, bias=qkv_bias)
        # self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = PyTorchMoE_FC(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        # B, N, C = x.shape
        # q = (
        #     self.q(x)
        #     .reshape(B, N, self.num_heads, C // self.num_heads)
        #     .permute(0, 2, 1, 3)
        # )
        # kv = (
        #         self.kv(x)
        #         .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        #         .permute(2, 0, 3, 1, 4)
        #     )
        # k, v = kv[0], kv[1]

        if self.sparse_reg:
            attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        # Quant
        scale1 = q.abs().mean()
        scale2 = k.abs().mean()
        # scale1 = 0.1
        # scale2 = 0.1

        binary_q_no_grad = torch.gt(q, 0).type(torch.float32) * scale1
        cliped_q = torch.clamp(q, 0, 1.0)
        q = binary_q_no_grad.detach() - cliped_q.detach() + cliped_q

        binary_k_no_grad = torch.gt(k, 0).type(torch.float32) * scale2
        cliped_k = torch.clamp(k, 0, 1.0)
        k = binary_k_no_grad.detach() - cliped_k.detach() + cliped_k
        # Quant end

        dconv_v = self.dconv(v)

        attn = torch.matmul(k.transpose(-2, -1), v) * scale2 if scale2 != None else torch.matmul(k.transpose(-2, -1), v)

        if self.sparse_reg:
            x = (
                torch.matmul(sparse, v)
                + 0.5 * v
                + 1.0 / math.pi * torch.matmul(q, attn)
            )
        else:
            attn = torch.matmul(q, attn) * scale1 if scale1 != None else torch.matmul(q, attn)
            x = 0.5 * v + 1.0 / math.pi * attn
        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp_FMoE(nn.Module):
    def __init__(
        self,
        d_model,
        d_hidden=None,
        out_features=None,
        activation=nn.GELU,
        drop=0.0,
        linear=False,
        world_size=1,
        top_k=1,
        gate=NaiveGate
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.out_features = out_features
        self.activation = activation
        self.drop = drop
        self.linear = linear

        out_features = out_features or d_model
        d_hidden = d_hidden or d_model

        self.fc1 = PyTorchMoE_FC(d_model, d_hidden, gate=gate)
        self.dwconv = DWConv(d_hidden)
        self.act = activation()
        self.fc2 = PyTorchMoE_FC(d_hidden, out_features, gate=gate)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)
        self.B = 0
        self.N = 0 
        self.C = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        self.shape = []
        self.H = H
        self.W = W
        self.shape.append(x.shape)
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        self.shape.append(x.shape)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        # print('ok')
        x = self.drop(x)
        self.shape.append(x.shape)
        x = self.fc2(x)
        x = self.drop(x)
            
        return x



# ===========================================================================
# ======================  shiftadd attention end ============================
# ===========================================================================
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        linear=False,
        last_stage=False,
        moe=False,
        world_size=0,
        flag = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if last_stage:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                linear=linear,
            )
        else:
            self.attn = LinAngularAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )


        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # if not moe or flag:

        num_expert = 2
        if num_expert % world_size != 0:
            print("experts number of {} is not divisible by world size of {}".format(num_expert, world_size))
        num_expert = num_expert // world_size
        self.mlp = Mlp_FMoE(
            d_model=dim,
            d_hidden=mlp_hidden_dim,
            activation=act_layer,
            drop=drop,
            linear=linear,
            gate=NaiveGate
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # print(torch.isnan(x).any())
        # print(torch.isnan(x.var()).any())
        # print(torch.isnan(self.norm1(x)).any())
        # print('ok')
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # print('ok')
        # print(torch.isnan(x).any())

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # print(torch.isnan(x).any())
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


@BACKBONES.register_module()
class PyramidVisionTransformerV2_ShiftAdd(BaseModule):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        num_stages=4,
        linear=False,
        moe=False,
        world_size=1,
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.depths = depths
        self.num_stages = num_stages
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        
        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages
        # self.use_performer = args.use_performer
        # if 1 in self.use_performer:
        #     self.feature_redraw_interval = 1
        #     self.register_buffer('calls_since_last_redraw', torch.tensor(0))

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        # if "rfa" in args.attn_type:
        #     self.auto_retrain = True
        #     self.feature_retrain_interval = args.k_iteration
        #     self.register_buffer("calls_since_last_retrain", torch.tensor(0))
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )
            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[i],
                        linear=linear,
                        last_stage=(i == num_stages - 1),
                        moe=moe,
                        world_size=world_size,
                        flag=((i==num_stages-1) or (i==num_stages-2 and j==depths[i]-1))
                    )
                    for j in range(depths[i])
                ]
            )
            # block = nn.ModuleList([Block(
            #     dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
            #     qk_scale=qk_scale,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
            #     sr_ratio=sr_ratios[i], linear=linear, use_performer=self.use_performer[i]==1)
            #     for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)




    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m.weight, std=.02)
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m.weight, 0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        constant_init(m.bias, 0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            checkpoint = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            logger.warn(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        outs = []

        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            
            for indx, blk in enumerate(block):
                x = blk(x, H, W)
            # print(i)
            x = norm(x)
            x = nlc_to_nchw(x, [H,W])
            if i in self.out_indices:
                outs.append(x)

        return outs
