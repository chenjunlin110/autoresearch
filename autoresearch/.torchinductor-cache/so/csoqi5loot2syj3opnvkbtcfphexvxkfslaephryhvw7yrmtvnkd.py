# AOT ID: ['0_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/4f/c4f3fjlhnqolnrtqqyzh5ywrzjwiucztsjp4cc352tvwak2xoxbr.py
# Topologically Sorted Source Nodes: [view_37, loss, logits, logits_1, truediv, tanh, logits_2, view_36], Original ATen: [aten.nll_loss_backward, aten.view, aten.nll_loss_forward, aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten._log_softmax, aten._log_softmax_backward_data, aten.tanh_backward]
# Source node to ATen node mapping:
#   logits => view_141
#   logits_1 => convert_element_type_243
#   logits_2 => mul_122
#   loss => full_default, full_default_1, sub, sub_1
#   tanh => tanh
#   truediv => div
#   view_36 => view_142
#   view_37 => view_143
# Graph fragment:
#   %primals_64 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_64]
#   %tangents_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_1]
#   %convert_element_type_244 : Tensor "f32[][]cuda:0" = PlaceHolder[target=convert_element_type_244]
#   %mm_52 : Tensor "bf16[262144, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_52]
#   %amax : Tensor "f32[262144, 1][1, 1]cuda:0" = PlaceHolder[target=amax]
#   %log : Tensor "f32[262144, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %sum_4 : Tensor "f32[262144, 1][1, 262144]cuda:0" = PlaceHolder[target=sum_4]
#   %sub_2 : Tensor "f32[262144, 8192][8192, 1]cuda:0" = PlaceHolder[target=sub_2]
#   %div_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_244), kwargs = {})
#   %view_143 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_64, [-1]), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[262144, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_143, 1), kwargs = {})
#   %ne_3 : Tensor "b8[262144, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_5, -1), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "i64[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze_5, %full_default), kwargs = {})
#   %scatter_upon_const_tensor : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [262144, 8192], background_val: 0, dtype: torch.float32, dim: 1, selector: %where_2, val: -1.0})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : Tensor "f32[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %div_2, %full_default_1), kwargs = {})
#   %mul_123 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_upon_const_tensor, %where_3), kwargs = {})
#   %view_141 : Tensor "bf16[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_52, [128, 2048, 8192]), kwargs = {})
#   %convert_element_type_243 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_141, torch.float32), kwargs = {})
#   %div : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_243, 15), kwargs = {})
#   %tanh : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_122 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 15), kwargs = {})
#   %view_142 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_122, [-1, 8192]), kwargs = {})
#   %sub : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_142, %amax), kwargs = {})
#   %sub_1 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %log), kwargs = {})
#   %exp_1 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_4 : Tensor "f32[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_123, [1], True), kwargs = {})
#   %mul_124 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %sum_4), kwargs = {})
#   %sub_2 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_123, %mul_124), kwargs = {})
#   %view_144 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_2, [128, 2048, 8192]), kwargs = {})
#   %mul_125 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_144, 15), kwargs = {})
#   %mul_126 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %tanh), kwargs = {})
#   %sub_3 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_126), kwargs = {})
#   %mul_127 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_125, %sub_3), kwargs = {})
#   %div_3 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_127, 15), kwargs = {})
#   %convert_element_type_245 : Tensor "bf16[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_3, torch.bfloat16), kwargs = {})
#   return %sum_4,%sub_2,%convert_element_type_245
triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_view_0 = async_compile.triton('triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i64', 'r0_numel': 'i64', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_view_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4194304, 'r0_': 12884901888}}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_view_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tl.full([1, 1], -1, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = r0_1
        tmp6 = tmp4 == tmp5
        tmp7 = -1.0
        tmp8 = 0.0
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp14 = (tmp11 / tmp13)
        tmp15 = tl.where(tmp2, tmp14, tmp8)
        tmp16 = tmp9 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp29 = tl.load(in_ptr1 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp31 = tl.load(in_ptr2 + (0))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp43 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp36 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.full([1, 1], -1, tl.int64)
        tmp21 = tmp0 != tmp20
        tmp22 = tl.full([1, 1], 0, tl.int64)
        tmp23 = tl.where(tmp21, tmp0, tmp22)
        tmp24 = r0_1
        tmp25 = tmp23 == tmp24
        tmp26 = -1.0
        tmp27 = 0.0
        tmp28 = tl.where(tmp25, tmp26, tmp27)
        tmp33 = (tmp30 / tmp32)
        tmp34 = tl.where(tmp21, tmp33, tmp27)
        tmp35 = tmp28 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = 0.06666666666666667
        tmp39 = tmp37 * tmp38
        tmp40 = libdevice.tanh(tmp39)
        tmp41 = 15.0
        tmp42 = tmp40 * tmp41
        tmp44 = tmp42 - tmp43
        tmp46 = tmp44 - tmp45
        tmp47 = libdevice.exp(tmp46)
        tmp48 = tmp47 * tmp18
        tmp49 = tmp35 - tmp48
        tmp50 = tmp49 * tmp41
        tmp51 = tmp40 * tmp40
        tmp52 = 1.0
        tmp53 = tmp52 - tmp51
        tmp54 = tmp50 * tmp53
        tmp55 = tmp54 * tmp38
        tmp56 = tmp55.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp56, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/tv/ctvzpsrhz4fnbcpdycze2pupvivo2vz2r3s67tlltxmj4b7wlyek.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_53 : Tensor "bf16[8192, 512][512, 1]cuda:0" = PlaceHolder[target=mm_53]
#   %convert_element_type_250 : Tensor "f32[8192, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_53, torch.float32), kwargs = {})
#   return %convert_element_type_250
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41943040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/jh/cjhrelllpbynmmpooor7zsp7ryv6fintnhzryjiplm4khvxhgzvb.py
# Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   x_50 => convert_element_type_238, mul_121
# Graph fragment:
#   %add_92 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_92]
#   %rsqrt_33 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_33]
#   %mm_54 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_54]
#   %sum_5 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_5]
#   %view_146 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_54, [128, 2048, 512]), kwargs = {})
#   %convert_element_type_251 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_146, torch.float32), kwargs = {})
#   %convert_element_type_238 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_92, torch.float32), kwargs = {})
#   %mul_121 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_238, %rsqrt_33), kwargs = {})
#   %mul_129 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, %convert_element_type_251), kwargs = {})
#   %sum_5 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_129, [2], True), kwargs = {})
#   %div_4 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_121, 512), kwargs = {})
#   %mul_130 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sum_5), kwargs = {})
#   %sub_4 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_251, %mul_130), kwargs = {})
#   %mul_131 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_33), kwargs = {})
#   %convert_element_type_253 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_131, torch.bfloat16), kwargs = {})
#   return %sum_5,%convert_element_type_253
triton_per_fused__fused_rms_norm_backward__to_copy_mul_view_2 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_mul_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_mul_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1073741824}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_mul_view_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp10 = 0.001953125
    tmp11 = tmp3 * tmp10
    tmp12 = tmp11 * tmp9
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp2
    tmp15 = tmp14.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp15, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/xd/cxdsm6ep5tgo3pmuqypdx4xqipu2hfxm4wxv7oa2i2xybtcq73dx.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_55 : Tensor "bf16[512, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_55]
#   %convert_element_type_259 : Tensor "f32[512, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_55, torch.float32), kwargs = {})
#   return %convert_element_type_259
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/v5/cv5p5dmixy432xmepuye7xzrkmzx6yq5rsdiwymhgkf6xkwgegah.py
# Topologically Sorted Source Nodes: [x_46, relu_7, x_47], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.threshold_backward]
# Source node to ATen node mapping:
#   relu_7 => relu_7
#   x_46 => view_137
#   x_47 => convert_element_type_233
# Graph fragment:
#   %mm_50 : Tensor "bf16[262144, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_50]
#   %mm_56 : Tensor "bf16[262144, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_56]
#   %view_148 : Tensor "bf16[128, 2048, 2048][4194304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_56, [128, 2048, 2048]), kwargs = {})
#   %convert_element_type_258 : Tensor "f32[128, 2048, 2048][4194304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_148, torch.float32), kwargs = {})
#   %view_137 : Tensor "bf16[128, 2048, 2048][4194304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_50, [128, 2048, 2048]), kwargs = {})
#   %relu_7 : Tensor "bf16[128, 2048, 2048][4194304, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_137,), kwargs = {})
#   %convert_element_type_233 : Tensor "f32[128, 2048, 2048][4194304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%relu_7, torch.float32), kwargs = {})
#   %pow_43 : Tensor "f32[128, 2048, 2048][4194304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_233, 1.0), kwargs = {})
#   %mul_132 : Tensor "f32[128, 2048, 2048][4194304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_43, 2.0), kwargs = {})
#   %mul_133 : Tensor "f32[128, 2048, 2048][4194304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_258, %mul_132), kwargs = {})
#   %convert_element_type_260 : Tensor "bf16[128, 2048, 2048][4194304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_133, torch.bfloat16), kwargs = {})
#   %le : Tensor "b8[128, 2048, 2048][4194304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_7, 0), kwargs = {})
#   %full_default_5 : Tensor "bf16[][]cuda:0"[num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "bf16[128, 2048, 2048][4194304, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default_5, %convert_element_type_260), kwargs = {})
#   return %where_4
triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4294967296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp2.to(tl.float32)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.where(tmp4, tmp3, tmp11)
    tl.store(in_out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/fr/cfr2astzi7qdlqsd2c5ag7nqtub4u6c47pxpqprkxkyzxz5kgoql.py
# Topologically Sorted Source Nodes: [rms_norm_32], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
# Source node to ATen node mapping:
#   rms_norm_32 => convert_element_type_228, mul_120
# Graph fragment:
#   %add_90 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_90]
#   %rsqrt_32 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_32]
#   %mm_58 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_58]
#   %convert_element_type_253 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=convert_element_type_253]
#   %sum_6 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_6]
#   %view_150 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_58, [128, 2048, 512]), kwargs = {})
#   %convert_element_type_266 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_150, torch.float32), kwargs = {})
#   %convert_element_type_228 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_90, torch.float32), kwargs = {})
#   %mul_120 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_228, %rsqrt_32), kwargs = {})
#   %mul_135 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_120, %convert_element_type_266), kwargs = {})
#   %sum_6 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_135, [2], True), kwargs = {})
#   %div_5 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_120, 512), kwargs = {})
#   %mul_136 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_5, %sum_6), kwargs = {})
#   %sub_5 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_266, %mul_136), kwargs = {})
#   %mul_137 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_32), kwargs = {})
#   %convert_element_type_268 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_137, torch.bfloat16), kwargs = {})
#   %add_94 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_253, %convert_element_type_268), kwargs = {})
#   return %sum_6,%add_94
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_5 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1342177280}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp10 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp11 = 0.001953125
    tmp12 = tmp3 * tmp11
    tmp13 = tmp12 * tmp9
    tmp14 = tmp5 - tmp13
    tmp15 = tmp14 * tmp2
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp10 + tmp16
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp17, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/5l/c5lh6xizaewiab53bcip4i7hsd4gngdv2ohjnwyln5mb365anvk7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_59 : Tensor "bf16[512, 512][512, 1]cuda:0" = PlaceHolder[target=mm_59]
#   %convert_element_type_273 : Tensor "f32[512, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_59, torch.float32), kwargs = {})
#   return %convert_element_type_273
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2621440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/wp/cwp5cltwapieafsnrdn733buzqnox75sztdvlrjzhptms6roxcq7.py
# Topologically Sorted Source Nodes: [k_23, q_23, cos, sin, neg], Original ATen: [aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.slice, aten.neg, aten.add, aten.slice_backward]
# Source node to ATen node mapping:
#   cos => slice_1
#   k_23 => convert_element_type_223, mul_119
#   neg => neg
#   q_23 => convert_element_type_221, mul_118
#   sin => slice_2
# Graph fragment:
#   %cat_14 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0" = PlaceHolder[target=cat_14]
#   %rsqrt_30 : Tensor "f32[128, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_30]
#   %getitem_32 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0" = PlaceHolder[target=getitem_32]
#   %cat_15 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0" = PlaceHolder[target=cat_15]
#   %rsqrt_31 : Tensor "f32[128, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_31]
#   %getitem_33 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0" = PlaceHolder[target=getitem_33]
#   %sum_7 : Tensor "f32[128, 2048, 4, 1][8192, 4, 1, 1048576]cuda:0" = PlaceHolder[target=sum_7]
#   %primals_2 : Tensor "bf16[1, 20480, 1, 64][1310720, 64, 64, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "bf16[1, 20480, 1, 64][1310720, 64, 64, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %slice_scatter_default : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0" = PlaceHolder[target=slice_scatter_default]
#   %slice_scatter_default_1 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0" = PlaceHolder[target=slice_scatter_default_1]
#   %sum_8 : Tensor "f32[128, 2048, 4, 1][8192, 4, 1, 1048576]cuda:0" = PlaceHolder[target=sum_8]
#   %slice_scatter_default_2 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0" = PlaceHolder[target=slice_scatter_default_2]
#   %slice_scatter_default_3 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0" = PlaceHolder[target=slice_scatter_default_3]
#   %convert_element_type_274 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_33, torch.float32), kwargs = {})
#   %convert_element_type_223 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_15, torch.float32), kwargs = {})
#   %mul_119 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_223, %rsqrt_31), kwargs = {})
#   %mul_139 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_119, %convert_element_type_274), kwargs = {})
#   %sum_7 : Tensor "f32[128, 2048, 4, 1][8192, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_139, [3], True), kwargs = {})
#   %div_6 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_119, 128), kwargs = {})
#   %mul_140 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, %sum_7), kwargs = {})
#   %sub_6 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_274, %mul_140), kwargs = {})
#   %mul_141 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_31), kwargs = {})
#   %convert_element_type_276 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_141, torch.bfloat16), kwargs = {})
#   %convert_element_type_277 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_32, torch.float32), kwargs = {})
#   %convert_element_type_221 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_14, torch.float32), kwargs = {})
#   %mul_118 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_221, %rsqrt_30), kwargs = {})
#   %mul_143 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %convert_element_type_277), kwargs = {})
#   %sum_8 : Tensor "f32[128, 2048, 4, 1][8192, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_143, [3], True), kwargs = {})
#   %div_7 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_118, 128), kwargs = {})
#   %mul_144 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_7, %sum_8), kwargs = {})
#   %sub_7 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_277, %mul_144), kwargs = {})
#   %mul_145 : Tensor "f32[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_30), kwargs = {})
#   %convert_element_type_279 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_145, torch.bfloat16), kwargs = {})
#   %slice_39 : Tensor "bf16[128, 2048, 4, 64][1048576, 512, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_276, 3, 0, 64), kwargs = {})
#   %slice_40 : Tensor "bf16[128, 2048, 4, 64][1048576, 512, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_276, 3, 64, 128), kwargs = {})
#   %slice_1 : Tensor "bf16[1, 2048, 1, 64][1310720, 64, 64, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_2, 1, 0, 2048), kwargs = {})
#   %mul_146 : Tensor "bf16[128, 2048, 4, 64][524288, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_40, %slice_1), kwargs = {})
#   %slice_2 : Tensor "bf16[1, 2048, 1, 64][1310720, 64, 64, 1]cuda:0"[num_users=17] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_3, 1, 0, 2048), kwargs = {})
#   %neg : Tensor "bf16[1, 2048, 1, 64][131072, 64, 64, 1]cuda:0"[num_users=16] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %mul_147 : Tensor "bf16[128, 2048, 4, 64][524288, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_40, %neg), kwargs = {})
#   %mul_148 : Tensor "bf16[128, 2048, 4, 64][524288, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_39, %slice_2), kwargs = {})
#   %add_95 : Tensor "bf16[128, 2048, 4, 64][524288, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_146, %mul_148), kwargs = {})
#   %mul_149 : Tensor "bf16[128, 2048, 4, 64][524288, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_39, %slice_1), kwargs = {})
#   %add_96 : Tensor "bf16[128, 2048, 4, 64][524288, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_147, %mul_149), kwargs = {})
#   %full_default_6 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.full.default](args = ([128, 2048, 4, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_6, %add_95, 3, 64, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_6, %add_96, 3, 0, 64), kwargs = {})
#   %add_97 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %slice_scatter_default_1), kwargs = {})
#   %slice_41 : Tensor "bf16[128, 2048, 4, 64][1048576, 512, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_279, 3, 0, 64), kwargs = {})
#   %slice_42 : Tensor "bf16[128, 2048, 4, 64][1048576, 512, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_279, 3, 64, 128), kwargs = {})
#   %mul_150 : Tensor "bf16[128, 2048, 4, 64][524288, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_42, %slice_1), kwargs = {})
#   %mul_151 : Tensor "bf16[128, 2048, 4, 64][524288, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_42, %neg), kwargs = {})
#   %mul_152 : Tensor "bf16[128, 2048, 4, 64][524288, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_41, %slice_2), kwargs = {})
#   %add_98 : Tensor "bf16[128, 2048, 4, 64][524288, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_150, %mul_152), kwargs = {})
#   %mul_153 : Tensor "bf16[128, 2048, 4, 64][524288, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_41, %slice_1), kwargs = {})
#   %add_99 : Tensor "bf16[128, 2048, 4, 64][524288, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_151, %mul_153), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_6, %add_98, 3, 64, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_3 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_6, %add_99, 3, 0, 64), kwargs = {})
#   %add_100 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_2, %slice_scatter_default_3), kwargs = {})
#   return %sum_8,%sum_7,%slice_scatter_default,%slice_scatter_default_1,%add_97,%slice_scatter_default_2,%slice_scatter_default_3,%add_100
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1048576, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 30, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8388608, 'r0_': 6442450944}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x3 = ((xindex // 4) % 2048)
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp20 = r0_1
    tmp21 = tl.full([1, 1], 64, tl.int64)
    tmp22 = tmp20 >= tmp21
    tmp23 = tl.load(in_ptr5 + (r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.load(in_ptr3 + (r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), tmp22, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = 0.0078125
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30 * tmp19
    tmp32 = tmp24 - tmp31
    tmp33 = tmp32 * tmp27
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.load(in_ptr6 + ((-64) + r0_1 + 64*x3), tmp22, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tl.load(in_ptr5 + ((-64) + r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.load(in_ptr3 + ((-64) + r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp40 * tmp27
    tmp42 = tmp41 * tmp29
    tmp43 = tmp42 * tmp19
    tmp44 = tmp38 - tmp43
    tmp45 = tmp44 * tmp27
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tl.load(in_ptr7 + ((-64) + r0_1 + 64*x3), tmp22, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp48 = tmp46 * tmp47
    tmp49 = tmp36 + tmp48
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp22, tmp49, tmp50)
    tmp52 = 0.0
    tmp53 = tl.where(tmp22, tmp51, tmp52)
    tmp54 = tmp20 < tmp21
    tmp55 = tl.load(in_ptr5 + (64 + r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tl.load(in_ptr3 + (64 + r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), tmp54, eviction_policy='evict_last', other=0.0)
    tmp60 = tmp58 * tmp59
    tmp61 = 0.0078125
    tmp62 = tmp60 * tmp61
    tmp63 = tmp62 * tmp19
    tmp64 = tmp56 - tmp63
    tmp65 = tmp64 * tmp59
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tl.load(in_ptr7 + (r0_1 + 64*x3), tmp54, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp68 = -tmp67
    tmp69 = tmp66 * tmp68
    tmp70 = tl.load(in_ptr5 + (r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tl.load(in_ptr3 + (r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp73 * tmp59
    tmp75 = tmp74 * tmp61
    tmp76 = tmp75 * tmp19
    tmp77 = tmp71 - tmp76
    tmp78 = tmp77 * tmp59
    tmp79 = tmp78.to(tl.float32)
    tmp80 = tl.load(in_ptr6 + (r0_1 + 64*x3), tmp54, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp81 = tmp79 * tmp80
    tmp82 = tmp69 + tmp81
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp54, tmp82, tmp83)
    tmp85 = tl.where(tmp54, tmp84, tmp52)
    tmp86 = tmp53 + tmp85
    tmp87 = tl.load(in_ptr2 + (r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp88 = tmp87.to(tl.float32)
    tmp89 = tl.load(in_ptr0 + (r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp90 = tmp89.to(tl.float32)
    tmp91 = tl.load(in_ptr1 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), tmp22, eviction_policy='evict_last', other=0.0)
    tmp92 = tmp90 * tmp91
    tmp93 = tmp92 * tmp29
    tmp94 = tmp93 * tmp9
    tmp95 = tmp88 - tmp94
    tmp96 = tmp95 * tmp91
    tmp97 = tmp96.to(tl.float32)
    tmp98 = tmp97 * tmp35
    tmp99 = tl.load(in_ptr2 + ((-64) + r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp100 = tmp99.to(tl.float32)
    tmp101 = tl.load(in_ptr0 + ((-64) + r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp102 = tmp101.to(tl.float32)
    tmp103 = tmp102 * tmp91
    tmp104 = tmp103 * tmp29
    tmp105 = tmp104 * tmp9
    tmp106 = tmp100 - tmp105
    tmp107 = tmp106 * tmp91
    tmp108 = tmp107.to(tl.float32)
    tmp109 = tmp108 * tmp47
    tmp110 = tmp98 + tmp109
    tmp111 = tl.full(tmp110.shape, 0.0, tmp110.dtype)
    tmp112 = tl.where(tmp22, tmp110, tmp111)
    tmp113 = tl.where(tmp22, tmp112, tmp52)
    tmp114 = tl.load(in_ptr2 + (64 + r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp115 = tmp114.to(tl.float32)
    tmp116 = tl.load(in_ptr0 + (64 + r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp117 = tmp116.to(tl.float32)
    tmp118 = tl.load(in_ptr1 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), tmp54, eviction_policy='evict_last', other=0.0)
    tmp119 = tmp117 * tmp118
    tmp120 = tmp119 * tmp61
    tmp121 = tmp120 * tmp9
    tmp122 = tmp115 - tmp121
    tmp123 = tmp122 * tmp118
    tmp124 = tmp123.to(tl.float32)
    tmp125 = tmp124 * tmp68
    tmp126 = tl.load(in_ptr2 + (r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp127 = tmp126.to(tl.float32)
    tmp128 = tl.load(in_ptr0 + (r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp129 = tmp128.to(tl.float32)
    tmp130 = tmp129 * tmp118
    tmp131 = tmp130 * tmp61
    tmp132 = tmp131 * tmp9
    tmp133 = tmp127 - tmp132
    tmp134 = tmp133 * tmp118
    tmp135 = tmp134.to(tl.float32)
    tmp136 = tmp135 * tmp80
    tmp137 = tmp125 + tmp136
    tmp138 = tl.full(tmp137.shape, 0.0, tmp137.dtype)
    tmp139 = tl.where(tmp54, tmp137, tmp138)
    tmp140 = tl.where(tmp54, tmp139, tmp52)
    tmp141 = tmp113 + tmp140
    tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp86, None)
    tl.store(in_out_ptr1 + (r0_1 + 128*x0), tmp141, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/pg/cpgvl64o5ziwvq4e23qjukxmpq32khgwdzosjqs6u7josycoppn6.py
# Topologically Sorted Source Nodes: [loss, linear_48, sigmoid_3, gate_3, unsqueeze_3], Original ATen: [aten.nll_loss_forward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   gate_3 => mul_108
#   linear_48 => view_132
#   loss => full_default_1
#   sigmoid_3 => sigmoid_3
#   unsqueeze_3 => unsqueeze_3
# Graph fragment:
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_132 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_48, [128, 2048, 4]), kwargs = {})
#   %sigmoid_3 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_132,), kwargs = {})
#   %mul_108 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_3, 2), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[128, 2048, 4, 1][8192, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_108, -1), kwargs = {})
#   %mul_154 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_34, %unsqueeze_3), kwargs = {})
#   %view_156 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_154, [128, 2048, 512]), kwargs = {})
#   %convert_element_type_307 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_156, torch.float32), kwargs = {})
#   %eq : Tensor "b8[128, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, -1), kwargs = {})
#   %unsqueeze_6 : Tensor "b8[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %where_5 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_6, %full_default_1, %convert_element_type_307), kwargs = {})
#   %full_default_12 : Tensor "f32[8192, 512][512, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([8192, 512], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : Tensor "f32[8192, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_12, [%primals_1], %where_5, True), kwargs = {})
#   return %index_put
triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8 = async_compile.triton('triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 33554432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/7l/c7lhcbnesvtanmwpriud67ucipkmkgmsieycggohnvd3wry4dbms.py
# Topologically Sorted Source Nodes: [loss, linear_48, sigmoid_3, gate_3, unsqueeze_3, ve_7], Original ATen: [aten.nll_loss_forward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   gate_3 => mul_108
#   linear_48 => view_132
#   loss => full_default_1
#   sigmoid_3 => sigmoid_3
#   unsqueeze_3 => unsqueeze_3
#   ve_7 => view_130
# Graph fragment:
#   %getitem_34 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0" = PlaceHolder[target=getitem_34]
#   %embedding_4 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=embedding_4]
#   %primals_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %mm_48 : Tensor "bf16[262144, 4][4, 1]cuda:0" = PlaceHolder[target=mm_48]
#   %index_put : Tensor "f32[8192, 512][512, 1]cuda:0" = PlaceHolder[target=index_put]
#   %sum_9 : Tensor "f32[128, 2048, 4, 1][8192, 4, 1, 1048576]cuda:0" = PlaceHolder[target=sum_9]
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_132 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_48, [128, 2048, 4]), kwargs = {})
#   %sigmoid_3 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_132,), kwargs = {})
#   %mul_108 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_3, 2), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[128, 2048, 4, 1][8192, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_108, -1), kwargs = {})
#   %mul_154 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_34, %unsqueeze_3), kwargs = {})
#   %view_130 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%embedding_4, [128, 2048, 4, 128]), kwargs = {})
#   %mul_155 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_34, %view_130), kwargs = {})
#   %sum_9 : Tensor "f32[128, 2048, 4, 1][8192, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_155, [3], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_280 : Tensor "bf16[128, 2048, 4, 1][8192, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_9, torch.bfloat16), kwargs = {})
#   %squeeze_1 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convert_element_type_280, -1), kwargs = {})
#   %mul_156 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, 2), kwargs = {})
#   %convert_element_type_281 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_156, torch.float32), kwargs = {})
#   %convert_element_type_282 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_3, torch.float32), kwargs = {})
#   %sub_8 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_282), kwargs = {})
#   %mul_157 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_282, %sub_8), kwargs = {})
#   %mul_158 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_281, %mul_157), kwargs = {})
#   %convert_element_type_283 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_158, torch.bfloat16), kwargs = {})
#   %view_156 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_154, [128, 2048, 512]), kwargs = {})
#   %convert_element_type_307 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_156, torch.float32), kwargs = {})
#   %eq : Tensor "b8[128, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, -1), kwargs = {})
#   %unsqueeze_6 : Tensor "b8[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %where_5 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_6, %full_default_1, %convert_element_type_307), kwargs = {})
#   %full_default_12 : Tensor "f32[8192, 512][512, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([8192, 512], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : Tensor "f32[8192, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_12, [%primals_1], %where_5, True), kwargs = {})
#   return %sum_9,%buf53,%convert_element_type_283
triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1048576, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x3 = xindex // 4
    x2 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp26 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 8192), "index out of bounds: 0 <= tmp11 < 8192")
    tmp13 = tl.full([1, 1], -1, tl.int64)
    tmp14 = tmp7 == tmp13
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = 2.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp0 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 0.0
    tmp22 = tl.where(tmp14, tmp21, tmp20)
    tmp23 = tmp6.to(tl.float32)
    tmp24 = tmp23 * tmp17
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tl.sigmoid(tmp26)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = 1.0
    tmp30 = tmp29 - tmp28
    tmp31 = tmp28 * tmp30
    tmp32 = tmp25 * tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.atomic_add(out_ptr1 + (r0_1 + 128*x2 + 512*tmp11), tmp22, None, sem='relaxed')
    tl.store(out_ptr2 + (x0), tmp33, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/sm/csmmix5zqsqvh5zc2lys3rngvlwhc4xq74ktvx2ug5rg6dxj5f2v.py
# Topologically Sorted Source Nodes: [linear_48, sigmoid_3], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
# Source node to ATen node mapping:
#   linear_48 => view_132
#   sigmoid_3 => sigmoid_3
# Graph fragment:
#   %convert_element_type_283 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0" = PlaceHolder[target=convert_element_type_283]
#   %view_132 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_48, [128, 2048, 4]), kwargs = {})
#   %sigmoid_3 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_132,), kwargs = {})
#   %convert_element_type_280 : Tensor "bf16[128, 2048, 4, 1][8192, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_9, torch.bfloat16), kwargs = {})
#   %squeeze_1 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convert_element_type_280, -1), kwargs = {})
#   %mul_156 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, 2), kwargs = {})
#   %convert_element_type_281 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_156, torch.float32), kwargs = {})
#   %convert_element_type_282 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_3, torch.float32), kwargs = {})
#   %sub_8 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_282), kwargs = {})
#   %mul_157 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_282, %sub_8), kwargs = {})
#   %mul_158 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_281, %mul_157), kwargs = {})
#   %convert_element_type_283 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_158, torch.bfloat16), kwargs = {})
#   %view_154 : Tensor "bf16[262144, 4][4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_283, [262144, 4]), kwargs = {})
#   %permute_69 : Tensor "bf16[4, 262144][1, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_154, [1, 0]), kwargs = {})
#   %constant_pad_nd_default_11 : Tensor "bf16[8, 262144][262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_69, [0, 0, 0, 4]), kwargs = {})
#   %constant_pad_nd_default_9 : Tensor "bf16[262144, 8][8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_154, [0, 4, 0, 0]), kwargs = {})
#   return %constant_pad_nd_default_11,%constant_pad_nd_default_9
triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20971520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 4*x1), tmp2, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp3, None)
    tl.store(out_ptr1 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/pr/cprsukkjzz6pneu7zzx5nwfjv37lmssjmq3gpxapcjjjgvm6dz3y.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %permute_71 : Tensor "bf16[4, 32][32, 1]cuda:0" = PlaceHolder[target=permute_71]
#   %constant_pad_nd_default_10 : Tensor "bf16[8, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_71, [0, 0, 0, 4]), kwargs = {})
#   return %constant_pad_nd_default_10
triton_poi_fused_mm_11 = async_compile.triton('triton_poi_fused_mm_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 32
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2 & xmask, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/ok/cokcipea5xenlh5kbsqlyy4h5wa3exenrxhe7x2lf4qrft4jrywn.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_default_7 : Tensor "bf16[8, 32][32, 1]cuda:0" = PlaceHolder[target=mm_default_7]
#   %slice_tensor_3 : Tensor "bf16[4, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_default_7, 0, 0, -4), kwargs = {})
#   %convert_element_type_288 : Tensor "f32[4, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_tensor_3, torch.float32), kwargs = {})
#   return %convert_element_type_288
triton_poi_fused__to_copy_mm_12 = async_compile.triton('triton_poi_fused__to_copy_mm_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mm_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mm_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/sr/csrtinkshg5rrxpujuhlqlw3uyi26ubgu6af43ati6nqsoj7lreb.py
# Topologically Sorted Source Nodes: [rms_norm_29, getitem_47], Original ATen: [aten.view, aten.slice_backward, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.select]
# Source node to ATen node mapping:
#   getitem_47 => select_14
#   rms_norm_29 => convert_element_type_207, mul_107
# Graph fragment:
#   %add_81 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_81]
#   %rsqrt_29 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_29]
#   %mm_default_6 : Tensor "bf16[262144, 32][32, 1]cuda:0" = PlaceHolder[target=mm_default_6]
#   %mm_64 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_64]
#   %mm_66 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_66]
#   %mm_68 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_68]
#   %add_94 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_94]
#   %sum_10 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_10]
#   %add_104 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_104]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %view_155 : Tensor "bf16[128, 2048, 32][65536, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_default_6, [128, 2048, 32]), kwargs = {})
#   %full_default_10 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([128, 2048, 512], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_4 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_155, 2, 0, 32), kwargs = {})
#   %view_159 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_64, [128, 2048, 512]), kwargs = {})
#   %add_101 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_4, %view_159), kwargs = {})
#   %view_162 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_66, [128, 2048, 512]), kwargs = {})
#   %add_102 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_101, %view_162), kwargs = {})
#   %view_165 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_68, [128, 2048, 512]), kwargs = {})
#   %add_103 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_102, %view_165), kwargs = {})
#   %convert_element_type_304 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_103, torch.float32), kwargs = {})
#   %convert_element_type_207 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_81, torch.float32), kwargs = {})
#   %mul_107 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_207, %rsqrt_29), kwargs = {})
#   %mul_160 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_107, %convert_element_type_304), kwargs = {})
#   %sum_10 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_160, [2], True), kwargs = {})
#   %div_8 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_107, 512), kwargs = {})
#   %mul_161 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_8, %sum_10), kwargs = {})
#   %sub_9 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_304, %mul_161), kwargs = {})
#   %mul_162 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_29), kwargs = {})
#   %convert_element_type_306 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_162, torch.bfloat16), kwargs = {})
#   %add_104 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, %convert_element_type_306), kwargs = {})
#   %select_14 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 7), kwargs = {})
#   %mul_165 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_104, %select_14), kwargs = {})
#   %view_166 : Tensor "bf16[262144, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_165, [262144, 512]), kwargs = {})
#   return %sum_10,%add_104,%view_166
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_13 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 2684354560}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr5 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = r0_1
        tmp5 = tl.full([1, 1], 32, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr2 + (r0_1 + 32*x0), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = 0.0
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp3 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp45 = tl.load(in_ptr6 + (7))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr0 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = r0_1
        tmp23 = tl.full([1, 1], 32, tl.int64)
        tmp24 = tmp22 < tmp23
        tmp25 = tl.load(in_ptr2 + (r0_1 + 32*x0), r0_mask & tmp24, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = 0.0
        tmp27 = tl.where(tmp24, tmp25, tmp26)
        tmp29 = tmp27 + tmp28
        tmp31 = tmp29 + tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp36 * tmp2
        tmp38 = 0.001953125
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39 * tmp19
        tmp41 = tmp34 - tmp40
        tmp42 = tmp41 * tmp2
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp21 + tmp43
        tmp47 = tmp46.to(tl.float32)
        tmp48 = tmp44 * tmp47
        tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp44, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 512*x0), tmp48, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/uj/cujstsqoadt7bgyshlvmohx2c25ovd673cowktmj5mz76a2sluuj.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf53 : Tensor "f32[8192, 512][512, 1]cuda:0" = PlaceHolder[target=buf53]
#   %convert_element_type_308 : Tensor "bf16[8192, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%index_put, torch.bfloat16), kwargs = {})
#   return %convert_element_type_308
triton_poi_fused_embedding_dense_backward_14 = async_compile.triton('triton_poi_fused_embedding_dense_backward_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16777216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/er/cerqwr2sl6dfko6xohumedglyawjtpujobyv7ngoxqkhyubilcfy.py
# Topologically Sorted Source Nodes: [getitem_47, rms_norm_28], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_47 => select_14
#   rms_norm_28 => convert_element_type_197, mul_104
# Graph fragment:
#   %add_78 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_78]
#   %rsqrt_28 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_28]
#   %mm_72 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_72]
#   %add_104 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_104]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_13 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_13]
#   %select_14 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 7), kwargs = {})
#   %mul_165 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_104, %select_14), kwargs = {})
#   %view_169 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_72, [128, 2048, 512]), kwargs = {})
#   %convert_element_type_321 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_169, torch.float32), kwargs = {})
#   %convert_element_type_197 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_78, torch.float32), kwargs = {})
#   %mul_104 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_197, %rsqrt_28), kwargs = {})
#   %mul_170 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_104, %convert_element_type_321), kwargs = {})
#   %sum_13 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_170, [2], True), kwargs = {})
#   %div_9 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_104, 512), kwargs = {})
#   %mul_171 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %sum_13), kwargs = {})
#   %sub_10 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_321, %mul_171), kwargs = {})
#   %mul_172 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_28), kwargs = {})
#   %convert_element_type_323 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_172, torch.bfloat16), kwargs = {})
#   %add_105 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_165, %convert_element_type_323), kwargs = {})
#   return %sum_13,%add_105
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_15 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1342177280}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (7))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 * tmp13
    tmp15 = 0.001953125
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 * tmp9
    tmp18 = tmp5 - tmp17
    tmp19 = tmp18 * tmp2
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp14 + tmp20
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp21, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/vx/cvxjoj4j6mir3y4qwxy54ers6oo4b5azteauhnjsj3kzks7ydxig.py
# Topologically Sorted Source Nodes: [x_1, rms_norm_25, getitem_41], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   getitem_41 => select_12
#   rms_norm_25 => convert_element_type_179, mul_93
#   x_1 => convert_element_type, convert_element_type_1, mul
# Graph fragment:
#   %add_70 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_70]
#   %rsqrt_25 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_25]
#   %mm_76 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_76]
#   %mm_78 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_78]
#   %mm_80 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_80]
#   %add_105 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_105]
#   %sum_16 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_16]
#   %add_114 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_114]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_104 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_104]
#   %embedding : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_80 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_80]
#   %add_69 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_69]
#   %convert_element_type : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=10] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_164 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_104, %convert_element_type_1), kwargs = {})
#   %sum_11 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_164,), kwargs = {dtype: torch.float32})
#   %mul_166 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_104, %add_80), kwargs = {})
#   %sum_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_166,), kwargs = {dtype: torch.float32})
#   %view_175 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_76, [128, 2048, 512]), kwargs = {})
#   %view_178 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_78, [128, 2048, 512]), kwargs = {})
#   %add_112 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_175, %view_178), kwargs = {})
#   %view_181 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_80, [128, 2048, 512]), kwargs = {})
#   %add_113 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_112, %view_181), kwargs = {})
#   %convert_element_type_350 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_113, torch.float32), kwargs = {})
#   %convert_element_type_179 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_70, torch.float32), kwargs = {})
#   %mul_93 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_179, %rsqrt_25), kwargs = {})
#   %mul_190 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, %convert_element_type_350), kwargs = {})
#   %sum_16 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_190, [2], True), kwargs = {})
#   %div_12 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_93, 512), kwargs = {})
#   %mul_191 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, %sum_16), kwargs = {})
#   %sub_13 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_350, %mul_191), kwargs = {})
#   %mul_192 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %rsqrt_25), kwargs = {})
#   %convert_element_type_352 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_192, torch.bfloat16), kwargs = {})
#   %add_114 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_105, %convert_element_type_352), kwargs = {})
#   %mul_194 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_114, %convert_element_type_1), kwargs = {})
#   %sum_17 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_194,), kwargs = {dtype: torch.float32})
#   %select_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 6), kwargs = {})
#   %mul_195 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_114, %select_12), kwargs = {})
#   %mul_196 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_114, %add_69), kwargs = {})
#   %sum_18 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_196,), kwargs = {dtype: torch.float32})
#   %view_182 : Tensor "bf16[262144, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_195, [262144, 512]), kwargs = {})
#   return %sum_16,%add_114,%view_182,%buf55,%buf96,%buf57,%buf98
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_16 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 12, 'num_reduction': 5, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760, 'r0_': 3489660928}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp22 = tl.load(in_ptr5 + (6))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp26 = tl.load(in_ptr6 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp27 = tl.load(in_ptr7 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp29 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr9 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp48 = tl.load(in_ptr10 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp3 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp15 = 0.001953125
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 * tmp13
    tmp18 = tmp9 - tmp17
    tmp19 = tmp18 * tmp2
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp14 + tmp20
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp21 * tmp24
    tmp28 = tmp27.to(tl.float32)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp26 * tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
    tmp36 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp37 = tmp21 * tmp31
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.sum(tmp39, 1)[:, None].to(tl.float32)
    tmp43 = tmp26 * tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
    tmp47 = tl.sum(tmp45, 1)[:, None].to(tl.float32)
    tmp49 = tmp21 * tmp48
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
    tmp53 = tl.sum(tmp51, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp21, None)
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp25, None)
    tl.store(out_ptr2 + (x0), tmp36, None)
    tl.store(out_ptr3 + (x0), tmp41, None)
    tl.store(out_ptr4 + (x0), tmp47, None)
    tl.store(out_ptr5 + (x0), tmp53, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/mj/cmjee5x662t7vnyd6abnuggp7djfqowemklwdyowhmk6wwfpfsfo.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   x_1 => convert_element_type, convert_element_type_1, mul
# Graph fragment:
#   %buf55 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf55]
#   %convert_element_type : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=10] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_164 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_104, %convert_element_type_1), kwargs = {})
#   %sum_11 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_164,), kwargs = {dtype: torch.float32})
#   return %sum_11
triton_red_fused__to_copy_mul_sum_17 = async_compile.triton('triton_red_fused__to_copy_mul_sum_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_sum_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 1048576}}
)
@triton.jit
def triton_red_fused__to_copy_mul_sum_17(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 262144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), None, eviction_policy='evict_first')
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tmp3
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/3c/c3cm4tc3eqlx5icptpspaeflm7sfscxykun6xt66byrilpsldxls.py
# Topologically Sorted Source Nodes: [getitem_41, rms_norm_24], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_41 => select_12
#   rms_norm_24 => convert_element_type_169, mul_90
# Graph fragment:
#   %add_67 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_67]
#   %rsqrt_24 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_24]
#   %mm_84 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_84]
#   %add_114 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_114]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_19 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_19]
#   %select_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 6), kwargs = {})
#   %mul_195 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_114, %select_12), kwargs = {})
#   %view_185 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_84, [128, 2048, 512]), kwargs = {})
#   %convert_element_type_365 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_185, torch.float32), kwargs = {})
#   %convert_element_type_169 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_67, torch.float32), kwargs = {})
#   %mul_90 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_169, %rsqrt_24), kwargs = {})
#   %mul_200 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %convert_element_type_365), kwargs = {})
#   %sum_19 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_200, [2], True), kwargs = {})
#   %div_13 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_90, 512), kwargs = {})
#   %mul_201 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_13, %sum_19), kwargs = {})
#   %sub_14 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_365, %mul_201), kwargs = {})
#   %mul_202 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_24), kwargs = {})
#   %convert_element_type_367 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_202, torch.bfloat16), kwargs = {})
#   %add_118 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_195, %convert_element_type_367), kwargs = {})
#   return %sum_19,%add_118
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_18 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1342177280}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (6))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 * tmp13
    tmp15 = 0.001953125
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 * tmp9
    tmp18 = tmp5 - tmp17
    tmp19 = tmp18 * tmp2
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp14 + tmp20
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp21, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/7u/c7uguzbc2smso7em4nuxdze6bpyjf72tcze5uihgxenbnmclx3i5.py
# Topologically Sorted Source Nodes: [rms_norm_21, getitem_34], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.select]
# Source node to ATen node mapping:
#   getitem_34 => select_10
#   rms_norm_21 => convert_element_type_148, mul_77
# Graph fragment:
#   %add_58 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_58]
#   %rsqrt_21 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_21]
#   %mm_default_4 : Tensor "bf16[262144, 32][32, 1]cuda:0" = PlaceHolder[target=mm_default_4]
#   %mm_90 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_90]
#   %mm_92 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_92]
#   %mm_94 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_94]
#   %add_118 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_118]
#   %sum_23 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_23]
#   %add_128 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_128]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %full_default_10 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([128, 2048, 512], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_190 : Tensor "bf16[128, 2048, 32][65536, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_default_4, [128, 2048, 32]), kwargs = {})
#   %slice_scatter_default_13 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_190, 2, 0, 32), kwargs = {})
#   %view_194 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_90, [128, 2048, 512]), kwargs = {})
#   %add_125 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_13, %view_194), kwargs = {})
#   %view_197 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_92, [128, 2048, 512]), kwargs = {})
#   %add_126 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_125, %view_197), kwargs = {})
#   %view_200 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_94, [128, 2048, 512]), kwargs = {})
#   %add_127 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_126, %view_200), kwargs = {})
#   %convert_element_type_403 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_127, torch.float32), kwargs = {})
#   %convert_element_type_148 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_58, torch.float32), kwargs = {})
#   %mul_77 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_148, %rsqrt_21), kwargs = {})
#   %mul_225 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_77, %convert_element_type_403), kwargs = {})
#   %sum_23 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_225, [2], True), kwargs = {})
#   %div_16 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_77, 512), kwargs = {})
#   %mul_226 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_16, %sum_23), kwargs = {})
#   %sub_18 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_403, %mul_226), kwargs = {})
#   %mul_227 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_21), kwargs = {})
#   %convert_element_type_405 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_227, torch.bfloat16), kwargs = {})
#   %add_128 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_118, %convert_element_type_405), kwargs = {})
#   %select_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 5), kwargs = {})
#   %mul_230 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_128, %select_10), kwargs = {})
#   %view_201 : Tensor "bf16[262144, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_230, [262144, 512]), kwargs = {})
#   return %sum_23,%add_128,%view_201
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_19 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 2684354560}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr5 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = r0_1
        tmp5 = tl.full([1, 1], 32, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr2 + (r0_1 + 32*x0), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = 0.0
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp3 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp45 = tl.load(in_ptr6 + (5))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr0 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = r0_1
        tmp23 = tl.full([1, 1], 32, tl.int64)
        tmp24 = tmp22 < tmp23
        tmp25 = tl.load(in_ptr2 + (r0_1 + 32*x0), r0_mask & tmp24, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = 0.0
        tmp27 = tl.where(tmp24, tmp25, tmp26)
        tmp29 = tmp27 + tmp28
        tmp31 = tmp29 + tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp36 * tmp2
        tmp38 = 0.001953125
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39 * tmp19
        tmp41 = tmp34 - tmp40
        tmp42 = tmp41 * tmp2
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp21 + tmp43
        tmp47 = tmp46.to(tl.float32)
        tmp48 = tmp44 * tmp47
        tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp44, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 512*x0), tmp48, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/zr/czrbkjrvd7tofxoh2air77rzwan5xydhe74vfpdvcicgncu3cxv4.py
# Topologically Sorted Source Nodes: [getitem_34, rms_norm_20], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_34 => select_10
#   rms_norm_20 => convert_element_type_138, mul_74
# Graph fragment:
#   %add_55 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_55]
#   %rsqrt_20 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_20]
#   %mm_98 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_98]
#   %add_128 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_128]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_26 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_26]
#   %select_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 5), kwargs = {})
#   %mul_230 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_128, %select_10), kwargs = {})
#   %view_204 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_98, [128, 2048, 512]), kwargs = {})
#   %convert_element_type_420 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_204, torch.float32), kwargs = {})
#   %convert_element_type_138 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_55, torch.float32), kwargs = {})
#   %mul_74 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_138, %rsqrt_20), kwargs = {})
#   %mul_235 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_74, %convert_element_type_420), kwargs = {})
#   %sum_26 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_235, [2], True), kwargs = {})
#   %div_17 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_74, 512), kwargs = {})
#   %mul_236 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_17, %sum_26), kwargs = {})
#   %sub_19 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_420, %mul_236), kwargs = {})
#   %mul_237 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %rsqrt_20), kwargs = {})
#   %convert_element_type_422 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_237, torch.bfloat16), kwargs = {})
#   %add_132 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_230, %convert_element_type_422), kwargs = {})
#   return %sum_26,%add_132
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_20 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1342177280}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (5))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 * tmp13
    tmp15 = 0.001953125
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 * tmp9
    tmp18 = tmp5 - tmp17
    tmp19 = tmp18 * tmp2
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp14 + tmp20
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp21, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/jy/cjynaloezzv5firue2f2ukujctqcskl536hc55pnbo3hcnynaw2g.py
# Topologically Sorted Source Nodes: [x_1, rms_norm_17, getitem_28], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   getitem_28 => select_8
#   rms_norm_17 => convert_element_type_120, mul_63
#   x_1 => convert_element_type, convert_element_type_1, mul
# Graph fragment:
#   %add_47 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_47]
#   %rsqrt_17 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_17]
#   %mm_102 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_102]
#   %mm_104 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_104]
#   %mm_106 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_106]
#   %add_132 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_132]
#   %sum_29 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_29]
#   %add_141 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_141]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_128 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_128]
#   %embedding : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_57 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_57]
#   %add_46 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_46]
#   %convert_element_type : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=10] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_229 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_128, %convert_element_type_1), kwargs = {})
#   %sum_24 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_229,), kwargs = {dtype: torch.float32})
#   %mul_231 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_128, %add_57), kwargs = {})
#   %sum_25 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_231,), kwargs = {dtype: torch.float32})
#   %view_210 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_102, [128, 2048, 512]), kwargs = {})
#   %view_213 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_104, [128, 2048, 512]), kwargs = {})
#   %add_139 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_210, %view_213), kwargs = {})
#   %view_216 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_106, [128, 2048, 512]), kwargs = {})
#   %add_140 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_139, %view_216), kwargs = {})
#   %convert_element_type_449 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_140, torch.float32), kwargs = {})
#   %convert_element_type_120 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_47, torch.float32), kwargs = {})
#   %mul_63 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_120, %rsqrt_17), kwargs = {})
#   %mul_255 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_63, %convert_element_type_449), kwargs = {})
#   %sum_29 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_255, [2], True), kwargs = {})
#   %div_20 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_63, 512), kwargs = {})
#   %mul_256 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_20, %sum_29), kwargs = {})
#   %sub_22 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_449, %mul_256), kwargs = {})
#   %mul_257 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt_17), kwargs = {})
#   %convert_element_type_451 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_257, torch.bfloat16), kwargs = {})
#   %add_141 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_132, %convert_element_type_451), kwargs = {})
#   %mul_259 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_141, %convert_element_type_1), kwargs = {})
#   %sum_30 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_259,), kwargs = {dtype: torch.float32})
#   %select_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 4), kwargs = {})
#   %mul_260 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_141, %select_8), kwargs = {})
#   %mul_261 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_141, %add_46), kwargs = {})
#   %sum_31 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_261,), kwargs = {dtype: torch.float32})
#   %view_217 : Tensor "bf16[262144, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_260, [262144, 512]), kwargs = {})
#   return %sum_29,%add_141,%view_217,%buf148,%buf189,%buf150,%buf191
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_21 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 12, 'num_reduction': 5, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760, 'r0_': 3489660928}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp22 = tl.load(in_ptr5 + (4))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp26 = tl.load(in_ptr6 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp27 = tl.load(in_ptr7 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp29 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr9 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp48 = tl.load(in_ptr10 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp3 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp15 = 0.001953125
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 * tmp13
    tmp18 = tmp9 - tmp17
    tmp19 = tmp18 * tmp2
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp14 + tmp20
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp21 * tmp24
    tmp28 = tmp27.to(tl.float32)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp26 * tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
    tmp36 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp37 = tmp21 * tmp31
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.sum(tmp39, 1)[:, None].to(tl.float32)
    tmp43 = tmp26 * tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
    tmp47 = tl.sum(tmp45, 1)[:, None].to(tl.float32)
    tmp49 = tmp21 * tmp48
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
    tmp53 = tl.sum(tmp51, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp21, None)
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp25, None)
    tl.store(out_ptr2 + (x0), tmp36, None)
    tl.store(out_ptr3 + (x0), tmp41, None)
    tl.store(out_ptr4 + (x0), tmp47, None)
    tl.store(out_ptr5 + (x0), tmp53, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/5m/c5mucritq2x2pisxvyzyx7wooxpgx2cviy7thrykpeaa53gcmbo3.py
# Topologically Sorted Source Nodes: [getitem_28, rms_norm_16], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_28 => select_8
#   rms_norm_16 => convert_element_type_110, mul_60
# Graph fragment:
#   %add_44 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_44]
#   %rsqrt_16 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_16]
#   %mm_110 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_110]
#   %add_141 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_141]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_32 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_32]
#   %select_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 4), kwargs = {})
#   %mul_260 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_141, %select_8), kwargs = {})
#   %view_220 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_110, [128, 2048, 512]), kwargs = {})
#   %convert_element_type_464 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_220, torch.float32), kwargs = {})
#   %convert_element_type_110 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_44, torch.float32), kwargs = {})
#   %mul_60 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_110, %rsqrt_16), kwargs = {})
#   %mul_265 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_60, %convert_element_type_464), kwargs = {})
#   %sum_32 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_265, [2], True), kwargs = {})
#   %div_21 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_60, 512), kwargs = {})
#   %mul_266 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_21, %sum_32), kwargs = {})
#   %sub_23 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_464, %mul_266), kwargs = {})
#   %mul_267 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %rsqrt_16), kwargs = {})
#   %convert_element_type_466 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_267, torch.bfloat16), kwargs = {})
#   %add_145 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_260, %convert_element_type_466), kwargs = {})
#   return %sum_32,%add_145
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_22 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1342177280}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (4))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 * tmp13
    tmp15 = 0.001953125
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 * tmp9
    tmp18 = tmp5 - tmp17
    tmp19 = tmp18 * tmp2
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp14 + tmp20
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp21, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/2i/c2izmmvseiefukafjyirynj2iaw6ey3phurstkenxgrw6zrd32ql.py
# Topologically Sorted Source Nodes: [getitem_48, getitem_42, getitem_35, getitem_29, rms_norm_13, getitem_22, getitem_21], Original ATen: [aten.slice_backward, aten.select, aten.mul, aten.add, aten.view, aten._fused_rms_norm_backward, aten._to_copy]
# Source node to ATen node mapping:
#   getitem_21 => select_6
#   getitem_22 => select_7
#   getitem_29 => select_9
#   getitem_35 => select_11
#   getitem_42 => select_13
#   getitem_48 => select_15
#   rms_norm_13 => convert_element_type_89, mul_47
# Graph fragment:
#   %add_35 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_35]
#   %rsqrt_13 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_13]
#   %mm_default_2 : Tensor "bf16[262144, 32][32, 1]cuda:0" = PlaceHolder[target=mm_default_2]
#   %mm_116 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_116]
#   %mm_118 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_118]
#   %mm_120 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_120]
#   %add_145 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_145]
#   %sum_36 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_36]
#   %add_104 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_104]
#   %primals_6 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_6]
#   %add_114 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_114]
#   %add_128 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_128]
#   %add_141 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_141]
#   %add_155 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_155]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %full_default_10 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([128, 2048, 512], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_15 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 7), kwargs = {})
#   %mul_163 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_104, %select_15), kwargs = {})
#   %select_13 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 6), kwargs = {})
#   %mul_193 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_114, %select_13), kwargs = {})
#   %add_115 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_163, %mul_193), kwargs = {})
#   %select_11 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 5), kwargs = {})
#   %mul_228 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_128, %select_11), kwargs = {})
#   %add_129 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_115, %mul_228), kwargs = {})
#   %select_9 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 4), kwargs = {})
#   %mul_258 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_141, %select_9), kwargs = {})
#   %add_142 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_129, %mul_258), kwargs = {})
#   %view_225 : Tensor "bf16[128, 2048, 32][65536, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_default_2, [128, 2048, 32]), kwargs = {})
#   %slice_scatter_default_22 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_225, 2, 0, 32), kwargs = {})
#   %view_229 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_116, [128, 2048, 512]), kwargs = {})
#   %add_152 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_22, %view_229), kwargs = {})
#   %view_232 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_118, [128, 2048, 512]), kwargs = {})
#   %add_153 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_152, %view_232), kwargs = {})
#   %view_235 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_120, [128, 2048, 512]), kwargs = {})
#   %add_154 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_153, %view_235), kwargs = {})
#   %convert_element_type_502 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_154, torch.float32), kwargs = {})
#   %convert_element_type_89 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_35, torch.float32), kwargs = {})
#   %mul_47 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_89, %rsqrt_13), kwargs = {})
#   %mul_290 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_47, %convert_element_type_502), kwargs = {})
#   %sum_36 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_290, [2], True), kwargs = {})
#   %div_24 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_47, 512), kwargs = {})
#   %mul_291 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_24, %sum_36), kwargs = {})
#   %sub_27 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_502, %mul_291), kwargs = {})
#   %mul_292 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_13), kwargs = {})
#   %convert_element_type_504 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_292, torch.bfloat16), kwargs = {})
#   %add_155 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_145, %convert_element_type_504), kwargs = {})
#   %select_7 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 3), kwargs = {})
#   %mul_293 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_155, %select_7), kwargs = {})
#   %add_156 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_142, %mul_293), kwargs = {})
#   %select_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 3), kwargs = {})
#   %mul_295 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_155, %select_6), kwargs = {})
#   %view_236 : Tensor "bf16[262144, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_295, [262144, 512]), kwargs = {})
#   return %sum_36,%add_155,%add_156,%view_236
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_23 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_23', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 22, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 4294967296}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_23(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr5 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = r0_1
        tmp5 = tl.full([1, 1], 32, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr2 + (r0_1 + 32*x0), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = 0.0
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp3 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp46 = tl.load(in_ptr6 + (7))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, R0_BLOCK])
    tmp51 = tl.load(in_ptr6 + (6))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK, R0_BLOCK])
    tmp57 = tl.load(in_ptr6 + (5))
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK, R0_BLOCK])
    tmp63 = tl.load(in_ptr6 + (4))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, R0_BLOCK])
    tmp68 = tl.load(in_ptr6 + (3))
    tmp69 = tl.broadcast_to(tmp68, [XBLOCK, R0_BLOCK])
    tmp73 = tl.load(in_ptr10 + (3))
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr0 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp45 = tl.load(in_out_ptr1 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp50 = tl.load(in_ptr7 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp56 = tl.load(in_ptr8 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp62 = tl.load(in_ptr9 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = r0_1
        tmp23 = tl.full([1, 1], 32, tl.int64)
        tmp24 = tmp22 < tmp23
        tmp25 = tl.load(in_ptr2 + (r0_1 + 32*x0), r0_mask & tmp24, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = 0.0
        tmp27 = tl.where(tmp24, tmp25, tmp26)
        tmp29 = tmp27 + tmp28
        tmp31 = tmp29 + tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp36 * tmp2
        tmp38 = 0.001953125
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39 * tmp19
        tmp41 = tmp34 - tmp40
        tmp42 = tmp41 * tmp2
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp21 + tmp43
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tmp45 * tmp48
        tmp53 = tmp52.to(tl.float32)
        tmp54 = tmp50 * tmp53
        tmp55 = tmp49 + tmp54
        tmp59 = tmp58.to(tl.float32)
        tmp60 = tmp56 * tmp59
        tmp61 = tmp55 + tmp60
        tmp65 = tmp64.to(tl.float32)
        tmp66 = tmp62 * tmp65
        tmp67 = tmp61 + tmp66
        tmp70 = tmp69.to(tl.float32)
        tmp71 = tmp44 * tmp70
        tmp72 = tmp67 + tmp71
        tmp75 = tmp74.to(tl.float32)
        tmp76 = tmp44 * tmp75
        tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp44, r0_mask)
        tl.store(in_out_ptr1 + (r0_1 + 512*x0), tmp72, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 512*x0), tmp76, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/z3/cz3vhftyivj56xiqo5gzfeqsuqv3fgnq3em7uwltp2bqaa7h2d7t.py
# Topologically Sorted Source Nodes: [getitem_21, rms_norm_12], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_21 => select_6
#   rms_norm_12 => convert_element_type_79, mul_44
# Graph fragment:
#   %add_32 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_32]
#   %rsqrt_12 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_12]
#   %mm_124 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_124]
#   %add_155 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_155]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_39 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_39]
#   %select_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 3), kwargs = {})
#   %mul_295 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_155, %select_6), kwargs = {})
#   %view_239 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_124, [128, 2048, 512]), kwargs = {})
#   %convert_element_type_519 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_239, torch.float32), kwargs = {})
#   %convert_element_type_79 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_32, torch.float32), kwargs = {})
#   %mul_44 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_79, %rsqrt_12), kwargs = {})
#   %mul_300 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, %convert_element_type_519), kwargs = {})
#   %sum_39 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_300, [2], True), kwargs = {})
#   %div_25 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_44, 512), kwargs = {})
#   %mul_301 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_25, %sum_39), kwargs = {})
#   %sub_28 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_519, %mul_301), kwargs = {})
#   %mul_302 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_12), kwargs = {})
#   %convert_element_type_521 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_302, torch.bfloat16), kwargs = {})
#   %add_159 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_295, %convert_element_type_521), kwargs = {})
#   return %sum_39,%add_159
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_24 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1342177280}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (3))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 * tmp13
    tmp15 = 0.001953125
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 * tmp9
    tmp18 = tmp5 - tmp17
    tmp19 = tmp18 * tmp2
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp14 + tmp20
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp21, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/g2/cg2k4ouis56tsogl2xkjtel3gr5rc5vuma2xgepuynoru722uf3h.py
# Topologically Sorted Source Nodes: [x_1, rms_norm_9, getitem_15], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   getitem_15 => select_4
#   rms_norm_9 => convert_element_type_61, mul_33
#   x_1 => convert_element_type, convert_element_type_1, mul
# Graph fragment:
#   %add_24 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_24]
#   %rsqrt_9 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_9]
#   %mm_128 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_128]
#   %mm_130 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_130]
#   %mm_132 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_132]
#   %add_159 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_159]
#   %sum_42 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_42]
#   %add_168 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_168]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_155 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_155]
#   %embedding : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_23 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_23]
#   %add_34 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_34]
#   %convert_element_type : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=10] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_294 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_155, %convert_element_type_1), kwargs = {})
#   %sum_37 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_294,), kwargs = {dtype: torch.float32})
#   %mul_296 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_155, %add_34), kwargs = {})
#   %sum_38 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_296,), kwargs = {dtype: torch.float32})
#   %view_245 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_128, [128, 2048, 512]), kwargs = {})
#   %view_248 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_130, [128, 2048, 512]), kwargs = {})
#   %add_166 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_245, %view_248), kwargs = {})
#   %view_251 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_132, [128, 2048, 512]), kwargs = {})
#   %add_167 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_166, %view_251), kwargs = {})
#   %convert_element_type_548 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_167, torch.float32), kwargs = {})
#   %convert_element_type_61 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_24, torch.float32), kwargs = {})
#   %mul_33 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_61, %rsqrt_9), kwargs = {})
#   %mul_320 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %convert_element_type_548), kwargs = {})
#   %sum_42 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_320, [2], True), kwargs = {})
#   %div_28 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_33, 512), kwargs = {})
#   %mul_321 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_28, %sum_42), kwargs = {})
#   %sub_31 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_548, %mul_321), kwargs = {})
#   %mul_322 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %rsqrt_9), kwargs = {})
#   %convert_element_type_550 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_322, torch.bfloat16), kwargs = {})
#   %add_168 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_159, %convert_element_type_550), kwargs = {})
#   %mul_324 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_168, %convert_element_type_1), kwargs = {})
#   %sum_43 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_324,), kwargs = {dtype: torch.float32})
#   %select_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 2), kwargs = {})
#   %mul_325 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_168, %select_4), kwargs = {})
#   %mul_326 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_168, %add_23), kwargs = {})
#   %sum_44 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_326,), kwargs = {dtype: torch.float32})
#   %view_252 : Tensor "bf16[262144, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_325, [262144, 512]), kwargs = {})
#   return %sum_42,%add_168,%view_252,%buf241,%buf283,%buf285,%buf244
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_25 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 12, 'num_reduction': 5, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760, 'r0_': 3489660928}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp22 = tl.load(in_ptr5 + (2))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp26 = tl.load(in_ptr6 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp27 = tl.load(in_ptr7 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp29 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr9 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp48 = tl.load(in_ptr10 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp3 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp15 = 0.001953125
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 * tmp13
    tmp18 = tmp9 - tmp17
    tmp19 = tmp18 * tmp2
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp14 + tmp20
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp21 * tmp24
    tmp28 = tmp27.to(tl.float32)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp26 * tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
    tmp36 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp37 = tmp21 * tmp31
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.sum(tmp39, 1)[:, None].to(tl.float32)
    tmp43 = tmp21 * tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
    tmp47 = tl.sum(tmp45, 1)[:, None].to(tl.float32)
    tmp49 = tmp26 * tmp48
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
    tmp53 = tl.sum(tmp51, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp21, None)
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp25, None)
    tl.store(out_ptr2 + (x0), tmp36, None)
    tl.store(out_ptr3 + (x0), tmp41, None)
    tl.store(out_ptr4 + (x0), tmp47, None)
    tl.store(out_ptr5 + (x0), tmp53, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/gd/cgdtsys7jjqloutcqovxi7ct7ncm4zjvg4ulwlqkgcbxvvprz6w5.py
# Topologically Sorted Source Nodes: [getitem_15, rms_norm_8], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_15 => select_4
#   rms_norm_8 => convert_element_type_51, mul_30
# Graph fragment:
#   %add_21 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_21]
#   %rsqrt_8 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_8]
#   %mm_136 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_136]
#   %add_168 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_168]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_45 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_45]
#   %select_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 2), kwargs = {})
#   %mul_325 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_168, %select_4), kwargs = {})
#   %view_255 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_136, [128, 2048, 512]), kwargs = {})
#   %convert_element_type_563 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_255, torch.float32), kwargs = {})
#   %convert_element_type_51 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_21, torch.float32), kwargs = {})
#   %mul_30 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_51, %rsqrt_8), kwargs = {})
#   %mul_330 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %convert_element_type_563), kwargs = {})
#   %sum_45 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_330, [2], True), kwargs = {})
#   %div_29 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_30, 512), kwargs = {})
#   %mul_331 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_29, %sum_45), kwargs = {})
#   %sub_32 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_563, %mul_331), kwargs = {})
#   %mul_332 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %rsqrt_8), kwargs = {})
#   %convert_element_type_565 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_332, torch.bfloat16), kwargs = {})
#   %add_172 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_325, %convert_element_type_565), kwargs = {})
#   return %sum_45,%add_172
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_26 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1342177280}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (2))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 * tmp13
    tmp15 = 0.001953125
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 * tmp9
    tmp18 = tmp5 - tmp17
    tmp19 = tmp18 * tmp2
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp14 + tmp20
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp21, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/if/cif5kgfankbvu4bxnni5xfb6uualgo7ilerdxagwcd7cmmasqvsa.py
# Topologically Sorted Source Nodes: [linear_9, sigmoid, ve_1], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.view, aten.mul, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward]
# Source node to ATen node mapping:
#   linear_9 => view_27
#   sigmoid => sigmoid
#   ve_1 => view_25
# Graph fragment:
#   %getitem_58 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0" = PlaceHolder[target=getitem_58]
#   %embedding_1 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=embedding_1]
#   %sum_48 : Tensor "f32[128, 2048, 4, 1][8192, 4, 1, 1048576]cuda:0" = PlaceHolder[target=sum_48]
#   %mm_9 : Tensor "bf16[262144, 4][4, 1]cuda:0" = PlaceHolder[target=mm_9]
#   %view_27 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [128, 2048, 4]), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_27,), kwargs = {})
#   %view_25 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%embedding_1, [128, 2048, 4, 128]), kwargs = {})
#   %mul_350 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_58, %view_25), kwargs = {})
#   %sum_48 : Tensor "f32[128, 2048, 4, 1][8192, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_350, [3], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_577 : Tensor "bf16[128, 2048, 4, 1][8192, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_48, torch.bfloat16), kwargs = {})
#   %squeeze_4 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convert_element_type_577, -1), kwargs = {})
#   %mul_351 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_4, 2), kwargs = {})
#   %convert_element_type_578 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_351, torch.float32), kwargs = {})
#   %convert_element_type_579 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid, torch.float32), kwargs = {})
#   %sub_35 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_579), kwargs = {})
#   %mul_352 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_579, %sub_35), kwargs = {})
#   %mul_353 : Tensor "f32[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_578, %mul_352), kwargs = {})
#   %convert_element_type_580 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_353, torch.bfloat16), kwargs = {})
#   return %sum_48,%convert_element_type_580
triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_view_27 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_view_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1048576, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_view_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456, 'r0_': 536870912}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_view_27(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = 1.0
    tmp15 = tmp14 - tmp13
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp18, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/gf/cgf2yjgv45i22tffrxoyxvyleatm7wbulttw5x64juvu6zmlpjzo.py
# Topologically Sorted Source Nodes: [rms_norm_5, getitem_8], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.select]
# Source node to ATen node mapping:
#   getitem_8 => select_2
#   rms_norm_5 => convert_element_type_30, mul_17
# Graph fragment:
#   %add_12 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_12]
#   %rsqrt_5 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_5]
#   %mm_default : Tensor "bf16[262144, 32][32, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %mm_142 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_142]
#   %mm_144 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_144]
#   %mm_146 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_146]
#   %add_172 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_172]
#   %sum_49 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_49]
#   %add_182 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_182]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %full_default_10 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([128, 2048, 512], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_260 : Tensor "bf16[128, 2048, 32][65536, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_default, [128, 2048, 32]), kwargs = {})
#   %slice_scatter_default_31 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_260, 2, 0, 32), kwargs = {})
#   %view_264 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_142, [128, 2048, 512]), kwargs = {})
#   %add_179 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_31, %view_264), kwargs = {})
#   %view_267 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_144, [128, 2048, 512]), kwargs = {})
#   %add_180 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_179, %view_267), kwargs = {})
#   %view_270 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_146, [128, 2048, 512]), kwargs = {})
#   %add_181 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_180, %view_270), kwargs = {})
#   %convert_element_type_601 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_181, torch.float32), kwargs = {})
#   %convert_element_type_30 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %mul_17 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_30, %rsqrt_5), kwargs = {})
#   %mul_355 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %convert_element_type_601), kwargs = {})
#   %sum_49 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_355, [2], True), kwargs = {})
#   %div_32 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_17, 512), kwargs = {})
#   %mul_356 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_32, %sum_49), kwargs = {})
#   %sub_36 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_601, %mul_356), kwargs = {})
#   %mul_357 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %rsqrt_5), kwargs = {})
#   %convert_element_type_603 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_357, torch.bfloat16), kwargs = {})
#   %add_182 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_172, %convert_element_type_603), kwargs = {})
#   %select_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 1), kwargs = {})
#   %mul_360 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_182, %select_2), kwargs = {})
#   %view_271 : Tensor "bf16[262144, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_360, [262144, 512]), kwargs = {})
#   return %sum_49,%add_182,%view_271
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_28 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 2684354560}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr5 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = r0_1
        tmp5 = tl.full([1, 1], 32, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr2 + (r0_1 + 32*x0), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = 0.0
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp3 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp45 = tl.load(in_ptr6 + (1))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr0 + (r0_1 + 512*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = r0_1
        tmp23 = tl.full([1, 1], 32, tl.int64)
        tmp24 = tmp22 < tmp23
        tmp25 = tl.load(in_ptr2 + (r0_1 + 32*x0), r0_mask & tmp24, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = 0.0
        tmp27 = tl.where(tmp24, tmp25, tmp26)
        tmp29 = tmp27 + tmp28
        tmp31 = tmp29 + tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp36 * tmp2
        tmp38 = 0.001953125
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39 * tmp19
        tmp41 = tmp34 - tmp40
        tmp42 = tmp41 * tmp2
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp21 + tmp43
        tmp47 = tmp46.to(tl.float32)
        tmp48 = tmp44 * tmp47
        tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp44, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 512*x0), tmp48, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/g6/cg6g557vbpwkqfqzzu22y2swda5ureuyrtnnplbs3f44ihvxwgeo.py
# Topologically Sorted Source Nodes: [getitem_8, rms_norm_4], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_8 => select_2
#   rms_norm_4 => convert_element_type_20, mul_14
# Graph fragment:
#   %add_9 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_9]
#   %rsqrt_4 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_4]
#   %mm_150 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_150]
#   %add_182 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_182]
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_52 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_52]
#   %select_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 1), kwargs = {})
#   %mul_360 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_182, %select_2), kwargs = {})
#   %view_274 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_150, [128, 2048, 512]), kwargs = {})
#   %convert_element_type_618 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_274, torch.float32), kwargs = {})
#   %convert_element_type_20 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.float32), kwargs = {})
#   %mul_14 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, %rsqrt_4), kwargs = {})
#   %mul_365 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %convert_element_type_618), kwargs = {})
#   %sum_52 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_365, [2], True), kwargs = {})
#   %div_33 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_14, 512), kwargs = {})
#   %mul_366 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_33, %sum_52), kwargs = {})
#   %sub_37 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_618, %mul_366), kwargs = {})
#   %mul_367 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %rsqrt_4), kwargs = {})
#   %convert_element_type_620 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_367, torch.bfloat16), kwargs = {})
#   %add_186 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_360, %convert_element_type_620), kwargs = {})
#   return %sum_52,%add_186
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1342177280}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (1))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 * tmp13
    tmp15 = 0.001953125
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 * tmp9
    tmp18 = tmp5 - tmp17
    tmp19 = tmp18 * tmp2
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp14 + tmp20
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp21, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/sz/cszrfyw7rbnzvkdgabgyiox4u7c57qbt26oqhpzrp36vz4vwz5fr.py
# Topologically Sorted Source Nodes: [loss, x_1, getitem_16, linear_9, sigmoid, gate, unsqueeze, getitem_9, getitem_2, mul, getitem_3, mul_1, x_2, rms_norm_1], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._to_copy, aten.mul, aten.select, aten.add, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.view, aten.sum, aten._fused_rms_norm_backward]
# Source node to ATen node mapping:
#   gate => mul_18
#   getitem_16 => select_5
#   getitem_2 => select
#   getitem_3 => select_1
#   getitem_9 => select_3
#   linear_9 => view_27
#   loss => full_default_1
#   mul => mul_1
#   mul_1 => mul_2
#   rms_norm_1 => convert_element_type_2, mul_3
#   sigmoid => sigmoid
#   unsqueeze => unsqueeze
#   x_1 => convert_element_type, convert_element_type_1, mul
#   x_2 => add_1
# Graph fragment:
#   %primals_5 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_5]
#   %embedding : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %primals_6 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_6]
#   %rsqrt_1 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %mul_3 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=mul_3]
#   %mm_154 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_154]
#   %mm_156 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_156]
#   %mm_158 : Tensor "bf16[262144, 512][512, 1]cuda:0" = PlaceHolder[target=mm_158]
#   %add_186 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_186]
#   %sum_55 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_55]
#   %add_156 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_156]
#   %add_168 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_168]
#   %add_182 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_182]
#   %add_195 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_195]
#   %convert_element_type_650 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=convert_element_type_650]
#   %add_11 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0" = PlaceHolder[target=add_11]
#   %primals_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %getitem_58 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0" = PlaceHolder[target=getitem_58]
#   %mm_9 : Tensor "bf16[262144, 4][4, 1]cuda:0" = PlaceHolder[target=mm_9]
#   %index_put_3 : Tensor "f32[8192, 512][512, 1]cuda:0" = PlaceHolder[target=index_put_3]
#   %sum_58 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_58]
#   %index_put_4 : Tensor "f32[8192, 512][512, 1]cuda:0" = PlaceHolder[target=index_put_4]
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq : Tensor "b8[128, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, -1), kwargs = {})
#   %unsqueeze_6 : Tensor "b8[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %full_default_12 : Tensor "f32[8192, 512][512, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([8192, 512], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=10] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %select_5 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 2), kwargs = {})
#   %mul_323 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_168, %select_5), kwargs = {})
#   %add_169 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_156, %mul_323), kwargs = {})
#   %view_27 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [128, 2048, 4]), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_27,), kwargs = {})
#   %mul_18 : Tensor "bf16[128, 2048, 4][8192, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, 2), kwargs = {})
#   %unsqueeze : Tensor "bf16[128, 2048, 4, 1][8192, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %mul_349 : Tensor "bf16[128, 2048, 4, 128][1048576, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_58, %unsqueeze), kwargs = {})
#   %view_261 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_349, [128, 2048, 512]), kwargs = {})
#   %convert_element_type_604 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_261, torch.float32), kwargs = {})
#   %where_14 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_6, %full_default_1, %convert_element_type_604), kwargs = {})
#   %index_put_3 : Tensor "f32[8192, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_12, [%primals_1], %where_14, True), kwargs = {})
#   %select_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 1), kwargs = {})
#   %mul_358 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_182, %select_3), kwargs = {})
#   %mul_359 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_182, %convert_element_type_1), kwargs = {})
#   %sum_50 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_359,), kwargs = {dtype: torch.float32})
#   %add_183 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_169, %mul_358), kwargs = {})
#   %mul_361 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_182, %add_11), kwargs = {})
#   %sum_51 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_361,), kwargs = {dtype: torch.float32})
#   %view_280 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_154, [128, 2048, 512]), kwargs = {})
#   %view_283 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_156, [128, 2048, 512]), kwargs = {})
#   %add_193 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_280, %view_283), kwargs = {})
#   %view_286 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_158, [128, 2048, 512]), kwargs = {})
#   %add_194 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_193, %view_286), kwargs = {})
#   %convert_element_type_647 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_194, torch.float32), kwargs = {})
#   %select : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 0), kwargs = {})
#   %mul_1 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %convert_element_type_1), kwargs = {})
#   %select_1 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 0), kwargs = {})
#   %mul_2 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %convert_element_type_1), kwargs = {})
#   %add_1 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %mul_3 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %rsqrt_1), kwargs = {})
#   %mul_385 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %convert_element_type_647), kwargs = {})
#   %sum_55 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_385, [2], True), kwargs = {})
#   %div_36 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_3, 512), kwargs = {})
#   %mul_386 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_36, %sum_55), kwargs = {})
#   %sub_40 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_647, %mul_386), kwargs = {})
#   %mul_387 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %rsqrt_1), kwargs = {})
#   %convert_element_type_649 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_387, torch.bfloat16), kwargs = {})
#   %add_195 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_186, %convert_element_type_649), kwargs = {})
#   %mul_388 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_195, %select_1), kwargs = {})
#   %mul_389 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_195, %convert_element_type_1), kwargs = {})
#   %sum_56 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_389,), kwargs = {dtype: torch.float32})
#   %add_196 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_183, %mul_388), kwargs = {})
#   %mul_390 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_195, %select), kwargs = {})
#   %add_198 : Tensor "bf16[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_196, %mul_390), kwargs = {})
#   %convert_element_type_650 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_198, torch.float32), kwargs = {})
#   %mul_393 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %convert_element_type_650), kwargs = {})
#   %sum_58 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_393, [2], True), kwargs = {})
#   %div_37 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 512), kwargs = {})
#   %mul_394 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_37, %sum_58), kwargs = {})
#   %sub_41 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_650, %mul_394), kwargs = {})
#   %mul_395 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %rsqrt), kwargs = {})
#   %where_16 : Tensor "f32[128, 2048, 512][1048576, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_6, %full_default_1, %mul_395), kwargs = {})
#   %index_put_4 : Tensor "f32[8192, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_12, [%primals_1], %where_16, True), kwargs = {})
#   return %mul_3,%sum_55,%add_195,%convert_element_type_650,%buf335,%buf377,%sum_58,%buf337,%buf333,%buf384
triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_select_sigmoid_sum_unsqueeze_view_30 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_select_sigmoid_sum_unsqueeze_view_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*bf16', 'in_ptr12': '*i64', 'in_ptr13': '*bf16', 'in_ptr14': '*bf16', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_select_sigmoid_sum_unsqueeze_view_30', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr7', 'out_ptr8'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 18, 'num_reduction': 5, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_select_sigmoid_sum_unsqueeze_view_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp18 = tl.load(in_ptr5 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp27 = tl.load(in_ptr7 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp35 = tl.load(in_ptr8 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp36 = tl.load(in_ptr9 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp37 = tl.load(in_ptr3 + (2))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, R0_BLOCK])
    tmp42 = tl.load(in_ptr10 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp43 = tl.load(in_ptr3 + (1))
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK, R0_BLOCK])
    tmp67 = tl.load(in_ptr11 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp73 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr13 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp82 = tl.load(in_ptr14 + (4*x0 + (r0_1 // 128)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp2 * tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp7
    tmp13 = tmp8 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp16 = tmp14 * tmp15
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp16 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp28 = 0.001953125
    tmp29 = tmp16 * tmp28
    tmp30 = tmp29 * tmp26
    tmp31 = tmp22 - tmp30
    tmp32 = tmp31 * tmp15
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp27 + tmp33
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp36 * tmp39
    tmp41 = tmp35 + tmp40
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp42 * tmp45
    tmp47 = tmp41 + tmp46
    tmp48 = tmp34 * tmp11
    tmp49 = tmp47 + tmp48
    tmp50 = tmp34 * tmp2
    tmp51 = tmp49 + tmp50
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp42 * tmp7
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, R0_BLOCK])
    tmp57 = tl.sum(tmp55, 1)[:, None].to(tl.float32)
    tmp58 = tmp34 * tmp7
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK])
    tmp62 = tl.sum(tmp60, 1)[:, None].to(tl.float32)
    tmp63 = tmp6 * tmp52
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, R0_BLOCK])
    tmp66 = tl.sum(tmp64, 1)[:, None].to(tl.float32)
    tmp68 = tmp42 * tmp67
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK, R0_BLOCK])
    tmp72 = tl.sum(tmp70, 1)[:, None].to(tl.float32)
    tmp74 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp75 = tmp73 + tmp74
    tmp76 = tmp73 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp73)
    tl.device_assert((0 <= tmp77) & (tmp77 < 8192), "index out of bounds: 0 <= tmp77 < 8192")
    tmp79 = tl.full([1, 1], -1, tl.int64)
    tmp80 = tmp73 == tmp79
    tmp83 = tl.sigmoid(tmp82)
    tmp84 = 2.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp81 * tmp85
    tmp87 = tmp86.to(tl.float32)
    tmp88 = 0.0
    tmp89 = tl.where(tmp80, tmp88, tmp87)
    tmp90 = tmp6 * tmp28
    tmp91 = tmp90 * tmp66
    tmp92 = tmp52 - tmp91
    tmp93 = tmp92 * tmp5
    tmp94 = tl.where(tmp80, tmp88, tmp93)
    tl.atomic_add(out_ptr7 + (tl.broadcast_to(r0_1 + 512*tmp77, [XBLOCK, R0_BLOCK])), tmp89, None, sem='relaxed')
    tl.atomic_add(out_ptr8 + (tl.broadcast_to(r0_1 + 512*tmp77, [XBLOCK, R0_BLOCK])), tmp94, None, sem='relaxed')
    tl.store(out_ptr3 + (x0), tmp57, None)
    tl.store(out_ptr4 + (x0), tmp62, None)
    tl.store(out_ptr6 + (x0), tmp72, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/z4/cz4bixmr57qmfjfsa2irsyjpkfsn4mkapegtjvxililss5r6mnnf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_11 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_11]
#   %sum_17 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_17]
#   %sum_24 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_24]
#   %sum_30 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_30]
#   %sum_37 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_37]
#   %sum_43 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_43]
#   %sum_50 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_50]
#   %sum_56 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_56]
#   %full_default_13 : Tensor "f32[8][1]cuda:0"[num_users=15] = call_function[target=torch.ops.aten.full.default](args = ([8], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_11, 0, 7), kwargs = {})
#   %select_scatter_default_2 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_17, 0, 6), kwargs = {})
#   %add_116 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_2), kwargs = {})
#   %select_scatter_default_4 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_24, 0, 5), kwargs = {})
#   %add_130 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_116, %select_scatter_default_4), kwargs = {})
#   %select_scatter_default_6 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_30, 0, 4), kwargs = {})
#   %add_143 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_130, %select_scatter_default_6), kwargs = {})
#   %select_scatter_default_8 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_37, 0, 3), kwargs = {})
#   %add_157 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_143, %select_scatter_default_8), kwargs = {})
#   %select_scatter_default_10 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_43, 0, 2), kwargs = {})
#   %add_170 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_157, %select_scatter_default_10), kwargs = {})
#   %select_scatter_default_12 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_50, 0, 1), kwargs = {})
#   %add_184 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_170, %select_scatter_default_12), kwargs = {})
#   %select_scatter_default_14 : Tensor "f32[8][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_56, 0, 0), kwargs = {})
#   %add_197 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_184, %select_scatter_default_14), kwargs = {})
#   return %add_197
triton_poi_fused_add_select_backward_31 = async_compile.triton('triton_poi_fused_add_select_backward_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_select_backward_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_select_backward_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp3 = tl.load(in_ptr0 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp9 = tl.load(in_ptr1 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp15 = tl.load(in_ptr2 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp27 = tl.load(in_ptr4 + (0))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp33 = tl.load(in_ptr5 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp39 = tl.load(in_ptr6 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp45 = tl.load(in_ptr7 + (0))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 7, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = 0.0
    tmp6 = tl.where(tmp2, tmp4, tmp5)
    tmp7 = tl.full([1], 6, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp11 = tl.where(tmp8, tmp10, tmp5)
    tmp12 = tmp6 + tmp11
    tmp13 = tl.full([1], 5, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp17 = tl.where(tmp14, tmp16, tmp5)
    tmp18 = tmp12 + tmp17
    tmp19 = tl.full([1], 4, tl.int32)
    tmp20 = tmp0 == tmp19
    tmp23 = tl.where(tmp20, tmp22, tmp5)
    tmp24 = tmp18 + tmp23
    tmp25 = tl.full([1], 3, tl.int32)
    tmp26 = tmp0 == tmp25
    tmp29 = tl.where(tmp26, tmp28, tmp5)
    tmp30 = tmp24 + tmp29
    tmp31 = tl.full([1], 2, tl.int32)
    tmp32 = tmp0 == tmp31
    tmp35 = tl.where(tmp32, tmp34, tmp5)
    tmp36 = tmp30 + tmp35
    tmp37 = tl.full([1], 1, tl.int32)
    tmp38 = tmp0 == tmp37
    tmp41 = tl.where(tmp38, tmp40, tmp5)
    tmp42 = tmp36 + tmp41
    tmp43 = tl.full([1], 0, tl.int32)
    tmp44 = tmp0 == tmp43
    tmp47 = tl.where(tmp44, tmp46, tmp5)
    tmp48 = tmp42 + tmp47
    tl.store(out_ptr0 + (x0), tmp48, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_5, primals_6, primals_64, embedding, rsqrt, rsqrt_1, view, view_8, cat, cat_1, rsqrt_2, convert_element_type_14, rsqrt_3, convert_element_type_16, getitem, getitem_1, add_9, rsqrt_4, view_12, mm_4, view_14, add_11, add_12, embedding_1, rsqrt_5, view_16, view_26, mm_9, add_14, cat_2, cat_3, rsqrt_6, convert_element_type_45, rsqrt_7, convert_element_type_47, getitem_4, getitem_5, add_21, rsqrt_8, view_31, mm_11, view_33, add_23, add_24, rsqrt_9, view_35, view_43, cat_4, cat_5, rsqrt_10, convert_element_type_73, rsqrt_11, convert_element_type_75, getitem_8, getitem_9, add_32, rsqrt_12, view_47, mm_17, view_49, add_34, add_35, embedding_2, rsqrt_13, view_51, view_61, mm_22, add_37, cat_6, cat_7, rsqrt_14, convert_element_type_104, rsqrt_15, convert_element_type_106, getitem_12, getitem_13, add_44, rsqrt_16, view_66, mm_24, view_68, add_46, add_47, rsqrt_17, view_70, view_78, cat_8, cat_9, rsqrt_18, convert_element_type_132, rsqrt_19, convert_element_type_134, getitem_16, getitem_17, add_55, rsqrt_20, view_82, mm_30, view_84, add_57, add_58, embedding_3, rsqrt_21, view_86, view_96, mm_35, add_60, cat_10, cat_11, rsqrt_22, convert_element_type_163, rsqrt_23, convert_element_type_165, getitem_20, getitem_21, add_67, rsqrt_24, view_101, mm_37, view_103, add_69, add_70, rsqrt_25, view_105, view_113, cat_12, cat_13, rsqrt_26, convert_element_type_191, rsqrt_27, convert_element_type_193, getitem_24, getitem_25, add_78, rsqrt_28, view_117, mm_43, view_119, add_80, add_81, embedding_4, rsqrt_29, view_121, view_131, mm_48, add_83, cat_14, cat_15, rsqrt_30, convert_element_type_222, rsqrt_31, convert_element_type_224, getitem_28, getitem_29, add_90, rsqrt_32, view_136, mm_50, view_138, add_92, rsqrt_33, view_140, mm_52, amax, log, convert_element_type_244, permute_55, permute_59, permute_63, permute_67, permute_71, permute_75, permute_79, permute_83, permute_87, permute_91, permute_95, permute_99, permute_103, permute_107, permute_111, permute_115, permute_119, permute_123, permute_127, permute_131, permute_135, permute_139, permute_143, permute_147, permute_151, permute_155, permute_159, permute_163, permute_167, permute_171, permute_175, permute_179, permute_183, permute_187, permute_191, permute_195, permute_199, permute_203, permute_207, permute_211, permute_215, permute_219, permute_223, permute_227, permute_231, permute_235, permute_239, permute_243, permute_247, permute_251, permute_255, permute_259, permute_263, tangents_1 = args
        args.clear()
        assert_size_stride(primals_1, (128, 2048), (2048, 1))
        assert_size_stride(primals_2, (1, 20480, 1, 64), (1310720, 64, 64, 1))
        assert_size_stride(primals_3, (1, 20480, 1, 64), (1310720, 64, 64, 1))
        assert_size_stride(primals_5, (8, ), (1, ))
        assert_size_stride(primals_6, (8, ), (1, ))
        assert_size_stride(primals_64, (128, 2048), (2048, 1))
        assert_size_stride(embedding, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(rsqrt_1, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view, (262144, 512), (512, 1))
        assert_size_stride(view_8, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_1, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_2, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_14, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_3, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_16, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_1, (128, 4, 2048), (8192, 2048, 1))
        assert_size_stride(add_9, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_4, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_12, (262144, 512), (512, 1))
        assert_size_stride(mm_4, (262144, 2048), (2048, 1))
        assert_size_stride(view_14, (262144, 2048), (2048, 1))
        assert_size_stride(add_11, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(add_12, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(embedding_1, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_5, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_16, (262144, 512), (512, 1))
        assert_size_stride(view_26, (262144, 32), (512, 1))
        assert_size_stride(mm_9, (262144, 4), (4, 1))
        assert_size_stride(add_14, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_2, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_3, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_6, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_45, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_7, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_47, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_4, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_5, (128, 4, 2048), (8192, 2048, 1))
        assert_size_stride(add_21, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_8, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_31, (262144, 512), (512, 1))
        assert_size_stride(mm_11, (262144, 2048), (2048, 1))
        assert_size_stride(view_33, (262144, 2048), (2048, 1))
        assert_size_stride(add_23, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(add_24, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_9, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_35, (262144, 512), (512, 1))
        assert_size_stride(view_43, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_4, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_5, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_10, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_73, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_11, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_75, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_8, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_9, (128, 4, 2048), (8192, 2048, 1))
        assert_size_stride(add_32, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_12, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_47, (262144, 512), (512, 1))
        assert_size_stride(mm_17, (262144, 2048), (2048, 1))
        assert_size_stride(view_49, (262144, 2048), (2048, 1))
        assert_size_stride(add_34, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(add_35, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(embedding_2, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_13, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_51, (262144, 512), (512, 1))
        assert_size_stride(view_61, (262144, 32), (512, 1))
        assert_size_stride(mm_22, (262144, 4), (4, 1))
        assert_size_stride(add_37, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_6, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_7, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_14, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_104, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_15, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_106, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_12, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_13, (128, 4, 2048), (8192, 2048, 1))
        assert_size_stride(add_44, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_16, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_66, (262144, 512), (512, 1))
        assert_size_stride(mm_24, (262144, 2048), (2048, 1))
        assert_size_stride(view_68, (262144, 2048), (2048, 1))
        assert_size_stride(add_46, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(add_47, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_17, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_70, (262144, 512), (512, 1))
        assert_size_stride(view_78, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_8, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_9, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_18, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_132, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_19, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_134, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_16, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_17, (128, 4, 2048), (8192, 2048, 1))
        assert_size_stride(add_55, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_20, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_82, (262144, 512), (512, 1))
        assert_size_stride(mm_30, (262144, 2048), (2048, 1))
        assert_size_stride(view_84, (262144, 2048), (2048, 1))
        assert_size_stride(add_57, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(add_58, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(embedding_3, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_21, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_86, (262144, 512), (512, 1))
        assert_size_stride(view_96, (262144, 32), (512, 1))
        assert_size_stride(mm_35, (262144, 4), (4, 1))
        assert_size_stride(add_60, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_10, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_11, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_22, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_163, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_23, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_165, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_20, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_21, (128, 4, 2048), (8192, 2048, 1))
        assert_size_stride(add_67, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_24, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_101, (262144, 512), (512, 1))
        assert_size_stride(mm_37, (262144, 2048), (2048, 1))
        assert_size_stride(view_103, (262144, 2048), (2048, 1))
        assert_size_stride(add_69, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(add_70, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_25, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_105, (262144, 512), (512, 1))
        assert_size_stride(view_113, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_12, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_13, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_26, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_191, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_27, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_193, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_24, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_25, (128, 4, 2048), (8192, 2048, 1))
        assert_size_stride(add_78, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_28, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_117, (262144, 512), (512, 1))
        assert_size_stride(mm_43, (262144, 2048), (2048, 1))
        assert_size_stride(view_119, (262144, 2048), (2048, 1))
        assert_size_stride(add_80, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(add_81, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(embedding_4, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_29, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_121, (262144, 512), (512, 1))
        assert_size_stride(view_131, (262144, 32), (512, 1))
        assert_size_stride(mm_48, (262144, 4), (4, 1))
        assert_size_stride(add_83, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_14, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(cat_15, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_30, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_222, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(rsqrt_31, (128, 2048, 4, 1), (8192, 4, 1, 1))
        assert_size_stride(convert_element_type_224, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_28, (128, 2048, 4, 128), (1048576, 512, 128, 1))
        assert_size_stride(getitem_29, (128, 4, 2048), (8192, 2048, 1))
        assert_size_stride(add_90, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_32, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_136, (262144, 512), (512, 1))
        assert_size_stride(mm_50, (262144, 2048), (2048, 1))
        assert_size_stride(view_138, (262144, 2048), (2048, 1))
        assert_size_stride(add_92, (128, 2048, 512), (1048576, 512, 1))
        assert_size_stride(rsqrt_33, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_140, (262144, 512), (512, 1))
        assert_size_stride(mm_52, (262144, 8192), (8192, 1))
        assert_size_stride(amax, (262144, 1), (1, 1))
        assert_size_stride(log, (262144, 1), (1, 1))
        assert_size_stride(convert_element_type_244, (), ())
        assert_size_stride(permute_55, (8192, 512), (512, 1))
        assert_size_stride(permute_59, (512, 2048), (2048, 1))
        assert_size_stride(permute_63, (2048, 512), (512, 1))
        assert_size_stride(permute_67, (512, 512), (512, 1))
        assert_size_stride(permute_71, (4, 32), (32, 1))
        assert_size_stride(permute_75, (512, 512), (512, 1))
        assert_size_stride(permute_79, (512, 512), (512, 1))
        assert_size_stride(permute_83, (512, 512), (512, 1))
        assert_size_stride(permute_87, (512, 2048), (2048, 1))
        assert_size_stride(permute_91, (2048, 512), (512, 1))
        assert_size_stride(permute_95, (512, 512), (512, 1))
        assert_size_stride(permute_99, (512, 512), (512, 1))
        assert_size_stride(permute_103, (512, 512), (512, 1))
        assert_size_stride(permute_107, (512, 512), (512, 1))
        assert_size_stride(permute_111, (512, 2048), (2048, 1))
        assert_size_stride(permute_115, (2048, 512), (512, 1))
        assert_size_stride(permute_119, (512, 512), (512, 1))
        assert_size_stride(permute_123, (4, 32), (32, 1))
        assert_size_stride(permute_127, (512, 512), (512, 1))
        assert_size_stride(permute_131, (512, 512), (512, 1))
        assert_size_stride(permute_135, (512, 512), (512, 1))
        assert_size_stride(permute_139, (512, 2048), (2048, 1))
        assert_size_stride(permute_143, (2048, 512), (512, 1))
        assert_size_stride(permute_147, (512, 512), (512, 1))
        assert_size_stride(permute_151, (512, 512), (512, 1))
        assert_size_stride(permute_155, (512, 512), (512, 1))
        assert_size_stride(permute_159, (512, 512), (512, 1))
        assert_size_stride(permute_163, (512, 2048), (2048, 1))
        assert_size_stride(permute_167, (2048, 512), (512, 1))
        assert_size_stride(permute_171, (512, 512), (512, 1))
        assert_size_stride(permute_175, (4, 32), (32, 1))
        assert_size_stride(permute_179, (512, 512), (512, 1))
        assert_size_stride(permute_183, (512, 512), (512, 1))
        assert_size_stride(permute_187, (512, 512), (512, 1))
        assert_size_stride(permute_191, (512, 2048), (2048, 1))
        assert_size_stride(permute_195, (2048, 512), (512, 1))
        assert_size_stride(permute_199, (512, 512), (512, 1))
        assert_size_stride(permute_203, (512, 512), (512, 1))
        assert_size_stride(permute_207, (512, 512), (512, 1))
        assert_size_stride(permute_211, (512, 512), (512, 1))
        assert_size_stride(permute_215, (512, 2048), (2048, 1))
        assert_size_stride(permute_219, (2048, 512), (512, 1))
        assert_size_stride(permute_223, (512, 512), (512, 1))
        assert_size_stride(permute_227, (4, 32), (32, 1))
        assert_size_stride(permute_231, (512, 512), (512, 1))
        assert_size_stride(permute_235, (512, 512), (512, 1))
        assert_size_stride(permute_239, (512, 512), (512, 1))
        assert_size_stride(permute_243, (512, 2048), (2048, 1))
        assert_size_stride(permute_247, (2048, 512), (512, 1))
        assert_size_stride(permute_251, (512, 512), (512, 1))
        assert_size_stride(permute_255, (512, 512), (512, 1))
        assert_size_stride(permute_259, (512, 512), (512, 1))
        assert_size_stride(permute_263, (512, 512), (512, 1))
        assert_size_stride(tangents_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf2 = reinterpret_tensor(mm_52, (128, 2048, 8192), (16777216, 8192, 1), 0); del mm_52  # reuse
            # Topologically Sorted Source Nodes: [view_37, loss, logits, logits_1, truediv, tanh, logits_2, view_36], Original ATen: [aten.nll_loss_backward, aten.view, aten.nll_loss_forward, aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten._log_softmax, aten._log_softmax_backward_data, aten.tanh_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_view_0.run(buf2, primals_64, tangents_1, convert_element_type_244, amax, log, 262144, 8192, stream=stream0)
            del amax
            del convert_element_type_244
            del log
            del primals_64
            del tangents_1
            buf3 = empty_strided_cuda((8192, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [logits, logits_1, truediv, tanh], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.view, aten.mul, aten.tanh_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf2, (8192, 262144), (1, 8192), 0), view_140, out=buf3)
            del view_140
            buf4 = empty_strided_cuda((262144, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [logits, logits_1, truediv, tanh], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.view, aten.mul, aten.tanh_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf2, (262144, 8192), (8192, 1), 0), permute_55, out=buf4)
            del buf2
            del permute_55
            buf5 = empty_strided_cuda((8192, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(buf3, buf5, 4194304, stream=stream0)
            buf7 = reinterpret_tensor(buf4, (128, 2048, 512), (1048576, 512, 1), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_view_2.run(buf7, add_92, rsqrt_33, 262144, 512, stream=stream0)
            del add_92
            del rsqrt_33
            buf8 = empty_strided_cuda((512, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf7, (512, 262144), (1, 512), 0), view_138, out=buf8)
            del view_138
            buf9 = empty_strided_cuda((262144, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf7, (262144, 512), (512, 1), 0), permute_59, out=buf9)
            del permute_59
            buf10 = empty_strided_cuda((512, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf8, buf10, 1048576, stream=stream0)
            buf11 = reinterpret_tensor(mm_50, (128, 2048, 2048), (4194304, 2048, 1), 0); del mm_50  # reuse
            # Topologically Sorted Source Nodes: [x_46, relu_7, x_47], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.threshold_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf11, buf9, 536870912, stream=stream0)
            del buf9
            buf12 = reinterpret_tensor(buf8, (2048, 512), (512, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [x_46, relu_7, x_47], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.threshold_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf11, (2048, 262144), (1, 2048), 0), view_136, out=buf12)
            del view_136
            buf13 = empty_strided_cuda((262144, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_46, relu_7, x_47], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.threshold_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf11, (262144, 2048), (2048, 1), 0), permute_63, out=buf13)
            del permute_63
            buf14 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf12, buf14, 1048576, stream=stream0)
            buf16 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_32], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_5.run(buf16, add_90, rsqrt_32, buf13, 262144, 512, stream=stream0)
            del add_90
            del rsqrt_32
            buf17 = empty_strided_cuda((512, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_22, y_23], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf16, (512, 262144), (1, 512), 0), reinterpret_tensor(getitem_28, (262144, 512), (512, 1), 0), out=buf17)
            buf18 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf16, (262144, 512), (512, 1), 0), permute_67, out=buf18)
            del permute_67
            buf19 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf17, buf19, 262144, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf20 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf18, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0), convert_element_type_222, convert_element_type_224, add_83, getitem_28, getitem_29, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_83
            del convert_element_type_222
            del convert_element_type_224
            del getitem_28
            del getitem_29
            buf21 = buf20[0]
            assert_size_stride(buf21, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf21, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf22 = buf20[1]
            assert_size_stride(buf22, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf22, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf23 = buf20[2]
            assert_size_stride(buf23, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf23, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf20
            buf27 = reinterpret_tensor(buf18, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0); del buf18  # reuse
            buf42 = buf27; del buf27  # reuse
            buf29 = empty_strided_cuda((128, 2048, 4, 128), (1048576, 512, 128, 1), torch.bfloat16)
            buf46 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [k_23, q_23, cos, sin, neg], Original ATen: [aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.slice, aten.neg, aten.add, aten.slice_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf42, buf46, cat_14, rsqrt_30, buf21, cat_15, rsqrt_31, buf22, primals_2, primals_3, 1048576, 128, stream=stream0)
            del cat_14
            del cat_15
            del rsqrt_30
            del rsqrt_31
            buf52 = empty_strided_cuda((8192, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [loss, linear_48, sigmoid_3, gate_3, unsqueeze_3], Original ATen: [aten.nll_loss_forward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8.run(buf52, 4194304, stream=stream0)
            buf32 = reinterpret_tensor(buf12, (128, 2048, 4), (8192, 4, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_48, sigmoid_3, gate_3, unsqueeze_3, ve_7], Original ATen: [aten.nll_loss_forward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9.run(buf23, embedding_4, primals_1, mm_48, buf52, buf32, 1048576, 128, stream=stream0)
            del embedding_4
            del mm_48
            buf33 = empty_strided_cuda((8, 262144), (1, 8), torch.bfloat16)
            buf35 = empty_strided_cuda((262144, 8), (8, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_48, sigmoid_3], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10.run(buf32, buf33, buf35, 2097152, stream=stream0)
            buf34 = empty_strided_cuda((8, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_48, sigmoid_3], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf33, view_131, out=buf34)
            del view_131
            buf36 = empty_strided_cuda((8, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mm_11.run(permute_71, buf36, 256, stream=stream0)
            del permute_71
            buf37 = empty_strided_cuda((262144, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_48, sigmoid_3], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(buf35, buf36, out=buf37)
            buf38 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mm_12.run(buf34, buf38, 128, stream=stream0)
            buf39 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf23, (512, 262144), (1, 512), 0), view_121, out=buf39)
            buf40 = reinterpret_tensor(buf22, (262144, 512), (512, 1), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf23, (262144, 512), (512, 1), 0), permute_75, out=buf40)
            del permute_75
            buf41 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf39, buf41, 262144, stream=stream0)
            buf43 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf42, (512, 262144), (1, 512), 0), view_121, out=buf43)
            buf44 = reinterpret_tensor(buf23, (262144, 512), (512, 1), 0); del buf23  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf42, (262144, 512), (512, 1), 0), permute_79, out=buf44)
            del permute_79
            buf45 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf43, buf45, 262144, stream=stream0)
            buf47 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf46, (512, 262144), (1, 512), 0), view_121, out=buf47)
            del view_121
            buf48 = reinterpret_tensor(buf42, (262144, 512), (512, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf46, (262144, 512), (512, 1), 0), permute_83, out=buf48)
            del permute_83
            buf49 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf47, buf49, 262144, stream=stream0)
            buf51 = buf16; del buf16  # reuse
            buf59 = reinterpret_tensor(buf46, (262144, 512), (512, 1), 0); del buf46  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_29, getitem_47], Original ATen: [aten.view, aten.slice_backward, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.select]
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_13.run(buf51, add_81, rsqrt_29, buf37, buf40, buf44, buf48, primals_5, buf59, 262144, 512, stream=stream0)
            del add_81
            del rsqrt_29
            buf54 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_14.run(buf52, buf54, 4194304, stream=stream0)
            buf61 = reinterpret_tensor(buf11, (262144, 2048), (2048, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf59, permute_87, out=buf61)
            del permute_87
            buf63 = reinterpret_tensor(mm_43, (128, 2048, 2048), (4194304, 2048, 1), 0); del mm_43  # reuse
            # Topologically Sorted Source Nodes: [x_40, relu_6, x_41], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf63, buf61, 536870912, stream=stream0)
            del buf61
            buf65 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [x_40, relu_6, x_41], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf63, (262144, 2048), (2048, 1), 0), permute_91, out=buf65)
            del permute_91
            buf68 = reinterpret_tensor(buf65, (128, 2048, 512), (1048576, 512, 1), 0); del buf65  # reuse
            # Topologically Sorted Source Nodes: [getitem_47, rms_norm_28], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_15.run(buf68, add_78, rsqrt_28, buf51, primals_5, 262144, 512, stream=stream0)
            del add_78
            del rsqrt_28
            buf70 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf68, (262144, 512), (512, 1), 0), permute_95, out=buf70)
            del permute_95
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf72 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf70, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0), convert_element_type_191, convert_element_type_193, view_113, getitem_24, getitem_25, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del convert_element_type_191
            del convert_element_type_193
            del getitem_25
            del view_113
            buf75 = buf72[2]
            assert_size_stride(buf75, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf75, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf84 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf75, (262144, 512), (512, 1), 0), permute_99, out=buf84)
            del permute_99
            buf73 = buf72[0]
            assert_size_stride(buf73, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf73, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf74 = buf72[1]
            assert_size_stride(buf74, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf74, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf72
            buf79 = reinterpret_tensor(buf40, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0); del buf40  # reuse
            buf86 = buf79; del buf79  # reuse
            buf81 = buf21; del buf21  # reuse
            buf90 = buf81; del buf81  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_20, q_20], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf86, buf90, cat_12, rsqrt_26, buf73, cat_13, rsqrt_27, buf74, primals_2, primals_3, 1048576, 128, stream=stream0)
            del cat_12
            del cat_13
            del rsqrt_26
            del rsqrt_27
            buf88 = reinterpret_tensor(buf74, (262144, 512), (512, 1), 0); del buf74  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf86, (262144, 512), (512, 1), 0), permute_103, out=buf88)
            del permute_103
            buf92 = reinterpret_tensor(buf73, (262144, 512), (512, 1), 0); del buf73  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf90, (262144, 512), (512, 1), 0), permute_107, out=buf92)
            del permute_107
            buf95 = reinterpret_tensor(buf84, (128, 2048, 512), (1048576, 512, 1), 0); del buf84  # reuse
            buf100 = empty_strided_cuda((262144, 512), (512, 1), torch.bfloat16)
            buf55 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf96 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf57 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf98 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, rms_norm_25, getitem_41], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_16.run(buf95, add_70, rsqrt_25, buf88, buf92, buf68, primals_5, buf51, embedding, rsqrt, add_80, add_69, buf100, buf55, buf96, buf57, buf98, 262144, 512, stream=stream0)
            del add_69
            del add_70
            del add_80
            del buf88
            del buf92
            del rsqrt_25
            buf56 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf55, buf56, 1, 262144, stream=stream0)
            buf58 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf57, buf58, 1, 262144, stream=stream0)
            buf60 = reinterpret_tensor(buf32, (512, 2048), (2048, 1), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf59, (512, 262144), (1, 512), 0), view_119, out=buf60)
            del buf59
            del view_119
            buf62 = empty_strided_cuda((512, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf60, buf62, 1048576, stream=stream0)
            buf64 = reinterpret_tensor(buf60, (2048, 512), (512, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [x_40, relu_6, x_41], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf63, (2048, 262144), (1, 2048), 0), view_117, out=buf64)
            del view_117
            buf66 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf64, buf66, 1048576, stream=stream0)
            buf69 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [y_19, y_20], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf68, (512, 262144), (1, 512), 0), reinterpret_tensor(getitem_24, (262144, 512), (512, 1), 0), out=buf69)
            del buf68
            del getitem_24
            buf71 = reinterpret_tensor(buf57, (512, 512), (512, 1), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf69, buf71, 262144, stream=stream0)
            buf83 = buf69; del buf69  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf75, (512, 262144), (1, 512), 0), view_105, out=buf83)
            buf85 = reinterpret_tensor(buf55, (512, 512), (512, 1), 0); del buf55  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf83, buf85, 262144, stream=stream0)
            buf87 = buf83; del buf83  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf86, (512, 262144), (1, 512), 0), view_105, out=buf87)
            buf89 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf87, buf89, 262144, stream=stream0)
            buf91 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf90, (512, 262144), (1, 512), 0), view_105, out=buf91)
            del view_105
            buf93 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf91, buf93, 262144, stream=stream0)
            buf97 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf96, buf97, 1, 262144, stream=stream0)
            buf99 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf98, buf99, 1, 262144, stream=stream0)
            buf101 = reinterpret_tensor(buf64, (512, 2048), (2048, 1), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf100, (512, 262144), (1, 512), 0), view_103, out=buf101)
            del view_103
            buf102 = reinterpret_tensor(buf63, (262144, 2048), (2048, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf100, permute_111, out=buf102)
            del permute_111
            buf103 = empty_strided_cuda((512, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf101, buf103, 1048576, stream=stream0)
            buf104 = reinterpret_tensor(mm_37, (128, 2048, 2048), (4194304, 2048, 1), 0); del mm_37  # reuse
            # Topologically Sorted Source Nodes: [x_34, relu_5, x_35], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf104, buf102, 536870912, stream=stream0)
            del buf102
            buf105 = reinterpret_tensor(buf101, (2048, 512), (512, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [x_34, relu_5, x_35], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf104, (2048, 262144), (1, 2048), 0), view_101, out=buf105)
            del view_101
            buf106 = buf100; del buf100  # reuse
            # Topologically Sorted Source Nodes: [x_34, relu_5, x_35], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf104, (262144, 2048), (2048, 1), 0), permute_115, out=buf106)
            del permute_115
            buf107 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf105, buf107, 1048576, stream=stream0)
            buf109 = reinterpret_tensor(buf106, (128, 2048, 512), (1048576, 512, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [getitem_41, rms_norm_24], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_18.run(buf109, add_67, rsqrt_24, buf95, primals_5, 262144, 512, stream=stream0)
            del add_67
            del rsqrt_24
            buf110 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [y_16, y_17], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf109, (512, 262144), (1, 512), 0), reinterpret_tensor(getitem_20, (262144, 512), (512, 1), 0), out=buf110)
            buf111 = reinterpret_tensor(buf90, (262144, 512), (512, 1), 0); del buf90  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf109, (262144, 512), (512, 1), 0), permute_119, out=buf111)
            del permute_119
            buf112 = reinterpret_tensor(buf98, (512, 512), (512, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf110, buf112, 262144, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf113 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf111, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0), convert_element_type_163, convert_element_type_165, add_60, getitem_20, getitem_21, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_60
            del convert_element_type_163
            del convert_element_type_165
            del getitem_20
            del getitem_21
            buf114 = buf113[0]
            assert_size_stride(buf114, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf114, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf115 = buf113[1]
            assert_size_stride(buf115, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf115, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf116 = buf113[2]
            assert_size_stride(buf116, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf116, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf113
            buf120 = reinterpret_tensor(buf111, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0); del buf111  # reuse
            buf135 = buf120; del buf120  # reuse
            buf122 = buf86; del buf86  # reuse
            buf139 = buf122; del buf122  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_17, q_17], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf135, buf139, cat_10, rsqrt_22, buf114, cat_11, rsqrt_23, buf115, primals_2, primals_3, 1048576, 128, stream=stream0)
            del cat_10
            del cat_11
            del rsqrt_22
            del rsqrt_23
            buf145 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_35, sigmoid_2, gate_2, unsqueeze_2], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8.run(buf145, 4194304, stream=stream0)
            buf125 = reinterpret_tensor(buf105, (128, 2048, 4), (8192, 4, 1), 0); del buf105  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_35, sigmoid_2, gate_2, unsqueeze_2, ve_5], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9.run(buf116, embedding_3, primals_1, mm_35, buf145, buf125, 1048576, 128, stream=stream0)
            del embedding_3
            del mm_35
            buf126 = reinterpret_tensor(buf35, (8, 262144), (1, 8), 0); del buf35  # reuse
            buf128 = reinterpret_tensor(buf33, (262144, 8), (8, 1), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [linear_35, sigmoid_2], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10.run(buf125, buf126, buf128, 2097152, stream=stream0)
            buf127 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [linear_35, sigmoid_2], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf126, view_96, out=buf127)
            del view_96
            buf129 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mm_11.run(permute_123, buf129, 256, stream=stream0)
            del permute_123
            buf130 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [linear_35, sigmoid_2], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(buf128, buf129, out=buf130)
            buf131 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mm_12.run(buf127, buf131, 128, stream=stream0)
            buf132 = buf110; del buf110  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf116, (512, 262144), (1, 512), 0), view_86, out=buf132)
            buf133 = reinterpret_tensor(buf115, (262144, 512), (512, 1), 0); del buf115  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf116, (262144, 512), (512, 1), 0), permute_127, out=buf133)
            del permute_127
            buf134 = reinterpret_tensor(buf96, (512, 512), (512, 1), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf132, buf134, 262144, stream=stream0)
            buf136 = buf132; del buf132  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf135, (512, 262144), (1, 512), 0), view_86, out=buf136)
            buf137 = reinterpret_tensor(buf116, (262144, 512), (512, 1), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf135, (262144, 512), (512, 1), 0), permute_131, out=buf137)
            del permute_131
            buf138 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf136, buf138, 262144, stream=stream0)
            buf140 = buf136; del buf136  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf139, (512, 262144), (1, 512), 0), view_86, out=buf140)
            del view_86
            buf141 = reinterpret_tensor(buf135, (262144, 512), (512, 1), 0); del buf135  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf139, (262144, 512), (512, 1), 0), permute_135, out=buf141)
            del permute_135
            buf142 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf140, buf142, 262144, stream=stream0)
            buf144 = buf109; del buf109  # reuse
            buf152 = reinterpret_tensor(buf139, (262144, 512), (512, 1), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_21, getitem_34], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.select]
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_19.run(buf144, add_58, rsqrt_21, buf130, buf133, buf137, buf141, primals_5, buf152, 262144, 512, stream=stream0)
            del add_58
            del rsqrt_21
            buf147 = empty_strided_cuda((8192, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_14.run(buf145, buf147, 4194304, stream=stream0)
            buf154 = reinterpret_tensor(buf104, (262144, 2048), (2048, 1), 0); del buf104  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf152, permute_139, out=buf154)
            del permute_139
            buf156 = reinterpret_tensor(mm_30, (128, 2048, 2048), (4194304, 2048, 1), 0); del mm_30  # reuse
            # Topologically Sorted Source Nodes: [x_28, relu_4, x_29], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf156, buf154, 536870912, stream=stream0)
            del buf154
            buf158 = buf141; del buf141  # reuse
            # Topologically Sorted Source Nodes: [x_28, relu_4, x_29], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf156, (262144, 2048), (2048, 1), 0), permute_143, out=buf158)
            del permute_143
            buf161 = reinterpret_tensor(buf158, (128, 2048, 512), (1048576, 512, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [getitem_34, rms_norm_20], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_20.run(buf161, add_55, rsqrt_20, buf144, primals_5, 262144, 512, stream=stream0)
            del add_55
            del rsqrt_20
            buf163 = buf137; del buf137  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf161, (262144, 512), (512, 1), 0), permute_147, out=buf163)
            del permute_147
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf165 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf163, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0), convert_element_type_132, convert_element_type_134, view_78, getitem_16, getitem_17, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del convert_element_type_132
            del convert_element_type_134
            del getitem_17
            del view_78
            buf168 = buf165[2]
            assert_size_stride(buf168, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf168, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf177 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf168, (262144, 512), (512, 1), 0), permute_151, out=buf177)
            del permute_151
            buf166 = buf165[0]
            assert_size_stride(buf166, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf166, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf167 = buf165[1]
            assert_size_stride(buf167, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf167, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf165
            buf172 = reinterpret_tensor(buf133, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0); del buf133  # reuse
            buf179 = buf172; del buf172  # reuse
            buf174 = buf114; del buf114  # reuse
            buf183 = buf174; del buf174  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_14, q_14], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf179, buf183, cat_8, rsqrt_18, buf166, cat_9, rsqrt_19, buf167, primals_2, primals_3, 1048576, 128, stream=stream0)
            del cat_8
            del cat_9
            del rsqrt_18
            del rsqrt_19
            buf181 = reinterpret_tensor(buf167, (262144, 512), (512, 1), 0); del buf167  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf179, (262144, 512), (512, 1), 0), permute_155, out=buf181)
            del permute_155
            buf185 = reinterpret_tensor(buf166, (262144, 512), (512, 1), 0); del buf166  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf183, (262144, 512), (512, 1), 0), permute_159, out=buf185)
            del permute_159
            buf188 = reinterpret_tensor(buf177, (128, 2048, 512), (1048576, 512, 1), 0); del buf177  # reuse
            buf193 = reinterpret_tensor(buf75, (262144, 512), (512, 1), 0); del buf75  # reuse
            buf148 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf189 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf150 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf191 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, rms_norm_17, getitem_28], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_21.run(buf188, add_47, rsqrt_17, buf181, buf185, buf161, primals_5, buf144, embedding, rsqrt, add_57, add_46, buf193, buf148, buf189, buf150, buf191, 262144, 512, stream=stream0)
            del add_46
            del add_47
            del add_57
            del buf181
            del buf185
            del rsqrt_17
            buf149 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf148, buf149, 1, 262144, stream=stream0)
            buf151 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf150, buf151, 1, 262144, stream=stream0)
            buf153 = reinterpret_tensor(buf125, (512, 2048), (2048, 1), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf152, (512, 262144), (1, 512), 0), view_84, out=buf153)
            del buf152
            del view_84
            buf155 = empty_strided_cuda((512, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf153, buf155, 1048576, stream=stream0)
            buf157 = reinterpret_tensor(buf153, (2048, 512), (512, 1), 0); del buf153  # reuse
            # Topologically Sorted Source Nodes: [x_28, relu_4, x_29], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf156, (2048, 262144), (1, 2048), 0), view_82, out=buf157)
            del view_82
            buf159 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf157, buf159, 1048576, stream=stream0)
            buf162 = buf140; del buf140  # reuse
            # Topologically Sorted Source Nodes: [y_13, y_14], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf161, (512, 262144), (1, 512), 0), reinterpret_tensor(getitem_16, (262144, 512), (512, 1), 0), out=buf162)
            del buf161
            del getitem_16
            buf164 = reinterpret_tensor(buf150, (512, 512), (512, 1), 0); del buf150  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf162, buf164, 262144, stream=stream0)
            buf176 = buf162; del buf162  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf168, (512, 262144), (1, 512), 0), view_70, out=buf176)
            del buf168
            buf178 = reinterpret_tensor(buf148, (512, 512), (512, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf176, buf178, 262144, stream=stream0)
            buf180 = buf176; del buf176  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf179, (512, 262144), (1, 512), 0), view_70, out=buf180)
            buf182 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf180, buf182, 262144, stream=stream0)
            buf184 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf183, (512, 262144), (1, 512), 0), view_70, out=buf184)
            del view_70
            buf186 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf184, buf186, 262144, stream=stream0)
            buf190 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf189, buf190, 1, 262144, stream=stream0)
            buf192 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf191, buf192, 1, 262144, stream=stream0)
            buf194 = reinterpret_tensor(buf157, (512, 2048), (2048, 1), 0); del buf157  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf193, (512, 262144), (1, 512), 0), view_68, out=buf194)
            del view_68
            buf195 = reinterpret_tensor(buf156, (262144, 2048), (2048, 1), 0); del buf156  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf193, permute_163, out=buf195)
            del permute_163
            buf196 = empty_strided_cuda((512, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf194, buf196, 1048576, stream=stream0)
            buf197 = reinterpret_tensor(mm_24, (128, 2048, 2048), (4194304, 2048, 1), 0); del mm_24  # reuse
            # Topologically Sorted Source Nodes: [x_22, relu_3, x_23], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf197, buf195, 536870912, stream=stream0)
            del buf195
            buf198 = reinterpret_tensor(buf194, (2048, 512), (512, 1), 0); del buf194  # reuse
            # Topologically Sorted Source Nodes: [x_22, relu_3, x_23], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf197, (2048, 262144), (1, 2048), 0), view_66, out=buf198)
            del view_66
            buf199 = buf193; del buf193  # reuse
            # Topologically Sorted Source Nodes: [x_22, relu_3, x_23], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf197, (262144, 2048), (2048, 1), 0), permute_167, out=buf199)
            del permute_167
            buf200 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf198, buf200, 1048576, stream=stream0)
            buf202 = reinterpret_tensor(buf199, (128, 2048, 512), (1048576, 512, 1), 0); del buf199  # reuse
            # Topologically Sorted Source Nodes: [getitem_28, rms_norm_16], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_22.run(buf202, add_44, rsqrt_16, buf188, primals_5, 262144, 512, stream=stream0)
            del add_44
            del rsqrt_16
            buf203 = buf184; del buf184  # reuse
            # Topologically Sorted Source Nodes: [y_10, y_11], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf202, (512, 262144), (1, 512), 0), reinterpret_tensor(getitem_12, (262144, 512), (512, 1), 0), out=buf203)
            buf204 = reinterpret_tensor(buf183, (262144, 512), (512, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf202, (262144, 512), (512, 1), 0), permute_171, out=buf204)
            del permute_171
            buf205 = reinterpret_tensor(buf191, (512, 512), (512, 1), 0); del buf191  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf203, buf205, 262144, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf206 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf204, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0), convert_element_type_104, convert_element_type_106, add_37, getitem_12, getitem_13, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_37
            del convert_element_type_104
            del convert_element_type_106
            del getitem_12
            del getitem_13
            buf207 = buf206[0]
            assert_size_stride(buf207, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf207, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf208 = buf206[1]
            assert_size_stride(buf208, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf208, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf209 = buf206[2]
            assert_size_stride(buf209, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf209, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf206
            buf213 = reinterpret_tensor(buf204, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0); del buf204  # reuse
            buf228 = buf213; del buf213  # reuse
            buf215 = buf179; del buf179  # reuse
            buf232 = buf215; del buf215  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_11, q_11], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf228, buf232, cat_6, rsqrt_14, buf207, cat_7, rsqrt_15, buf208, primals_2, primals_3, 1048576, 128, stream=stream0)
            del buf207
            del cat_6
            del cat_7
            del rsqrt_14
            del rsqrt_15
            buf238 = buf145; del buf145  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_22, sigmoid_1, gate_1, unsqueeze_1], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8.run(buf238, 4194304, stream=stream0)
            buf218 = reinterpret_tensor(buf198, (128, 2048, 4), (8192, 4, 1), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_22, sigmoid_1, gate_1, unsqueeze_1, ve_3], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9.run(buf209, embedding_2, primals_1, mm_22, buf238, buf218, 1048576, 128, stream=stream0)
            del embedding_2
            del mm_22
            buf219 = reinterpret_tensor(buf128, (8, 262144), (1, 8), 0); del buf128  # reuse
            buf221 = reinterpret_tensor(buf126, (262144, 8), (8, 1), 0); del buf126  # reuse
            # Topologically Sorted Source Nodes: [linear_22, sigmoid_1], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10.run(buf218, buf219, buf221, 2097152, stream=stream0)
            buf220 = buf127; del buf127  # reuse
            # Topologically Sorted Source Nodes: [linear_22, sigmoid_1], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf219, view_61, out=buf220)
            del view_61
            buf222 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mm_11.run(permute_175, buf222, 256, stream=stream0)
            del permute_175
            buf223 = buf130; del buf130  # reuse
            # Topologically Sorted Source Nodes: [linear_22, sigmoid_1], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(buf221, buf222, out=buf223)
            buf224 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mm_12.run(buf220, buf224, 128, stream=stream0)
            buf225 = buf203; del buf203  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf209, (512, 262144), (1, 512), 0), view_51, out=buf225)
            buf226 = reinterpret_tensor(buf208, (262144, 512), (512, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf209, (262144, 512), (512, 1), 0), permute_179, out=buf226)
            del permute_179
            buf227 = reinterpret_tensor(buf189, (512, 512), (512, 1), 0); del buf189  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf225, buf227, 262144, stream=stream0)
            buf229 = buf225; del buf225  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf228, (512, 262144), (1, 512), 0), view_51, out=buf229)
            buf230 = reinterpret_tensor(buf209, (262144, 512), (512, 1), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf228, (262144, 512), (512, 1), 0), permute_183, out=buf230)
            del permute_183
            buf231 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf229, buf231, 262144, stream=stream0)
            buf233 = buf229; del buf229  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf232, (512, 262144), (1, 512), 0), view_51, out=buf233)
            del view_51
            buf234 = reinterpret_tensor(buf228, (262144, 512), (512, 1), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf232, (262144, 512), (512, 1), 0), permute_187, out=buf234)
            del permute_187
            buf235 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf233, buf235, 262144, stream=stream0)
            buf237 = buf202; del buf202  # reuse
            buf243 = buf51; del buf51  # reuse
            buf246 = reinterpret_tensor(buf232, (262144, 512), (512, 1), 0); del buf232  # reuse
            # Topologically Sorted Source Nodes: [getitem_48, getitem_42, getitem_35, getitem_29, rms_norm_13, getitem_22, getitem_21], Original ATen: [aten.slice_backward, aten.select, aten.mul, aten.add, aten.view, aten._fused_rms_norm_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_23.run(buf237, buf243, add_35, rsqrt_13, buf223, buf226, buf230, buf234, primals_6, buf95, buf144, buf188, primals_5, buf246, 262144, 512, stream=stream0)
            del add_35
            del buf144
            del rsqrt_13
            buf240 = empty_strided_cuda((8192, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_14.run(buf238, buf240, 4194304, stream=stream0)
            buf248 = reinterpret_tensor(buf197, (262144, 2048), (2048, 1), 0); del buf197  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf246, permute_191, out=buf248)
            del permute_191
            buf250 = reinterpret_tensor(mm_17, (128, 2048, 2048), (4194304, 2048, 1), 0); del mm_17  # reuse
            # Topologically Sorted Source Nodes: [x_16, relu_2, x_17], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf250, buf248, 536870912, stream=stream0)
            del buf248
            buf252 = reinterpret_tensor(buf95, (262144, 512), (512, 1), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [x_16, relu_2, x_17], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf250, (262144, 2048), (2048, 1), 0), permute_195, out=buf252)
            del permute_195
            buf255 = reinterpret_tensor(buf252, (128, 2048, 512), (1048576, 512, 1), 0); del buf252  # reuse
            # Topologically Sorted Source Nodes: [getitem_21, rms_norm_12], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_24.run(buf255, add_32, rsqrt_12, buf237, primals_5, 262144, 512, stream=stream0)
            del add_32
            del rsqrt_12
            buf257 = buf234; del buf234  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf255, (262144, 512), (512, 1), 0), permute_199, out=buf257)
            del permute_199
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf259 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf257, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0), convert_element_type_73, convert_element_type_75, view_43, getitem_8, getitem_9, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del convert_element_type_73
            del convert_element_type_75
            del getitem_9
            del view_43
            buf262 = buf259[2]
            assert_size_stride(buf262, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf262, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf271 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf262, (262144, 512), (512, 1), 0), permute_203, out=buf271)
            del permute_203
            buf260 = buf259[0]
            assert_size_stride(buf260, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf260, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf261 = buf259[1]
            assert_size_stride(buf261, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf261, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf259
            buf266 = reinterpret_tensor(buf230, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0); del buf230  # reuse
            buf273 = buf266; del buf266  # reuse
            buf268 = reinterpret_tensor(buf226, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0); del buf226  # reuse
            buf277 = buf268; del buf268  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_8, q_8], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf273, buf277, cat_4, rsqrt_10, buf260, cat_5, rsqrt_11, buf261, primals_2, primals_3, 1048576, 128, stream=stream0)
            del cat_4
            del cat_5
            del rsqrt_10
            del rsqrt_11
            buf275 = reinterpret_tensor(buf261, (262144, 512), (512, 1), 0); del buf261  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf273, (262144, 512), (512, 1), 0), permute_207, out=buf275)
            del permute_207
            buf279 = reinterpret_tensor(buf260, (262144, 512), (512, 1), 0); del buf260  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf277, (262144, 512), (512, 1), 0), permute_211, out=buf279)
            del permute_211
            buf282 = reinterpret_tensor(buf271, (128, 2048, 512), (1048576, 512, 1), 0); del buf271  # reuse
            buf287 = reinterpret_tensor(buf188, (262144, 512), (512, 1), 0); del buf188  # reuse
            buf241 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf283 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf285 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf244 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, rms_norm_9, getitem_15], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_25.run(buf282, add_24, rsqrt_9, buf275, buf279, buf255, primals_5, buf237, embedding, rsqrt, add_23, add_34, buf287, buf241, buf283, buf285, buf244, 262144, 512, stream=stream0)
            del add_23
            del add_24
            del add_34
            del buf237
            del buf275
            del buf279
            del rsqrt_9
            buf242 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf241, buf242, 1, 262144, stream=stream0)
            buf245 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf244, buf245, 1, 262144, stream=stream0)
            buf247 = reinterpret_tensor(buf218, (512, 2048), (2048, 1), 0); del buf218  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf246, (512, 262144), (1, 512), 0), view_49, out=buf247)
            del buf246
            del view_49
            buf249 = empty_strided_cuda((512, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf247, buf249, 1048576, stream=stream0)
            buf251 = reinterpret_tensor(buf247, (2048, 512), (512, 1), 0); del buf247  # reuse
            # Topologically Sorted Source Nodes: [x_16, relu_2, x_17], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf250, (2048, 262144), (1, 2048), 0), view_47, out=buf251)
            del view_47
            buf253 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf251, buf253, 1048576, stream=stream0)
            buf256 = buf233; del buf233  # reuse
            # Topologically Sorted Source Nodes: [y_7, y_8], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf255, (512, 262144), (1, 512), 0), reinterpret_tensor(getitem_8, (262144, 512), (512, 1), 0), out=buf256)
            del buf255
            del getitem_8
            buf258 = reinterpret_tensor(buf244, (512, 512), (512, 1), 0); del buf244  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf256, buf258, 262144, stream=stream0)
            buf270 = buf256; del buf256  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf262, (512, 262144), (1, 512), 0), view_35, out=buf270)
            buf272 = reinterpret_tensor(buf241, (512, 512), (512, 1), 0); del buf241  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf270, buf272, 262144, stream=stream0)
            buf274 = buf270; del buf270  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf273, (512, 262144), (1, 512), 0), view_35, out=buf274)
            buf276 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf274, buf276, 262144, stream=stream0)
            buf278 = buf274; del buf274  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf277, (512, 262144), (1, 512), 0), view_35, out=buf278)
            del view_35
            buf280 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf278, buf280, 262144, stream=stream0)
            buf284 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf283, buf284, 1, 262144, stream=stream0)
            buf286 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf285, buf286, 1, 262144, stream=stream0)
            buf288 = reinterpret_tensor(buf251, (512, 2048), (2048, 1), 0); del buf251  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf287, (512, 262144), (1, 512), 0), view_33, out=buf288)
            del view_33
            buf289 = reinterpret_tensor(buf250, (262144, 2048), (2048, 1), 0); del buf250  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf287, permute_215, out=buf289)
            del permute_215
            buf290 = empty_strided_cuda((512, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf288, buf290, 1048576, stream=stream0)
            buf291 = reinterpret_tensor(mm_11, (128, 2048, 2048), (4194304, 2048, 1), 0); del mm_11  # reuse
            # Topologically Sorted Source Nodes: [x_10, relu_1, x_11], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf291, buf289, 536870912, stream=stream0)
            del buf289
            buf292 = reinterpret_tensor(buf288, (2048, 512), (512, 1), 0); del buf288  # reuse
            # Topologically Sorted Source Nodes: [x_10, relu_1, x_11], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf291, (2048, 262144), (1, 2048), 0), view_31, out=buf292)
            del view_31
            buf293 = buf287; del buf287  # reuse
            # Topologically Sorted Source Nodes: [x_10, relu_1, x_11], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf291, (262144, 2048), (2048, 1), 0), permute_219, out=buf293)
            del permute_219
            buf294 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf292, buf294, 1048576, stream=stream0)
            buf296 = reinterpret_tensor(buf293, (128, 2048, 512), (1048576, 512, 1), 0); del buf293  # reuse
            # Topologically Sorted Source Nodes: [getitem_15, rms_norm_8], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_26.run(buf296, add_21, rsqrt_8, buf282, primals_5, 262144, 512, stream=stream0)
            del add_21
            del rsqrt_8
            buf297 = buf278; del buf278  # reuse
            # Topologically Sorted Source Nodes: [y_4, y_5], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf296, (512, 262144), (1, 512), 0), reinterpret_tensor(getitem_4, (262144, 512), (512, 1), 0), out=buf297)
            buf298 = reinterpret_tensor(buf277, (262144, 512), (512, 1), 0); del buf277  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf296, (262144, 512), (512, 1), 0), permute_223, out=buf298)
            del permute_223
            buf299 = reinterpret_tensor(buf285, (512, 512), (512, 1), 0); del buf285  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf297, buf299, 262144, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf300 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf298, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0), convert_element_type_45, convert_element_type_47, add_14, getitem_4, getitem_5, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_14
            del convert_element_type_45
            del convert_element_type_47
            del getitem_4
            del getitem_5
            buf301 = buf300[0]
            assert_size_stride(buf301, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf301, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf302 = buf300[1]
            assert_size_stride(buf302, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf302, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf303 = buf300[2]
            assert_size_stride(buf303, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf303, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf300
            buf307 = reinterpret_tensor(buf298, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0); del buf298  # reuse
            buf322 = buf307; del buf307  # reuse
            buf309 = buf273; del buf273  # reuse
            buf326 = buf309; del buf309  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_5, q_5], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf322, buf326, cat_2, rsqrt_6, buf301, cat_3, rsqrt_7, buf302, primals_2, primals_3, 1048576, 128, stream=stream0)
            del cat_2
            del cat_3
            del rsqrt_6
            del rsqrt_7
            buf312 = reinterpret_tensor(buf292, (128, 2048, 4), (8192, 4, 1), 0); del buf292  # reuse
            # Topologically Sorted Source Nodes: [linear_9, sigmoid, ve_1], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.view, aten.mul, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_view_27.run(buf303, embedding_1, mm_9, buf312, 1048576, 128, stream=stream0)
            del embedding_1
            buf313 = reinterpret_tensor(buf221, (8, 262144), (1, 8), 0); del buf221  # reuse
            buf315 = reinterpret_tensor(buf219, (262144, 8), (8, 1), 0); del buf219  # reuse
            # Topologically Sorted Source Nodes: [linear_9, sigmoid], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10.run(buf312, buf313, buf315, 2097152, stream=stream0)
            buf314 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [linear_9, sigmoid], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf313, view_26, out=buf314)
            del buf313
            del view_26
            buf316 = buf222; del buf222  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mm_11.run(permute_227, buf316, 256, stream=stream0)
            del permute_227
            buf317 = buf223; del buf223  # reuse
            # Topologically Sorted Source Nodes: [linear_9, sigmoid], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(buf315, buf316, out=buf317)
            del buf315
            del buf316
            buf318 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mm_12.run(buf314, buf318, 128, stream=stream0)
            del buf314
            buf319 = buf297; del buf297  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf303, (512, 262144), (1, 512), 0), view_16, out=buf319)
            buf320 = reinterpret_tensor(buf302, (262144, 512), (512, 1), 0); del buf302  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf303, (262144, 512), (512, 1), 0), permute_231, out=buf320)
            del permute_231
            buf321 = reinterpret_tensor(buf283, (512, 512), (512, 1), 0); del buf283  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf319, buf321, 262144, stream=stream0)
            buf323 = buf319; del buf319  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf322, (512, 262144), (1, 512), 0), view_16, out=buf323)
            buf324 = reinterpret_tensor(buf301, (262144, 512), (512, 1), 0); del buf301  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf322, (262144, 512), (512, 1), 0), permute_235, out=buf324)
            del permute_235
            buf325 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf323, buf325, 262144, stream=stream0)
            buf327 = buf323; del buf323  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf326, (512, 262144), (1, 512), 0), view_16, out=buf327)
            del view_16
            buf328 = reinterpret_tensor(buf322, (262144, 512), (512, 1), 0); del buf322  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf326, (262144, 512), (512, 1), 0), permute_239, out=buf328)
            del permute_239
            buf329 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf327, buf329, 262144, stream=stream0)
            buf331 = buf296; del buf296  # reuse
            buf339 = reinterpret_tensor(buf326, (262144, 512), (512, 1), 0); del buf326  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_5, getitem_8], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.select]
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_28.run(buf331, add_12, rsqrt_5, buf317, buf320, buf324, buf328, primals_5, buf339, 262144, 512, stream=stream0)
            del add_12
            del buf317
            del rsqrt_5
            buf332 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_9, sigmoid, gate, unsqueeze], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8.run(buf332, 4194304, stream=stream0)
            buf341 = reinterpret_tensor(buf291, (262144, 2048), (2048, 1), 0); del buf291  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf339, permute_243, out=buf341)
            del permute_243
            buf343 = reinterpret_tensor(mm_4, (128, 2048, 2048), (4194304, 2048, 1), 0); del mm_4  # reuse
            # Topologically Sorted Source Nodes: [x_4, relu, x_5], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf343, buf341, 536870912, stream=stream0)
            del buf341
            buf345 = buf328; del buf328  # reuse
            # Topologically Sorted Source Nodes: [x_4, relu, x_5], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf343, (262144, 2048), (2048, 1), 0), permute_247, out=buf345)
            del permute_247
            buf348 = reinterpret_tensor(buf345, (128, 2048, 512), (1048576, 512, 1), 0); del buf345  # reuse
            # Topologically Sorted Source Nodes: [getitem_8, rms_norm_4], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29.run(buf348, add_9, rsqrt_4, buf331, primals_5, 262144, 512, stream=stream0)
            del add_9
            del rsqrt_4
            buf350 = buf324; del buf324  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf348, (262144, 512), (512, 1), 0), permute_251, out=buf350)
            del permute_251
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf352 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf350, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0), convert_element_type_14, convert_element_type_16, view_8, getitem, getitem_1, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del convert_element_type_14
            del convert_element_type_16
            del getitem_1
            del view_8
            buf355 = buf352[2]
            assert_size_stride(buf355, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf355, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf364 = buf350; del buf350  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf355, (262144, 512), (512, 1), 0), permute_255, out=buf364)
            del permute_255
            buf353 = buf352[0]
            assert_size_stride(buf353, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf353, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf354 = buf352[1]
            assert_size_stride(buf354, (128, 2048, 4, 128), (1048576, 512, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf354, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf352
            buf359 = reinterpret_tensor(buf320, (128, 2048, 4, 128), (1048576, 512, 128, 1), 0); del buf320  # reuse
            buf366 = buf359; del buf359  # reuse
            buf361 = buf262; del buf262  # reuse
            buf370 = buf361; del buf361  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_2, q_2], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf366, buf370, cat, rsqrt_2, buf353, cat_1, rsqrt_3, buf354, primals_2, primals_3, 1048576, 128, stream=stream0)
            del cat
            del cat_1
            del primals_2
            del primals_3
            del rsqrt_2
            del rsqrt_3
            buf368 = reinterpret_tensor(buf354, (262144, 512), (512, 1), 0); del buf354  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf366, (262144, 512), (512, 1), 0), permute_259, out=buf368)
            del permute_259
            buf372 = reinterpret_tensor(buf353, (262144, 512), (512, 1), 0); del buf353  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf370, (262144, 512), (512, 1), 0), permute_263, out=buf372)
            del permute_263
            buf383 = empty_strided_cuda((8192, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8.run(buf383, 4194304, stream=stream0)
            buf376 = reinterpret_tensor(buf364, (128, 2048, 512), (1048576, 512, 1), 0); del buf364  # reuse
            buf335 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf377 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf337 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            # Topologically Sorted Source Nodes: [loss, x_1, getitem_16, linear_9, sigmoid, gate, unsqueeze, getitem_9, getitem_2, mul, getitem_3, mul_1, x_2, rms_norm_1], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._to_copy, aten.mul, aten.select, aten.add, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.view, aten.sum, aten._fused_rms_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_select_sigmoid_sum_unsqueeze_view_30.run(buf376, primals_5, embedding, rsqrt, primals_6, rsqrt_1, buf368, buf372, buf348, buf243, buf282, buf331, add_11, primals_1, buf303, mm_9, buf335, buf377, buf337, buf332, buf383, 262144, 512, stream=stream0)
            del add_11
            del buf243
            del buf282
            del buf303
            del buf331
            del buf368
            del buf372
            del buf376
            del embedding
            del mm_9
            del primals_1
            del primals_5
            del primals_6
            del rsqrt
            del rsqrt_1
            buf334 = empty_strided_cuda((8192, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_14.run(buf332, buf334, 4194304, stream=stream0)
            del buf332
            buf336 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf335, buf336, 1, 262144, stream=stream0)
            buf338 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf337, buf338, 1, 262144, stream=stream0)
            buf340 = reinterpret_tensor(buf312, (512, 2048), (2048, 1), 0); del buf312  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf339, (512, 262144), (1, 512), 0), view_14, out=buf340)
            del buf339
            del view_14
            buf342 = empty_strided_cuda((512, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf340, buf342, 1048576, stream=stream0)
            buf344 = reinterpret_tensor(buf340, (2048, 512), (512, 1), 0); del buf340  # reuse
            # Topologically Sorted Source Nodes: [x_4, relu, x_5], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf343, (2048, 262144), (1, 2048), 0), view_12, out=buf344)
            del buf343
            del view_12
            buf346 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf344, buf346, 1048576, stream=stream0)
            del buf344
            buf349 = buf327; del buf327  # reuse
            # Topologically Sorted Source Nodes: [y_1, y_2], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf348, (512, 262144), (1, 512), 0), reinterpret_tensor(getitem, (262144, 512), (512, 1), 0), out=buf349)
            del buf348
            del getitem
            buf351 = reinterpret_tensor(buf337, (512, 512), (512, 1), 0); del buf337  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf349, buf351, 262144, stream=stream0)
            buf363 = buf349; del buf349  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf355, (512, 262144), (1, 512), 0), view, out=buf363)
            del buf355
            buf365 = reinterpret_tensor(buf335, (512, 512), (512, 1), 0); del buf335  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf363, buf365, 262144, stream=stream0)
            buf367 = buf363; del buf363  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf366, (512, 262144), (1, 512), 0), view, out=buf367)
            del buf366
            buf369 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf367, buf369, 262144, stream=stream0)
            buf371 = buf367; del buf367  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf370, (512, 262144), (1, 512), 0), view, out=buf371)
            del buf370
            del view
            buf373 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf371, buf373, 262144, stream=stream0)
            del buf371
            buf378 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_17.run(buf377, buf378, 1, 262144, stream=stream0)
            del buf377
            buf379 = empty_strided_cuda((8, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_select_backward_31.run(buf56, buf97, buf149, buf190, buf242, buf284, buf336, buf378, buf379, 8, stream=stream0)
            del buf149
            del buf190
            del buf242
            del buf284
            del buf336
            del buf56
            del buf97
            buf380 = empty_strided_cuda((8, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_select_backward_31.run(buf58, buf99, buf151, buf192, buf245, buf286, buf338, buf378, buf380, 8, stream=stream0)
            del buf151
            del buf192
            del buf245
            del buf286
            del buf338
            del buf378
            del buf58
            del buf99
            buf385 = empty_strided_cuda((8192, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_14.run(buf383, buf385, 4194304, stream=stream0)
            del buf383
        return (None, None, None, buf385, buf380, buf379, buf373, buf369, buf365, buf351, buf346, buf342, buf334, buf329, buf325, buf321, buf318, buf299, buf294, buf290, buf280, buf276, buf272, buf258, buf253, buf249, buf240, buf235, buf231, buf227, buf224, buf205, buf200, buf196, buf186, buf182, buf178, buf164, buf159, buf155, buf147, buf142, buf138, buf134, buf131, buf112, buf107, buf103, buf93, buf89, buf85, buf71, buf66, buf62, buf54, buf49, buf45, buf41, buf38, buf19, buf14, buf10, buf5, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((1, 20480, 1, 64), (1310720, 64, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((1, 20480, 1, 64), (1310720, 64, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    embedding = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_8 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_1 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_2 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_14 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_3 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_16 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((128, 4, 2048), (8192, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_9 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_4 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_4 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    view_14 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    add_11 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    add_12 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_1 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_5 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_26 = rand_strided((262144, 32), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_9 = rand_strided((262144, 4), (4, 1), device='cuda:0', dtype=torch.bfloat16)
    add_14 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_2 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_3 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_6 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_45 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_7 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_47 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_4 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_5 = rand_strided((128, 4, 2048), (8192, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_21 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_8 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_11 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    view_33 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    add_23 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    add_24 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_9 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_43 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_4 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_5 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_10 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_73 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_11 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_75 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_8 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_9 = rand_strided((128, 4, 2048), (8192, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_32 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_12 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_17 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    view_49 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    add_34 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    add_35 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_2 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_13 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_61 = rand_strided((262144, 32), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_22 = rand_strided((262144, 4), (4, 1), device='cuda:0', dtype=torch.bfloat16)
    add_37 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_6 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_7 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_14 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_104 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_15 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_106 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_12 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_13 = rand_strided((128, 4, 2048), (8192, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_44 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_16 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_24 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    view_68 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    add_46 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    add_47 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_17 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_70 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_78 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_8 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_9 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_18 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_132 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_19 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_134 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_16 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_17 = rand_strided((128, 4, 2048), (8192, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_55 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_20 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_30 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    view_84 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    add_57 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    add_58 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_3 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_21 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_96 = rand_strided((262144, 32), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_35 = rand_strided((262144, 4), (4, 1), device='cuda:0', dtype=torch.bfloat16)
    add_60 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_10 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_11 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_22 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_163 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_23 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_165 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_20 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_21 = rand_strided((128, 4, 2048), (8192, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_67 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_24 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_37 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    view_103 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    add_69 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    add_70 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_25 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_113 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_12 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_13 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_26 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_191 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_27 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_193 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_24 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_25 = rand_strided((128, 4, 2048), (8192, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_78 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_28 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_43 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    view_119 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    add_80 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    add_81 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_4 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_29 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_121 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    view_131 = rand_strided((262144, 32), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_48 = rand_strided((262144, 4), (4, 1), device='cuda:0', dtype=torch.bfloat16)
    add_83 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_14 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_15 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_30 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_222 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_31 = rand_strided((128, 2048, 4, 1), (8192, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_224 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_28 = rand_strided((128, 2048, 4, 128), (1048576, 512, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_29 = rand_strided((128, 4, 2048), (8192, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_90 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_32 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_136 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_50 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    view_138 = rand_strided((262144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    add_92 = rand_strided((128, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_33 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((262144, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_52 = rand_strided((262144, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    amax = rand_strided((262144, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((262144, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_244 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_55 = rand_strided((8192, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_59 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_63 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_67 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_71 = rand_strided((4, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_75 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_79 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_83 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_87 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_91 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_95 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_99 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_103 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_107 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_111 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_115 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_119 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_123 = rand_strided((4, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_127 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_131 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_135 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_139 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_143 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_147 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_151 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_155 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_159 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_163 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_167 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_171 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_175 = rand_strided((4, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_179 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_183 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_187 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_191 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_195 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_199 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_203 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_207 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_211 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_215 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_219 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_223 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_227 = rand_strided((4, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_231 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_235 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_239 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_243 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_247 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_251 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_255 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_259 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_263 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_5, primals_6, primals_64, embedding, rsqrt, rsqrt_1, view, view_8, cat, cat_1, rsqrt_2, convert_element_type_14, rsqrt_3, convert_element_type_16, getitem, getitem_1, add_9, rsqrt_4, view_12, mm_4, view_14, add_11, add_12, embedding_1, rsqrt_5, view_16, view_26, mm_9, add_14, cat_2, cat_3, rsqrt_6, convert_element_type_45, rsqrt_7, convert_element_type_47, getitem_4, getitem_5, add_21, rsqrt_8, view_31, mm_11, view_33, add_23, add_24, rsqrt_9, view_35, view_43, cat_4, cat_5, rsqrt_10, convert_element_type_73, rsqrt_11, convert_element_type_75, getitem_8, getitem_9, add_32, rsqrt_12, view_47, mm_17, view_49, add_34, add_35, embedding_2, rsqrt_13, view_51, view_61, mm_22, add_37, cat_6, cat_7, rsqrt_14, convert_element_type_104, rsqrt_15, convert_element_type_106, getitem_12, getitem_13, add_44, rsqrt_16, view_66, mm_24, view_68, add_46, add_47, rsqrt_17, view_70, view_78, cat_8, cat_9, rsqrt_18, convert_element_type_132, rsqrt_19, convert_element_type_134, getitem_16, getitem_17, add_55, rsqrt_20, view_82, mm_30, view_84, add_57, add_58, embedding_3, rsqrt_21, view_86, view_96, mm_35, add_60, cat_10, cat_11, rsqrt_22, convert_element_type_163, rsqrt_23, convert_element_type_165, getitem_20, getitem_21, add_67, rsqrt_24, view_101, mm_37, view_103, add_69, add_70, rsqrt_25, view_105, view_113, cat_12, cat_13, rsqrt_26, convert_element_type_191, rsqrt_27, convert_element_type_193, getitem_24, getitem_25, add_78, rsqrt_28, view_117, mm_43, view_119, add_80, add_81, embedding_4, rsqrt_29, view_121, view_131, mm_48, add_83, cat_14, cat_15, rsqrt_30, convert_element_type_222, rsqrt_31, convert_element_type_224, getitem_28, getitem_29, add_90, rsqrt_32, view_136, mm_50, view_138, add_92, rsqrt_33, view_140, mm_52, amax, log, convert_element_type_244, permute_55, permute_59, permute_63, permute_67, permute_71, permute_75, permute_79, permute_83, permute_87, permute_91, permute_95, permute_99, permute_103, permute_107, permute_111, permute_115, permute_119, permute_123, permute_127, permute_131, permute_135, permute_139, permute_143, permute_147, permute_151, permute_155, permute_159, permute_163, permute_167, permute_171, permute_175, permute_179, permute_183, permute_187, permute_191, permute_195, permute_199, permute_203, permute_207, permute_211, permute_215, permute_219, permute_223, permute_227, permute_231, permute_235, permute_239, permute_243, permute_247, permute_251, permute_255, permute_259, permute_263, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
