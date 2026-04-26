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
# Topologically Sorted Source Nodes: [view_46, loss, logits, logits_1, truediv, tanh, logits_2, view_45], Original ATen: [aten.nll_loss_backward, aten.view, aten.nll_loss_forward, aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten._log_softmax, aten._log_softmax_backward_data, aten.tanh_backward]
# Source node to ATen node mapping:
#   logits => view_176
#   logits_1 => convert_element_type_302
#   logits_2 => mul_152
#   loss => full_default, full_default_1, sub, sub_1
#   tanh => tanh
#   truediv => div
#   view_45 => view_177
#   view_46 => view_178
# Graph fragment:
#   %primals_78 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_78]
#   %tangents_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_1]
#   %convert_element_type_303 : Tensor "f32[][]cuda:0" = PlaceHolder[target=convert_element_type_303]
#   %mm_65 : Tensor "bf16[262144, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_65]
#   %amax : Tensor "f32[262144, 1][1, 1]cuda:0" = PlaceHolder[target=amax]
#   %log : Tensor "f32[262144, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %sum_4 : Tensor "f32[262144, 1][1, 262144]cuda:0" = PlaceHolder[target=sum_4]
#   %sub_2 : Tensor "f32[262144, 8192][8192, 1]cuda:0" = PlaceHolder[target=sub_2]
#   %div_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_303), kwargs = {})
#   %view_178 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_78, [-1]), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[262144, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_178, 1), kwargs = {})
#   %ne_3 : Tensor "b8[262144, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_6, -1), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "i64[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze_6, %full_default), kwargs = {})
#   %scatter_upon_const_tensor : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [262144, 8192], background_val: 0, dtype: torch.float32, dim: 1, selector: %where_2, val: -1.0})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : Tensor "f32[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %div_2, %full_default_1), kwargs = {})
#   %mul_153 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_upon_const_tensor, %where_3), kwargs = {})
#   %view_176 : Tensor "bf16[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_65, [128, 2048, 8192]), kwargs = {})
#   %convert_element_type_302 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_176, torch.float32), kwargs = {})
#   %div : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_302, 15), kwargs = {})
#   %tanh : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_152 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 15), kwargs = {})
#   %view_177 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_152, [-1, 8192]), kwargs = {})
#   %sub : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_177, %amax), kwargs = {})
#   %sub_1 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %log), kwargs = {})
#   %exp_1 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_4 : Tensor "f32[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_153, [1], True), kwargs = {})
#   %mul_154 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %sum_4), kwargs = {})
#   %sub_2 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_153, %mul_154), kwargs = {})
#   %view_179 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_2, [128, 2048, 8192]), kwargs = {})
#   %mul_155 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_179, 15), kwargs = {})
#   %mul_156 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %tanh), kwargs = {})
#   %sub_3 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_156), kwargs = {})
#   %mul_157 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_155, %sub_3), kwargs = {})
#   %div_3 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_157, 15), kwargs = {})
#   %convert_element_type_304 : Tensor "bf16[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_3, torch.bfloat16), kwargs = {})
#   return %sum_4,%sub_2,%convert_element_type_304
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


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/ro/cros5lxu4lpqdlumnip22nmazbskghii37qcgin6rrmx22ittbrz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_66 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=mm_66]
#   %convert_element_type_309 : Tensor "f32[8192, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_66, torch.float32), kwargs = {})
#   return %convert_element_type_309
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 52428800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/dx/cdx3ddikxyw3dubwteipyh4ier5g4teccpe32uqctmotdeyietxy.py
# Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   x_62 => convert_element_type_297, mul_151
# Graph fragment:
#   %add_115 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_115]
#   %rsqrt_41 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_41]
#   %mm_67 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_67]
#   %sum_5 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_5]
#   %view_181 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_67, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_310 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_181, torch.float32), kwargs = {})
#   %convert_element_type_297 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_115, torch.float32), kwargs = {})
#   %mul_151 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_297, %rsqrt_41), kwargs = {})
#   %mul_159 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_151, %convert_element_type_310), kwargs = {})
#   %sum_5 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_159, [2], True), kwargs = {})
#   %div_4 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_151, 640), kwargs = {})
#   %mul_160 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sum_5), kwargs = {})
#   %sub_4 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_310, %mul_160), kwargs = {})
#   %mul_161 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_41), kwargs = {})
#   %convert_element_type_312 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_161, torch.bfloat16), kwargs = {})
#   return %sum_5,%convert_element_type_312
triton_per_fused__fused_rms_norm_backward__to_copy_mul_view_2 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_mul_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_mul_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1342177280}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_mul_view_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = 0.0015625
    tmp12 = tmp3 * tmp11
    tmp13 = tmp12 * tmp10
    tmp14 = tmp5 - tmp13
    tmp15 = tmp14 * tmp2
    tmp16 = tmp15.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp16, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/52/c52ebozgrux5gfifl4pqcfscdlf6xuhozlbr2kim7htmyfps65xb.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_68 : Tensor "bf16[640, 2560][2560, 1]cuda:0" = PlaceHolder[target=mm_68]
#   %convert_element_type_318 : Tensor "f32[640, 2560][2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_68, torch.float32), kwargs = {})
#   return %convert_element_type_318
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16384000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/y3/cy3cpy5e7kj4rz7zifrox7ecimmhlnft3sg7b45bfvsomrz62dcy.py
# Topologically Sorted Source Nodes: [x_58, relu_9, x_59], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.threshold_backward]
# Source node to ATen node mapping:
#   relu_9 => relu_9
#   x_58 => view_172
#   x_59 => convert_element_type_292
# Graph fragment:
#   %mm_63 : Tensor "bf16[262144, 2560][2560, 1]cuda:0" = PlaceHolder[target=mm_63]
#   %mm_69 : Tensor "bf16[262144, 2560][2560, 1]cuda:0" = PlaceHolder[target=mm_69]
#   %view_183 : Tensor "bf16[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_69, [128, 2048, 2560]), kwargs = {})
#   %convert_element_type_317 : Tensor "f32[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_183, torch.float32), kwargs = {})
#   %view_172 : Tensor "bf16[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_63, [128, 2048, 2560]), kwargs = {})
#   %relu_9 : Tensor "bf16[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_172,), kwargs = {})
#   %convert_element_type_292 : Tensor "f32[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%relu_9, torch.float32), kwargs = {})
#   %pow_53 : Tensor "f32[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_292, 1.0), kwargs = {})
#   %mul_162 : Tensor "f32[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_53, 2.0), kwargs = {})
#   %mul_163 : Tensor "f32[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_317, %mul_162), kwargs = {})
#   %convert_element_type_319 : Tensor "bf16[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_163, torch.bfloat16), kwargs = {})
#   %le : Tensor "b8[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_9, 0), kwargs = {})
#   %full_default_5 : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "bf16[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default_5, %convert_element_type_319), kwargs = {})
#   return %where_4
triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5368709120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 671088640
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


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/rg/crgrvmfs57zlt2fn5temjf2rie4uptjn2t72cpm3eu2vhvsqtyao.py
# Topologically Sorted Source Nodes: [rms_norm_40], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
# Source node to ATen node mapping:
#   rms_norm_40 => convert_element_type_287, mul_150
# Graph fragment:
#   %add_113 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_113]
#   %rsqrt_40 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_40]
#   %mm_71 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_71]
#   %convert_element_type_312 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=convert_element_type_312]
#   %sum_6 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_6]
#   %view_185 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_71, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_325 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_185, torch.float32), kwargs = {})
#   %convert_element_type_287 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_113, torch.float32), kwargs = {})
#   %mul_150 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_287, %rsqrt_40), kwargs = {})
#   %mul_165 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_150, %convert_element_type_325), kwargs = {})
#   %sum_6 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_165, [2], True), kwargs = {})
#   %div_5 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_150, 640), kwargs = {})
#   %mul_166 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_5, %sum_6), kwargs = {})
#   %sub_5 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_325, %mul_166), kwargs = {})
#   %mul_167 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_40), kwargs = {})
#   %convert_element_type_327 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_167, torch.bfloat16), kwargs = {})
#   %add_117 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_312, %convert_element_type_327), kwargs = {})
#   return %sum_6,%add_117
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_5 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1677721600}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp12 = 0.0015625
    tmp13 = tmp3 * tmp12
    tmp14 = tmp13 * tmp10
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp2
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp11 + tmp17
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp18, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/ze/czexobl6iv43dfdnsc4ia3a5t4uoqteatxp2mudmbdghklp4ay4y.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_72 : Tensor "bf16[640, 640][640, 1]cuda:0" = PlaceHolder[target=mm_72]
#   %convert_element_type_332 : Tensor "f32[640, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_72, torch.float32), kwargs = {})
#   return %convert_element_type_332
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/7t/c7trtfakqtccgtzb5is7pv7t6iw6noiv52ojdfnvclinsbwt6rh5.py
# Topologically Sorted Source Nodes: [k_29, q_29, cos, sin, neg], Original ATen: [aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.slice, aten.neg, aten.add, aten.slice_backward]
# Source node to ATen node mapping:
#   cos => slice_1
#   k_29 => convert_element_type_282, mul_149
#   neg => neg
#   q_29 => convert_element_type_280, mul_148
#   sin => slice_2
# Graph fragment:
#   %cat_18 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=cat_18]
#   %rsqrt_38 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_38]
#   %getitem_40 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=getitem_40]
#   %cat_19 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=cat_19]
#   %rsqrt_39 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_39]
#   %getitem_41 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=getitem_41]
#   %sum_7 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1310720]cuda:0" = PlaceHolder[target=sum_7]
#   %primals_2 : Tensor "bf16[1, 20480, 1, 64][1310720, 64, 64, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "bf16[1, 20480, 1, 64][1310720, 64, 64, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %slice_scatter_default : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=slice_scatter_default]
#   %slice_scatter_default_1 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=slice_scatter_default_1]
#   %sum_8 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1310720]cuda:0" = PlaceHolder[target=sum_8]
#   %slice_scatter_default_2 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=slice_scatter_default_2]
#   %slice_scatter_default_3 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=slice_scatter_default_3]
#   %convert_element_type_333 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_41, torch.float32), kwargs = {})
#   %convert_element_type_282 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_19, torch.float32), kwargs = {})
#   %mul_149 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_282, %rsqrt_39), kwargs = {})
#   %mul_169 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_149, %convert_element_type_333), kwargs = {})
#   %sum_7 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_169, [3], True), kwargs = {})
#   %div_6 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_149, 128), kwargs = {})
#   %mul_170 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, %sum_7), kwargs = {})
#   %sub_6 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_333, %mul_170), kwargs = {})
#   %mul_171 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_39), kwargs = {})
#   %convert_element_type_335 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_171, torch.bfloat16), kwargs = {})
#   %convert_element_type_336 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_40, torch.float32), kwargs = {})
#   %convert_element_type_280 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_18, torch.float32), kwargs = {})
#   %mul_148 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_280, %rsqrt_38), kwargs = {})
#   %mul_173 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %convert_element_type_336), kwargs = {})
#   %sum_8 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_173, [3], True), kwargs = {})
#   %div_7 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_148, 128), kwargs = {})
#   %mul_174 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_7, %sum_8), kwargs = {})
#   %sub_7 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_336, %mul_174), kwargs = {})
#   %mul_175 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_38), kwargs = {})
#   %convert_element_type_338 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_175, torch.bfloat16), kwargs = {})
#   %slice_48 : Tensor "bf16[128, 2048, 5, 64][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_335, 3, 0, 64), kwargs = {})
#   %slice_49 : Tensor "bf16[128, 2048, 5, 64][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_335, 3, 64, 128), kwargs = {})
#   %slice_1 : Tensor "bf16[1, 2048, 1, 64][1310720, 64, 64, 1]cuda:0"[num_users=40] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_2, 1, 0, 2048), kwargs = {})
#   %mul_176 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_49, %slice_1), kwargs = {})
#   %slice_2 : Tensor "bf16[1, 2048, 1, 64][1310720, 64, 64, 1]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_3, 1, 0, 2048), kwargs = {})
#   %neg : Tensor "bf16[1, 2048, 1, 64][131072, 64, 64, 1]cuda:0"[num_users=20] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %mul_177 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_49, %neg), kwargs = {})
#   %mul_178 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_48, %slice_2), kwargs = {})
#   %add_118 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %mul_178), kwargs = {})
#   %mul_179 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_48, %slice_1), kwargs = {})
#   %add_119 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_177, %mul_179), kwargs = {})
#   %full_default_6 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=40] = call_function[target=torch.ops.aten.full.default](args = ([128, 2048, 5, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_6, %add_118, 3, 64, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_6, %add_119, 3, 0, 64), kwargs = {})
#   %add_120 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %slice_scatter_default_1), kwargs = {})
#   %slice_50 : Tensor "bf16[128, 2048, 5, 64][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_338, 3, 0, 64), kwargs = {})
#   %slice_51 : Tensor "bf16[128, 2048, 5, 64][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_338, 3, 64, 128), kwargs = {})
#   %mul_180 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_51, %slice_1), kwargs = {})
#   %mul_181 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_51, %neg), kwargs = {})
#   %mul_182 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_50, %slice_2), kwargs = {})
#   %add_121 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_180, %mul_182), kwargs = {})
#   %mul_183 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_50, %slice_1), kwargs = {})
#   %add_122 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_181, %mul_183), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_6, %add_121, 3, 64, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_3 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_6, %add_122, 3, 0, 64), kwargs = {})
#   %add_123 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_2, %slice_scatter_default_3), kwargs = {})
#   return %sum_8,%sum_7,%slice_scatter_default,%slice_scatter_default_1,%add_120,%slice_scatter_default_2,%slice_scatter_default_3,%add_123
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2097152, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 30, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760, 'r0_': 8053063680}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1310720
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
    x3 = ((xindex // 5) % 2048)
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


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/zp/czptwmqftmzxu36o2vzmfmqjttzdzkdojr5ilheqq7ljwwfcg5z7.py
# Topologically Sorted Source Nodes: [loss, linear_61, sigmoid_4, gate_4, unsqueeze_4], Original ATen: [aten.nll_loss_forward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   gate_4 => mul_138
#   linear_61 => view_167
#   loss => full_default_1
#   sigmoid_4 => sigmoid_4
#   unsqueeze_4 => unsqueeze_4
# Graph fragment:
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_167 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_61, [128, 2048, 5]), kwargs = {})
#   %sigmoid_4 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_167,), kwargs = {})
#   %mul_138 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_4, 2), kwargs = {})
#   %unsqueeze_4 : Tensor "bf16[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_138, -1), kwargs = {})
#   %mul_184 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_42, %unsqueeze_4), kwargs = {})
#   %view_191 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_184, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_366 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_191, torch.float32), kwargs = {})
#   %eq : Tensor "b8[128, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "b8[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %where_5 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_7, %full_default_1, %convert_element_type_366), kwargs = {})
#   %full_default_12 : Tensor "f32[8192, 640][640, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([8192, 640], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : Tensor "f32[8192, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_12, [%primals_1], %where_5, True), kwargs = {})
#   return %index_put
triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8 = async_compile.triton('triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41943040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/vw/cvwhejxkqfc45yac7lm6l6vasghnrx6s7ghyagvwcepfnek7f4aw.py
# Topologically Sorted Source Nodes: [loss, linear_61, sigmoid_4, gate_4, unsqueeze_4, ve_9], Original ATen: [aten.nll_loss_forward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   gate_4 => mul_138
#   linear_61 => view_167
#   loss => full_default_1
#   sigmoid_4 => sigmoid_4
#   unsqueeze_4 => unsqueeze_4
#   ve_9 => view_165
# Graph fragment:
#   %getitem_42 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=getitem_42]
#   %embedding_5 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding_5]
#   %primals_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %mm_61 : Tensor "bf16[262144, 5][5, 1]cuda:0" = PlaceHolder[target=mm_61]
#   %index_put : Tensor "f32[8192, 640][640, 1]cuda:0" = PlaceHolder[target=index_put]
#   %sum_9 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1310720]cuda:0" = PlaceHolder[target=sum_9]
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_167 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_61, [128, 2048, 5]), kwargs = {})
#   %sigmoid_4 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_167,), kwargs = {})
#   %mul_138 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_4, 2), kwargs = {})
#   %unsqueeze_4 : Tensor "bf16[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_138, -1), kwargs = {})
#   %mul_184 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_42, %unsqueeze_4), kwargs = {})
#   %view_165 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%embedding_5, [128, 2048, 5, 128]), kwargs = {})
#   %mul_185 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_42, %view_165), kwargs = {})
#   %sum_9 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_185, [3], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_339 : Tensor "bf16[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_9, torch.bfloat16), kwargs = {})
#   %squeeze_1 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convert_element_type_339, -1), kwargs = {})
#   %mul_186 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, 2), kwargs = {})
#   %convert_element_type_340 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_186, torch.float32), kwargs = {})
#   %convert_element_type_341 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_4, torch.float32), kwargs = {})
#   %sub_8 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_341), kwargs = {})
#   %mul_187 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_341, %sub_8), kwargs = {})
#   %mul_188 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_340, %mul_187), kwargs = {})
#   %convert_element_type_342 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_188, torch.bfloat16), kwargs = {})
#   %view_191 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_184, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_366 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_191, torch.float32), kwargs = {})
#   %eq : Tensor "b8[128, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "b8[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %where_5 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_7, %full_default_1, %convert_element_type_366), kwargs = {})
#   %full_default_12 : Tensor "f32[8192, 640][640, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([8192, 640], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : Tensor "f32[8192, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_12, [%primals_1], %where_5, True), kwargs = {})
#   return %sum_9,%buf51,%convert_element_type_342
triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2097152, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1310720
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
    x3 = xindex // 5
    x2 = (xindex % 5)
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
    tl.atomic_add(out_ptr1 + (r0_1 + 128*x2 + 640*tmp11), tmp22, None, sem='relaxed')
    tl.store(out_ptr2 + (x0), tmp33, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/t4/ct46rwj3nsmkh2ea6zqgvsufirvfw7ywijahsydzgq2ieaahmfqb.py
# Topologically Sorted Source Nodes: [linear_61, sigmoid_4], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
# Source node to ATen node mapping:
#   linear_61 => view_167
#   sigmoid_4 => sigmoid_4
# Graph fragment:
#   %convert_element_type_342 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0" = PlaceHolder[target=convert_element_type_342]
#   %view_167 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_61, [128, 2048, 5]), kwargs = {})
#   %sigmoid_4 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_167,), kwargs = {})
#   %convert_element_type_339 : Tensor "bf16[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_9, torch.bfloat16), kwargs = {})
#   %squeeze_1 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convert_element_type_339, -1), kwargs = {})
#   %mul_186 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, 2), kwargs = {})
#   %convert_element_type_340 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_186, torch.float32), kwargs = {})
#   %convert_element_type_341 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_4, torch.float32), kwargs = {})
#   %sub_8 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_341), kwargs = {})
#   %mul_187 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_341, %sub_8), kwargs = {})
#   %mul_188 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_340, %mul_187), kwargs = {})
#   %convert_element_type_342 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_188, torch.bfloat16), kwargs = {})
#   %view_189 : Tensor "bf16[262144, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_342, [262144, 5]), kwargs = {})
#   %permute_82 : Tensor "bf16[5, 262144][1, 5]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_189, [1, 0]), kwargs = {})
#   %constant_pad_nd_default_4 : Tensor "bf16[8, 262144][262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_82, [0, 0, 0, 3]), kwargs = {})
#   return %constant_pad_nd_default_4
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12582912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 5, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 5*x1), tmp2, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/fb/cfbd6rx7m2q3km4e4br5ilfr7jejuj5n6cmngqn5s6pqwvejehbz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_default_4 : Tensor "bf16[8, 32][32, 1]cuda:0" = PlaceHolder[target=mm_default_4]
#   %slice_tensor_4 : Tensor "bf16[5, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_default_4, 0, 0, -3), kwargs = {})
#   %convert_element_type_347 : Tensor "f32[5, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_tensor_4, torch.float32), kwargs = {})
#   return %convert_element_type_347
triton_poi_fused__to_copy_mm_11 = async_compile.triton('triton_poi_fused__to_copy_mm_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mm_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mm_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/c3/cc3nrkfvhudjrfgqhfh377dlw2xktabj4ruuf33tk424rjh7pfsb.py
# Topologically Sorted Source Nodes: [rms_norm_37, getitem_60], Original ATen: [aten.view, aten.slice_backward, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.select]
# Source node to ATen node mapping:
#   getitem_60 => select_18
#   rms_norm_37 => convert_element_type_266, mul_137
# Graph fragment:
#   %add_104 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_104]
#   %rsqrt_37 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_37]
#   %mm_75 : Tensor "bf16[262144, 32][32, 1]cuda:0" = PlaceHolder[target=mm_75]
#   %mm_77 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_77]
#   %mm_79 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_79]
#   %mm_81 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_81]
#   %add_117 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_117]
#   %sum_10 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_10]
#   %add_127 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_127]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %view_190 : Tensor "bf16[128, 2048, 32][65536, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_75, [128, 2048, 32]), kwargs = {})
#   %full_default_10 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([128, 2048, 640], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_4 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_190, 2, 0, 32), kwargs = {})
#   %view_194 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_77, [128, 2048, 640]), kwargs = {})
#   %add_124 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_4, %view_194), kwargs = {})
#   %view_197 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_79, [128, 2048, 640]), kwargs = {})
#   %add_125 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_124, %view_197), kwargs = {})
#   %view_200 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_81, [128, 2048, 640]), kwargs = {})
#   %add_126 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_125, %view_200), kwargs = {})
#   %convert_element_type_363 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_126, torch.float32), kwargs = {})
#   %convert_element_type_266 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_104, torch.float32), kwargs = {})
#   %mul_137 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_266, %rsqrt_37), kwargs = {})
#   %mul_190 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_137, %convert_element_type_363), kwargs = {})
#   %sum_10 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_190, [2], True), kwargs = {})
#   %div_8 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_137, 640), kwargs = {})
#   %mul_191 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_8, %sum_10), kwargs = {})
#   %sub_9 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_363, %mul_191), kwargs = {})
#   %mul_192 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_37), kwargs = {})
#   %convert_element_type_365 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_192, torch.bfloat16), kwargs = {})
#   %add_127 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_117, %convert_element_type_365), kwargs = {})
#   %select_18 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 9), kwargs = {})
#   %mul_195 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_127, %select_18), kwargs = {})
#   %view_201 : Tensor "bf16[262144, 640][640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_195, [262144, 640]), kwargs = {})
#   return %sum_10,%add_127,%view_201
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_12 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 3355443200}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr5 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp45 = tl.load(in_ptr6 + (9))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
        tmp38 = 0.0015625
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39 * tmp19
        tmp41 = tmp34 - tmp40
        tmp42 = tmp41 * tmp2
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp21 + tmp43
        tmp47 = tmp46.to(tl.float32)
        tmp48 = tmp44 * tmp47
        tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp44, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 640*x0), tmp48, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/5f/c5fkejouluzlum5was5b3gy4a4tkh744jeb23buj3fvwqgmgef6n.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf51 : Tensor "f32[8192, 640][640, 1]cuda:0" = PlaceHolder[target=buf51]
#   %convert_element_type_367 : Tensor "bf16[8192, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%index_put, torch.bfloat16), kwargs = {})
#   return %convert_element_type_367
triton_poi_fused_embedding_dense_backward_13 = async_compile.triton('triton_poi_fused_embedding_dense_backward_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20971520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/xd/cxdlvojh2rdybu4agpwhv3b4aeyvt3j7rf4hliqk62lyleef2bne.py
# Topologically Sorted Source Nodes: [getitem_60, rms_norm_36], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_60 => select_18
#   rms_norm_36 => convert_element_type_256, mul_134
# Graph fragment:
#   %add_101 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_101]
#   %rsqrt_36 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_36]
#   %mm_85 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_85]
#   %add_127 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_127]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_13 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_13]
#   %select_18 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 9), kwargs = {})
#   %mul_195 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_127, %select_18), kwargs = {})
#   %view_204 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_85, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_380 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_204, torch.float32), kwargs = {})
#   %convert_element_type_256 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_101, torch.float32), kwargs = {})
#   %mul_134 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_256, %rsqrt_36), kwargs = {})
#   %mul_200 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_134, %convert_element_type_380), kwargs = {})
#   %sum_13 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_200, [2], True), kwargs = {})
#   %div_9 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_134, 640), kwargs = {})
#   %mul_201 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %sum_13), kwargs = {})
#   %sub_10 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_380, %mul_201), kwargs = {})
#   %mul_202 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_36), kwargs = {})
#   %convert_element_type_382 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_202, torch.bfloat16), kwargs = {})
#   %add_128 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_195, %convert_element_type_382), kwargs = {})
#   return %sum_13,%add_128
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_14 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1677721600}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (9))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/do/cdo35dygeepvhec2nkzjbwnqfwmwfwcdtwy6sn2rfw3lgtdfqgxt.py
# Topologically Sorted Source Nodes: [x_1, rms_norm_33, getitem_54], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   getitem_54 => select_16
#   rms_norm_33 => convert_element_type_238, mul_123
#   x_1 => convert_element_type, convert_element_type_1, mul
# Graph fragment:
#   %add_93 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_93]
#   %rsqrt_33 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_33]
#   %mm_89 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_89]
#   %mm_91 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_91]
#   %mm_93 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_93]
#   %add_128 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_128]
#   %sum_16 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_16]
#   %add_137 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_137]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_127 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_127]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_103 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_103]
#   %add_92 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_92]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_194 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_127, %convert_element_type_1), kwargs = {})
#   %sum_11 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_194,), kwargs = {dtype: torch.float32})
#   %mul_196 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_127, %add_103), kwargs = {})
#   %sum_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_196,), kwargs = {dtype: torch.float32})
#   %view_210 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_89, [128, 2048, 640]), kwargs = {})
#   %view_213 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_91, [128, 2048, 640]), kwargs = {})
#   %add_135 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_210, %view_213), kwargs = {})
#   %view_216 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_93, [128, 2048, 640]), kwargs = {})
#   %add_136 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_135, %view_216), kwargs = {})
#   %convert_element_type_409 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_136, torch.float32), kwargs = {})
#   %convert_element_type_238 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_93, torch.float32), kwargs = {})
#   %mul_123 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_238, %rsqrt_33), kwargs = {})
#   %mul_220 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_123, %convert_element_type_409), kwargs = {})
#   %sum_16 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_220, [2], True), kwargs = {})
#   %div_12 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_123, 640), kwargs = {})
#   %mul_221 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, %sum_16), kwargs = {})
#   %sub_13 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_409, %mul_221), kwargs = {})
#   %mul_222 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %rsqrt_33), kwargs = {})
#   %convert_element_type_411 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_222, torch.bfloat16), kwargs = {})
#   %add_137 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_128, %convert_element_type_411), kwargs = {})
#   %mul_224 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, %convert_element_type_1), kwargs = {})
#   %sum_17 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_224,), kwargs = {dtype: torch.float32})
#   %select_16 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 8), kwargs = {})
#   %mul_225 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, %select_16), kwargs = {})
#   %mul_226 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, %add_92), kwargs = {})
#   %sum_18 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_226,), kwargs = {dtype: torch.float32})
#   %view_217 : Tensor "bf16[262144, 640][640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_225, [262144, 640]), kwargs = {})
#   return %sum_16,%add_137,%view_217,%buf53,%buf94,%buf55,%buf96
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_15 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 12, 'num_reduction': 5, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760, 'r0_': 4362076160}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp23 = tl.load(in_ptr5 + (8))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp27 = tl.load(in_ptr6 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr7 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr9 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp52 = tl.load(in_ptr10 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp3 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp14
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp22 * tmp25
    tmp29 = tmp28.to(tl.float32)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp27 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
    tmp37 = tl.where(r0_mask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None].to(tl.float32)
    tmp39 = tmp22 * tmp32
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
    tmp43 = tl.where(r0_mask, tmp41, 0)
    tmp44 = tl.sum(tmp43, 1)[:, None].to(tl.float32)
    tmp46 = tmp27 * tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, R0_BLOCK])
    tmp50 = tl.where(r0_mask, tmp48, 0)
    tmp51 = tl.sum(tmp50, 1)[:, None].to(tl.float32)
    tmp53 = tmp22 * tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, R0_BLOCK])
    tmp57 = tl.where(r0_mask, tmp55, 0)
    tmp58 = tl.sum(tmp57, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp26, r0_mask)
    tl.store(out_ptr2 + (x0), tmp38, None)
    tl.store(out_ptr3 + (x0), tmp44, None)
    tl.store(out_ptr4 + (x0), tmp51, None)
    tl.store(out_ptr5 + (x0), tmp58, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/lg/clgqckwvbszrjlefp6rl4n6ntg26qzywpjoq7ogrgpnzfhqbicnt.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   x_1 => convert_element_type, convert_element_type_1, mul
# Graph fragment:
#   %buf53 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf53]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_194 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_127, %convert_element_type_1), kwargs = {})
#   %sum_11 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_194,), kwargs = {dtype: torch.float32})
#   return %sum_11
triton_red_fused__to_copy_mul_sum_16 = async_compile.triton('triton_red_fused__to_copy_mul_sum_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_sum_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 1048576}}
)
@triton.jit
def triton_red_fused__to_copy_mul_sum_16(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/43/c43lohtwzmcqeayhawson62gvsiz37evjymbhggmg2v3du4ejjub.py
# Topologically Sorted Source Nodes: [getitem_54, rms_norm_32], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_54 => select_16
#   rms_norm_32 => convert_element_type_228, mul_120
# Graph fragment:
#   %add_90 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_90]
#   %rsqrt_32 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_32]
#   %mm_97 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_97]
#   %add_137 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_137]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_19 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_19]
#   %select_16 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 8), kwargs = {})
#   %mul_225 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, %select_16), kwargs = {})
#   %view_220 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_97, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_424 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_220, torch.float32), kwargs = {})
#   %convert_element_type_228 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_90, torch.float32), kwargs = {})
#   %mul_120 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_228, %rsqrt_32), kwargs = {})
#   %mul_230 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_120, %convert_element_type_424), kwargs = {})
#   %sum_19 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_230, [2], True), kwargs = {})
#   %div_13 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_120, 640), kwargs = {})
#   %mul_231 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_13, %sum_19), kwargs = {})
#   %sub_14 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_424, %mul_231), kwargs = {})
#   %mul_232 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_32), kwargs = {})
#   %convert_element_type_426 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_232, torch.bfloat16), kwargs = {})
#   %add_141 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_225, %convert_element_type_426), kwargs = {})
#   return %sum_19,%add_141
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_17 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1677721600}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (8))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/cz/cczhwngbao2upcb7rri6p7f4gfth2x5vje64uwpx5zj6pnjr5635.py
# Topologically Sorted Source Nodes: [rms_norm_29, getitem_47], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.select]
# Source node to ATen node mapping:
#   getitem_47 => select_14
#   rms_norm_29 => convert_element_type_207, mul_107
# Graph fragment:
#   %add_81 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_81]
#   %rsqrt_29 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_29]
#   %mm_101 : Tensor "bf16[262144, 32][32, 1]cuda:0" = PlaceHolder[target=mm_101]
#   %mm_103 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_103]
#   %mm_105 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_105]
#   %mm_107 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_107]
#   %add_141 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_141]
#   %sum_23 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_23]
#   %add_151 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_151]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %full_default_10 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([128, 2048, 640], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_225 : Tensor "bf16[128, 2048, 32][65536, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_101, [128, 2048, 32]), kwargs = {})
#   %slice_scatter_default_13 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_225, 2, 0, 32), kwargs = {})
#   %view_229 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_103, [128, 2048, 640]), kwargs = {})
#   %add_148 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_13, %view_229), kwargs = {})
#   %view_232 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_105, [128, 2048, 640]), kwargs = {})
#   %add_149 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_148, %view_232), kwargs = {})
#   %view_235 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_107, [128, 2048, 640]), kwargs = {})
#   %add_150 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_149, %view_235), kwargs = {})
#   %convert_element_type_462 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_150, torch.float32), kwargs = {})
#   %convert_element_type_207 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_81, torch.float32), kwargs = {})
#   %mul_107 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_207, %rsqrt_29), kwargs = {})
#   %mul_255 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_107, %convert_element_type_462), kwargs = {})
#   %sum_23 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_255, [2], True), kwargs = {})
#   %div_16 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_107, 640), kwargs = {})
#   %mul_256 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_16, %sum_23), kwargs = {})
#   %sub_18 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_462, %mul_256), kwargs = {})
#   %mul_257 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_29), kwargs = {})
#   %convert_element_type_464 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_257, torch.bfloat16), kwargs = {})
#   %add_151 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_141, %convert_element_type_464), kwargs = {})
#   %select_14 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 7), kwargs = {})
#   %mul_260 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_151, %select_14), kwargs = {})
#   %view_236 : Tensor "bf16[262144, 640][640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_260, [262144, 640]), kwargs = {})
#   return %sum_23,%add_151,%view_236
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_18 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 3355443200}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr5 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp21 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
        tmp38 = 0.0015625
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39 * tmp19
        tmp41 = tmp34 - tmp40
        tmp42 = tmp41 * tmp2
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp21 + tmp43
        tmp47 = tmp46.to(tl.float32)
        tmp48 = tmp44 * tmp47
        tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp44, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 640*x0), tmp48, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/tz/ctzf6fklscngmosdkmj7l37ajrqx5bc4hdnpdhp2dprje6xes766.py
# Topologically Sorted Source Nodes: [getitem_47, rms_norm_28], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_47 => select_14
#   rms_norm_28 => convert_element_type_197, mul_104
# Graph fragment:
#   %add_78 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_78]
#   %rsqrt_28 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_28]
#   %mm_111 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_111]
#   %add_151 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_151]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_26 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_26]
#   %select_14 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 7), kwargs = {})
#   %mul_260 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_151, %select_14), kwargs = {})
#   %view_239 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_111, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_479 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_239, torch.float32), kwargs = {})
#   %convert_element_type_197 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_78, torch.float32), kwargs = {})
#   %mul_104 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_197, %rsqrt_28), kwargs = {})
#   %mul_265 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_104, %convert_element_type_479), kwargs = {})
#   %sum_26 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_265, [2], True), kwargs = {})
#   %div_17 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_104, 640), kwargs = {})
#   %mul_266 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_17, %sum_26), kwargs = {})
#   %sub_19 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_479, %mul_266), kwargs = {})
#   %mul_267 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %rsqrt_28), kwargs = {})
#   %convert_element_type_481 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_267, torch.bfloat16), kwargs = {})
#   %add_155 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_260, %convert_element_type_481), kwargs = {})
#   return %sum_26,%add_155
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_19 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1677721600}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (7))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/pq/cpqdhuus5d4pr342zahpqiweg47mmbeyrnzxjq5gze5sblmhn4uu.py
# Topologically Sorted Source Nodes: [x_1, rms_norm_25, getitem_41], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   getitem_41 => select_12
#   rms_norm_25 => convert_element_type_179, mul_93
#   x_1 => convert_element_type, convert_element_type_1, mul
# Graph fragment:
#   %add_70 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_70]
#   %rsqrt_25 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_25]
#   %mm_115 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_115]
#   %mm_117 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_117]
#   %mm_119 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_119]
#   %add_155 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_155]
#   %sum_29 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_29]
#   %add_164 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_164]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_151 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_151]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_80 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_80]
#   %add_69 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_69]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_259 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_151, %convert_element_type_1), kwargs = {})
#   %sum_24 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_259,), kwargs = {dtype: torch.float32})
#   %mul_261 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_151, %add_80), kwargs = {})
#   %sum_25 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_261,), kwargs = {dtype: torch.float32})
#   %view_245 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_115, [128, 2048, 640]), kwargs = {})
#   %view_248 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_117, [128, 2048, 640]), kwargs = {})
#   %add_162 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_245, %view_248), kwargs = {})
#   %view_251 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_119, [128, 2048, 640]), kwargs = {})
#   %add_163 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_162, %view_251), kwargs = {})
#   %convert_element_type_508 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_163, torch.float32), kwargs = {})
#   %convert_element_type_179 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_70, torch.float32), kwargs = {})
#   %mul_93 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_179, %rsqrt_25), kwargs = {})
#   %mul_285 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, %convert_element_type_508), kwargs = {})
#   %sum_29 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_285, [2], True), kwargs = {})
#   %div_20 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_93, 640), kwargs = {})
#   %mul_286 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_20, %sum_29), kwargs = {})
#   %sub_22 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_508, %mul_286), kwargs = {})
#   %mul_287 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt_25), kwargs = {})
#   %convert_element_type_510 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_287, torch.bfloat16), kwargs = {})
#   %add_164 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_155, %convert_element_type_510), kwargs = {})
#   %mul_289 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_164, %convert_element_type_1), kwargs = {})
#   %sum_30 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_289,), kwargs = {dtype: torch.float32})
#   %select_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 6), kwargs = {})
#   %mul_290 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_164, %select_12), kwargs = {})
#   %mul_291 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_164, %add_69), kwargs = {})
#   %sum_31 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_291,), kwargs = {dtype: torch.float32})
#   %view_252 : Tensor "bf16[262144, 640][640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_290, [262144, 640]), kwargs = {})
#   return %sum_29,%add_164,%view_252,%buf144,%buf185,%buf146,%buf187
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_20 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 12, 'num_reduction': 5, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760, 'r0_': 4362076160}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp23 = tl.load(in_ptr5 + (6))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp27 = tl.load(in_ptr6 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr7 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr9 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp52 = tl.load(in_ptr10 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp3 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp14
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp22 * tmp25
    tmp29 = tmp28.to(tl.float32)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp27 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
    tmp37 = tl.where(r0_mask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None].to(tl.float32)
    tmp39 = tmp22 * tmp32
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
    tmp43 = tl.where(r0_mask, tmp41, 0)
    tmp44 = tl.sum(tmp43, 1)[:, None].to(tl.float32)
    tmp46 = tmp27 * tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, R0_BLOCK])
    tmp50 = tl.where(r0_mask, tmp48, 0)
    tmp51 = tl.sum(tmp50, 1)[:, None].to(tl.float32)
    tmp53 = tmp22 * tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, R0_BLOCK])
    tmp57 = tl.where(r0_mask, tmp55, 0)
    tmp58 = tl.sum(tmp57, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp26, r0_mask)
    tl.store(out_ptr2 + (x0), tmp38, None)
    tl.store(out_ptr3 + (x0), tmp44, None)
    tl.store(out_ptr4 + (x0), tmp51, None)
    tl.store(out_ptr5 + (x0), tmp58, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/dp/cdp4vf746ya6hmw3pccdiz2p2iacz673hev6tng5pbkffy2rb2dp.py
# Topologically Sorted Source Nodes: [getitem_41, rms_norm_24], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_41 => select_12
#   rms_norm_24 => convert_element_type_169, mul_90
# Graph fragment:
#   %add_67 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_67]
#   %rsqrt_24 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_24]
#   %mm_123 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_123]
#   %add_164 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_164]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_32 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_32]
#   %select_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 6), kwargs = {})
#   %mul_290 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_164, %select_12), kwargs = {})
#   %view_255 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_123, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_523 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_255, torch.float32), kwargs = {})
#   %convert_element_type_169 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_67, torch.float32), kwargs = {})
#   %mul_90 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_169, %rsqrt_24), kwargs = {})
#   %mul_295 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %convert_element_type_523), kwargs = {})
#   %sum_32 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_295, [2], True), kwargs = {})
#   %div_21 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_90, 640), kwargs = {})
#   %mul_296 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_21, %sum_32), kwargs = {})
#   %sub_23 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_523, %mul_296), kwargs = {})
#   %mul_297 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %rsqrt_24), kwargs = {})
#   %convert_element_type_525 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_297, torch.bfloat16), kwargs = {})
#   %add_168 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %convert_element_type_525), kwargs = {})
#   return %sum_32,%add_168
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_21 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1677721600}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (6))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/xl/cxlwo36czotyni252ti44ciduh3bt2oviy4pwo7w46dbwfzfko6w.py
# Topologically Sorted Source Nodes: [getitem_61, getitem_55, getitem_48, getitem_42, rms_norm_21, getitem_35, getitem_34], Original ATen: [aten.slice_backward, aten.select, aten.mul, aten.add, aten.view, aten._fused_rms_norm_backward, aten._to_copy]
# Source node to ATen node mapping:
#   getitem_34 => select_10
#   getitem_35 => select_11
#   getitem_42 => select_13
#   getitem_48 => select_15
#   getitem_55 => select_17
#   getitem_61 => select_19
#   rms_norm_21 => convert_element_type_148, mul_77
# Graph fragment:
#   %add_58 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_58]
#   %rsqrt_21 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_21]
#   %mm_127 : Tensor "bf16[262144, 32][32, 1]cuda:0" = PlaceHolder[target=mm_127]
#   %mm_129 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_129]
#   %mm_131 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_131]
#   %mm_133 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_133]
#   %add_168 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_168]
#   %sum_36 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_36]
#   %add_127 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_127]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %add_137 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_137]
#   %add_151 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_151]
#   %add_164 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_164]
#   %add_178 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_178]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %full_default_10 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([128, 2048, 640], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_19 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 9), kwargs = {})
#   %mul_193 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_127, %select_19), kwargs = {})
#   %select_17 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 8), kwargs = {})
#   %mul_223 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, %select_17), kwargs = {})
#   %add_138 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_193, %mul_223), kwargs = {})
#   %select_15 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 7), kwargs = {})
#   %mul_258 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_151, %select_15), kwargs = {})
#   %add_152 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_138, %mul_258), kwargs = {})
#   %select_13 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 6), kwargs = {})
#   %mul_288 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_164, %select_13), kwargs = {})
#   %add_165 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_152, %mul_288), kwargs = {})
#   %view_260 : Tensor "bf16[128, 2048, 32][65536, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_127, [128, 2048, 32]), kwargs = {})
#   %slice_scatter_default_22 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_260, 2, 0, 32), kwargs = {})
#   %view_264 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_129, [128, 2048, 640]), kwargs = {})
#   %add_175 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_22, %view_264), kwargs = {})
#   %view_267 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_131, [128, 2048, 640]), kwargs = {})
#   %add_176 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_175, %view_267), kwargs = {})
#   %view_270 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_133, [128, 2048, 640]), kwargs = {})
#   %add_177 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_176, %view_270), kwargs = {})
#   %convert_element_type_561 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_177, torch.float32), kwargs = {})
#   %convert_element_type_148 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_58, torch.float32), kwargs = {})
#   %mul_77 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_148, %rsqrt_21), kwargs = {})
#   %mul_320 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_77, %convert_element_type_561), kwargs = {})
#   %sum_36 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_320, [2], True), kwargs = {})
#   %div_24 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_77, 640), kwargs = {})
#   %mul_321 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_24, %sum_36), kwargs = {})
#   %sub_27 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_561, %mul_321), kwargs = {})
#   %mul_322 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_21), kwargs = {})
#   %convert_element_type_563 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_322, torch.bfloat16), kwargs = {})
#   %add_178 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_168, %convert_element_type_563), kwargs = {})
#   %select_11 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 5), kwargs = {})
#   %mul_323 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_178, %select_11), kwargs = {})
#   %add_179 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_165, %mul_323), kwargs = {})
#   %select_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 5), kwargs = {})
#   %mul_325 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_178, %select_10), kwargs = {})
#   %view_271 : Tensor "bf16[262144, 640][640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_325, [262144, 640]), kwargs = {})
#   return %sum_36,%add_178,%add_179,%view_271
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_22 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 22, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 5368709120}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_22(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr5 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp46 = tl.load(in_ptr6 + (9))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, R0_BLOCK])
    tmp51 = tl.load(in_ptr6 + (8))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK, R0_BLOCK])
    tmp57 = tl.load(in_ptr6 + (7))
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK, R0_BLOCK])
    tmp63 = tl.load(in_ptr6 + (6))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, R0_BLOCK])
    tmp68 = tl.load(in_ptr6 + (5))
    tmp69 = tl.broadcast_to(tmp68, [XBLOCK, R0_BLOCK])
    tmp73 = tl.load(in_ptr10 + (5))
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp45 = tl.load(in_out_ptr1 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp50 = tl.load(in_ptr7 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp56 = tl.load(in_ptr8 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp62 = tl.load(in_ptr9 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
        tmp38 = 0.0015625
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
        tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp44, r0_mask)
        tl.store(in_out_ptr1 + (r0_1 + 640*x0), tmp72, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 640*x0), tmp76, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/zn/czndsw5fb2kd2gv2jeqxbgs6t7oi2eqcsjtjjvw2hvl4kgsm3v7w.py
# Topologically Sorted Source Nodes: [getitem_34, rms_norm_20], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_34 => select_10
#   rms_norm_20 => convert_element_type_138, mul_74
# Graph fragment:
#   %add_55 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_55]
#   %rsqrt_20 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_20]
#   %mm_137 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_137]
#   %add_178 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_178]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_39 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_39]
#   %select_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 5), kwargs = {})
#   %mul_325 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_178, %select_10), kwargs = {})
#   %view_274 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_137, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_578 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_274, torch.float32), kwargs = {})
#   %convert_element_type_138 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_55, torch.float32), kwargs = {})
#   %mul_74 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_138, %rsqrt_20), kwargs = {})
#   %mul_330 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_74, %convert_element_type_578), kwargs = {})
#   %sum_39 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_330, [2], True), kwargs = {})
#   %div_25 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_74, 640), kwargs = {})
#   %mul_331 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_25, %sum_39), kwargs = {})
#   %sub_28 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_578, %mul_331), kwargs = {})
#   %mul_332 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_20), kwargs = {})
#   %convert_element_type_580 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_332, torch.bfloat16), kwargs = {})
#   %add_182 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_325, %convert_element_type_580), kwargs = {})
#   return %sum_39,%add_182
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_23 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1677721600}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (5))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/ah/cahc4kmaw2ohlrjizumdsccnu4p3gd3cfeou4u7fikxtanuxmvcl.py
# Topologically Sorted Source Nodes: [x_1, rms_norm_17, getitem_28], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   getitem_28 => select_8
#   rms_norm_17 => convert_element_type_120, mul_63
#   x_1 => convert_element_type, convert_element_type_1, mul
# Graph fragment:
#   %add_47 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_47]
#   %rsqrt_17 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_17]
#   %mm_141 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_141]
#   %mm_143 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_143]
#   %mm_145 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_145]
#   %add_182 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_182]
#   %sum_42 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_42]
#   %add_191 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_191]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_178 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_178]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_46 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_46]
#   %add_57 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_57]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_324 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_178, %convert_element_type_1), kwargs = {})
#   %sum_37 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_324,), kwargs = {dtype: torch.float32})
#   %mul_326 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_178, %add_57), kwargs = {})
#   %sum_38 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_326,), kwargs = {dtype: torch.float32})
#   %view_280 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_141, [128, 2048, 640]), kwargs = {})
#   %view_283 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_143, [128, 2048, 640]), kwargs = {})
#   %add_189 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_280, %view_283), kwargs = {})
#   %view_286 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_145, [128, 2048, 640]), kwargs = {})
#   %add_190 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_189, %view_286), kwargs = {})
#   %convert_element_type_607 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_190, torch.float32), kwargs = {})
#   %convert_element_type_120 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_47, torch.float32), kwargs = {})
#   %mul_63 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_120, %rsqrt_17), kwargs = {})
#   %mul_350 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_63, %convert_element_type_607), kwargs = {})
#   %sum_42 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_350, [2], True), kwargs = {})
#   %div_28 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_63, 640), kwargs = {})
#   %mul_351 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_28, %sum_42), kwargs = {})
#   %sub_31 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_607, %mul_351), kwargs = {})
#   %mul_352 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %rsqrt_17), kwargs = {})
#   %convert_element_type_609 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_352, torch.bfloat16), kwargs = {})
#   %add_191 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_182, %convert_element_type_609), kwargs = {})
#   %mul_354 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_191, %convert_element_type_1), kwargs = {})
#   %sum_43 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_354,), kwargs = {dtype: torch.float32})
#   %select_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 4), kwargs = {})
#   %mul_355 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_191, %select_8), kwargs = {})
#   %mul_356 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_191, %add_46), kwargs = {})
#   %sum_44 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_356,), kwargs = {dtype: torch.float32})
#   %view_287 : Tensor "bf16[262144, 640][640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_355, [262144, 640]), kwargs = {})
#   return %sum_42,%add_191,%view_287,%buf235,%buf277,%buf279,%buf238
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_24 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 12, 'num_reduction': 5, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760, 'r0_': 4362076160}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp23 = tl.load(in_ptr5 + (4))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp27 = tl.load(in_ptr6 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr7 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr9 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp52 = tl.load(in_ptr10 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp3 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp14
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp22 * tmp25
    tmp29 = tmp28.to(tl.float32)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp27 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
    tmp37 = tl.where(r0_mask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None].to(tl.float32)
    tmp39 = tmp22 * tmp32
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
    tmp43 = tl.where(r0_mask, tmp41, 0)
    tmp44 = tl.sum(tmp43, 1)[:, None].to(tl.float32)
    tmp46 = tmp22 * tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, R0_BLOCK])
    tmp50 = tl.where(r0_mask, tmp48, 0)
    tmp51 = tl.sum(tmp50, 1)[:, None].to(tl.float32)
    tmp53 = tmp27 * tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, R0_BLOCK])
    tmp57 = tl.where(r0_mask, tmp55, 0)
    tmp58 = tl.sum(tmp57, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp26, r0_mask)
    tl.store(out_ptr2 + (x0), tmp38, None)
    tl.store(out_ptr3 + (x0), tmp44, None)
    tl.store(out_ptr4 + (x0), tmp51, None)
    tl.store(out_ptr5 + (x0), tmp58, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/hk/chknqvdo75h457vjtcgc7wgtjwhvlslfafn3lyvbohrrjaziwvtf.py
# Topologically Sorted Source Nodes: [getitem_28, rms_norm_16], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_28 => select_8
#   rms_norm_16 => convert_element_type_110, mul_60
# Graph fragment:
#   %add_44 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_44]
#   %rsqrt_16 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_16]
#   %mm_149 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_149]
#   %add_191 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_191]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_45 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_45]
#   %select_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 4), kwargs = {})
#   %mul_355 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_191, %select_8), kwargs = {})
#   %view_290 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_149, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_622 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_290, torch.float32), kwargs = {})
#   %convert_element_type_110 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_44, torch.float32), kwargs = {})
#   %mul_60 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_110, %rsqrt_16), kwargs = {})
#   %mul_360 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_60, %convert_element_type_622), kwargs = {})
#   %sum_45 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_360, [2], True), kwargs = {})
#   %div_29 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_60, 640), kwargs = {})
#   %mul_361 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_29, %sum_45), kwargs = {})
#   %sub_32 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_622, %mul_361), kwargs = {})
#   %mul_362 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %rsqrt_16), kwargs = {})
#   %convert_element_type_624 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_362, torch.bfloat16), kwargs = {})
#   %add_195 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_355, %convert_element_type_624), kwargs = {})
#   return %sum_45,%add_195
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_25 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1677721600}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (4))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/kl/cklqtin57tdjt5ybaqxm43rcg6wjtfwzszife2vyvjnwvpdvgvql.py
# Topologically Sorted Source Nodes: [rms_norm_13, getitem_21], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.select]
# Source node to ATen node mapping:
#   getitem_21 => select_6
#   rms_norm_13 => convert_element_type_89, mul_47
# Graph fragment:
#   %add_35 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_35]
#   %rsqrt_13 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_13]
#   %mm_153 : Tensor "bf16[262144, 32][32, 1]cuda:0" = PlaceHolder[target=mm_153]
#   %mm_155 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_155]
#   %mm_157 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_157]
#   %mm_159 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_159]
#   %add_195 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_195]
#   %sum_49 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_49]
#   %add_205 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_205]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %full_default_10 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([128, 2048, 640], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_295 : Tensor "bf16[128, 2048, 32][65536, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_153, [128, 2048, 32]), kwargs = {})
#   %slice_scatter_default_31 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_295, 2, 0, 32), kwargs = {})
#   %view_299 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_155, [128, 2048, 640]), kwargs = {})
#   %add_202 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_31, %view_299), kwargs = {})
#   %view_302 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_157, [128, 2048, 640]), kwargs = {})
#   %add_203 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_202, %view_302), kwargs = {})
#   %view_305 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_159, [128, 2048, 640]), kwargs = {})
#   %add_204 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_203, %view_305), kwargs = {})
#   %convert_element_type_660 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_204, torch.float32), kwargs = {})
#   %convert_element_type_89 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_35, torch.float32), kwargs = {})
#   %mul_47 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_89, %rsqrt_13), kwargs = {})
#   %mul_385 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_47, %convert_element_type_660), kwargs = {})
#   %sum_49 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_385, [2], True), kwargs = {})
#   %div_32 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_47, 640), kwargs = {})
#   %mul_386 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_32, %sum_49), kwargs = {})
#   %sub_36 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_660, %mul_386), kwargs = {})
#   %mul_387 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %rsqrt_13), kwargs = {})
#   %convert_element_type_662 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_387, torch.bfloat16), kwargs = {})
#   %add_205 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_195, %convert_element_type_662), kwargs = {})
#   %select_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 3), kwargs = {})
#   %mul_390 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_205, %select_6), kwargs = {})
#   %view_306 : Tensor "bf16[262144, 640][640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_390, [262144, 640]), kwargs = {})
#   return %sum_49,%add_205,%view_306
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_26 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 3355443200}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr5 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp45 = tl.load(in_ptr6 + (3))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
        tmp38 = 0.0015625
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39 * tmp19
        tmp41 = tmp34 - tmp40
        tmp42 = tmp41 * tmp2
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp21 + tmp43
        tmp47 = tmp46.to(tl.float32)
        tmp48 = tmp44 * tmp47
        tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp44, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 640*x0), tmp48, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/tj/ctj33cpqakfccakwayk6qal526w7izdj3jarcjbk325iz3miwhoc.py
# Topologically Sorted Source Nodes: [getitem_21, rms_norm_12], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_21 => select_6
#   rms_norm_12 => convert_element_type_79, mul_44
# Graph fragment:
#   %add_32 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_32]
#   %rsqrt_12 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_12]
#   %mm_163 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_163]
#   %add_205 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_205]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_52 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_52]
#   %select_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 3), kwargs = {})
#   %mul_390 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_205, %select_6), kwargs = {})
#   %view_309 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_163, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_677 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_309, torch.float32), kwargs = {})
#   %convert_element_type_79 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_32, torch.float32), kwargs = {})
#   %mul_44 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_79, %rsqrt_12), kwargs = {})
#   %mul_395 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, %convert_element_type_677), kwargs = {})
#   %sum_52 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_395, [2], True), kwargs = {})
#   %div_33 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_44, 640), kwargs = {})
#   %mul_396 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_33, %sum_52), kwargs = {})
#   %sub_37 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_677, %mul_396), kwargs = {})
#   %mul_397 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %rsqrt_12), kwargs = {})
#   %convert_element_type_679 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_397, torch.bfloat16), kwargs = {})
#   %add_209 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_390, %convert_element_type_679), kwargs = {})
#   return %sum_52,%add_209
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_27 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1677721600}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (3))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/k6/ck6ldkrrvcromicmildnzlg3dxvhp4b34zsmmu2pjayvrpkled2m.py
# Topologically Sorted Source Nodes: [x_1, rms_norm_9, getitem_15], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   getitem_15 => select_4
#   rms_norm_9 => convert_element_type_61, mul_33
#   x_1 => convert_element_type, convert_element_type_1, mul
# Graph fragment:
#   %add_24 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_24]
#   %rsqrt_9 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_9]
#   %mm_167 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_167]
#   %mm_169 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_169]
#   %mm_171 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_171]
#   %add_209 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_209]
#   %sum_55 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_55]
#   %add_218 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_218]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_205 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_205]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_34 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_34]
#   %add_23 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_23]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_389 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_205, %convert_element_type_1), kwargs = {})
#   %sum_50 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_389,), kwargs = {dtype: torch.float32})
#   %mul_391 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_205, %add_34), kwargs = {})
#   %sum_51 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_391,), kwargs = {dtype: torch.float32})
#   %view_315 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_167, [128, 2048, 640]), kwargs = {})
#   %view_318 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_169, [128, 2048, 640]), kwargs = {})
#   %add_216 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_315, %view_318), kwargs = {})
#   %view_321 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_171, [128, 2048, 640]), kwargs = {})
#   %add_217 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_216, %view_321), kwargs = {})
#   %convert_element_type_706 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_217, torch.float32), kwargs = {})
#   %convert_element_type_61 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_24, torch.float32), kwargs = {})
#   %mul_33 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_61, %rsqrt_9), kwargs = {})
#   %mul_415 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %convert_element_type_706), kwargs = {})
#   %sum_55 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_415, [2], True), kwargs = {})
#   %div_36 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_33, 640), kwargs = {})
#   %mul_416 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_36, %sum_55), kwargs = {})
#   %sub_40 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_706, %mul_416), kwargs = {})
#   %mul_417 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %rsqrt_9), kwargs = {})
#   %convert_element_type_708 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_417, torch.bfloat16), kwargs = {})
#   %add_218 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_209, %convert_element_type_708), kwargs = {})
#   %mul_419 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_218, %convert_element_type_1), kwargs = {})
#   %sum_56 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_419,), kwargs = {dtype: torch.float32})
#   %select_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 2), kwargs = {})
#   %mul_420 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_218, %select_4), kwargs = {})
#   %mul_421 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_218, %add_23), kwargs = {})
#   %sum_57 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_421,), kwargs = {dtype: torch.float32})
#   %view_322 : Tensor "bf16[262144, 640][640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_420, [262144, 640]), kwargs = {})
#   return %sum_55,%add_218,%view_322,%buf327,%buf368,%buf329,%buf370
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_28 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 12, 'num_reduction': 5, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760, 'r0_': 4362076160}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp23 = tl.load(in_ptr5 + (2))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp27 = tl.load(in_ptr6 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr7 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr9 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp52 = tl.load(in_ptr10 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp3 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp14
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp22 * tmp25
    tmp29 = tmp28.to(tl.float32)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp27 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
    tmp37 = tl.where(r0_mask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None].to(tl.float32)
    tmp39 = tmp22 * tmp32
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
    tmp43 = tl.where(r0_mask, tmp41, 0)
    tmp44 = tl.sum(tmp43, 1)[:, None].to(tl.float32)
    tmp46 = tmp27 * tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, R0_BLOCK])
    tmp50 = tl.where(r0_mask, tmp48, 0)
    tmp51 = tl.sum(tmp50, 1)[:, None].to(tl.float32)
    tmp53 = tmp22 * tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, R0_BLOCK])
    tmp57 = tl.where(r0_mask, tmp55, 0)
    tmp58 = tl.sum(tmp57, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp26, r0_mask)
    tl.store(out_ptr2 + (x0), tmp38, None)
    tl.store(out_ptr3 + (x0), tmp44, None)
    tl.store(out_ptr4 + (x0), tmp51, None)
    tl.store(out_ptr5 + (x0), tmp58, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/r4/cr4oxlw3t6c5fat7nrvnpsczebvl3r46w3lmljp2gomqppjlsi5a.py
# Topologically Sorted Source Nodes: [getitem_15, rms_norm_8], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_15 => select_4
#   rms_norm_8 => convert_element_type_51, mul_30
# Graph fragment:
#   %add_21 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_21]
#   %rsqrt_8 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_8]
#   %mm_175 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_175]
#   %add_218 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_218]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_58 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_58]
#   %select_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 2), kwargs = {})
#   %mul_420 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_218, %select_4), kwargs = {})
#   %view_325 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_175, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_721 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_325, torch.float32), kwargs = {})
#   %convert_element_type_51 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_21, torch.float32), kwargs = {})
#   %mul_30 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_51, %rsqrt_8), kwargs = {})
#   %mul_425 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %convert_element_type_721), kwargs = {})
#   %sum_58 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_425, [2], True), kwargs = {})
#   %div_37 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_30, 640), kwargs = {})
#   %mul_426 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_37, %sum_58), kwargs = {})
#   %sub_41 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_721, %mul_426), kwargs = {})
#   %mul_427 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %rsqrt_8), kwargs = {})
#   %convert_element_type_723 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_427, torch.bfloat16), kwargs = {})
#   %add_222 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_420, %convert_element_type_723), kwargs = {})
#   return %sum_58,%add_222
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1677721600}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (2))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/fd/cfd4ppiomfdnlzkgvavdt6fwqeilhgcgsf6p45lyaj2m5emoyeri.py
# Topologically Sorted Source Nodes: [linear_9, sigmoid, ve_1], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.view, aten.mul, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward]
# Source node to ATen node mapping:
#   linear_9 => view_27
#   sigmoid => sigmoid
#   ve_1 => view_25
# Graph fragment:
#   %getitem_74 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=getitem_74]
#   %embedding_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding_1]
#   %sum_61 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1310720]cuda:0" = PlaceHolder[target=sum_61]
#   %mm_9 : Tensor "bf16[262144, 5][5, 1]cuda:0" = PlaceHolder[target=mm_9]
#   %view_27 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [128, 2048, 5]), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_27,), kwargs = {})
#   %view_25 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%embedding_1, [128, 2048, 5, 128]), kwargs = {})
#   %mul_445 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_74, %view_25), kwargs = {})
#   %sum_61 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_445, [3], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_735 : Tensor "bf16[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_61, torch.bfloat16), kwargs = {})
#   %squeeze_5 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convert_element_type_735, -1), kwargs = {})
#   %mul_446 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_5, 2), kwargs = {})
#   %convert_element_type_736 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_446, torch.float32), kwargs = {})
#   %convert_element_type_737 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid, torch.float32), kwargs = {})
#   %sub_44 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_737), kwargs = {})
#   %mul_447 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_737, %sub_44), kwargs = {})
#   %mul_448 : Tensor "f32[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_736, %mul_447), kwargs = {})
#   %convert_element_type_738 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_448, torch.bfloat16), kwargs = {})
#   return %sum_61,%convert_element_type_738
triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_view_30 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_view_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2097152, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_view_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7864320, 'r0_': 671088640}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_view_30(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1310720
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


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/wt/cwthamhgr4pur4q2ipglvhhw7kbmggjb6sslzn74z257dyeaww2x.py
# Topologically Sorted Source Nodes: [getitem_29, getitem_22, getitem_16, rms_norm_5, getitem_9, getitem_8], Original ATen: [aten.slice_backward, aten.select, aten.mul, aten.add, aten.view, aten._fused_rms_norm_backward, aten._to_copy]
# Source node to ATen node mapping:
#   getitem_16 => select_5
#   getitem_22 => select_7
#   getitem_29 => select_9
#   getitem_8 => select_2
#   getitem_9 => select_3
#   rms_norm_5 => convert_element_type_30, mul_17
# Graph fragment:
#   %add_12 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_12]
#   %rsqrt_5 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_5]
#   %mm_179 : Tensor "bf16[262144, 32][32, 1]cuda:0" = PlaceHolder[target=mm_179]
#   %mm_181 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_181]
#   %mm_183 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_183]
#   %mm_185 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_185]
#   %add_222 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_222]
#   %sum_62 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_62]
#   %add_179 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_179]
#   %add_191 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_191]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %add_205 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_205]
#   %add_218 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_218]
#   %add_232 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_232]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %full_default_10 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([128, 2048, 640], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_9 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 4), kwargs = {})
#   %mul_353 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_191, %select_9), kwargs = {})
#   %add_192 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_179, %mul_353), kwargs = {})
#   %select_7 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 3), kwargs = {})
#   %mul_388 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_205, %select_7), kwargs = {})
#   %add_206 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_192, %mul_388), kwargs = {})
#   %select_5 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 2), kwargs = {})
#   %mul_418 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_218, %select_5), kwargs = {})
#   %add_219 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_206, %mul_418), kwargs = {})
#   %view_330 : Tensor "bf16[128, 2048, 32][65536, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_179, [128, 2048, 32]), kwargs = {})
#   %slice_scatter_default_40 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_330, 2, 0, 32), kwargs = {})
#   %view_334 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_181, [128, 2048, 640]), kwargs = {})
#   %add_229 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_40, %view_334), kwargs = {})
#   %view_337 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_183, [128, 2048, 640]), kwargs = {})
#   %add_230 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_229, %view_337), kwargs = {})
#   %view_340 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_185, [128, 2048, 640]), kwargs = {})
#   %add_231 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_230, %view_340), kwargs = {})
#   %convert_element_type_759 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_231, torch.float32), kwargs = {})
#   %convert_element_type_30 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %mul_17 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_30, %rsqrt_5), kwargs = {})
#   %mul_450 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %convert_element_type_759), kwargs = {})
#   %sum_62 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_450, [2], True), kwargs = {})
#   %div_40 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_17, 640), kwargs = {})
#   %mul_451 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_40, %sum_62), kwargs = {})
#   %sub_45 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_759, %mul_451), kwargs = {})
#   %mul_452 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %rsqrt_5), kwargs = {})
#   %convert_element_type_761 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_452, torch.bfloat16), kwargs = {})
#   %add_232 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_222, %convert_element_type_761), kwargs = {})
#   %select_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 1), kwargs = {})
#   %mul_453 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_232, %select_3), kwargs = {})
#   %add_233 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_219, %mul_453), kwargs = {})
#   %select_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 1), kwargs = {})
#   %mul_455 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_232, %select_2), kwargs = {})
#   %view_341 : Tensor "bf16[262144, 640][640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_455, [262144, 640]), kwargs = {})
#   return %sum_62,%add_232,%add_233,%view_341
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_31 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*fp32', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 21, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 5368709120}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_31(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr5 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp47 = tl.load(in_ptr7 + (4))
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, R0_BLOCK])
    tmp53 = tl.load(in_ptr7 + (3))
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, R0_BLOCK])
    tmp59 = tl.load(in_ptr7 + (2))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK])
    tmp64 = tl.load(in_ptr7 + (1))
    tmp65 = tl.broadcast_to(tmp64, [XBLOCK, R0_BLOCK])
    tmp69 = tl.load(in_ptr10 + (1))
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp45 = tl.load(in_out_ptr1 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp46 = tl.load(in_ptr6 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp52 = tl.load(in_ptr8 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp58 = tl.load(in_ptr9 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
        tmp38 = 0.0015625
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39 * tmp19
        tmp41 = tmp34 - tmp40
        tmp42 = tmp41 * tmp2
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp21 + tmp43
        tmp49 = tmp48.to(tl.float32)
        tmp50 = tmp46 * tmp49
        tmp51 = tmp45 + tmp50
        tmp55 = tmp54.to(tl.float32)
        tmp56 = tmp52 * tmp55
        tmp57 = tmp51 + tmp56
        tmp61 = tmp60.to(tl.float32)
        tmp62 = tmp58 * tmp61
        tmp63 = tmp57 + tmp62
        tmp66 = tmp65.to(tl.float32)
        tmp67 = tmp44 * tmp66
        tmp68 = tmp63 + tmp67
        tmp71 = tmp70.to(tl.float32)
        tmp72 = tmp44 * tmp71
        tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp44, r0_mask)
        tl.store(in_out_ptr1 + (r0_1 + 640*x0), tmp68, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 640*x0), tmp72, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/rs/crsxqp3qtvcube3rhsxl62re7d6d27qalt4qlq2b5gcbvczgr5ug.py
# Topologically Sorted Source Nodes: [getitem_8, rms_norm_4], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   getitem_8 => select_2
#   rms_norm_4 => convert_element_type_20, mul_14
# Graph fragment:
#   %add_9 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_9]
#   %rsqrt_4 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_4]
#   %mm_189 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_189]
#   %add_232 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_232]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_65 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_65]
#   %select_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 1), kwargs = {})
#   %mul_455 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_232, %select_2), kwargs = {})
#   %view_344 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_189, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_776 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_344, torch.float32), kwargs = {})
#   %convert_element_type_20 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.float32), kwargs = {})
#   %mul_14 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, %rsqrt_4), kwargs = {})
#   %mul_460 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %convert_element_type_776), kwargs = {})
#   %sum_65 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_460, [2], True), kwargs = {})
#   %div_41 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_14, 640), kwargs = {})
#   %mul_461 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_41, %sum_65), kwargs = {})
#   %sub_46 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_776, %mul_461), kwargs = {})
#   %mul_462 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %rsqrt_4), kwargs = {})
#   %convert_element_type_778 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_462, torch.bfloat16), kwargs = {})
#   %add_236 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_455, %convert_element_type_778), kwargs = {})
#   return %sum_65,%add_236
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_32 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 1677721600}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (1))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0015625
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/r5/cr5vw7rgyl465m6pghdlhgurpqnsshoihwyfh4ovs45qyceuqkph.py
# Topologically Sorted Source Nodes: [loss, x_1, linear_9, sigmoid, gate, unsqueeze, getitem_2, mul, getitem_3, mul_1, x_2, rms_norm_1], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._to_copy, aten.mul, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.view, aten.sum, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   gate => mul_18
#   getitem_2 => select
#   getitem_3 => select_1
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
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %rsqrt_1 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %mul_3 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=mul_3]
#   %mm_193 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_193]
#   %mm_195 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_195]
#   %mm_197 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_197]
#   %add_236 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_236]
#   %sum_68 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_68]
#   %add_232 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_232]
#   %add_245 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_245]
#   %add_233 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_233]
#   %add_11 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_11]
#   %primals_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %getitem_74 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=getitem_74]
#   %mm_9 : Tensor "bf16[262144, 5][5, 1]cuda:0" = PlaceHolder[target=mm_9]
#   %index_put_4 : Tensor "f32[8192, 640][640, 1]cuda:0" = PlaceHolder[target=index_put_4]
#   %sum_71 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=sum_71]
#   %index_put_5 : Tensor "f32[8192, 640][640, 1]cuda:0" = PlaceHolder[target=index_put_5]
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq : Tensor "b8[128, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "b8[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %full_default_12 : Tensor "f32[8192, 640][640, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([8192, 640], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %view_27 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [128, 2048, 5]), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_27,), kwargs = {})
#   %mul_18 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, 2), kwargs = {})
#   %unsqueeze : Tensor "bf16[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %mul_444 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_74, %unsqueeze), kwargs = {})
#   %view_331 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_444, [128, 2048, 640]), kwargs = {})
#   %convert_element_type_762 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_331, torch.float32), kwargs = {})
#   %where_17 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_7, %full_default_1, %convert_element_type_762), kwargs = {})
#   %index_put_4 : Tensor "f32[8192, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_12, [%primals_1], %where_17, True), kwargs = {})
#   %mul_454 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_232, %convert_element_type_1), kwargs = {})
#   %sum_63 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_454,), kwargs = {dtype: torch.float32})
#   %mul_456 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_232, %add_11), kwargs = {})
#   %sum_64 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_456,), kwargs = {dtype: torch.float32})
#   %view_350 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_193, [128, 2048, 640]), kwargs = {})
#   %view_353 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_195, [128, 2048, 640]), kwargs = {})
#   %add_243 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_350, %view_353), kwargs = {})
#   %view_356 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_197, [128, 2048, 640]), kwargs = {})
#   %add_244 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_243, %view_356), kwargs = {})
#   %convert_element_type_805 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_244, torch.float32), kwargs = {})
#   %select : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 0), kwargs = {})
#   %mul_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %convert_element_type_1), kwargs = {})
#   %select_1 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 0), kwargs = {})
#   %mul_2 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %convert_element_type_1), kwargs = {})
#   %add_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %mul_3 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %rsqrt_1), kwargs = {})
#   %mul_480 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %convert_element_type_805), kwargs = {})
#   %sum_68 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_480, [2], True), kwargs = {})
#   %div_44 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_3, 640), kwargs = {})
#   %mul_481 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_44, %sum_68), kwargs = {})
#   %sub_49 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_805, %mul_481), kwargs = {})
#   %mul_482 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %rsqrt_1), kwargs = {})
#   %convert_element_type_807 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_482, torch.bfloat16), kwargs = {})
#   %add_245 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_236, %convert_element_type_807), kwargs = {})
#   %mul_483 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_245, %select_1), kwargs = {})
#   %mul_484 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_245, %convert_element_type_1), kwargs = {})
#   %sum_69 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_484,), kwargs = {dtype: torch.float32})
#   %add_246 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_233, %mul_483), kwargs = {})
#   %mul_485 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_245, %select), kwargs = {})
#   %add_248 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_246, %mul_485), kwargs = {})
#   %convert_element_type_808 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_248, torch.float32), kwargs = {})
#   %mul_488 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %convert_element_type_808), kwargs = {})
#   %sum_71 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_488, [2], True), kwargs = {})
#   %div_45 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 640), kwargs = {})
#   %mul_489 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_45, %sum_71), kwargs = {})
#   %sub_50 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_808, %mul_489), kwargs = {})
#   %mul_490 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %rsqrt), kwargs = {})
#   %where_19 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_7, %full_default_1, %mul_490), kwargs = {})
#   %index_put_5 : Tensor "f32[8192, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_12, [%primals_1], %where_19, True), kwargs = {})
#   return %mul_3,%sum_68,%add_245,%buf418,%buf461,%sum_71,%buf421,%buf416,%buf467
triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_select_sigmoid_sum_unsqueeze_view_33 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_select_sigmoid_sum_unsqueeze_view_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*i64', 'in_ptr12': '*bf16', 'in_ptr13': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_select_sigmoid_sum_unsqueeze_view_33', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr6', 'out_ptr7'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 15, 'num_reduction': 5, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_select_sigmoid_sum_unsqueeze_view_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr2, out_ptr3, out_ptr5, out_ptr6, out_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr5 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr7 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr8 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp49 = tl.load(in_ptr9 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp60 = tl.load(in_ptr10 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp67 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr12 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp76 = tl.load(in_ptr13 + (5*x0 + (r0_1 // 128)), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp26 = tl.where(r0_mask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tmp29 = 0.0015625
    tmp30 = tmp16 * tmp29
    tmp31 = tmp30 * tmp27
    tmp32 = tmp22 - tmp31
    tmp33 = tmp32 * tmp15
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp28 + tmp34
    tmp37 = tmp36 * tmp7
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.where(r0_mask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None].to(tl.float32)
    tmp43 = tmp35 * tmp7
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
    tmp47 = tl.where(r0_mask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None].to(tl.float32)
    tmp50 = tmp35 * tmp11
    tmp51 = tmp49 + tmp50
    tmp52 = tmp35 * tmp2
    tmp53 = tmp51 + tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp6 * tmp54
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, R0_BLOCK])
    tmp58 = tl.where(r0_mask, tmp56, 0)
    tmp59 = tl.sum(tmp58, 1)[:, None].to(tl.float32)
    tmp61 = tmp36 * tmp60
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK, R0_BLOCK])
    tmp65 = tl.where(r0_mask, tmp63, 0)
    tmp66 = tl.sum(tmp65, 1)[:, None].to(tl.float32)
    tmp68 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp69 = tmp67 + tmp68
    tmp70 = tmp67 < 0
    tmp71 = tl.where(tmp70, tmp69, tmp67)
    tl.device_assert((0 <= tmp71) & (tmp71 < 8192), "index out of bounds: 0 <= tmp71 < 8192")
    tmp73 = tl.full([1, 1], -1, tl.int64)
    tmp74 = tmp67 == tmp73
    tmp77 = tl.sigmoid(tmp76)
    tmp78 = 2.0
    tmp79 = tmp77 * tmp78
    tmp80 = tmp75 * tmp79
    tmp81 = tmp80.to(tl.float32)
    tmp82 = 0.0
    tmp83 = tl.where(tmp74, tmp82, tmp81)
    tmp84 = tmp6 * tmp29
    tmp85 = tmp84 * tmp59
    tmp86 = tmp54 - tmp85
    tmp87 = tmp86 * tmp5
    tmp88 = tl.where(tmp74, tmp82, tmp87)
    tl.atomic_add(out_ptr6 + (tl.broadcast_to(r0_1 + 640*tmp71, [XBLOCK, R0_BLOCK])), tmp83, r0_mask, sem='relaxed')
    tl.atomic_add(out_ptr7 + (tl.broadcast_to(r0_1 + 640*tmp71, [XBLOCK, R0_BLOCK])), tmp88, r0_mask, sem='relaxed')
    tl.store(out_ptr2 + (x0), tmp42, None)
    tl.store(out_ptr3 + (x0), tmp48, None)
    tl.store(out_ptr5 + (x0), tmp66, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/qu/cqu3trccgvxbf5fnxvf4caqydxc27nlba2mmwgamqycagkbb2nmq.py
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
#   %sum_63 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_63]
#   %sum_69 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_69]
#   %full_default_13 : Tensor "f32[10][1]cuda:0"[num_users=19] = call_function[target=torch.ops.aten.full.default](args = ([10], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_11, 0, 9), kwargs = {})
#   %select_scatter_default_2 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_17, 0, 8), kwargs = {})
#   %add_139 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_2), kwargs = {})
#   %select_scatter_default_4 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_24, 0, 7), kwargs = {})
#   %add_153 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_139, %select_scatter_default_4), kwargs = {})
#   %select_scatter_default_6 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_30, 0, 6), kwargs = {})
#   %add_166 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_153, %select_scatter_default_6), kwargs = {})
#   %select_scatter_default_8 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_37, 0, 5), kwargs = {})
#   %add_180 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_166, %select_scatter_default_8), kwargs = {})
#   %select_scatter_default_10 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_43, 0, 4), kwargs = {})
#   %add_193 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_180, %select_scatter_default_10), kwargs = {})
#   %select_scatter_default_12 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_50, 0, 3), kwargs = {})
#   %add_207 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_193, %select_scatter_default_12), kwargs = {})
#   %select_scatter_default_14 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_56, 0, 2), kwargs = {})
#   %add_220 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_207, %select_scatter_default_14), kwargs = {})
#   %select_scatter_default_16 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_63, 0, 1), kwargs = {})
#   %add_234 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_220, %select_scatter_default_16), kwargs = {})
#   %select_scatter_default_18 : Tensor "f32[10][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %sum_69, 0, 0), kwargs = {})
#   %add_247 : Tensor "f32[10][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_234, %select_scatter_default_18), kwargs = {})
#   return %add_247
triton_poi_fused_add_select_backward_34 = async_compile.triton('triton_poi_fused_add_select_backward_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_select_backward_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 80}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_select_backward_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10
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
    tmp51 = tl.load(in_ptr8 + (0))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp57 = tl.load(in_ptr9 + (0))
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 9, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = 0.0
    tmp6 = tl.where(tmp2, tmp4, tmp5)
    tmp7 = tl.full([1], 8, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp11 = tl.where(tmp8, tmp10, tmp5)
    tmp12 = tmp6 + tmp11
    tmp13 = tl.full([1], 7, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp17 = tl.where(tmp14, tmp16, tmp5)
    tmp18 = tmp12 + tmp17
    tmp19 = tl.full([1], 6, tl.int32)
    tmp20 = tmp0 == tmp19
    tmp23 = tl.where(tmp20, tmp22, tmp5)
    tmp24 = tmp18 + tmp23
    tmp25 = tl.full([1], 5, tl.int32)
    tmp26 = tmp0 == tmp25
    tmp29 = tl.where(tmp26, tmp28, tmp5)
    tmp30 = tmp24 + tmp29
    tmp31 = tl.full([1], 4, tl.int32)
    tmp32 = tmp0 == tmp31
    tmp35 = tl.where(tmp32, tmp34, tmp5)
    tmp36 = tmp30 + tmp35
    tmp37 = tl.full([1], 3, tl.int32)
    tmp38 = tmp0 == tmp37
    tmp41 = tl.where(tmp38, tmp40, tmp5)
    tmp42 = tmp36 + tmp41
    tmp43 = tl.full([1], 2, tl.int32)
    tmp44 = tmp0 == tmp43
    tmp47 = tl.where(tmp44, tmp46, tmp5)
    tmp48 = tmp42 + tmp47
    tmp49 = tl.full([1], 1, tl.int32)
    tmp50 = tmp0 == tmp49
    tmp53 = tl.where(tmp50, tmp52, tmp5)
    tmp54 = tmp48 + tmp53
    tmp55 = tl.full([1], 0, tl.int32)
    tmp56 = tmp0 == tmp55
    tmp59 = tl.where(tmp56, tmp58, tmp5)
    tmp60 = tmp54 + tmp59
    tl.store(out_ptr0 + (x0), tmp60, xmask)
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
        primals_1, primals_2, primals_3, primals_5, primals_6, primals_78, embedding, rsqrt, rsqrt_1, view, view_8, cat, cat_1, rsqrt_2, convert_element_type_14, rsqrt_3, convert_element_type_16, getitem, getitem_1, add_9, rsqrt_4, view_12, mm_4, view_14, add_11, add_12, embedding_1, rsqrt_5, view_16, view_26, mm_9, add_14, cat_2, cat_3, rsqrt_6, convert_element_type_45, rsqrt_7, convert_element_type_47, getitem_4, getitem_5, add_21, rsqrt_8, view_31, mm_11, view_33, add_23, add_24, rsqrt_9, view_35, view_43, cat_4, cat_5, rsqrt_10, convert_element_type_73, rsqrt_11, convert_element_type_75, getitem_8, getitem_9, add_32, rsqrt_12, view_47, mm_17, view_49, add_34, add_35, embedding_2, rsqrt_13, view_51, view_61, mm_22, add_37, cat_6, cat_7, rsqrt_14, convert_element_type_104, rsqrt_15, convert_element_type_106, getitem_12, getitem_13, add_44, rsqrt_16, view_66, mm_24, view_68, add_46, add_47, rsqrt_17, view_70, view_78, cat_8, cat_9, rsqrt_18, convert_element_type_132, rsqrt_19, convert_element_type_134, getitem_16, getitem_17, add_55, rsqrt_20, view_82, mm_30, view_84, add_57, add_58, embedding_3, rsqrt_21, view_86, view_96, mm_35, add_60, cat_10, cat_11, rsqrt_22, convert_element_type_163, rsqrt_23, convert_element_type_165, getitem_20, getitem_21, add_67, rsqrt_24, view_101, mm_37, view_103, add_69, add_70, rsqrt_25, view_105, view_113, cat_12, cat_13, rsqrt_26, convert_element_type_191, rsqrt_27, convert_element_type_193, getitem_24, getitem_25, add_78, rsqrt_28, view_117, mm_43, view_119, add_80, add_81, embedding_4, rsqrt_29, view_121, view_131, mm_48, add_83, cat_14, cat_15, rsqrt_30, convert_element_type_222, rsqrt_31, convert_element_type_224, getitem_28, getitem_29, add_90, rsqrt_32, view_136, mm_50, view_138, add_92, add_93, rsqrt_33, view_140, view_148, cat_16, cat_17, rsqrt_34, convert_element_type_250, rsqrt_35, convert_element_type_252, getitem_32, getitem_33, add_101, rsqrt_36, view_152, mm_56, view_154, add_103, add_104, embedding_5, rsqrt_37, view_156, view_166, mm_61, add_106, cat_18, cat_19, rsqrt_38, convert_element_type_281, rsqrt_39, convert_element_type_283, getitem_36, getitem_37, add_113, rsqrt_40, view_171, mm_63, view_173, add_115, rsqrt_41, view_175, mm_65, amax, log, convert_element_type_303, permute_68, permute_72, permute_76, permute_80, permute_84, permute_88, permute_92, permute_96, permute_100, permute_104, permute_108, permute_112, permute_116, permute_120, permute_124, permute_128, permute_132, permute_136, permute_140, permute_144, permute_148, permute_152, permute_156, permute_160, permute_164, permute_168, permute_172, permute_176, permute_180, permute_184, permute_188, permute_192, permute_196, permute_200, permute_204, permute_208, permute_212, permute_216, permute_220, permute_224, permute_228, permute_232, permute_236, permute_240, permute_244, permute_248, permute_252, permute_256, permute_260, permute_264, permute_268, permute_272, permute_276, permute_280, permute_284, permute_288, permute_292, permute_296, permute_300, permute_304, permute_308, permute_312, permute_316, permute_320, permute_324, permute_328, tangents_1 = args
        args.clear()
        assert_size_stride(primals_1, (128, 2048), (2048, 1))
        assert_size_stride(primals_2, (1, 20480, 1, 64), (1310720, 64, 64, 1))
        assert_size_stride(primals_3, (1, 20480, 1, 64), (1310720, 64, 64, 1))
        assert_size_stride(primals_5, (10, ), (1, ))
        assert_size_stride(primals_6, (10, ), (1, ))
        assert_size_stride(primals_78, (128, 2048), (2048, 1))
        assert_size_stride(embedding, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(rsqrt_1, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view, (262144, 640), (640, 1))
        assert_size_stride(view_8, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_1, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_2, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_14, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_3, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_16, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_1, (128, 5, 2048), (10240, 2048, 1))
        assert_size_stride(add_9, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_4, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_12, (262144, 640), (640, 1))
        assert_size_stride(mm_4, (262144, 2560), (2560, 1))
        assert_size_stride(view_14, (262144, 2560), (2560, 1))
        assert_size_stride(add_11, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(add_12, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(embedding_1, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_5, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_16, (262144, 640), (640, 1))
        assert_size_stride(view_26, (262144, 32), (640, 1))
        assert_size_stride(mm_9, (262144, 5), (5, 1))
        assert_size_stride(add_14, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_2, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_3, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_6, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_45, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_7, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_47, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_4, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_5, (128, 5, 2048), (10240, 2048, 1))
        assert_size_stride(add_21, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_8, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_31, (262144, 640), (640, 1))
        assert_size_stride(mm_11, (262144, 2560), (2560, 1))
        assert_size_stride(view_33, (262144, 2560), (2560, 1))
        assert_size_stride(add_23, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(add_24, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_9, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_35, (262144, 640), (640, 1))
        assert_size_stride(view_43, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_4, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_5, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_10, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_73, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_11, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_75, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_8, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_9, (128, 5, 2048), (10240, 2048, 1))
        assert_size_stride(add_32, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_12, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_47, (262144, 640), (640, 1))
        assert_size_stride(mm_17, (262144, 2560), (2560, 1))
        assert_size_stride(view_49, (262144, 2560), (2560, 1))
        assert_size_stride(add_34, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(add_35, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(embedding_2, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_13, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_51, (262144, 640), (640, 1))
        assert_size_stride(view_61, (262144, 32), (640, 1))
        assert_size_stride(mm_22, (262144, 5), (5, 1))
        assert_size_stride(add_37, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_6, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_7, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_14, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_104, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_15, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_106, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_12, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_13, (128, 5, 2048), (10240, 2048, 1))
        assert_size_stride(add_44, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_16, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_66, (262144, 640), (640, 1))
        assert_size_stride(mm_24, (262144, 2560), (2560, 1))
        assert_size_stride(view_68, (262144, 2560), (2560, 1))
        assert_size_stride(add_46, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(add_47, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_17, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_70, (262144, 640), (640, 1))
        assert_size_stride(view_78, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_8, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_9, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_18, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_132, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_19, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_134, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_16, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_17, (128, 5, 2048), (10240, 2048, 1))
        assert_size_stride(add_55, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_20, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_82, (262144, 640), (640, 1))
        assert_size_stride(mm_30, (262144, 2560), (2560, 1))
        assert_size_stride(view_84, (262144, 2560), (2560, 1))
        assert_size_stride(add_57, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(add_58, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(embedding_3, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_21, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_86, (262144, 640), (640, 1))
        assert_size_stride(view_96, (262144, 32), (640, 1))
        assert_size_stride(mm_35, (262144, 5), (5, 1))
        assert_size_stride(add_60, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_10, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_11, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_22, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_163, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_23, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_165, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_20, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_21, (128, 5, 2048), (10240, 2048, 1))
        assert_size_stride(add_67, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_24, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_101, (262144, 640), (640, 1))
        assert_size_stride(mm_37, (262144, 2560), (2560, 1))
        assert_size_stride(view_103, (262144, 2560), (2560, 1))
        assert_size_stride(add_69, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(add_70, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_25, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_105, (262144, 640), (640, 1))
        assert_size_stride(view_113, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_12, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_13, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_26, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_191, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_27, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_193, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_24, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_25, (128, 5, 2048), (10240, 2048, 1))
        assert_size_stride(add_78, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_28, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_117, (262144, 640), (640, 1))
        assert_size_stride(mm_43, (262144, 2560), (2560, 1))
        assert_size_stride(view_119, (262144, 2560), (2560, 1))
        assert_size_stride(add_80, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(add_81, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(embedding_4, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_29, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_121, (262144, 640), (640, 1))
        assert_size_stride(view_131, (262144, 32), (640, 1))
        assert_size_stride(mm_48, (262144, 5), (5, 1))
        assert_size_stride(add_83, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_14, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_15, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_30, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_222, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_31, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_224, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_28, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_29, (128, 5, 2048), (10240, 2048, 1))
        assert_size_stride(add_90, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_32, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_136, (262144, 640), (640, 1))
        assert_size_stride(mm_50, (262144, 2560), (2560, 1))
        assert_size_stride(view_138, (262144, 2560), (2560, 1))
        assert_size_stride(add_92, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(add_93, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_33, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_140, (262144, 640), (640, 1))
        assert_size_stride(view_148, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_16, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_17, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_34, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_250, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_35, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_252, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_32, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_33, (128, 5, 2048), (10240, 2048, 1))
        assert_size_stride(add_101, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_36, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_152, (262144, 640), (640, 1))
        assert_size_stride(mm_56, (262144, 2560), (2560, 1))
        assert_size_stride(view_154, (262144, 2560), (2560, 1))
        assert_size_stride(add_103, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(add_104, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(embedding_5, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_37, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_156, (262144, 640), (640, 1))
        assert_size_stride(view_166, (262144, 32), (640, 1))
        assert_size_stride(mm_61, (262144, 5), (5, 1))
        assert_size_stride(add_106, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_18, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(cat_19, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_38, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_281, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(rsqrt_39, (128, 2048, 5, 1), (10240, 5, 1, 1))
        assert_size_stride(convert_element_type_283, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_36, (128, 2048, 5, 128), (1310720, 640, 128, 1))
        assert_size_stride(getitem_37, (128, 5, 2048), (10240, 2048, 1))
        assert_size_stride(add_113, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_40, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_171, (262144, 640), (640, 1))
        assert_size_stride(mm_63, (262144, 2560), (2560, 1))
        assert_size_stride(view_173, (262144, 2560), (2560, 1))
        assert_size_stride(add_115, (128, 2048, 640), (1310720, 640, 1))
        assert_size_stride(rsqrt_41, (128, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_175, (262144, 640), (640, 1))
        assert_size_stride(mm_65, (262144, 8192), (8192, 1))
        assert_size_stride(amax, (262144, 1), (1, 1))
        assert_size_stride(log, (262144, 1), (1, 1))
        assert_size_stride(convert_element_type_303, (), ())
        assert_size_stride(permute_68, (8192, 640), (640, 1))
        assert_size_stride(permute_72, (640, 2560), (2560, 1))
        assert_size_stride(permute_76, (2560, 640), (640, 1))
        assert_size_stride(permute_80, (640, 640), (640, 1))
        assert_size_stride(permute_84, (5, 32), (32, 1))
        assert_size_stride(permute_88, (640, 640), (640, 1))
        assert_size_stride(permute_92, (640, 640), (640, 1))
        assert_size_stride(permute_96, (640, 640), (640, 1))
        assert_size_stride(permute_100, (640, 2560), (2560, 1))
        assert_size_stride(permute_104, (2560, 640), (640, 1))
        assert_size_stride(permute_108, (640, 640), (640, 1))
        assert_size_stride(permute_112, (640, 640), (640, 1))
        assert_size_stride(permute_116, (640, 640), (640, 1))
        assert_size_stride(permute_120, (640, 640), (640, 1))
        assert_size_stride(permute_124, (640, 2560), (2560, 1))
        assert_size_stride(permute_128, (2560, 640), (640, 1))
        assert_size_stride(permute_132, (640, 640), (640, 1))
        assert_size_stride(permute_136, (5, 32), (32, 1))
        assert_size_stride(permute_140, (640, 640), (640, 1))
        assert_size_stride(permute_144, (640, 640), (640, 1))
        assert_size_stride(permute_148, (640, 640), (640, 1))
        assert_size_stride(permute_152, (640, 2560), (2560, 1))
        assert_size_stride(permute_156, (2560, 640), (640, 1))
        assert_size_stride(permute_160, (640, 640), (640, 1))
        assert_size_stride(permute_164, (640, 640), (640, 1))
        assert_size_stride(permute_168, (640, 640), (640, 1))
        assert_size_stride(permute_172, (640, 640), (640, 1))
        assert_size_stride(permute_176, (640, 2560), (2560, 1))
        assert_size_stride(permute_180, (2560, 640), (640, 1))
        assert_size_stride(permute_184, (640, 640), (640, 1))
        assert_size_stride(permute_188, (5, 32), (32, 1))
        assert_size_stride(permute_192, (640, 640), (640, 1))
        assert_size_stride(permute_196, (640, 640), (640, 1))
        assert_size_stride(permute_200, (640, 640), (640, 1))
        assert_size_stride(permute_204, (640, 2560), (2560, 1))
        assert_size_stride(permute_208, (2560, 640), (640, 1))
        assert_size_stride(permute_212, (640, 640), (640, 1))
        assert_size_stride(permute_216, (640, 640), (640, 1))
        assert_size_stride(permute_220, (640, 640), (640, 1))
        assert_size_stride(permute_224, (640, 640), (640, 1))
        assert_size_stride(permute_228, (640, 2560), (2560, 1))
        assert_size_stride(permute_232, (2560, 640), (640, 1))
        assert_size_stride(permute_236, (640, 640), (640, 1))
        assert_size_stride(permute_240, (5, 32), (32, 1))
        assert_size_stride(permute_244, (640, 640), (640, 1))
        assert_size_stride(permute_248, (640, 640), (640, 1))
        assert_size_stride(permute_252, (640, 640), (640, 1))
        assert_size_stride(permute_256, (640, 2560), (2560, 1))
        assert_size_stride(permute_260, (2560, 640), (640, 1))
        assert_size_stride(permute_264, (640, 640), (640, 1))
        assert_size_stride(permute_268, (640, 640), (640, 1))
        assert_size_stride(permute_272, (640, 640), (640, 1))
        assert_size_stride(permute_276, (640, 640), (640, 1))
        assert_size_stride(permute_280, (640, 2560), (2560, 1))
        assert_size_stride(permute_284, (2560, 640), (640, 1))
        assert_size_stride(permute_288, (640, 640), (640, 1))
        assert_size_stride(permute_292, (5, 32), (32, 1))
        assert_size_stride(permute_296, (640, 640), (640, 1))
        assert_size_stride(permute_300, (640, 640), (640, 1))
        assert_size_stride(permute_304, (640, 640), (640, 1))
        assert_size_stride(permute_308, (640, 2560), (2560, 1))
        assert_size_stride(permute_312, (2560, 640), (640, 1))
        assert_size_stride(permute_316, (640, 640), (640, 1))
        assert_size_stride(permute_320, (640, 640), (640, 1))
        assert_size_stride(permute_324, (640, 640), (640, 1))
        assert_size_stride(permute_328, (640, 640), (640, 1))
        assert_size_stride(tangents_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf2 = reinterpret_tensor(mm_65, (128, 2048, 8192), (16777216, 8192, 1), 0); del mm_65  # reuse
            # Topologically Sorted Source Nodes: [view_46, loss, logits, logits_1, truediv, tanh, logits_2, view_45], Original ATen: [aten.nll_loss_backward, aten.view, aten.nll_loss_forward, aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten._log_softmax, aten._log_softmax_backward_data, aten.tanh_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_view_0.run(buf2, primals_78, tangents_1, convert_element_type_303, amax, log, 262144, 8192, stream=stream0)
            del amax
            del convert_element_type_303
            del log
            del primals_78
            del tangents_1
            buf3 = empty_strided_cuda((8192, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [logits, logits_1, truediv, tanh], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.view, aten.mul, aten.tanh_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf2, (8192, 262144), (1, 8192), 0), view_175, out=buf3)
            del view_175
            buf4 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [logits, logits_1, truediv, tanh], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.view, aten.mul, aten.tanh_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf2, (262144, 8192), (8192, 1), 0), permute_68, out=buf4)
            del buf2
            del permute_68
            buf5 = empty_strided_cuda((8192, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(buf3, buf5, 5242880, stream=stream0)
            buf7 = reinterpret_tensor(buf4, (128, 2048, 640), (1310720, 640, 1), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_view_2.run(buf7, add_115, rsqrt_41, 262144, 640, stream=stream0)
            del add_115
            del rsqrt_41
            buf8 = empty_strided_cuda((640, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf7, (640, 262144), (1, 640), 0), view_173, out=buf8)
            del view_173
            buf9 = empty_strided_cuda((262144, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf7, (262144, 640), (640, 1), 0), permute_72, out=buf9)
            del permute_72
            buf10 = empty_strided_cuda((640, 2560), (2560, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf8, buf10, 1638400, stream=stream0)
            buf11 = reinterpret_tensor(mm_63, (128, 2048, 2560), (5242880, 2560, 1), 0); del mm_63  # reuse
            # Topologically Sorted Source Nodes: [x_58, relu_9, x_59], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.threshold_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf11, buf9, 671088640, stream=stream0)
            del buf9
            buf12 = reinterpret_tensor(buf8, (2560, 640), (640, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [x_58, relu_9, x_59], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.threshold_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf11, (2560, 262144), (1, 2560), 0), view_171, out=buf12)
            del view_171
            buf13 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_58, relu_9, x_59], Original ATen: [aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.threshold_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf11, (262144, 2560), (2560, 1), 0), permute_76, out=buf13)
            del permute_76
            buf14 = empty_strided_cuda((2560, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf12, buf14, 1638400, stream=stream0)
            buf16 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_40], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_5.run(buf16, add_113, rsqrt_40, buf13, 262144, 640, stream=stream0)
            del add_113
            del rsqrt_40
            buf17 = empty_strided_cuda((640, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_28, y_29], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf16, (640, 262144), (1, 640), 0), reinterpret_tensor(getitem_36, (262144, 640), (640, 1), 0), out=buf17)
            buf18 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf16, (262144, 640), (640, 1), 0), permute_80, out=buf18)
            del permute_80
            buf19 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf17, buf19, 409600, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf20 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf18, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), convert_element_type_281, convert_element_type_283, add_106, getitem_36, getitem_37, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_106
            del convert_element_type_281
            del convert_element_type_283
            del getitem_36
            del getitem_37
            buf21 = buf20[0]
            assert_size_stride(buf21, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf21, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf22 = buf20[1]
            assert_size_stride(buf22, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf22, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf23 = buf20[2]
            assert_size_stride(buf23, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf23, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf20
            buf27 = reinterpret_tensor(buf18, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf18  # reuse
            buf40 = buf27; del buf27  # reuse
            buf29 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            buf44 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [k_29, q_29, cos, sin, neg], Original ATen: [aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.slice, aten.neg, aten.add, aten.slice_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf40, buf44, cat_18, rsqrt_38, buf21, cat_19, rsqrt_39, buf22, primals_2, primals_3, 1310720, 128, stream=stream0)
            del cat_18
            del cat_19
            del rsqrt_38
            del rsqrt_39
            buf50 = empty_strided_cuda((8192, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [loss, linear_61, sigmoid_4, gate_4, unsqueeze_4], Original ATen: [aten.nll_loss_forward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8.run(buf50, 5242880, stream=stream0)
            buf32 = empty_strided_cuda((128, 2048, 5), (10240, 5, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [loss, linear_61, sigmoid_4, gate_4, unsqueeze_4, ve_9], Original ATen: [aten.nll_loss_forward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9.run(buf23, embedding_5, primals_1, mm_61, buf50, buf32, 1310720, 128, stream=stream0)
            del embedding_5
            del mm_61
            buf33 = empty_strided_cuda((8, 262144), (1, 8), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_61, sigmoid_4], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10.run(buf32, buf33, 2097152, stream=stream0)
            buf34 = empty_strided_cuda((8, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_61, sigmoid_4], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf33, view_166, out=buf34)
            del view_166
            buf35 = empty_strided_cuda((262144, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_61, sigmoid_4], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf32, (262144, 5), (5, 1), 0), permute_84, out=buf35)
            del permute_84
            buf36 = empty_strided_cuda((5, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mm_11.run(buf34, buf36, 160, stream=stream0)
            buf37 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf23, (640, 262144), (1, 640), 0), view_156, out=buf37)
            buf38 = reinterpret_tensor(buf22, (262144, 640), (640, 1), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf23, (262144, 640), (640, 1), 0), permute_88, out=buf38)
            del permute_88
            buf39 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf37, buf39, 409600, stream=stream0)
            buf41 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (640, 262144), (1, 640), 0), view_156, out=buf41)
            buf42 = reinterpret_tensor(buf23, (262144, 640), (640, 1), 0); del buf23  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (262144, 640), (640, 1), 0), permute_92, out=buf42)
            del permute_92
            buf43 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf41, buf43, 409600, stream=stream0)
            buf45 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf44, (640, 262144), (1, 640), 0), view_156, out=buf45)
            del view_156
            buf46 = reinterpret_tensor(buf40, (262144, 640), (640, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf44, (262144, 640), (640, 1), 0), permute_96, out=buf46)
            del permute_96
            buf47 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf45, buf47, 409600, stream=stream0)
            buf49 = buf16; del buf16  # reuse
            buf57 = reinterpret_tensor(buf44, (262144, 640), (640, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_37, getitem_60], Original ATen: [aten.view, aten.slice_backward, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.select]
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_12.run(buf49, add_104, rsqrt_37, buf35, buf38, buf42, buf46, primals_5, buf57, 262144, 640, stream=stream0)
            del add_104
            del rsqrt_37
            buf52 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_13.run(buf50, buf52, 5242880, stream=stream0)
            buf59 = reinterpret_tensor(buf11, (262144, 2560), (2560, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf57, permute_100, out=buf59)
            del permute_100
            buf61 = reinterpret_tensor(mm_56, (128, 2048, 2560), (5242880, 2560, 1), 0); del mm_56  # reuse
            # Topologically Sorted Source Nodes: [x_52, relu_8, x_53], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf61, buf59, 671088640, stream=stream0)
            del buf59
            buf63 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [x_52, relu_8, x_53], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf61, (262144, 2560), (2560, 1), 0), permute_104, out=buf63)
            del permute_104
            buf66 = reinterpret_tensor(buf63, (128, 2048, 640), (1310720, 640, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [getitem_60, rms_norm_36], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_14.run(buf66, add_101, rsqrt_36, buf49, primals_5, 262144, 640, stream=stream0)
            del add_101
            del rsqrt_36
            buf68 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf66, (262144, 640), (640, 1), 0), permute_108, out=buf68)
            del permute_108
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf70 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf68, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), convert_element_type_250, convert_element_type_252, view_148, getitem_32, getitem_33, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del convert_element_type_250
            del convert_element_type_252
            del getitem_33
            del view_148
            buf73 = buf70[2]
            assert_size_stride(buf73, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf73, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf82 = buf68; del buf68  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf73, (262144, 640), (640, 1), 0), permute_112, out=buf82)
            del permute_112
            buf71 = buf70[0]
            assert_size_stride(buf71, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf71, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf72 = buf70[1]
            assert_size_stride(buf72, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf72, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf70
            buf77 = reinterpret_tensor(buf38, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf38  # reuse
            buf84 = buf77; del buf77  # reuse
            buf79 = buf21; del buf21  # reuse
            buf88 = buf79; del buf79  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_26, q_26], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf84, buf88, cat_16, rsqrt_34, buf71, cat_17, rsqrt_35, buf72, primals_2, primals_3, 1310720, 128, stream=stream0)
            del cat_16
            del cat_17
            del rsqrt_34
            del rsqrt_35
            buf86 = reinterpret_tensor(buf72, (262144, 640), (640, 1), 0); del buf72  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf84, (262144, 640), (640, 1), 0), permute_116, out=buf86)
            del permute_116
            buf90 = reinterpret_tensor(buf71, (262144, 640), (640, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf88, (262144, 640), (640, 1), 0), permute_120, out=buf90)
            del permute_120
            buf93 = reinterpret_tensor(buf82, (128, 2048, 640), (1310720, 640, 1), 0); del buf82  # reuse
            buf98 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            buf53 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf94 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf55 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf96 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, rms_norm_33, getitem_54], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_15.run(buf93, add_93, rsqrt_33, buf86, buf90, buf66, primals_5, buf49, embedding, rsqrt, add_103, add_92, buf98, buf53, buf94, buf55, buf96, 262144, 640, stream=stream0)
            del add_103
            del add_92
            del add_93
            del buf86
            del buf90
            del rsqrt_33
            buf54 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf53, buf54, 1, 262144, stream=stream0)
            buf56 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf55, buf56, 1, 262144, stream=stream0)
            buf58 = reinterpret_tensor(buf12, (640, 2560), (2560, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf57, (640, 262144), (1, 640), 0), view_154, out=buf58)
            del buf57
            del view_154
            buf60 = empty_strided_cuda((640, 2560), (2560, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf58, buf60, 1638400, stream=stream0)
            buf62 = reinterpret_tensor(buf58, (2560, 640), (640, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [x_52, relu_8, x_53], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf61, (2560, 262144), (1, 2560), 0), view_152, out=buf62)
            del view_152
            buf64 = empty_strided_cuda((2560, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf62, buf64, 1638400, stream=stream0)
            buf67 = buf45; del buf45  # reuse
            # Topologically Sorted Source Nodes: [y_25, y_26], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf66, (640, 262144), (1, 640), 0), reinterpret_tensor(getitem_32, (262144, 640), (640, 1), 0), out=buf67)
            del buf66
            del getitem_32
            buf69 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf67, buf69, 409600, stream=stream0)
            buf81 = buf67; del buf67  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf73, (640, 262144), (1, 640), 0), view_140, out=buf81)
            buf83 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf81, buf83, 409600, stream=stream0)
            buf85 = buf81; del buf81  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf84, (640, 262144), (1, 640), 0), view_140, out=buf85)
            buf87 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf85, buf87, 409600, stream=stream0)
            buf89 = buf85; del buf85  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf88, (640, 262144), (1, 640), 0), view_140, out=buf89)
            del view_140
            buf91 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf89, buf91, 409600, stream=stream0)
            buf95 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf94, buf95, 1, 262144, stream=stream0)
            buf97 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf96, buf97, 1, 262144, stream=stream0)
            buf99 = reinterpret_tensor(buf62, (640, 2560), (2560, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf98, (640, 262144), (1, 640), 0), view_138, out=buf99)
            del view_138
            buf100 = reinterpret_tensor(buf61, (262144, 2560), (2560, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf98, permute_124, out=buf100)
            del permute_124
            buf101 = empty_strided_cuda((640, 2560), (2560, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf99, buf101, 1638400, stream=stream0)
            buf102 = reinterpret_tensor(mm_50, (128, 2048, 2560), (5242880, 2560, 1), 0); del mm_50  # reuse
            # Topologically Sorted Source Nodes: [x_46, relu_7, x_47], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf102, buf100, 671088640, stream=stream0)
            del buf100
            buf103 = reinterpret_tensor(buf99, (2560, 640), (640, 1), 0); del buf99  # reuse
            # Topologically Sorted Source Nodes: [x_46, relu_7, x_47], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf102, (2560, 262144), (1, 2560), 0), view_136, out=buf103)
            del view_136
            buf104 = buf98; del buf98  # reuse
            # Topologically Sorted Source Nodes: [x_46, relu_7, x_47], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf102, (262144, 2560), (2560, 1), 0), permute_128, out=buf104)
            del permute_128
            buf105 = empty_strided_cuda((2560, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf103, buf105, 1638400, stream=stream0)
            buf107 = reinterpret_tensor(buf104, (128, 2048, 640), (1310720, 640, 1), 0); del buf104  # reuse
            # Topologically Sorted Source Nodes: [getitem_54, rms_norm_32], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_17.run(buf107, add_90, rsqrt_32, buf93, primals_5, 262144, 640, stream=stream0)
            del add_90
            del rsqrt_32
            buf108 = buf89; del buf89  # reuse
            # Topologically Sorted Source Nodes: [y_22, y_23], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf107, (640, 262144), (1, 640), 0), reinterpret_tensor(getitem_28, (262144, 640), (640, 1), 0), out=buf108)
            buf109 = reinterpret_tensor(buf88, (262144, 640), (640, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf107, (262144, 640), (640, 1), 0), permute_132, out=buf109)
            del permute_132
            buf110 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf108, buf110, 409600, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf111 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf109, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), convert_element_type_222, convert_element_type_224, add_83, getitem_28, getitem_29, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_83
            del convert_element_type_222
            del convert_element_type_224
            del getitem_28
            del getitem_29
            buf112 = buf111[0]
            assert_size_stride(buf112, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf112, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf113 = buf111[1]
            assert_size_stride(buf113, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf113, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf114 = buf111[2]
            assert_size_stride(buf114, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf114, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf111
            buf118 = reinterpret_tensor(buf109, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf109  # reuse
            buf131 = buf118; del buf118  # reuse
            buf120 = buf84; del buf84  # reuse
            buf135 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_23, q_23], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf131, buf135, cat_14, rsqrt_30, buf112, cat_15, rsqrt_31, buf113, primals_2, primals_3, 1310720, 128, stream=stream0)
            del cat_14
            del cat_15
            del rsqrt_30
            del rsqrt_31
            buf141 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_48, sigmoid_3, gate_3, unsqueeze_3], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8.run(buf141, 5242880, stream=stream0)
            buf123 = buf32; del buf32  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_48, sigmoid_3, gate_3, unsqueeze_3, ve_7], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9.run(buf114, embedding_4, primals_1, mm_48, buf141, buf123, 1310720, 128, stream=stream0)
            del embedding_4
            del mm_48
            buf124 = buf33; del buf33  # reuse
            # Topologically Sorted Source Nodes: [linear_48, sigmoid_3], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10.run(buf123, buf124, 2097152, stream=stream0)
            buf125 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [linear_48, sigmoid_3], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf124, view_131, out=buf125)
            del view_131
            buf126 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [linear_48, sigmoid_3], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf123, (262144, 5), (5, 1), 0), permute_136, out=buf126)
            del permute_136
            buf127 = empty_strided_cuda((5, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mm_11.run(buf125, buf127, 160, stream=stream0)
            buf128 = buf108; del buf108  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf114, (640, 262144), (1, 640), 0), view_121, out=buf128)
            buf129 = reinterpret_tensor(buf113, (262144, 640), (640, 1), 0); del buf113  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf114, (262144, 640), (640, 1), 0), permute_140, out=buf129)
            del permute_140
            buf130 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf128, buf130, 409600, stream=stream0)
            buf132 = buf128; del buf128  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf131, (640, 262144), (1, 640), 0), view_121, out=buf132)
            buf133 = reinterpret_tensor(buf114, (262144, 640), (640, 1), 0); del buf114  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf131, (262144, 640), (640, 1), 0), permute_144, out=buf133)
            del permute_144
            buf134 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf132, buf134, 409600, stream=stream0)
            buf136 = buf132; del buf132  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf135, (640, 262144), (1, 640), 0), view_121, out=buf136)
            del view_121
            buf137 = reinterpret_tensor(buf131, (262144, 640), (640, 1), 0); del buf131  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf135, (262144, 640), (640, 1), 0), permute_148, out=buf137)
            del permute_148
            buf138 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf136, buf138, 409600, stream=stream0)
            buf140 = buf107; del buf107  # reuse
            buf148 = reinterpret_tensor(buf135, (262144, 640), (640, 1), 0); del buf135  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_29, getitem_47], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.select]
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_18.run(buf140, add_81, rsqrt_29, buf126, buf129, buf133, buf137, primals_5, buf148, 262144, 640, stream=stream0)
            del add_81
            del rsqrt_29
            buf143 = empty_strided_cuda((8192, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_13.run(buf141, buf143, 5242880, stream=stream0)
            buf150 = reinterpret_tensor(buf102, (262144, 2560), (2560, 1), 0); del buf102  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf148, permute_152, out=buf150)
            del permute_152
            buf152 = reinterpret_tensor(mm_43, (128, 2048, 2560), (5242880, 2560, 1), 0); del mm_43  # reuse
            # Topologically Sorted Source Nodes: [x_40, relu_6, x_41], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf152, buf150, 671088640, stream=stream0)
            del buf150
            buf154 = buf137; del buf137  # reuse
            # Topologically Sorted Source Nodes: [x_40, relu_6, x_41], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf152, (262144, 2560), (2560, 1), 0), permute_156, out=buf154)
            del permute_156
            buf157 = reinterpret_tensor(buf154, (128, 2048, 640), (1310720, 640, 1), 0); del buf154  # reuse
            # Topologically Sorted Source Nodes: [getitem_47, rms_norm_28], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_19.run(buf157, add_78, rsqrt_28, buf140, primals_5, 262144, 640, stream=stream0)
            del add_78
            del rsqrt_28
            buf159 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf157, (262144, 640), (640, 1), 0), permute_160, out=buf159)
            del permute_160
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf161 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf159, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), convert_element_type_191, convert_element_type_193, view_113, getitem_24, getitem_25, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del convert_element_type_191
            del convert_element_type_193
            del getitem_25
            del view_113
            buf164 = buf161[2]
            assert_size_stride(buf164, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf164, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf173 = buf159; del buf159  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf164, (262144, 640), (640, 1), 0), permute_164, out=buf173)
            del permute_164
            buf162 = buf161[0]
            assert_size_stride(buf162, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf162, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf163 = buf161[1]
            assert_size_stride(buf163, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf163, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf161
            buf168 = reinterpret_tensor(buf129, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf129  # reuse
            buf175 = buf168; del buf168  # reuse
            buf170 = buf112; del buf112  # reuse
            buf179 = buf170; del buf170  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_20, q_20], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf175, buf179, cat_12, rsqrt_26, buf162, cat_13, rsqrt_27, buf163, primals_2, primals_3, 1310720, 128, stream=stream0)
            del cat_12
            del cat_13
            del rsqrt_26
            del rsqrt_27
            buf177 = reinterpret_tensor(buf163, (262144, 640), (640, 1), 0); del buf163  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf175, (262144, 640), (640, 1), 0), permute_168, out=buf177)
            del permute_168
            buf181 = reinterpret_tensor(buf162, (262144, 640), (640, 1), 0); del buf162  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf179, (262144, 640), (640, 1), 0), permute_172, out=buf181)
            del permute_172
            buf184 = reinterpret_tensor(buf173, (128, 2048, 640), (1310720, 640, 1), 0); del buf173  # reuse
            buf189 = reinterpret_tensor(buf73, (262144, 640), (640, 1), 0); del buf73  # reuse
            buf144 = buf96; del buf96  # reuse
            buf185 = buf94; del buf94  # reuse
            buf146 = buf55; del buf55  # reuse
            buf187 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [x_1, rms_norm_25, getitem_41], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_20.run(buf184, add_70, rsqrt_25, buf177, buf181, buf157, primals_5, buf140, embedding, rsqrt, add_80, add_69, buf189, buf144, buf185, buf146, buf187, 262144, 640, stream=stream0)
            del add_69
            del add_70
            del add_80
            del buf177
            del buf181
            del rsqrt_25
            buf145 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf144, buf145, 1, 262144, stream=stream0)
            buf147 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf146, buf147, 1, 262144, stream=stream0)
            buf149 = reinterpret_tensor(buf103, (640, 2560), (2560, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf148, (640, 262144), (1, 640), 0), view_119, out=buf149)
            del buf148
            del view_119
            buf151 = empty_strided_cuda((640, 2560), (2560, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf149, buf151, 1638400, stream=stream0)
            buf153 = reinterpret_tensor(buf149, (2560, 640), (640, 1), 0); del buf149  # reuse
            # Topologically Sorted Source Nodes: [x_40, relu_6, x_41], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf152, (2560, 262144), (1, 2560), 0), view_117, out=buf153)
            del view_117
            buf155 = empty_strided_cuda((2560, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf153, buf155, 1638400, stream=stream0)
            buf158 = buf136; del buf136  # reuse
            # Topologically Sorted Source Nodes: [y_19, y_20], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf157, (640, 262144), (1, 640), 0), reinterpret_tensor(getitem_24, (262144, 640), (640, 1), 0), out=buf158)
            del buf157
            del getitem_24
            buf160 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf158, buf160, 409600, stream=stream0)
            buf172 = buf158; del buf158  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf164, (640, 262144), (1, 640), 0), view_105, out=buf172)
            del buf164
            buf174 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf172, buf174, 409600, stream=stream0)
            buf176 = buf172; del buf172  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf175, (640, 262144), (1, 640), 0), view_105, out=buf176)
            buf178 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf176, buf178, 409600, stream=stream0)
            buf180 = buf176; del buf176  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf179, (640, 262144), (1, 640), 0), view_105, out=buf180)
            del view_105
            buf182 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf180, buf182, 409600, stream=stream0)
            buf186 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf185, buf186, 1, 262144, stream=stream0)
            buf188 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf187, buf188, 1, 262144, stream=stream0)
            buf190 = reinterpret_tensor(buf153, (640, 2560), (2560, 1), 0); del buf153  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf189, (640, 262144), (1, 640), 0), view_103, out=buf190)
            del view_103
            buf191 = reinterpret_tensor(buf152, (262144, 2560), (2560, 1), 0); del buf152  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf189, permute_176, out=buf191)
            del permute_176
            buf192 = empty_strided_cuda((640, 2560), (2560, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf190, buf192, 1638400, stream=stream0)
            buf193 = reinterpret_tensor(mm_37, (128, 2048, 2560), (5242880, 2560, 1), 0); del mm_37  # reuse
            # Topologically Sorted Source Nodes: [x_34, relu_5, x_35], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf193, buf191, 671088640, stream=stream0)
            del buf191
            buf194 = reinterpret_tensor(buf190, (2560, 640), (640, 1), 0); del buf190  # reuse
            # Topologically Sorted Source Nodes: [x_34, relu_5, x_35], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf193, (2560, 262144), (1, 2560), 0), view_101, out=buf194)
            del view_101
            buf195 = buf189; del buf189  # reuse
            # Topologically Sorted Source Nodes: [x_34, relu_5, x_35], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf193, (262144, 2560), (2560, 1), 0), permute_180, out=buf195)
            del permute_180
            buf196 = empty_strided_cuda((2560, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf194, buf196, 1638400, stream=stream0)
            buf198 = reinterpret_tensor(buf195, (128, 2048, 640), (1310720, 640, 1), 0); del buf195  # reuse
            # Topologically Sorted Source Nodes: [getitem_41, rms_norm_24], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_21.run(buf198, add_67, rsqrt_24, buf184, primals_5, 262144, 640, stream=stream0)
            del add_67
            del rsqrt_24
            buf199 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [y_16, y_17], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf198, (640, 262144), (1, 640), 0), reinterpret_tensor(getitem_20, (262144, 640), (640, 1), 0), out=buf199)
            buf200 = reinterpret_tensor(buf179, (262144, 640), (640, 1), 0); del buf179  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf198, (262144, 640), (640, 1), 0), permute_184, out=buf200)
            del permute_184
            buf201 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf199, buf201, 409600, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf202 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf200, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), convert_element_type_163, convert_element_type_165, add_60, getitem_20, getitem_21, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_60
            del convert_element_type_163
            del convert_element_type_165
            del getitem_20
            del getitem_21
            buf203 = buf202[0]
            assert_size_stride(buf203, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf203, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf204 = buf202[1]
            assert_size_stride(buf204, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf204, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf205 = buf202[2]
            assert_size_stride(buf205, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf205, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf202
            buf209 = reinterpret_tensor(buf200, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf200  # reuse
            buf222 = buf209; del buf209  # reuse
            buf211 = buf175; del buf175  # reuse
            buf226 = buf211; del buf211  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_17, q_17], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf222, buf226, cat_10, rsqrt_22, buf203, cat_11, rsqrt_23, buf204, primals_2, primals_3, 1310720, 128, stream=stream0)
            del buf203
            del cat_10
            del cat_11
            del rsqrt_22
            del rsqrt_23
            buf232 = buf141; del buf141  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_35, sigmoid_2, gate_2, unsqueeze_2], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8.run(buf232, 5242880, stream=stream0)
            buf214 = buf123; del buf123  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_35, sigmoid_2, gate_2, unsqueeze_2, ve_5], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9.run(buf205, embedding_3, primals_1, mm_35, buf232, buf214, 1310720, 128, stream=stream0)
            del embedding_3
            del mm_35
            buf215 = buf124; del buf124  # reuse
            # Topologically Sorted Source Nodes: [linear_35, sigmoid_2], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10.run(buf214, buf215, 2097152, stream=stream0)
            buf216 = buf125; del buf125  # reuse
            # Topologically Sorted Source Nodes: [linear_35, sigmoid_2], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf215, view_96, out=buf216)
            del view_96
            buf217 = buf126; del buf126  # reuse
            # Topologically Sorted Source Nodes: [linear_35, sigmoid_2], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf214, (262144, 5), (5, 1), 0), permute_188, out=buf217)
            del permute_188
            buf218 = empty_strided_cuda((5, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mm_11.run(buf216, buf218, 160, stream=stream0)
            buf219 = buf199; del buf199  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf205, (640, 262144), (1, 640), 0), view_86, out=buf219)
            buf220 = reinterpret_tensor(buf204, (262144, 640), (640, 1), 0); del buf204  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf205, (262144, 640), (640, 1), 0), permute_192, out=buf220)
            del permute_192
            buf221 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf219, buf221, 409600, stream=stream0)
            buf223 = buf219; del buf219  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf222, (640, 262144), (1, 640), 0), view_86, out=buf223)
            buf224 = reinterpret_tensor(buf205, (262144, 640), (640, 1), 0); del buf205  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf222, (262144, 640), (640, 1), 0), permute_196, out=buf224)
            del permute_196
            buf225 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf223, buf225, 409600, stream=stream0)
            buf227 = buf223; del buf223  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf226, (640, 262144), (1, 640), 0), view_86, out=buf227)
            del view_86
            buf228 = reinterpret_tensor(buf222, (262144, 640), (640, 1), 0); del buf222  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf226, (262144, 640), (640, 1), 0), permute_200, out=buf228)
            del permute_200
            buf229 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf227, buf229, 409600, stream=stream0)
            buf231 = buf198; del buf198  # reuse
            buf237 = buf49; del buf49  # reuse
            buf240 = reinterpret_tensor(buf226, (262144, 640), (640, 1), 0); del buf226  # reuse
            # Topologically Sorted Source Nodes: [getitem_61, getitem_55, getitem_48, getitem_42, rms_norm_21, getitem_35, getitem_34], Original ATen: [aten.slice_backward, aten.select, aten.mul, aten.add, aten.view, aten._fused_rms_norm_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_22.run(buf231, buf237, add_58, rsqrt_21, buf217, buf220, buf224, buf228, primals_6, buf93, buf140, buf184, primals_5, buf240, 262144, 640, stream=stream0)
            del add_58
            del buf140
            del rsqrt_21
            buf234 = empty_strided_cuda((8192, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_13.run(buf232, buf234, 5242880, stream=stream0)
            buf242 = reinterpret_tensor(buf193, (262144, 2560), (2560, 1), 0); del buf193  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf240, permute_204, out=buf242)
            del permute_204
            buf244 = reinterpret_tensor(mm_30, (128, 2048, 2560), (5242880, 2560, 1), 0); del mm_30  # reuse
            # Topologically Sorted Source Nodes: [x_28, relu_4, x_29], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf244, buf242, 671088640, stream=stream0)
            del buf242
            buf246 = reinterpret_tensor(buf93, (262144, 640), (640, 1), 0); del buf93  # reuse
            # Topologically Sorted Source Nodes: [x_28, relu_4, x_29], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf244, (262144, 2560), (2560, 1), 0), permute_208, out=buf246)
            del permute_208
            buf249 = reinterpret_tensor(buf246, (128, 2048, 640), (1310720, 640, 1), 0); del buf246  # reuse
            # Topologically Sorted Source Nodes: [getitem_34, rms_norm_20], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_23.run(buf249, add_55, rsqrt_20, buf231, primals_5, 262144, 640, stream=stream0)
            del add_55
            del rsqrt_20
            buf251 = buf228; del buf228  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf249, (262144, 640), (640, 1), 0), permute_212, out=buf251)
            del permute_212
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf253 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf251, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), convert_element_type_132, convert_element_type_134, view_78, getitem_16, getitem_17, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del convert_element_type_132
            del convert_element_type_134
            del getitem_17
            del view_78
            buf256 = buf253[2]
            assert_size_stride(buf256, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf256, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf265 = buf251; del buf251  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf256, (262144, 640), (640, 1), 0), permute_216, out=buf265)
            del permute_216
            buf254 = buf253[0]
            assert_size_stride(buf254, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf254, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf255 = buf253[1]
            assert_size_stride(buf255, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf255, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf253
            buf260 = reinterpret_tensor(buf224, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf224  # reuse
            buf267 = buf260; del buf260  # reuse
            buf262 = reinterpret_tensor(buf220, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf220  # reuse
            buf271 = buf262; del buf262  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_14, q_14], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf267, buf271, cat_8, rsqrt_18, buf254, cat_9, rsqrt_19, buf255, primals_2, primals_3, 1310720, 128, stream=stream0)
            del cat_8
            del cat_9
            del rsqrt_18
            del rsqrt_19
            buf269 = reinterpret_tensor(buf255, (262144, 640), (640, 1), 0); del buf255  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf267, (262144, 640), (640, 1), 0), permute_220, out=buf269)
            del permute_220
            buf273 = reinterpret_tensor(buf254, (262144, 640), (640, 1), 0); del buf254  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf271, (262144, 640), (640, 1), 0), permute_224, out=buf273)
            del permute_224
            buf276 = reinterpret_tensor(buf265, (128, 2048, 640), (1310720, 640, 1), 0); del buf265  # reuse
            buf281 = reinterpret_tensor(buf184, (262144, 640), (640, 1), 0); del buf184  # reuse
            buf235 = buf187; del buf187  # reuse
            buf277 = buf185; del buf185  # reuse
            buf279 = buf146; del buf146  # reuse
            buf238 = buf144; del buf144  # reuse
            # Topologically Sorted Source Nodes: [x_1, rms_norm_17, getitem_28], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_24.run(buf276, add_47, rsqrt_17, buf269, buf273, buf249, primals_5, buf231, embedding, rsqrt, add_46, add_57, buf281, buf235, buf277, buf279, buf238, 262144, 640, stream=stream0)
            del add_46
            del add_47
            del add_57
            del buf231
            del buf269
            del buf273
            del rsqrt_17
            buf236 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf235, buf236, 1, 262144, stream=stream0)
            buf239 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf238, buf239, 1, 262144, stream=stream0)
            buf241 = reinterpret_tensor(buf194, (640, 2560), (2560, 1), 0); del buf194  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf240, (640, 262144), (1, 640), 0), view_84, out=buf241)
            del buf240
            del view_84
            buf243 = empty_strided_cuda((640, 2560), (2560, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf241, buf243, 1638400, stream=stream0)
            buf245 = reinterpret_tensor(buf241, (2560, 640), (640, 1), 0); del buf241  # reuse
            # Topologically Sorted Source Nodes: [x_28, relu_4, x_29], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf244, (2560, 262144), (1, 2560), 0), view_82, out=buf245)
            del view_82
            buf247 = empty_strided_cuda((2560, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf245, buf247, 1638400, stream=stream0)
            buf250 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [y_13, y_14], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf249, (640, 262144), (1, 640), 0), reinterpret_tensor(getitem_16, (262144, 640), (640, 1), 0), out=buf250)
            del buf249
            del getitem_16
            buf252 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf250, buf252, 409600, stream=stream0)
            buf264 = buf250; del buf250  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf256, (640, 262144), (1, 640), 0), view_70, out=buf264)
            buf266 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf264, buf266, 409600, stream=stream0)
            buf268 = buf264; del buf264  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf267, (640, 262144), (1, 640), 0), view_70, out=buf268)
            buf270 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf268, buf270, 409600, stream=stream0)
            buf272 = buf268; del buf268  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf271, (640, 262144), (1, 640), 0), view_70, out=buf272)
            del view_70
            buf274 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf272, buf274, 409600, stream=stream0)
            buf278 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf277, buf278, 1, 262144, stream=stream0)
            buf280 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf279, buf280, 1, 262144, stream=stream0)
            buf282 = reinterpret_tensor(buf245, (640, 2560), (2560, 1), 0); del buf245  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf281, (640, 262144), (1, 640), 0), view_68, out=buf282)
            del view_68
            buf283 = reinterpret_tensor(buf244, (262144, 2560), (2560, 1), 0); del buf244  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf281, permute_228, out=buf283)
            del permute_228
            buf284 = empty_strided_cuda((640, 2560), (2560, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf282, buf284, 1638400, stream=stream0)
            buf285 = reinterpret_tensor(mm_24, (128, 2048, 2560), (5242880, 2560, 1), 0); del mm_24  # reuse
            # Topologically Sorted Source Nodes: [x_22, relu_3, x_23], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf285, buf283, 671088640, stream=stream0)
            del buf283
            buf286 = reinterpret_tensor(buf282, (2560, 640), (640, 1), 0); del buf282  # reuse
            # Topologically Sorted Source Nodes: [x_22, relu_3, x_23], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf285, (2560, 262144), (1, 2560), 0), view_66, out=buf286)
            del view_66
            buf287 = buf281; del buf281  # reuse
            # Topologically Sorted Source Nodes: [x_22, relu_3, x_23], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf285, (262144, 2560), (2560, 1), 0), permute_232, out=buf287)
            del permute_232
            buf288 = empty_strided_cuda((2560, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf286, buf288, 1638400, stream=stream0)
            buf290 = reinterpret_tensor(buf287, (128, 2048, 640), (1310720, 640, 1), 0); del buf287  # reuse
            # Topologically Sorted Source Nodes: [getitem_28, rms_norm_16], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_25.run(buf290, add_44, rsqrt_16, buf276, primals_5, 262144, 640, stream=stream0)
            del add_44
            del rsqrt_16
            buf291 = buf272; del buf272  # reuse
            # Topologically Sorted Source Nodes: [y_10, y_11], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf290, (640, 262144), (1, 640), 0), reinterpret_tensor(getitem_12, (262144, 640), (640, 1), 0), out=buf291)
            buf292 = reinterpret_tensor(buf271, (262144, 640), (640, 1), 0); del buf271  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf290, (262144, 640), (640, 1), 0), permute_236, out=buf292)
            del permute_236
            buf293 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf291, buf293, 409600, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf294 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf292, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), convert_element_type_104, convert_element_type_106, add_37, getitem_12, getitem_13, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_37
            del convert_element_type_104
            del convert_element_type_106
            del getitem_12
            del getitem_13
            buf295 = buf294[0]
            assert_size_stride(buf295, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf295, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf296 = buf294[1]
            assert_size_stride(buf296, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf296, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf297 = buf294[2]
            assert_size_stride(buf297, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf297, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf294
            buf301 = reinterpret_tensor(buf292, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf292  # reuse
            buf314 = buf301; del buf301  # reuse
            buf303 = buf267; del buf267  # reuse
            buf318 = buf303; del buf303  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_11, q_11], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf314, buf318, cat_6, rsqrt_14, buf295, cat_7, rsqrt_15, buf296, primals_2, primals_3, 1310720, 128, stream=stream0)
            del cat_6
            del cat_7
            del rsqrt_14
            del rsqrt_15
            buf324 = buf232; del buf232  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_22, sigmoid_1, gate_1, unsqueeze_1], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8.run(buf324, 5242880, stream=stream0)
            buf306 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_22, sigmoid_1, gate_1, unsqueeze_1, ve_3], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9.run(buf297, embedding_2, primals_1, mm_22, buf324, buf306, 1310720, 128, stream=stream0)
            del embedding_2
            del mm_22
            buf307 = buf215; del buf215  # reuse
            # Topologically Sorted Source Nodes: [linear_22, sigmoid_1], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10.run(buf306, buf307, 2097152, stream=stream0)
            buf308 = buf216; del buf216  # reuse
            # Topologically Sorted Source Nodes: [linear_22, sigmoid_1], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf307, view_61, out=buf308)
            del view_61
            buf309 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [linear_22, sigmoid_1], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf306, (262144, 5), (5, 1), 0), permute_240, out=buf309)
            del permute_240
            buf310 = empty_strided_cuda((5, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mm_11.run(buf308, buf310, 160, stream=stream0)
            buf311 = buf291; del buf291  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf297, (640, 262144), (1, 640), 0), view_51, out=buf311)
            buf312 = reinterpret_tensor(buf296, (262144, 640), (640, 1), 0); del buf296  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf297, (262144, 640), (640, 1), 0), permute_244, out=buf312)
            del permute_244
            buf313 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf311, buf313, 409600, stream=stream0)
            buf315 = buf311; del buf311  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf314, (640, 262144), (1, 640), 0), view_51, out=buf315)
            buf316 = reinterpret_tensor(buf297, (262144, 640), (640, 1), 0); del buf297  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf314, (262144, 640), (640, 1), 0), permute_248, out=buf316)
            del permute_248
            buf317 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf315, buf317, 409600, stream=stream0)
            buf319 = buf315; del buf315  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf318, (640, 262144), (1, 640), 0), view_51, out=buf319)
            del view_51
            buf320 = reinterpret_tensor(buf314, (262144, 640), (640, 1), 0); del buf314  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf318, (262144, 640), (640, 1), 0), permute_252, out=buf320)
            del permute_252
            buf321 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf319, buf321, 409600, stream=stream0)
            buf323 = buf290; del buf290  # reuse
            buf331 = reinterpret_tensor(buf318, (262144, 640), (640, 1), 0); del buf318  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_13, getitem_21], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.select]
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_26.run(buf323, add_35, rsqrt_13, buf309, buf312, buf316, buf320, primals_5, buf331, 262144, 640, stream=stream0)
            del add_35
            del rsqrt_13
            buf326 = empty_strided_cuda((8192, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_13.run(buf324, buf326, 5242880, stream=stream0)
            buf333 = reinterpret_tensor(buf285, (262144, 2560), (2560, 1), 0); del buf285  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf331, permute_256, out=buf333)
            del permute_256
            buf335 = reinterpret_tensor(mm_17, (128, 2048, 2560), (5242880, 2560, 1), 0); del mm_17  # reuse
            # Topologically Sorted Source Nodes: [x_16, relu_2, x_17], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf335, buf333, 671088640, stream=stream0)
            del buf333
            buf337 = buf320; del buf320  # reuse
            # Topologically Sorted Source Nodes: [x_16, relu_2, x_17], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf335, (262144, 2560), (2560, 1), 0), permute_260, out=buf337)
            del permute_260
            buf340 = reinterpret_tensor(buf337, (128, 2048, 640), (1310720, 640, 1), 0); del buf337  # reuse
            # Topologically Sorted Source Nodes: [getitem_21, rms_norm_12], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_27.run(buf340, add_32, rsqrt_12, buf323, primals_5, 262144, 640, stream=stream0)
            del add_32
            del rsqrt_12
            buf342 = buf316; del buf316  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf340, (262144, 640), (640, 1), 0), permute_264, out=buf342)
            del permute_264
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf344 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf342, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), convert_element_type_73, convert_element_type_75, view_43, getitem_8, getitem_9, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del convert_element_type_73
            del convert_element_type_75
            del getitem_9
            del view_43
            buf347 = buf344[2]
            assert_size_stride(buf347, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf347, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf356 = buf342; del buf342  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf347, (262144, 640), (640, 1), 0), permute_268, out=buf356)
            del permute_268
            buf345 = buf344[0]
            assert_size_stride(buf345, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf345, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf346 = buf344[1]
            assert_size_stride(buf346, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf346, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf344
            buf351 = reinterpret_tensor(buf312, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf312  # reuse
            buf358 = buf351; del buf351  # reuse
            buf353 = buf295; del buf295  # reuse
            buf362 = buf353; del buf353  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_8, q_8], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf358, buf362, cat_4, rsqrt_10, buf345, cat_5, rsqrt_11, buf346, primals_2, primals_3, 1310720, 128, stream=stream0)
            del cat_4
            del cat_5
            del rsqrt_10
            del rsqrt_11
            buf360 = reinterpret_tensor(buf346, (262144, 640), (640, 1), 0); del buf346  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf358, (262144, 640), (640, 1), 0), permute_272, out=buf360)
            del permute_272
            buf364 = reinterpret_tensor(buf345, (262144, 640), (640, 1), 0); del buf345  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf362, (262144, 640), (640, 1), 0), permute_276, out=buf364)
            del permute_276
            buf367 = reinterpret_tensor(buf356, (128, 2048, 640), (1310720, 640, 1), 0); del buf356  # reuse
            buf372 = reinterpret_tensor(buf256, (262144, 640), (640, 1), 0); del buf256  # reuse
            buf327 = buf279; del buf279  # reuse
            buf368 = buf277; del buf277  # reuse
            buf329 = buf238; del buf238  # reuse
            buf370 = buf235; del buf235  # reuse
            # Topologically Sorted Source Nodes: [x_1, rms_norm_9, getitem_15], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_28.run(buf367, add_24, rsqrt_9, buf360, buf364, buf340, primals_5, buf323, embedding, rsqrt, add_34, add_23, buf372, buf327, buf368, buf329, buf370, 262144, 640, stream=stream0)
            del add_23
            del add_24
            del add_34
            del buf360
            del buf364
            del rsqrt_9
            buf328 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf327, buf328, 1, 262144, stream=stream0)
            del buf327
            buf330 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf329, buf330, 1, 262144, stream=stream0)
            buf332 = reinterpret_tensor(buf286, (640, 2560), (2560, 1), 0); del buf286  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf331, (640, 262144), (1, 640), 0), view_49, out=buf332)
            del buf331
            del view_49
            buf334 = empty_strided_cuda((640, 2560), (2560, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf332, buf334, 1638400, stream=stream0)
            buf336 = reinterpret_tensor(buf332, (2560, 640), (640, 1), 0); del buf332  # reuse
            # Topologically Sorted Source Nodes: [x_16, relu_2, x_17], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf335, (2560, 262144), (1, 2560), 0), view_47, out=buf336)
            del view_47
            buf338 = empty_strided_cuda((2560, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf336, buf338, 1638400, stream=stream0)
            buf341 = buf319; del buf319  # reuse
            # Topologically Sorted Source Nodes: [y_7, y_8], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf340, (640, 262144), (1, 640), 0), reinterpret_tensor(getitem_8, (262144, 640), (640, 1), 0), out=buf341)
            del buf340
            del getitem_8
            buf343 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf341, buf343, 409600, stream=stream0)
            buf355 = buf341; del buf341  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf347, (640, 262144), (1, 640), 0), view_35, out=buf355)
            del buf347
            buf357 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf355, buf357, 409600, stream=stream0)
            buf359 = buf355; del buf355  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf358, (640, 262144), (1, 640), 0), view_35, out=buf359)
            buf361 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf359, buf361, 409600, stream=stream0)
            buf363 = buf359; del buf359  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf362, (640, 262144), (1, 640), 0), view_35, out=buf363)
            del view_35
            buf365 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf363, buf365, 409600, stream=stream0)
            buf369 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf368, buf369, 1, 262144, stream=stream0)
            buf371 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf370, buf371, 1, 262144, stream=stream0)
            buf373 = reinterpret_tensor(buf336, (640, 2560), (2560, 1), 0); del buf336  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf372, (640, 262144), (1, 640), 0), view_33, out=buf373)
            del view_33
            buf374 = reinterpret_tensor(buf335, (262144, 2560), (2560, 1), 0); del buf335  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf372, permute_280, out=buf374)
            del permute_280
            buf375 = empty_strided_cuda((640, 2560), (2560, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf373, buf375, 1638400, stream=stream0)
            buf376 = reinterpret_tensor(mm_11, (128, 2048, 2560), (5242880, 2560, 1), 0); del mm_11  # reuse
            # Topologically Sorted Source Nodes: [x_10, relu_1, x_11], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf376, buf374, 671088640, stream=stream0)
            del buf374
            buf377 = reinterpret_tensor(buf373, (2560, 640), (640, 1), 0); del buf373  # reuse
            # Topologically Sorted Source Nodes: [x_10, relu_1, x_11], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf376, (2560, 262144), (1, 2560), 0), view_31, out=buf377)
            del view_31
            buf378 = buf372; del buf372  # reuse
            # Topologically Sorted Source Nodes: [x_10, relu_1, x_11], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf376, (262144, 2560), (2560, 1), 0), permute_284, out=buf378)
            del permute_284
            buf379 = empty_strided_cuda((2560, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf377, buf379, 1638400, stream=stream0)
            buf381 = reinterpret_tensor(buf378, (128, 2048, 640), (1310720, 640, 1), 0); del buf378  # reuse
            # Topologically Sorted Source Nodes: [getitem_15, rms_norm_8], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29.run(buf381, add_21, rsqrt_8, buf367, primals_5, 262144, 640, stream=stream0)
            del add_21
            del rsqrt_8
            buf382 = buf363; del buf363  # reuse
            # Topologically Sorted Source Nodes: [y_4, y_5], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf381, (640, 262144), (1, 640), 0), reinterpret_tensor(getitem_4, (262144, 640), (640, 1), 0), out=buf382)
            buf383 = reinterpret_tensor(buf362, (262144, 640), (640, 1), 0); del buf362  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf381, (262144, 640), (640, 1), 0), permute_288, out=buf383)
            del permute_288
            buf384 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf382, buf384, 409600, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf385 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf383, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), convert_element_type_45, convert_element_type_47, add_14, getitem_4, getitem_5, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_14
            del convert_element_type_45
            del convert_element_type_47
            del getitem_4
            del getitem_5
            buf386 = buf385[0]
            assert_size_stride(buf386, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf386, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf387 = buf385[1]
            assert_size_stride(buf387, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf387, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf388 = buf385[2]
            assert_size_stride(buf388, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf388, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf385
            buf392 = reinterpret_tensor(buf383, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf383  # reuse
            buf405 = buf392; del buf392  # reuse
            buf394 = buf358; del buf358  # reuse
            buf409 = buf394; del buf394  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_5, q_5], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf405, buf409, cat_2, rsqrt_6, buf386, cat_3, rsqrt_7, buf387, primals_2, primals_3, 1310720, 128, stream=stream0)
            del cat_2
            del cat_3
            del rsqrt_6
            del rsqrt_7
            buf397 = buf306; del buf306  # reuse
            # Topologically Sorted Source Nodes: [linear_9, sigmoid, ve_1], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.view, aten.mul, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_squeeze_sum_view_30.run(buf388, embedding_1, mm_9, buf397, 1310720, 128, stream=stream0)
            del embedding_1
            buf398 = buf307; del buf307  # reuse
            # Topologically Sorted Source Nodes: [linear_9, sigmoid], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mm_mul_sigmoid_sigmoid_backward_squeeze_t_view_10.run(buf397, buf398, 2097152, stream=stream0)
            buf399 = buf308; del buf308  # reuse
            # Topologically Sorted Source Nodes: [linear_9, sigmoid], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(buf398, view_26, out=buf399)
            del buf398
            del view_26
            buf400 = buf309; del buf309  # reuse
            # Topologically Sorted Source Nodes: [linear_9, sigmoid], Original ATen: [aten._unsafe_view, aten.sigmoid, aten._to_copy, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf397, (262144, 5), (5, 1), 0), permute_292, out=buf400)
            del buf397
            del permute_292
            buf401 = empty_strided_cuda((5, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mm_11.run(buf399, buf401, 160, stream=stream0)
            del buf399
            buf402 = buf382; del buf382  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf388, (640, 262144), (1, 640), 0), view_16, out=buf402)
            buf403 = reinterpret_tensor(buf387, (262144, 640), (640, 1), 0); del buf387  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf388, (262144, 640), (640, 1), 0), permute_296, out=buf403)
            del permute_296
            buf404 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf402, buf404, 409600, stream=stream0)
            buf406 = buf402; del buf402  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf405, (640, 262144), (1, 640), 0), view_16, out=buf406)
            buf407 = reinterpret_tensor(buf386, (262144, 640), (640, 1), 0); del buf386  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf405, (262144, 640), (640, 1), 0), permute_300, out=buf407)
            del permute_300
            buf408 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf406, buf408, 409600, stream=stream0)
            buf410 = buf406; del buf406  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf409, (640, 262144), (1, 640), 0), view_16, out=buf410)
            del view_16
            buf411 = reinterpret_tensor(buf405, (262144, 640), (640, 1), 0); del buf405  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf409, (262144, 640), (640, 1), 0), permute_304, out=buf411)
            del permute_304
            buf412 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf410, buf412, 409600, stream=stream0)
            buf414 = buf381; del buf381  # reuse
            buf420 = buf237; del buf237  # reuse
            buf423 = reinterpret_tensor(buf409, (262144, 640), (640, 1), 0); del buf409  # reuse
            # Topologically Sorted Source Nodes: [getitem_29, getitem_22, getitem_16, rms_norm_5, getitem_9, getitem_8], Original ATen: [aten.slice_backward, aten.select, aten.mul, aten.add, aten.view, aten._fused_rms_norm_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_31.run(buf414, buf420, add_12, rsqrt_5, buf400, buf403, buf407, buf411, buf276, primals_6, buf323, buf367, primals_5, buf423, 262144, 640, stream=stream0)
            del add_12
            del buf276
            del buf323
            del buf400
            del rsqrt_5
            buf415 = buf324; del buf324  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_9, sigmoid, gate, unsqueeze], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8.run(buf415, 5242880, stream=stream0)
            buf425 = reinterpret_tensor(buf376, (262144, 2560), (2560, 1), 0); del buf376  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf423, permute_308, out=buf425)
            del permute_308
            buf427 = reinterpret_tensor(mm_4, (128, 2048, 2560), (5242880, 2560, 1), 0); del mm_4  # reuse
            # Topologically Sorted Source Nodes: [x_4, relu, x_5], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_pow_relu_threshold_backward_view_4.run(buf427, buf425, 671088640, stream=stream0)
            del buf425
            buf429 = buf411; del buf411  # reuse
            # Topologically Sorted Source Nodes: [x_4, relu, x_5], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf427, (262144, 2560), (2560, 1), 0), permute_312, out=buf429)
            del permute_312
            buf432 = reinterpret_tensor(buf429, (128, 2048, 640), (1310720, 640, 1), 0); del buf429  # reuse
            # Topologically Sorted Source Nodes: [getitem_8, rms_norm_4], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_32.run(buf432, add_9, rsqrt_4, buf414, primals_5, 262144, 640, stream=stream0)
            del add_9
            del rsqrt_4
            buf434 = buf407; del buf407  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf432, (262144, 640), (640, 1), 0), permute_316, out=buf434)
            del permute_316
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf436 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf434, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), convert_element_type_14, convert_element_type_16, view_8, getitem, getitem_1, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del convert_element_type_14
            del convert_element_type_16
            del getitem_1
            del view_8
            buf439 = buf436[2]
            assert_size_stride(buf439, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf439, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf448 = buf434; del buf434  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf439, (262144, 640), (640, 1), 0), permute_320, out=buf448)
            del permute_320
            buf437 = buf436[0]
            assert_size_stride(buf437, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf437, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf438 = buf436[1]
            assert_size_stride(buf438, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf438, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf436
            buf443 = reinterpret_tensor(buf403, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf403  # reuse
            buf450 = buf443; del buf443  # reuse
            buf445 = reinterpret_tensor(buf367, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf367  # reuse
            buf454 = buf445; del buf445  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, k_2, q_2], Original ATen: [aten.slice, aten.neg, aten.slice_backward, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7.run(buf450, buf454, cat, rsqrt_2, buf437, cat_1, rsqrt_3, buf438, primals_2, primals_3, 1310720, 128, stream=stream0)
            del cat
            del cat_1
            del primals_2
            del primals_3
            del rsqrt_2
            del rsqrt_3
            buf452 = reinterpret_tensor(buf438, (262144, 640), (640, 1), 0); del buf438  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf450, (262144, 640), (640, 1), 0), permute_324, out=buf452)
            del permute_324
            buf456 = reinterpret_tensor(buf437, (262144, 640), (640, 1), 0); del buf437  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf454, (262144, 640), (640, 1), 0), permute_328, out=buf456)
            del permute_328
            buf466 = empty_strided_cuda((8192, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_8.run(buf466, 5242880, stream=stream0)
            buf460 = reinterpret_tensor(buf448, (128, 2048, 640), (1310720, 640, 1), 0); del buf448  # reuse
            buf418 = buf370; del buf370  # reuse
            buf461 = buf368; del buf368  # reuse
            buf421 = buf329; del buf329  # reuse
            # Topologically Sorted Source Nodes: [loss, x_1, linear_9, sigmoid, gate, unsqueeze, getitem_2, mul, getitem_3, mul_1, x_2, rms_norm_1], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._to_copy, aten.mul, aten._unsafe_view, aten.sigmoid, aten.unsqueeze, aten.view, aten.sum, aten.add, aten._fused_rms_norm_backward, aten.select]
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_select_sigmoid_sum_unsqueeze_view_33.run(buf460, primals_5, embedding, rsqrt, primals_6, rsqrt_1, buf452, buf456, buf432, buf414, buf420, add_11, primals_1, buf388, mm_9, buf418, buf461, buf421, buf415, buf466, 262144, 640, stream=stream0)
            del add_11
            del buf388
            del buf414
            del buf420
            del buf452
            del buf456
            del buf460
            del embedding
            del mm_9
            del primals_1
            del primals_5
            del primals_6
            del rsqrt
            del rsqrt_1
            buf417 = empty_strided_cuda((8192, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_13.run(buf415, buf417, 5242880, stream=stream0)
            del buf415
            buf419 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf418, buf419, 1, 262144, stream=stream0)
            del buf418
            buf422 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf421, buf422, 1, 262144, stream=stream0)
            del buf421
            buf424 = reinterpret_tensor(buf377, (640, 2560), (2560, 1), 0); del buf377  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf423, (640, 262144), (1, 640), 0), view_14, out=buf424)
            del buf423
            del view_14
            buf426 = empty_strided_cuda((640, 2560), (2560, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf424, buf426, 1638400, stream=stream0)
            buf428 = reinterpret_tensor(buf424, (2560, 640), (640, 1), 0); del buf424  # reuse
            # Topologically Sorted Source Nodes: [x_4, relu, x_5], Original ATen: [aten.threshold_backward, aten.view, aten._to_copy, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf427, (2560, 262144), (1, 2560), 0), view_12, out=buf428)
            del buf427
            del view_12
            buf430 = empty_strided_cuda((2560, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf428, buf430, 1638400, stream=stream0)
            del buf428
            buf433 = buf410; del buf410  # reuse
            # Topologically Sorted Source Nodes: [y_1, y_2], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf432, (640, 262144), (1, 640), 0), reinterpret_tensor(getitem, (262144, 640), (640, 1), 0), out=buf433)
            del buf432
            del getitem
            buf435 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf433, buf435, 409600, stream=stream0)
            buf447 = buf433; del buf433  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf439, (640, 262144), (1, 640), 0), view, out=buf447)
            del buf439
            buf449 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf447, buf449, 409600, stream=stream0)
            buf451 = buf447; del buf447  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf450, (640, 262144), (1, 640), 0), view, out=buf451)
            del buf450
            buf453 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf451, buf453, 409600, stream=stream0)
            buf455 = buf451; del buf451  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf454, (640, 262144), (1, 640), 0), view, out=buf455)
            del buf454
            del view
            buf457 = empty_strided_cuda((640, 640), (640, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(buf455, buf457, 409600, stream=stream0)
            del buf455
            buf462 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_16.run(buf461, buf462, 1, 262144, stream=stream0)
            del buf461
            buf463 = empty_strided_cuda((10, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_select_backward_34.run(buf54, buf95, buf145, buf186, buf236, buf278, buf328, buf369, buf419, buf462, buf463, 10, stream=stream0)
            del buf145
            del buf186
            del buf236
            del buf278
            del buf328
            del buf369
            del buf419
            del buf54
            del buf95
            buf464 = empty_strided_cuda((10, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_select_backward_34.run(buf56, buf97, buf147, buf188, buf239, buf280, buf330, buf371, buf422, buf462, buf464, 10, stream=stream0)
            del buf147
            del buf188
            del buf239
            del buf280
            del buf330
            del buf371
            del buf422
            del buf462
            del buf56
            del buf97
            buf468 = empty_strided_cuda((8192, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_13.run(buf466, buf468, 5242880, stream=stream0)
            del buf466
        return (None, None, None, buf468, buf464, buf463, buf457, buf453, buf449, buf435, buf430, buf426, buf417, buf412, buf408, buf404, buf401, buf384, buf379, buf375, buf365, buf361, buf357, buf343, buf338, buf334, buf326, buf321, buf317, buf313, buf310, buf293, buf288, buf284, buf274, buf270, buf266, buf252, buf247, buf243, buf234, buf229, buf225, buf221, buf218, buf201, buf196, buf192, buf182, buf178, buf174, buf160, buf155, buf151, buf143, buf138, buf134, buf130, buf127, buf110, buf105, buf101, buf91, buf87, buf83, buf69, buf64, buf60, buf52, buf47, buf43, buf39, buf36, buf19, buf14, buf10, buf5, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((1, 20480, 1, 64), (1310720, 64, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((1, 20480, 1, 64), (1310720, 64, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    embedding = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    view_8 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_1 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_2 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_14 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_3 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_16 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((128, 5, 2048), (10240, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_9 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_4 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_4 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_14 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    add_11 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    add_12 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_1 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_5 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    view_26 = rand_strided((262144, 32), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_9 = rand_strided((262144, 5), (5, 1), device='cuda:0', dtype=torch.bfloat16)
    add_14 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_2 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_3 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_6 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_45 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_7 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_47 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_4 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_5 = rand_strided((128, 5, 2048), (10240, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_21 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_8 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_11 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_33 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    add_23 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    add_24 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_9 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    view_43 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_4 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_5 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_10 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_73 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_11 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_75 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_8 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_9 = rand_strided((128, 5, 2048), (10240, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_32 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_12 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_17 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_49 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    add_34 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    add_35 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_2 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_13 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    view_61 = rand_strided((262144, 32), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_22 = rand_strided((262144, 5), (5, 1), device='cuda:0', dtype=torch.bfloat16)
    add_37 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_6 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_7 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_14 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_104 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_15 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_106 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_12 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_13 = rand_strided((128, 5, 2048), (10240, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_44 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_16 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_24 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_68 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    add_46 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    add_47 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_17 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_70 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    view_78 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_8 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_9 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_18 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_132 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_19 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_134 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_16 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_17 = rand_strided((128, 5, 2048), (10240, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_55 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_20 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_30 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_84 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    add_57 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    add_58 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_3 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_21 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    view_96 = rand_strided((262144, 32), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_35 = rand_strided((262144, 5), (5, 1), device='cuda:0', dtype=torch.bfloat16)
    add_60 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_10 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_11 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_22 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_163 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_23 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_165 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_20 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_21 = rand_strided((128, 5, 2048), (10240, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_67 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_24 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_37 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_103 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    add_69 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    add_70 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_25 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    view_113 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_12 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_13 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_26 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_191 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_27 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_193 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_24 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_25 = rand_strided((128, 5, 2048), (10240, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_78 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_28 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_43 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_119 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    add_80 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    add_81 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_4 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_29 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_121 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    view_131 = rand_strided((262144, 32), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_48 = rand_strided((262144, 5), (5, 1), device='cuda:0', dtype=torch.bfloat16)
    add_83 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_14 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_15 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_30 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_222 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_31 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_224 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_28 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_29 = rand_strided((128, 5, 2048), (10240, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_90 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_32 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_136 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_50 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_138 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    add_92 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    add_93 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_33 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    view_148 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_16 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_17 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_34 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_250 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_35 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_252 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_32 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_33 = rand_strided((128, 5, 2048), (10240, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_101 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_36 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_56 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_154 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    add_103 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    add_104 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_5 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_37 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_156 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    view_166 = rand_strided((262144, 32), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_61 = rand_strided((262144, 5), (5, 1), device='cuda:0', dtype=torch.bfloat16)
    add_106 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_18 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_19 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_38 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_281 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_39 = rand_strided((128, 2048, 5, 1), (10240, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_283 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_36 = rand_strided((128, 2048, 5, 128), (1310720, 640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_37 = rand_strided((128, 5, 2048), (10240, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_113 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_40 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_171 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_63 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_173 = rand_strided((262144, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    add_115 = rand_strided((128, 2048, 640), (1310720, 640, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_41 = rand_strided((128, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_175 = rand_strided((262144, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_65 = rand_strided((262144, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    amax = rand_strided((262144, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((262144, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_303 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_68 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_72 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_76 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_80 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_84 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_88 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_92 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_96 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_100 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_104 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_108 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_112 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_116 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_120 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_124 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_128 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_132 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_136 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_140 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_144 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_148 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_152 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_156 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_160 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_164 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_168 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_172 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_176 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_180 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_184 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_188 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_192 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_196 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_200 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_204 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_208 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_212 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_216 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_220 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_224 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_228 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_232 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_236 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_240 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_244 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_248 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_252 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_256 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_260 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_264 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_268 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_272 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_276 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_280 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_284 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_288 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_292 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_296 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_300 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_304 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_308 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_312 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_316 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_320 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_324 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_328 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_5, primals_6, primals_78, embedding, rsqrt, rsqrt_1, view, view_8, cat, cat_1, rsqrt_2, convert_element_type_14, rsqrt_3, convert_element_type_16, getitem, getitem_1, add_9, rsqrt_4, view_12, mm_4, view_14, add_11, add_12, embedding_1, rsqrt_5, view_16, view_26, mm_9, add_14, cat_2, cat_3, rsqrt_6, convert_element_type_45, rsqrt_7, convert_element_type_47, getitem_4, getitem_5, add_21, rsqrt_8, view_31, mm_11, view_33, add_23, add_24, rsqrt_9, view_35, view_43, cat_4, cat_5, rsqrt_10, convert_element_type_73, rsqrt_11, convert_element_type_75, getitem_8, getitem_9, add_32, rsqrt_12, view_47, mm_17, view_49, add_34, add_35, embedding_2, rsqrt_13, view_51, view_61, mm_22, add_37, cat_6, cat_7, rsqrt_14, convert_element_type_104, rsqrt_15, convert_element_type_106, getitem_12, getitem_13, add_44, rsqrt_16, view_66, mm_24, view_68, add_46, add_47, rsqrt_17, view_70, view_78, cat_8, cat_9, rsqrt_18, convert_element_type_132, rsqrt_19, convert_element_type_134, getitem_16, getitem_17, add_55, rsqrt_20, view_82, mm_30, view_84, add_57, add_58, embedding_3, rsqrt_21, view_86, view_96, mm_35, add_60, cat_10, cat_11, rsqrt_22, convert_element_type_163, rsqrt_23, convert_element_type_165, getitem_20, getitem_21, add_67, rsqrt_24, view_101, mm_37, view_103, add_69, add_70, rsqrt_25, view_105, view_113, cat_12, cat_13, rsqrt_26, convert_element_type_191, rsqrt_27, convert_element_type_193, getitem_24, getitem_25, add_78, rsqrt_28, view_117, mm_43, view_119, add_80, add_81, embedding_4, rsqrt_29, view_121, view_131, mm_48, add_83, cat_14, cat_15, rsqrt_30, convert_element_type_222, rsqrt_31, convert_element_type_224, getitem_28, getitem_29, add_90, rsqrt_32, view_136, mm_50, view_138, add_92, add_93, rsqrt_33, view_140, view_148, cat_16, cat_17, rsqrt_34, convert_element_type_250, rsqrt_35, convert_element_type_252, getitem_32, getitem_33, add_101, rsqrt_36, view_152, mm_56, view_154, add_103, add_104, embedding_5, rsqrt_37, view_156, view_166, mm_61, add_106, cat_18, cat_19, rsqrt_38, convert_element_type_281, rsqrt_39, convert_element_type_283, getitem_36, getitem_37, add_113, rsqrt_40, view_171, mm_63, view_173, add_115, rsqrt_41, view_175, mm_65, amax, log, convert_element_type_303, permute_68, permute_72, permute_76, permute_80, permute_84, permute_88, permute_92, permute_96, permute_100, permute_104, permute_108, permute_112, permute_116, permute_120, permute_124, permute_128, permute_132, permute_136, permute_140, permute_144, permute_148, permute_152, permute_156, permute_160, permute_164, permute_168, permute_172, permute_176, permute_180, permute_184, permute_188, permute_192, permute_196, permute_200, permute_204, permute_208, permute_212, permute_216, permute_220, permute_224, permute_228, permute_232, permute_236, permute_240, permute_244, permute_248, permute_252, permute_256, permute_260, permute_264, permute_268, permute_272, permute_276, permute_280, permute_284, permute_288, permute_292, permute_296, permute_300, permute_304, permute_308, permute_312, permute_316, permute_320, permute_324, permute_328, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
