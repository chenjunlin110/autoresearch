# AOT ID: ['4_inference']
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


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/ph/cphppt3o67vz7spis6wlhr54zafe3hlvq6qd2laievhqdtbir2jd.py
# Topologically Sorted Source Nodes: [g, sub, lerp_, X, norm, mul, add, X_1], Original ATen: [aten.lerp, aten.rsub, aten._to_copy, aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten.copy_]
# Source node to ATen node mapping:
#   X => convert_element_type
#   X_1 => div
#   add => add_2
#   g => abs_2, add_1, ge_1, mul_1, sub_3, sub_4, where_2, where_3
#   lerp_ => abs_1, add, ge, mul, sub_1, sub_2, where, where_1
#   mul => mul_2
#   norm => convert_element_type_1, convert_element_type_2, pow_1, pow_2, sum_1
#   sub => sub
# Graph fragment:
#   %arg0_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg0_1]
#   %copy_ : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=copy_]
#   %copy__1 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=copy__1]
#   %sum_1 : Tensor "f32[5, 1, 1][1, 5, 5]cuda:0" = PlaceHolder[target=sum_1]
#   %add_1 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=add_1]
#   %expand_5 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=expand_5]
#   %add : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=add]
#   %abs_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg0_1,), kwargs = {})
#   %ge_1 : Tensor "b8[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_2, 0.5), kwargs = {})
#   %sub_3 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, 1), kwargs = {})
#   %where_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %sub_3, %arg0_1), kwargs = {})
#   %sub : Tensor "f32[][]cpu"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg0_1), kwargs = {})
#   %abs_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %ge : Tensor "b8[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %sub_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, 1), kwargs = {})
#   %where : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %sub_1, %sub), kwargs = {})
#   %sub_2 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %mul : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %sub_2), kwargs = {})
#   %where_1 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %arg1_1, %arg2_1), kwargs = {})
#   %add : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %where_1), kwargs = {})
#   %sub_4 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %arg1_1), kwargs = {})
#   %mul_1 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_2, %sub_4), kwargs = {})
#   %where_3 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %add, %arg1_1), kwargs = {})
#   %add_1 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %where_3), kwargs = {})
#   %convert_element_type : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1, 2), kwargs = {})
#   %sum_1 : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [-2, -1], True), kwargs = {})
#   %pow_2 : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_2, torch.bfloat16), kwargs = {})
#   %mul_2 : Tensor "bf16[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, 1.02), kwargs = {})
#   %add_2 : Tensor "bf16[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 1e-06), kwargs = {})
#   %div : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %add_2), kwargs = {})
#   %copy_ : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg1_1, %add_1), kwargs = {})
#   %copy__1 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg2_1, %add), kwargs = {})
#   return %sum_1,%expand_5,%add,%add_1,%buf41,%buf42
triton_per_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_mul_rsub_0 = async_compile.triton('triton_per_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_mul_rsub_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': 'fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_mul_rsub_0', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr5'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 22400}}
)
@triton.jit
def triton_per_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_mul_rsub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 5
    r0_numel = 160
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = in_ptr0
    tmp12 = tl.load(in_ptr1 + (r0_1 + 160*x0), r0_mask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr2 + (r0_1 + 160*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 0.5
    tmp3 = tmp1 >= tmp2
    tmp4 = 1.0
    tmp5 = tmp0 - tmp4
    tmp6 = tl.where(tmp3, tmp5, tmp0)
    tmp7 = tmp4 - tmp0
    tmp8 = tl_math.abs(tmp7)
    tmp9 = tmp8 >= tmp2
    tmp10 = tmp7 - tmp4
    tmp11 = tl.where(tmp9, tmp10, tmp7)
    tmp14 = tmp12 - tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp9, tmp12, tmp13)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17 - tmp12
    tmp19 = tmp6 * tmp18
    tmp20 = tl.where(tmp3, tmp17, tmp12)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(r0_mask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = 1.02
    tmp32 = tmp30 * tmp31
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = (tmp22 / tmp34)
    tl.store(out_ptr1 + (r0_1 + 160*x0), tmp35, r0_mask & xmask)
    tl.store(out_ptr4 + (r0_1 + 160*x0), tmp21, r0_mask & xmask)
    tl.store(out_ptr5 + (r0_1 + 160*x0), tmp17, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/mf/cmfbdkc2ggfk7md6omcqusrdfov2jwoq3ezqx3b5viquzllufa7j.py
# Topologically Sorted Source Nodes: [mul_1, mul_2, B], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B => add_3
#   mul_1 => mul_3
#   mul_2 => mul_4
# Graph fragment:
#   %expand_3 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0" = PlaceHolder[target=expand_3]
#   %bmm_1 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %mul_3 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm, -22.48329292557795), kwargs = {})
#   %mul_4 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_1, 15.878769915207462), kwargs = {})
#   %add_3 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_4), kwargs = {})
#   return %expand_4
triton_poi_fused_add_mul_1 = async_compile.triton('triton_poi_fused_add_mul_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = -22.48329292557795
    tmp2 = tmp0 * tmp1
    tmp4 = 15.878769915207462
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/dn/cdnyt7flbeo7mnhhvd42cojk3gx37f3zzmqigq33j6s4nyim4bkl.py
# Topologically Sorted Source Nodes: [mul_3, X_2], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   X_2 => add_4
#   mul_3 => mul_5
# Graph fragment:
#   %expand_5 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=expand_5]
#   %bmm_2 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %mul_5 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 8.156554524902461), kwargs = {})
#   %add_4 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %bmm_2), kwargs = {})
#   return %expand_11
triton_poi_fused_add_mul_2 = async_compile.triton('triton_poi_fused_add_mul_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp1 = 8.156554524902461
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/m6/cm6ytchdkavcrrlrwtg6uywejzghbwhvpubkwdig7hywdrawdhvy.py
# Topologically Sorted Source Nodes: [mul_4, mul_5, B_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_1 => add_5
#   mul_4 => mul_6
#   mul_5 => mul_7
# Graph fragment:
#   %expand_9 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0" = PlaceHolder[target=expand_9]
#   %bmm_4 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %mul_6 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, -2.808917465908714), kwargs = {})
#   %mul_7 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_4, 0.5000178451051316), kwargs = {})
#   %add_5 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   return %expand_10
triton_poi_fused_add_mul_3 = async_compile.triton('triton_poi_fused_add_mul_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = -2.808917465908714
    tmp2 = tmp0 * tmp1
    tmp4 = 0.5000178451051316
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/2c/c2ceea7y625qmbs5lfn65butv6qhgwastuijior6dchgamwnonjb.py
# Topologically Sorted Source Nodes: [mul_3, X_2, mul_6, X_3], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   X_2 => add_4
#   X_3 => add_6
#   mul_3 => mul_5
#   mul_6 => mul_8
# Graph fragment:
#   %expand_5 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=expand_5]
#   %bmm_2 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %bmm_5 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %mul_5 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 8.156554524902461), kwargs = {})
#   %add_4 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %bmm_2), kwargs = {})
#   %mul_8 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 4.042929935166739), kwargs = {})
#   %add_6 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %bmm_5), kwargs = {})
#   return %expand_17
triton_poi_fused_add_mul_4 = async_compile.triton('triton_poi_fused_add_mul_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp1 = 8.156554524902461
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 4.042929935166739
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/l7/cl73h3cgwhb6gybpjbhbideogbnltt7qj6kyr7xhlhjtcnikf66w.py
# Topologically Sorted Source Nodes: [mul_7, mul_8, B_2], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_2 => add_7
#   mul_7 => mul_9
#   mul_8 => mul_10
# Graph fragment:
#   %expand_15 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0" = PlaceHolder[target=expand_15]
#   %bmm_7 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %mul_9 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_6, -2.772484153217685), kwargs = {})
#   %mul_10 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_7, 0.5060648178503393), kwargs = {})
#   %add_7 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %mul_10), kwargs = {})
#   return %expand_16
triton_poi_fused_add_mul_5 = async_compile.triton('triton_poi_fused_add_mul_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = -2.772484153217685
    tmp2 = tmp0 * tmp1
    tmp4 = 0.5060648178503393
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/kc/ckcjhe3p2mzpmikh23mutdjanmgdk52d5dvlots33dkaevf5gmvm.py
# Topologically Sorted Source Nodes: [mul_3, X_2, mul_6, X_3, mul_9, X_4], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   X_2 => add_4
#   X_3 => add_6
#   X_4 => add_8
#   mul_3 => mul_5
#   mul_6 => mul_8
#   mul_9 => mul_11
# Graph fragment:
#   %expand_5 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=expand_5]
#   %bmm_2 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %bmm_5 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %bmm_8 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %mul_5 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 8.156554524902461), kwargs = {})
#   %add_4 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %bmm_2), kwargs = {})
#   %mul_8 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 4.042929935166739), kwargs = {})
#   %add_6 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %bmm_5), kwargs = {})
#   %mul_11 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 3.8916678022926607), kwargs = {})
#   %add_8 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %bmm_8), kwargs = {})
#   return %expand_23
triton_poi_fused_add_mul_6 = async_compile.triton('triton_poi_fused_add_mul_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (x0), xmask).to(tl.float32)
    tmp1 = 8.156554524902461
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 4.042929935166739
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = 3.8916678022926607
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/3e/c3ehxjw6qdmogsmakaff34eiegc3n2l3cf4ebhdxkzlp6cnrysvf.py
# Topologically Sorted Source Nodes: [mul_10, mul_11, B_3], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_3 => add_9
#   mul_10 => mul_12
#   mul_11 => mul_13
# Graph fragment:
#   %expand_21 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0" = PlaceHolder[target=expand_21]
#   %bmm_10 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0" = PlaceHolder[target=bmm_10]
#   %mul_12 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, -2.3681294933425376), kwargs = {})
#   %mul_13 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_10, 0.46449024233003106), kwargs = {})
#   %add_9 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_13), kwargs = {})
#   return %expand_22
triton_poi_fused_add_mul_7 = async_compile.triton('triton_poi_fused_add_mul_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = -2.3681294933425376
    tmp2 = tmp0 * tmp1
    tmp4 = 0.46449024233003106
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/7w/c7wjc2tv6crrfzveeuvhk5bypywqepjk3mkqcm5e4z3kifnimvhh.py
# Topologically Sorted Source Nodes: [mul_3, X_2, mul_6, X_3, mul_9, X_4, mul_12, X_5], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   X_2 => add_4
#   X_3 => add_6
#   X_4 => add_8
#   X_5 => add_10
#   mul_12 => mul_14
#   mul_3 => mul_5
#   mul_6 => mul_8
#   mul_9 => mul_11
# Graph fragment:
#   %expand_5 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=expand_5]
#   %bmm_2 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %bmm_5 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %bmm_8 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %bmm_11 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %mul_5 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 8.156554524902461), kwargs = {})
#   %add_4 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %bmm_2), kwargs = {})
#   %mul_8 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 4.042929935166739), kwargs = {})
#   %add_6 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %bmm_5), kwargs = {})
#   %mul_11 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 3.8916678022926607), kwargs = {})
#   %add_8 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %bmm_8), kwargs = {})
#   %mul_14 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 3.285753657755655), kwargs = {})
#   %add_10 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %bmm_11), kwargs = {})
#   return %expand_29
triton_poi_fused_add_mul_8 = async_compile.triton('triton_poi_fused_add_mul_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (x0), xmask).to(tl.float32)
    tmp1 = 8.156554524902461
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 4.042929935166739
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = 3.8916678022926607
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = 3.285753657755655
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tl.store(in_out_ptr0 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/2i/c2i5mid6j5fnktw2ivo3gijlknjdeeinypm5nriuo63yhcmzzskx.py
# Topologically Sorted Source Nodes: [mul_13, mul_14, B_4], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_4 => add_11
#   mul_13 => mul_15
#   mul_14 => mul_16
# Graph fragment:
#   %expand_27 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0" = PlaceHolder[target=expand_27]
#   %bmm_13 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0" = PlaceHolder[target=bmm_13]
#   %mul_15 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_12, -1.7097828382687081), kwargs = {})
#   %mul_16 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_13, 0.42323551169305323), kwargs = {})
#   %add_11 : Tensor "bf16[5, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %mul_16), kwargs = {})
#   return %expand_28
triton_poi_fused_add_mul_9 = async_compile.triton('triton_poi_fused_add_mul_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = -1.7097828382687081
    tmp2 = tmp0 * tmp1
    tmp4 = 0.42323551169305323
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/2u/c2u7cwlba67yrt355jaoz23bpbahre44hutxdp6codhhkdvzvfsh.py
# Topologically Sorted Source Nodes: [mul_15, X_6, float_1, square, v_mean, sum_1, v_norm_sq, v_norm, mul_17, beta2, sub_1, lerp__2, clamp_min, step_size, square_1, scaled_sq_sum, sum_2, v_norm_new, clamp_min_1, truediv_1, final_scale, to_3], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.sum, aten.sqrt, aten.rsub, aten.lerp, aten.clamp_min, aten.rsqrt, aten.div, aten.copy_]
# Source node to ATen node mapping:
#   X_6 => add_12
#   beta2 => convert_element_type_33
#   clamp_min => clamp_min
#   clamp_min_1 => clamp_min_1
#   final_scale => mul_22
#   float_1 => convert_element_type_34
#   lerp__2 => abs_3, add_13, convert_element_type_35, ge_2, mul_19, sub_6, sub_7, where_4, where_5
#   mul_15 => mul_17
#   mul_17 => mul_20
#   scaled_sq_sum => mul_21
#   square => pow_3
#   square_1 => pow_4
#   step_size => rsqrt
#   sub_1 => sub_5
#   sum_1 => sum_2
#   sum_2 => sum_3
#   to_3 => convert_element_type_36
#   truediv_1 => div_1
#   v_mean => mean
#   v_norm => sqrt
#   v_norm_new => sqrt_1
#   v_norm_sq => mul_18
# Graph fragment:
#   %expand_29 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=expand_29]
#   %bmm_14 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=bmm_14]
#   %mean : Tensor "f32[5, 1, 32][32, 160, 1]cuda:0" = PlaceHolder[target=mean]
#   %arg3_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg3_1]
#   %copy__2 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0" = PlaceHolder[target=copy__2]
#   %sum_2 : Tensor "f32[5, 1, 1][1, 5, 5]cuda:0" = PlaceHolder[target=sum_2]
#   %sum_3 : Tensor "f32[5, 1, 1][1, 5, 5]cuda:0" = PlaceHolder[target=sum_3]
#   %add_13 : Tensor "f32[5, 1, 32][32, 160, 1]cuda:0" = PlaceHolder[target=add_13]
#   %convert_element_type_36 : Tensor "bf16[5, 1, 32][32, 160, 1]cuda:0" = PlaceHolder[target=convert_element_type_36]
#   %mul_17 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 2.3465413258596377), kwargs = {})
#   %add_12 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %bmm_14), kwargs = {})
#   %convert_element_type_34 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_34, 2), kwargs = {})
#   %mean : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-2], True), kwargs = {})
#   %sum_2 : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mean, [-2, -1], True), kwargs = {})
#   %mul_18 : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, 5), kwargs = {})
#   %sqrt : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mul_18,), kwargs = {})
#   %mul_20 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 5), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.bfloat16), kwargs = {})
#   %sub_5 : Tensor "bf16[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_33), kwargs = {})
#   %convert_element_type_35 : Tensor "f32[][]cpu"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_5, torch.float32), kwargs = {})
#   %abs_3 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type_35,), kwargs = {})
#   %ge_2 : Tensor "b8[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_3, 0.5), kwargs = {})
#   %sub_6 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_35, 1), kwargs = {})
#   %where_4 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_2, %sub_6, %convert_element_type_35), kwargs = {})
#   %sub_7 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean, %arg4_1), kwargs = {})
#   %mul_19 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_4, %sub_7), kwargs = {})
#   %where_5 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_2, %mean, %arg4_1), kwargs = {})
#   %add_13 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %where_5), kwargs = {})
#   %clamp_min : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_13, 1e-10), kwargs = {})
#   %rsqrt : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%clamp_min,), kwargs = {})
#   %pow_4 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%rsqrt, 2), kwargs = {})
#   %mul_21 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %pow_4), kwargs = {})
#   %sum_3 : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_21, [-2, -1], True), kwargs = {})
#   %sqrt_1 : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_3,), kwargs = {})
#   %clamp_min_1 : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sqrt_1, 1e-10), kwargs = {})
#   %div_1 : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sqrt, %clamp_min_1), kwargs = {})
#   %mul_22 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%rsqrt, %div_1), kwargs = {})
#   %convert_element_type_36 : Tensor "bf16[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_22, torch.bfloat16), kwargs = {})
#   %copy__2 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg4_1, %add_13), kwargs = {})
#   return %mean,%sum_2,%sum_3,%convert_element_type_36,%add_13,%buf52
triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_rsub_sqrt_sum_10 = async_compile.triton('triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_rsub_sqrt_sum_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': 'fp32', 'in_ptr3': '*fp32', 'out_ptr3': '*bf16', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_rsub_sqrt_sum_10', 'mutated_arg_names': ['in_ptr3', 'out_ptr5'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 12, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 5760}}
)
@triton.jit
def triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_rsub_sqrt_sum_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 5
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (32 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (32 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (64 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr1 + (64 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr0 + (96 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp23 = tl.load(in_ptr1 + (96 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr0 + (128 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr1 + (128 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp42 = in_ptr2
    tmp52 = tl.load(in_ptr3 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp1 = 2.3465413258596377
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp5
    tmp8 = tmp7 * tmp1
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp11
    tmp13 = tmp6 + tmp12
    tmp15 = tmp14 * tmp1
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp18
    tmp20 = tmp13 + tmp19
    tmp22 = tmp21 * tmp1
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp25
    tmp27 = tmp20 + tmp26
    tmp29 = tmp28 * tmp1
    tmp31 = tmp29 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 * tmp32
    tmp34 = tmp27 + tmp33
    tmp35 = 5.0
    tmp36 = (tmp34 / tmp35)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp39 = tl.where(xmask, tmp37, 0)
    tmp40 = tl.sum(tmp39, 1)[:, None].to(tl.float32)
    tmp41 = tmp36 * tmp35
    tmp43 = tmp42.to(tl.float32)
    tmp44 = 1.0
    tmp45 = tmp44 - tmp43
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tl_math.abs(tmp46)
    tmp48 = 0.5
    tmp49 = tmp47 >= tmp48
    tmp50 = tmp46 - tmp44
    tmp51 = tl.where(tmp49, tmp50, tmp46)
    tmp53 = tmp36 - tmp52
    tmp54 = tmp51 * tmp53
    tmp55 = tl.where(tmp49, tmp36, tmp52)
    tmp56 = tmp54 + tmp55
    tmp57 = 1e-10
    tmp58 = triton_helpers.maximum(tmp56, tmp57)
    tmp59 = libdevice.rsqrt(tmp58)
    tmp60 = tmp59 * tmp59
    tmp61 = tmp41 * tmp60
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK, R0_BLOCK])
    tmp64 = tl.where(xmask, tmp62, 0)
    tmp65 = tl.sum(tmp64, 1)[:, None].to(tl.float32)
    tmp66 = tmp40 * tmp35
    tmp67 = libdevice.sqrt(tmp66)
    tmp68 = libdevice.sqrt(tmp65)
    tmp69 = triton_helpers.maximum(tmp68, tmp57)
    tmp70 = (tmp67 / tmp69)
    tmp71 = tmp59 * tmp70
    tmp72 = tmp71.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 32*x0), tmp72, xmask)
    tl.store(out_ptr5 + (r0_1 + 32*x0), tmp56, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/lk/clk34wiqhy4yeuhpyb44pzo6nwbcxcind6fnjn6fgauaksmhkjdu.py
# Topologically Sorted Source Nodes: [lr, mul_15, X_6, v_norm_sq, v_norm, beta2, sub_1, lerp__2, clamp_min, step_size, v_norm_new, clamp_min_1, truediv_1, final_scale, to_3, g_1, mul_22, wd, mul_23, mul_24, mul_21, mask, mul_25, add_11, sub_], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.sqrt, aten.rsub, aten.lerp, aten.clamp_min, aten.rsqrt, aten.div, aten.ge, aten.sub, aten.copy_]
# Source node to ATen node mapping:
#   X_6 => add_12
#   add_11 => add_14
#   beta2 => convert_element_type_33
#   clamp_min => clamp_min
#   clamp_min_1 => clamp_min_1
#   final_scale => mul_22
#   g_1 => mul_23
#   lerp__2 => abs_3, add_13, convert_element_type_35, ge_2, mul_19, sub_6, sub_7, where_4, where_5
#   lr => convert_element_type_37
#   mask => ge_3
#   mul_15 => mul_17
#   mul_21 => mul_24
#   mul_22 => mul_25
#   mul_23 => mul_26
#   mul_24 => mul_27
#   mul_25 => mul_28
#   step_size => rsqrt
#   sub_ => sub_8
#   sub_1 => sub_5
#   to_3 => convert_element_type_36
#   truediv_1 => div_1
#   v_norm => sqrt
#   v_norm_new => sqrt_1
#   v_norm_sq => mul_18
#   wd => convert_element_type_38
# Graph fragment:
#   %copy__3 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=copy__3]
#   %arg5_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg5_1]
#   %expand_29 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=expand_29]
#   %bmm_14 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=bmm_14]
#   %convert_element_type_36 : Tensor "bf16[5, 1, 32][32, 160, 1]cuda:0" = PlaceHolder[target=convert_element_type_36]
#   %arg6_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg6_1]
#   %sub_8 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0" = PlaceHolder[target=sub_8]
#   %convert_element_type_37 : Tensor "bf16[][]cpu"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.bfloat16), kwargs = {})
#   %mul_17 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 2.3465413258596377), kwargs = {})
#   %add_12 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %bmm_14), kwargs = {})
#   %mul_18 : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, 5), kwargs = {})
#   %sqrt : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mul_18,), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.bfloat16), kwargs = {})
#   %sub_5 : Tensor "bf16[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_33), kwargs = {})
#   %convert_element_type_35 : Tensor "f32[][]cpu"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_5, torch.float32), kwargs = {})
#   %abs_3 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type_35,), kwargs = {})
#   %ge_2 : Tensor "b8[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_3, 0.5), kwargs = {})
#   %sub_6 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_35, 1), kwargs = {})
#   %where_4 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_2, %sub_6, %convert_element_type_35), kwargs = {})
#   %sub_7 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean, %arg4_1), kwargs = {})
#   %mul_19 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_4, %sub_7), kwargs = {})
#   %where_5 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_2, %mean, %arg4_1), kwargs = {})
#   %add_13 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %where_5), kwargs = {})
#   %clamp_min : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_13, 1e-10), kwargs = {})
#   %rsqrt : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%clamp_min,), kwargs = {})
#   %sqrt_1 : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_3,), kwargs = {})
#   %clamp_min_1 : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sqrt_1, 1e-10), kwargs = {})
#   %div_1 : Tensor "f32[5, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sqrt, %clamp_min_1), kwargs = {})
#   %mul_22 : Tensor "f32[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%rsqrt, %div_1), kwargs = {})
#   %convert_element_type_36 : Tensor "bf16[5, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_22, torch.bfloat16), kwargs = {})
#   %mul_23 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %convert_element_type_36), kwargs = {})
#   %mul_25 : Tensor "bf16[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_37, %mul_23), kwargs = {})
#   %convert_element_type_38 : Tensor "bf16[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.bfloat16), kwargs = {})
#   %mul_26 : Tensor "bf16[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_37, %convert_element_type_38), kwargs = {})
#   %mul_27 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %arg7_1), kwargs = {})
#   %mul_24 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %arg7_1), kwargs = {})
#   %ge_3 : Tensor "b8[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%mul_24, 0), kwargs = {})
#   %mul_28 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %ge_3), kwargs = {})
#   %add_14 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %mul_28), kwargs = {})
#   %sub_8 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg7_1, %add_14), kwargs = {})
#   %copy__3 : Tensor "f32[5, 5, 32][160, 32, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg7_1, %sub_8), kwargs = {})
#   return %sub_8,%buf59
triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_mul_rsqrt_rsub_sqrt_sub_11 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_mul_rsqrt_rsub_sqrt_sub_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': 'fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_mul_rsqrt_rsub_sqrt_sub_11', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 14400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_mul_rsqrt_rsub_sqrt_sub_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 32)
    x2 = xindex // 160
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = in_ptr1
    tmp3 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (x0 + 32*x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = in_ptr5
    tmp2 = tmp1.to(tl.float32)
    tmp4 = 2.3465413258596377
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp2 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp0
    tmp17 = tmp9.to(tl.float32)
    tmp18 = tmp17 * tmp0
    tmp19 = 0.0
    tmp20 = tmp18 >= tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp16 * tmp21
    tmp23 = tmp11 + tmp22
    tmp24 = tmp0 - tmp23
    tl.store(out_ptr1 + (x3), tmp24, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
        args.clear()
        assert_size_stride(arg0_1, (), ())
        assert_size_stride(arg1_1, (5, 5, 32), (160, 32, 1))
        assert_size_stride(arg2_1, (5, 5, 32), (160, 32, 1))
        assert_size_stride(arg3_1, (), ())
        assert_size_stride(arg4_1, (5, 1, 32), (32, 32, 1))
        assert_size_stride(arg5_1, (), ())
        assert_size_stride(arg6_1, (), ())
        assert_size_stride(arg7_1, (5, 5, 32), (160, 32, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((5, 5, 32), (160, 32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [g, sub, lerp_, X, norm, mul, add, X_1], Original ATen: [aten.lerp, aten.rsub, aten._to_copy, aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_mul_rsub_0.run(arg0_1.item(), arg1_1, arg2_1, buf1, arg1_1, arg2_1, 5, 160, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            buf2 = empty_strided_cuda((5, 5, 5), (25, 5, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getattr_1, A], Original ATen: [aten.transpose, aten.bmm]
            extern_kernels.bmm(buf1, reinterpret_tensor(buf1, (5, 32, 5), (160, 1, 32), 0), out=buf2)
            buf3 = empty_strided_cuda((5, 5, 5), (25, 5, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf2, buf2, out=buf3)
            buf4 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [mul_1, mul_2, B], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_1.run(buf4, buf3, 125, stream=stream0)
            buf5 = empty_strided_cuda((5, 5, 32), (160, 32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_1, mul_2, B, matmul_2], Original ATen: [aten.mul, aten.add, aten.bmm]
            extern_kernels.bmm(buf4, buf1, out=buf5)
            buf6 = empty_strided_cuda((5, 5, 32), (160, 32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_3, X_2], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_2.run(buf1, buf5, buf6, 800, stream=stream0)
            buf7 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [mul_3, X_2, getattr_2, A_1], Original ATen: [aten.mul, aten.add, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf6, reinterpret_tensor(buf6, (5, 32, 5), (160, 1, 32), 0), out=buf7)
            buf8 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf7, buf7, out=buf8)
            buf9 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [mul_4, mul_5, B_1], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_3.run(buf9, buf8, 125, stream=stream0)
            buf10 = empty_strided_cuda((5, 5, 32), (160, 32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_4, mul_5, B_1, matmul_5], Original ATen: [aten.mul, aten.add, aten.bmm]
            extern_kernels.bmm(buf9, buf6, out=buf10)
            buf11 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [mul_3, X_2, mul_6, X_3], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_4.run(buf1, buf5, buf10, buf11, 800, stream=stream0)
            buf12 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [mul_3, X_2, mul_6, X_3, getattr_3, A_2], Original ATen: [aten.mul, aten.add, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf11, reinterpret_tensor(buf11, (5, 32, 5), (160, 1, 32), 0), out=buf12)
            buf13 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf12, buf12, out=buf13)
            buf14 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [mul_7, mul_8, B_2], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_5.run(buf14, buf13, 125, stream=stream0)
            buf15 = empty_strided_cuda((5, 5, 32), (160, 32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_7, mul_8, B_2, matmul_8], Original ATen: [aten.mul, aten.add, aten.bmm]
            extern_kernels.bmm(buf14, buf11, out=buf15)
            buf16 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [mul_3, X_2, mul_6, X_3, mul_9, X_4], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_6.run(buf1, buf5, buf10, buf15, buf16, 800, stream=stream0)
            buf17 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [mul_3, X_2, mul_6, X_3, mul_9, X_4, getattr_4, A_3], Original ATen: [aten.mul, aten.add, aten.transpose, aten.bmm]
            extern_kernels.bmm(buf16, reinterpret_tensor(buf16, (5, 32, 5), (160, 1, 32), 0), out=buf17)
            buf18 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf17, buf17, out=buf18)
            buf19 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [mul_10, mul_11, B_3], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_7.run(buf19, buf18, 125, stream=stream0)
            del buf18
            buf20 = empty_strided_cuda((5, 5, 32), (160, 32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_10, mul_11, B_3, matmul_11], Original ATen: [aten.mul, aten.add, aten.bmm]
            extern_kernels.bmm(buf19, buf16, out=buf20)
            del buf16
            buf21 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [mul_3, X_2, mul_6, X_3, mul_9, X_4, mul_12, X_5], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_8.run(buf21, buf5, buf10, buf15, buf20, 800, stream=stream0)
            del buf10
            del buf15
            del buf20
            buf22 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [getattr_5, A_4], Original ATen: [aten.transpose, aten.bmm]
            extern_kernels.bmm(buf21, reinterpret_tensor(buf21, (5, 32, 5), (160, 1, 32), 0), out=buf22)
            buf23 = empty_strided_cuda((5, 5, 5), (25, 5, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf22, buf22, out=buf23)
            buf24 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [mul_13, mul_14, B_4], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf24, buf23, 125, stream=stream0)
            del buf23
            buf25 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [mul_13, mul_14, B_4, matmul_14], Original ATen: [aten.mul, aten.add, aten.bmm]
            extern_kernels.bmm(buf24, buf21, out=buf25)
            del buf24
            buf29 = empty_strided_cuda((5, 1, 32), (32, 160, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_15, X_6, float_1, square, v_mean, sum_1, v_norm_sq, v_norm, mul_17, beta2, sub_1, lerp__2, clamp_min, step_size, square_1, scaled_sq_sum, sum_2, v_norm_new, clamp_min_1, truediv_1, final_scale, to_3], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.sum, aten.sqrt, aten.rsub, aten.lerp, aten.clamp_min, aten.rsqrt, aten.div, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_rsub_sqrt_sum_10.run(buf21, buf25, arg3_1.item(), arg4_1, buf29, arg4_1, 5, 32, stream=stream0)
            del arg3_1
            del arg4_1
            # Topologically Sorted Source Nodes: [lr, mul_15, X_6, v_norm_sq, v_norm, beta2, sub_1, lerp__2, clamp_min, step_size, v_norm_new, clamp_min_1, truediv_1, final_scale, to_3, g_1, mul_22, wd, mul_23, mul_24, mul_21, mask, mul_25, add_11, sub_], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.sqrt, aten.rsub, aten.lerp, aten.clamp_min, aten.rsqrt, aten.div, aten.ge, aten.sub, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_mul_rsqrt_rsub_sqrt_sub_11.run(arg7_1, arg5_1.item(), buf21, buf25, buf29, arg6_1.item(), arg7_1, 800, stream=stream0)
            del arg5_1
            del arg6_1
            del arg7_1
            del buf21
            del buf25
            del buf29
        return ()

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((5, 5, 32), (160, 32, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((5, 5, 32), (160, 32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((5, 1, 32), (32, 32, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((5, 5, 32), (160, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
