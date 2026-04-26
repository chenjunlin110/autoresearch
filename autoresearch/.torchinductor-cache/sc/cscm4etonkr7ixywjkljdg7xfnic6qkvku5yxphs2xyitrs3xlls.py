# AOT ID: ['8_inference']
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


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/qh/cqh44frxcfy3ift7j4edwclwin7ivlvwuqg5kitg3blps5wyod2s.py
# Topologically Sorted Source Nodes: [getitem_2, x, x_1, mul, getitem_3, mul_1, x_2, rms_norm_1], Original ATen: [aten.select, aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   getitem_2 => select
#   getitem_3 => select_1
#   mul => mul_1
#   mul_1 => mul_2
#   rms_norm_1 => add_2, convert_element_type_2, convert_element_type_3, mean_1, mul_3, pow_2, rsqrt_1
#   x => embedding
#   x_1 => add, convert_element_type, convert_element_type_1, mean, mul, pow_1, rsqrt
#   x_2 => add_1
# Graph fragment:
#   %arg0_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf0 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf0]
#   %arg5_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %add_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_1]
#   %buf2 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf2]
#   %select : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg4_1, 0, 0), kwargs = {})
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1), kwargs = {})
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %convert_element_type_1), kwargs = {})
#   %select_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg5_1, 0, 0), kwargs = {})
#   %mul_2 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %convert_element_type_1), kwargs = {})
#   %add_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_2, 2), kwargs = {})
#   %mean_1 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_2 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_1 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_3 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %rsqrt_1), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   return %buf0,%add_1,%buf2,%convert_element_type_3
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_select_0 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_select_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_select_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_select_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert((0 <= tmp4) & (tmp4 < 8192), "index out of bounds: 0 <= tmp4 < 8192")
        tmp6 = tl.load(in_ptr1 + (r0_1 + 640*tmp4), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, None)
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp30 = tl.load(in_ptr3 + (0))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
    _tmp38 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
        tmp16 = tmp0 + tmp15
        tmp17 = tmp0 < 0
        tmp18 = tl.where(tmp17, tmp16, tmp0)
        tl.device_assert((0 <= tmp18) & (tmp18 < 8192), "index out of bounds: 0 <= tmp18 < 8192")
        tmp20 = tl.load(in_ptr1 + (r0_1 + 640*tmp18), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 640.0
        tmp23 = (tmp10 / tmp22)
        tmp24 = 1.1920928955078125e-07
        tmp25 = tmp23 + tmp24
        tmp26 = libdevice.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp14 * tmp28
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp32 * tmp28
        tmp34 = tmp29 + tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp35 * tmp35
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(r0_mask, tmp39, _tmp38)
        tl.store(out_ptr1 + (r0_1 + 640*x0), tmp34, r0_mask)
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp40 = tl.load(out_ptr1 + (r0_1 + 640*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = 640.0
        tmp43 = (tmp38 / tmp42)
        tmp44 = 1.1920928955078125e-07
        tmp45 = tmp43 + tmp44
        tmp46 = libdevice.rsqrt(tmp45)
        tmp47 = tmp41 * tmp46
        tmp48 = tmp47.to(tl.float32)
        tl.store(out_ptr3 + (r0_1 + 640*x0), tmp48, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/fw/cfwn7stkp7z525upojj3nk3v5okwq6cwlxzn7zj5myldikqoo5zw.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_4
# Graph fragment:
#   %arg6_1 : Tensor "f32[640, 640][640, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convert_element_type_4 : Tensor "bf16[640, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_4
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3276800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/xz/cxzxnmpuom5fv6y62gzcap42qe5lqmdsd4x6a7mpqpnrycod2ay2.py
# Topologically Sorted Source Nodes: [linear, q, x1, cos, mul_2, x2, sin, mul_3, y1, neg, mul_4, mul_5, y2, q_1, q_2, k_2, linear_2, v, _flash_attn_forward_default], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, flash_attn_3._flash_attn_forward]
# Source node to ATen node mapping:
#   _flash_attn_forward_default => _flash_attn_forward
#   cos => slice_1
#   k_2 => add_8, convert_element_type_16, mean_3, mul_13, pow_4, rsqrt_3
#   linear => view_1
#   linear_2 => view_7
#   mul_2 => mul_4
#   mul_3 => mul_5
#   mul_4 => mul_6
#   mul_5 => mul_7
#   neg => neg
#   q => view_2
#   q_1 => cat
#   q_2 => add_7, convert_element_type_13, convert_element_type_14, mean_2, mul_12, pow_3, rsqrt_2
#   sin => slice_2
#   v => view_8
#   x1 => slice_3
#   x2 => slice_4
#   y1 => add_3
#   y2 => add_4
# Graph fragment:
#   %mm : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg1_1 : Tensor "bf16[1, 20480, 1, 64][1310720, 64, 64, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "bf16[1, 20480, 1, 64][1310720, 64, 64, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convert_element_type_13 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=convert_element_type_13]
#   %buf7 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1310720]cuda:0" = PlaceHolder[target=buf7]
#   %view_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 2048, 640]), kwargs = {})
#   %view_2 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [128, 2048, 5, 128]), kwargs = {})
#   %slice_3 : Tensor "bf16[128, 2048, 5, 64][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_2, 3, 0, 64), kwargs = {})
#   %slice_1 : Tensor "bf16[1, 2048, 1, 64][1310720, 64, 64, 1]cuda:0"[num_users=40] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 0, 2048), kwargs = {})
#   %mul_4 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_3, %slice_1), kwargs = {})
#   %slice_4 : Tensor "bf16[128, 2048, 5, 64][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_2, 3, 64, 9223372036854775807), kwargs = {})
#   %slice_2 : Tensor "bf16[1, 2048, 1, 64][1310720, 64, 64, 1]cuda:0"[num_users=40] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg2_1, 1, 0, 2048), kwargs = {})
#   %mul_5 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_4, %slice_2), kwargs = {})
#   %add_3 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %neg : Tensor "bf16[1, 2048, 1, 64][131072, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %mul_6 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_3, %neg), kwargs = {})
#   %mul_7 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_4, %slice_1), kwargs = {})
#   %add_4 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %cat : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_3, %add_4], 3), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_13, 2), kwargs = {})
#   %mean_2 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [3], True), kwargs = {})
#   %add_7 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_2, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_2 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %mul_12 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_13, %rsqrt_2), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12, torch.bfloat16), kwargs = {})
#   %pow_4 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_15, 2), kwargs = {})
#   %mean_3 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [3], True), kwargs = {})
#   %add_8 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_3, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_3 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_13 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_15, %rsqrt_3), kwargs = {})
#   %convert_element_type_16 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.bfloat16), kwargs = {})
#   %view_7 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [128, 2048, 640]), kwargs = {})
#   %view_8 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_7, [128, 2048, 5, 128]), kwargs = {})
#   %_flash_attn_forward : [num_users=1] = call_function[target=torch.ops.flash_attn_3._flash_attn_forward.default](args = (%convert_element_type_14, %convert_element_type_16, %view_8, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0), kwargs = {})
#   return %convert_element_type_13,%buf7,%buf14
triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2 = async_compile.triton('triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 1946157056}}
)
@triton.jit
def triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_3 = r0_index
    x4 = xindex
    x1 = ((xindex // 5) % 2048)
    tmp0 = r0_3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x4 + (r0_3)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (64*x1 + (r0_3)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.load(in_ptr0 + (64 + 128*x4 + (r0_3)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (64*x1 + (r0_3)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 128, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr0 + (128*x4 + ((-64) + r0_3)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (64*x1 + ((-64) + r0_3)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = -tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tl.load(in_ptr0 + (64 + 128*x4 + ((-64) + r0_3)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr1 + (64*x1 + ((-64) + r0_3)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp14, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp13, tmp26)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp28
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp32 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
    tmp33 = 128.0
    tmp34 = (tmp32 / tmp33)
    tmp35 = 1.1920928955078125e-07
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp28 * tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr2 + (r0_3 + 128*x4), tmp39, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/76/c76t4vnfp6xzmugyte3mxnove233gsc3bvbb6xmeqyo7qf3dnivm.py
# Topologically Sorted Source Nodes: [y_1, x_3, rms_norm_4], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   rms_norm_4 => add_10, convert_element_type_20, convert_element_type_21, mean_4, mul_14, pow_5, rsqrt_4
#   x_3 => add_9
#   y_1 => view_11
# Graph fragment:
#   %add_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_1]
#   %mm_3 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %buf23 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf23]
#   %view_11 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [128, 2048, 640]), kwargs = {})
#   %add_9 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_11), kwargs = {})
#   %convert_element_type_20 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.float32), kwargs = {})
#   %pow_5 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_20, 2), kwargs = {})
#   %mean_4 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [2], True), kwargs = {})
#   %add_10 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_4, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_4 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_14 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, %rsqrt_4), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14, torch.bfloat16), kwargs = {})
#   return %buf23,%convert_element_type_21
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 1342177280}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = 640.0
    tmp10 = (tmp8 / tmp9)
    tmp11 = 1.1920928955078125e-07
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp15, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/w5/cw5diknm2ztvr77pififqkjcfhrxeirlzgarlhe55ccfm42bssdv.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_4 => convert_element_type_22
# Graph fragment:
#   %arg10_1 : Tensor "f32[2560, 640][640, 1]cuda:0" = PlaceHolder[target=arg10_1]
#   %convert_element_type_22 : Tensor "bf16[2560, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg10_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_22
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 13107200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/bn/cbnstcepm5lhylg3yv7e43bhqysv3553hbtkxpp3dyu2wkzrv74m.py
# Topologically Sorted Source Nodes: [x_4, relu, x_5, x_6], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
# Source node to ATen node mapping:
#   relu => relu
#   x_4 => view_13
#   x_5 => convert_element_type_25, pow_6
#   x_6 => convert_element_type_27
# Graph fragment:
#   %mm_4 : Tensor "bf16[262144, 2560][2560, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %view_13 : Tensor "bf16[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [128, 2048, 2560]), kwargs = {})
#   %relu : Tensor "bf16[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_13,), kwargs = {})
#   %convert_element_type_25 : Tensor "f32[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%relu, torch.float32), kwargs = {})
#   %pow_6 : Tensor "f32[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_25, 2), kwargs = {})
#   %convert_element_type_27 : Tensor "bf16[128, 2048, 2560][5242880, 2560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_6, torch.bfloat16), kwargs = {})
#   return %convert_element_type_27
triton_poi_fused__to_copy__unsafe_view_pow_relu_5 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_pow_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_pow_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4026531840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_pow_relu_5(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 671088640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp5, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/ys/cysre73fasdx6ycrtja6rcn3rbxxxiio7txhsmxr2koz7moeqi2d.py
# Topologically Sorted Source Nodes: [x, x_1, getitem_12, y_1, x_3, x_6, x_7, mul_10, getitem_13, mul_11, x_8, rms_norm_5], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
# Source node to ATen node mapping:
#   getitem_12 => select_2
#   getitem_13 => select_3
#   mul_10 => mul_15
#   mul_11 => mul_16
#   rms_norm_5 => add_13, convert_element_type_30, convert_element_type_31, mean_5, mul_17, pow_7, rsqrt_5
#   x => embedding
#   x_1 => add, convert_element_type, convert_element_type_1, mean, mul, pow_1, rsqrt
#   x_3 => add_9
#   x_6 => view_15
#   x_7 => add_11
#   x_8 => add_12
#   y_1 => view_11
# Graph fragment:
#   %arg4_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_1]
#   %mm_3 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %mm_5 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %arg5_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg0_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %buf0 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf0]
#   %add_12 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_12]
#   %buf31 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf31]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1), kwargs = {})
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %select_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg4_1, 0, 1), kwargs = {})
#   %view_11 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [128, 2048, 640]), kwargs = {})
#   %add_9 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_11), kwargs = {})
#   %view_15 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [128, 2048, 640]), kwargs = {})
#   %add_11 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_15), kwargs = {})
#   %mul_15 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %add_11), kwargs = {})
#   %select_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg5_1, 0, 1), kwargs = {})
#   %mul_16 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %convert_element_type_1), kwargs = {})
#   %add_12 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %mul_16), kwargs = {})
#   %convert_element_type_30 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %pow_7 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_30, 2), kwargs = {})
#   %mean_5 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [2], True), kwargs = {})
#   %add_13 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_5, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_5 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_17 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_30, %rsqrt_5), kwargs = {})
#   %convert_element_type_31 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17, torch.bfloat16), kwargs = {})
#   return %add_12,%buf31,%convert_element_type_31
triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_6 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (1))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (1))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert((0 <= tmp16) & (tmp16 < 8192), "index out of bounds: 0 <= tmp16 < 8192")
    tmp18 = tl.load(in_ptr5 + (r0_1 + 640*tmp16), r0_mask, other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = 640.0
    tmp22 = (tmp20 / tmp21)
    tmp23 = 1.1920928955078125e-07
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp19 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp11 * tmp27
    tmp29 = tmp8 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(r0_mask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp36 = (tmp35 / tmp21)
    tmp37 = tmp36 + tmp23
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp30 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp29, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp40, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/rn/crnsuei6t7uzli3blt7boakx3hgmyg7ddoksi6apbkt3gmtbr7q7.py
# Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_9 => convert_element_type_41
# Graph fragment:
#   %arg16_1 : Tensor "f32[5, 32][32, 1]cuda:0" = PlaceHolder[target=arg16_1]
#   %convert_element_type_41 : Tensor "bf16[5, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg16_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_41
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/qt/cqtla37q4lj4ea6q55m5izrdl2sr44icjwycheo2snc2wdc53da6.py
# Topologically Sorted Source Nodes: [q_5, k_5, linear_8, v_1, sigmoid, gate, unsqueeze, ve, ve_1, mul_13, v_2, _flash_attn_forward_default_1], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
# Source node to ATen node mapping:
#   _flash_attn_forward_default_1 => _flash_attn_forward_1
#   gate => mul_18
#   k_5 => add_20, convert_element_type_47, mean_7, mul_29, pow_9, rsqrt_7
#   linear_8 => view_23
#   mul_13 => mul_19
#   q_5 => add_19, convert_element_type_45, mean_6, mul_28, pow_8, rsqrt_6
#   sigmoid => sigmoid
#   unsqueeze => unsqueeze
#   v_1 => view_24
#   v_2 => add_14
#   ve => embedding_1
#   ve_1 => view_25
# Graph fragment:
#   %mm_8 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %bmm : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0" = PlaceHolder[target=bmm]
#   %arg0_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg12_1 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=arg12_1]
#   %pow_8 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_44, 2), kwargs = {})
#   %mean_6 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [3], True), kwargs = {})
#   %add_19 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_6, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_6 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_28 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_44, %rsqrt_6), kwargs = {})
#   %convert_element_type_45 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_28, torch.bfloat16), kwargs = {})
#   %pow_9 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_46, 2), kwargs = {})
#   %mean_7 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [3], True), kwargs = {})
#   %add_20 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_7, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_7 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_20,), kwargs = {})
#   %mul_29 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_46, %rsqrt_7), kwargs = {})
#   %convert_element_type_47 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_29, torch.bfloat16), kwargs = {})
#   %view_23 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [128, 2048, 640]), kwargs = {})
#   %view_24 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_23, [128, 2048, 5, 128]), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%bmm,), kwargs = {})
#   %mul_18 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, 2), kwargs = {})
#   %unsqueeze : Tensor "bf16[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %embedding_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg12_1, %arg0_1), kwargs = {})
#   %view_25 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%embedding_1, [128, 2048, 5, 128]), kwargs = {})
#   %mul_19 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %view_25), kwargs = {})
#   %add_14 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_24, %mul_19), kwargs = {})
#   %_flash_attn_forward_1 : [num_users=1] = call_function[target=torch.ops.flash_attn_3._flash_attn_forward.default](args = (%convert_element_type_45, %convert_element_type_47, %add_14, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0), kwargs = {})
#   return %buf47
triton_poi_fused__flash_attn_forward__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_sigmoid_unsqueeze_view_8 = async_compile.triton('triton_poi_fused__flash_attn_forward__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_sigmoid_unsqueeze_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__flash_attn_forward__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_sigmoid_unsqueeze_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__flash_attn_forward__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_sigmoid_unsqueeze_view_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 167772160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 128
    x2 = xindex // 640
    x5 = (xindex % 640)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp6 = tl.full([XBLOCK], 8192, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tl.device_assert((0 <= tmp9) & (tmp9 < 8192), "index out of bounds: 0 <= tmp9 < 8192")
    tmp11 = tl.load(in_ptr2 + (x5 + 640*tmp9), None).to(tl.float32)
    tmp12 = tmp4 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/tv/ctv66su7djglpa6p5u7ix7o3nejqgmy2zjjhbcanw2v6lvfjohvs.py
# Topologically Sorted Source Nodes: [x, x_1, getitem_23, y_3, x_9, x_12, x_13, mul_22, getitem_24, mul_23, x_14, rms_norm_9], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
# Source node to ATen node mapping:
#   getitem_23 => select_4
#   getitem_24 => select_5
#   mul_22 => mul_31
#   mul_23 => mul_32
#   rms_norm_9 => add_25, convert_element_type_61, convert_element_type_62, mean_9, mul_33, pow_12, rsqrt_9
#   x => embedding
#   x_1 => add, convert_element_type, convert_element_type_1, mean, mul, pow_1, rsqrt
#   x_12 => view_35
#   x_13 => add_23
#   x_14 => add_24
#   x_9 => add_21
#   y_3 => view_31
# Graph fragment:
#   %arg4_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add_12 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_12]
#   %mm_9 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_9]
#   %mm_11 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_11]
#   %arg5_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg0_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %buf0 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf0]
#   %add_24 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_24]
#   %buf63 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf63]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1), kwargs = {})
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %select_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg4_1, 0, 2), kwargs = {})
#   %view_31 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [128, 2048, 640]), kwargs = {})
#   %add_21 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %view_31), kwargs = {})
#   %view_35 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_11, [128, 2048, 640]), kwargs = {})
#   %add_23 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_35), kwargs = {})
#   %mul_31 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, %add_23), kwargs = {})
#   %select_5 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg5_1, 0, 2), kwargs = {})
#   %mul_32 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %convert_element_type_1), kwargs = {})
#   %add_24 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %mul_32), kwargs = {})
#   %convert_element_type_61 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_24, torch.float32), kwargs = {})
#   %pow_12 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_61, 2), kwargs = {})
#   %mean_9 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_12, [2], True), kwargs = {})
#   %add_25 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_9, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_9 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_33 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_61, %rsqrt_9), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_33, torch.bfloat16), kwargs = {})
#   return %add_24,%buf63,%convert_element_type_62
triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_9 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (2))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (2))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert((0 <= tmp16) & (tmp16 < 8192), "index out of bounds: 0 <= tmp16 < 8192")
    tmp18 = tl.load(in_ptr5 + (r0_1 + 640*tmp16), r0_mask, other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = 640.0
    tmp22 = (tmp20 / tmp21)
    tmp23 = 1.1920928955078125e-07
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp19 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp11 * tmp27
    tmp29 = tmp8 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(r0_mask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp36 = (tmp35 / tmp21)
    tmp37 = tmp36 + tmp23
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp30 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp29, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp40, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/xg/cxgiyh4bw2o6r2n5uiectnanh7xy6dy4tp64jo7j6ouqd42gy2xw.py
# Topologically Sorted Source Nodes: [x, x_1, getitem_33, y_5, x_15, x_18, x_19, mul_32, getitem_34, mul_33, x_20, rms_norm_13], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
# Source node to ATen node mapping:
#   getitem_33 => select_6
#   getitem_34 => select_7
#   mul_32 => mul_45
#   mul_33 => mul_46
#   rms_norm_13 => add_36, convert_element_type_89, convert_element_type_90, mean_13, mul_47, pow_17, rsqrt_13
#   x => embedding
#   x_1 => add, convert_element_type, convert_element_type_1, mean, mul, pow_1, rsqrt
#   x_15 => add_32
#   x_18 => view_51
#   x_19 => add_34
#   x_20 => add_35
#   y_5 => view_47
# Graph fragment:
#   %arg4_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add_24 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_24]
#   %mm_15 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_15]
#   %mm_17 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_17]
#   %arg5_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg0_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %buf0 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf0]
#   %add_35 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_35]
#   %buf92 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf92]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1), kwargs = {})
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %select_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg4_1, 0, 3), kwargs = {})
#   %view_47 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_15, [128, 2048, 640]), kwargs = {})
#   %add_32 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %view_47), kwargs = {})
#   %view_51 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_17, [128, 2048, 640]), kwargs = {})
#   %add_34 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_32, %view_51), kwargs = {})
#   %mul_45 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_6, %add_34), kwargs = {})
#   %select_7 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg5_1, 0, 3), kwargs = {})
#   %mul_46 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_7, %convert_element_type_1), kwargs = {})
#   %add_35 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_46), kwargs = {})
#   %convert_element_type_89 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_35, torch.float32), kwargs = {})
#   %pow_17 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_89, 2), kwargs = {})
#   %mean_13 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_17, [2], True), kwargs = {})
#   %add_36 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_13, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_13 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_36,), kwargs = {})
#   %mul_47 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_89, %rsqrt_13), kwargs = {})
#   %convert_element_type_90 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_47, torch.bfloat16), kwargs = {})
#   return %add_35,%buf92,%convert_element_type_90
triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_10 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (3))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (3))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert((0 <= tmp16) & (tmp16 < 8192), "index out of bounds: 0 <= tmp16 < 8192")
    tmp18 = tl.load(in_ptr5 + (r0_1 + 640*tmp16), r0_mask, other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = 640.0
    tmp22 = (tmp20 / tmp21)
    tmp23 = 1.1920928955078125e-07
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp19 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp11 * tmp27
    tmp29 = tmp8 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(r0_mask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp36 = (tmp35 / tmp21)
    tmp37 = tmp36 + tmp23
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp30 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp29, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp40, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/j3/cj32lcpzcn23ap42zjcnsomxn44pxwl6kiniamdc2p6clxumzzek.py
# Topologically Sorted Source Nodes: [x, x_1, getitem_44, y_7, x_21, x_24, x_25, mul_44, getitem_45, mul_45, x_26, rms_norm_17], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
# Source node to ATen node mapping:
#   getitem_44 => select_8
#   getitem_45 => select_9
#   mul_44 => mul_61
#   mul_45 => mul_62
#   rms_norm_17 => add_48, convert_element_type_120, convert_element_type_121, mean_17, mul_63, pow_22, rsqrt_17
#   x => embedding
#   x_1 => add, convert_element_type, convert_element_type_1, mean, mul, pow_1, rsqrt
#   x_21 => add_44
#   x_24 => view_71
#   x_25 => add_46
#   x_26 => add_47
#   y_7 => view_67
# Graph fragment:
#   %arg4_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add_35 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_35]
#   %mm_21 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_21]
#   %mm_23 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_23]
#   %arg5_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg0_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %buf0 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf0]
#   %add_47 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_47]
#   %buf124 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf124]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1), kwargs = {})
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %select_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg4_1, 0, 4), kwargs = {})
#   %view_67 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_21, [128, 2048, 640]), kwargs = {})
#   %add_44 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_35, %view_67), kwargs = {})
#   %view_71 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_23, [128, 2048, 640]), kwargs = {})
#   %add_46 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_44, %view_71), kwargs = {})
#   %mul_61 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %add_46), kwargs = {})
#   %select_9 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg5_1, 0, 4), kwargs = {})
#   %mul_62 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, %convert_element_type_1), kwargs = {})
#   %add_47 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_61, %mul_62), kwargs = {})
#   %convert_element_type_120 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_47, torch.float32), kwargs = {})
#   %pow_22 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_120, 2), kwargs = {})
#   %mean_17 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_22, [2], True), kwargs = {})
#   %add_48 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_17, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_17 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_48,), kwargs = {})
#   %mul_63 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_120, %rsqrt_17), kwargs = {})
#   %convert_element_type_121 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_63, torch.bfloat16), kwargs = {})
#   return %add_47,%buf124,%convert_element_type_121
triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_11 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (4))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (4))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert((0 <= tmp16) & (tmp16 < 8192), "index out of bounds: 0 <= tmp16 < 8192")
    tmp18 = tl.load(in_ptr5 + (r0_1 + 640*tmp16), r0_mask, other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = 640.0
    tmp22 = (tmp20 / tmp21)
    tmp23 = 1.1920928955078125e-07
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp19 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp11 * tmp27
    tmp29 = tmp8 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(r0_mask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp36 = (tmp35 / tmp21)
    tmp37 = tmp36 + tmp23
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp30 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp29, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp40, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/uk/cukyu2tyzeqxmtywwnya2f4wdtnrtzjbahzzejak5tj6duahqcfs.py
# Topologically Sorted Source Nodes: [x, x_1, getitem_54, y_9, x_27, x_30, x_31, mul_54, getitem_55, mul_55, x_32, rms_norm_21], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
# Source node to ATen node mapping:
#   getitem_54 => select_10
#   getitem_55 => select_11
#   mul_54 => mul_75
#   mul_55 => mul_76
#   rms_norm_21 => add_59, convert_element_type_148, convert_element_type_149, mean_21, mul_77, pow_27, rsqrt_21
#   x => embedding
#   x_1 => add, convert_element_type, convert_element_type_1, mean, mul, pow_1, rsqrt
#   x_27 => add_55
#   x_30 => view_87
#   x_31 => add_57
#   x_32 => add_58
#   y_9 => view_83
# Graph fragment:
#   %arg4_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add_47 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_47]
#   %mm_27 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_27]
#   %mm_29 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_29]
#   %arg5_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg0_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %buf0 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf0]
#   %add_58 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_58]
#   %buf153 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf153]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1), kwargs = {})
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %select_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg4_1, 0, 5), kwargs = {})
#   %view_83 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_27, [128, 2048, 640]), kwargs = {})
#   %add_55 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_47, %view_83), kwargs = {})
#   %view_87 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_29, [128, 2048, 640]), kwargs = {})
#   %add_57 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_55, %view_87), kwargs = {})
#   %mul_75 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_10, %add_57), kwargs = {})
#   %select_11 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg5_1, 0, 5), kwargs = {})
#   %mul_76 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_11, %convert_element_type_1), kwargs = {})
#   %add_58 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, %mul_76), kwargs = {})
#   %convert_element_type_148 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_58, torch.float32), kwargs = {})
#   %pow_27 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_148, 2), kwargs = {})
#   %mean_21 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_27, [2], True), kwargs = {})
#   %add_59 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_21, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_21 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_59,), kwargs = {})
#   %mul_77 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_148, %rsqrt_21), kwargs = {})
#   %convert_element_type_149 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_77, torch.bfloat16), kwargs = {})
#   return %add_58,%buf153,%convert_element_type_149
triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_12 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (5))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (5))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert((0 <= tmp16) & (tmp16 < 8192), "index out of bounds: 0 <= tmp16 < 8192")
    tmp18 = tl.load(in_ptr5 + (r0_1 + 640*tmp16), r0_mask, other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = 640.0
    tmp22 = (tmp20 / tmp21)
    tmp23 = 1.1920928955078125e-07
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp19 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp11 * tmp27
    tmp29 = tmp8 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(r0_mask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp36 = (tmp35 / tmp21)
    tmp37 = tmp36 + tmp23
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp30 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp29, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp40, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/ga/cga2b36j6mb6qaj2ucfmtfmtmuzog776hux3zm75wecwkfr4mlsu.py
# Topologically Sorted Source Nodes: [x, x_1, getitem_65, y_11, x_33, x_36, x_37, mul_66, getitem_66, mul_67, x_38, rms_norm_25], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
# Source node to ATen node mapping:
#   getitem_65 => select_12
#   getitem_66 => select_13
#   mul_66 => mul_91
#   mul_67 => mul_92
#   rms_norm_25 => add_71, convert_element_type_179, convert_element_type_180, mean_25, mul_93, pow_32, rsqrt_25
#   x => embedding
#   x_1 => add, convert_element_type, convert_element_type_1, mean, mul, pow_1, rsqrt
#   x_33 => add_67
#   x_36 => view_107
#   x_37 => add_69
#   x_38 => add_70
#   y_11 => view_103
# Graph fragment:
#   %arg4_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add_58 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_58]
#   %mm_33 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_33]
#   %mm_35 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_35]
#   %arg5_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg0_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %buf0 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf0]
#   %add_70 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_70]
#   %buf185 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf185]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1), kwargs = {})
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %select_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg4_1, 0, 6), kwargs = {})
#   %view_103 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_33, [128, 2048, 640]), kwargs = {})
#   %add_67 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_58, %view_103), kwargs = {})
#   %view_107 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_35, [128, 2048, 640]), kwargs = {})
#   %add_69 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_67, %view_107), kwargs = {})
#   %mul_91 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_12, %add_69), kwargs = {})
#   %select_13 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg5_1, 0, 6), kwargs = {})
#   %mul_92 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_13, %convert_element_type_1), kwargs = {})
#   %add_70 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %mul_92), kwargs = {})
#   %convert_element_type_179 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_70, torch.float32), kwargs = {})
#   %pow_32 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_179, 2), kwargs = {})
#   %mean_25 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_32, [2], True), kwargs = {})
#   %add_71 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_25, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_25 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_71,), kwargs = {})
#   %mul_93 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_179, %rsqrt_25), kwargs = {})
#   %convert_element_type_180 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_93, torch.bfloat16), kwargs = {})
#   return %add_70,%buf185,%convert_element_type_180
triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_13 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (6))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (6))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert((0 <= tmp16) & (tmp16 < 8192), "index out of bounds: 0 <= tmp16 < 8192")
    tmp18 = tl.load(in_ptr5 + (r0_1 + 640*tmp16), r0_mask, other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = 640.0
    tmp22 = (tmp20 / tmp21)
    tmp23 = 1.1920928955078125e-07
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp19 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp11 * tmp27
    tmp29 = tmp8 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(r0_mask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp36 = (tmp35 / tmp21)
    tmp37 = tmp36 + tmp23
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp30 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp29, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp40, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/nd/cndfaiwjt4anyd7zye3y57iv5tkiuwhg7gbkyhkdq2uh7uncy4e2.py
# Topologically Sorted Source Nodes: [x, x_1, getitem_75, y_13, x_39, x_42, x_43, mul_76, getitem_76, mul_77, x_44, rms_norm_29], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
# Source node to ATen node mapping:
#   getitem_75 => select_14
#   getitem_76 => select_15
#   mul_76 => mul_105
#   mul_77 => mul_106
#   rms_norm_29 => add_82, convert_element_type_207, convert_element_type_208, mean_29, mul_107, pow_37, rsqrt_29
#   x => embedding
#   x_1 => add, convert_element_type, convert_element_type_1, mean, mul, pow_1, rsqrt
#   x_39 => add_78
#   x_42 => view_123
#   x_43 => add_80
#   x_44 => add_81
#   y_13 => view_119
# Graph fragment:
#   %arg4_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add_70 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_70]
#   %mm_39 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_39]
#   %mm_41 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_41]
#   %arg5_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg0_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %buf0 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf0]
#   %add_81 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_81]
#   %buf214 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf214]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1), kwargs = {})
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %select_14 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg4_1, 0, 7), kwargs = {})
#   %view_119 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_39, [128, 2048, 640]), kwargs = {})
#   %add_78 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_70, %view_119), kwargs = {})
#   %view_123 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_41, [128, 2048, 640]), kwargs = {})
#   %add_80 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_78, %view_123), kwargs = {})
#   %mul_105 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_14, %add_80), kwargs = {})
#   %select_15 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg5_1, 0, 7), kwargs = {})
#   %mul_106 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_15, %convert_element_type_1), kwargs = {})
#   %add_81 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_105, %mul_106), kwargs = {})
#   %convert_element_type_207 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_81, torch.float32), kwargs = {})
#   %pow_37 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_207, 2), kwargs = {})
#   %mean_29 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_37, [2], True), kwargs = {})
#   %add_82 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_29, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_29 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_82,), kwargs = {})
#   %mul_107 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_207, %rsqrt_29), kwargs = {})
#   %convert_element_type_208 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_107, torch.bfloat16), kwargs = {})
#   return %add_81,%buf214,%convert_element_type_208
triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_14 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (7))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (7))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert((0 <= tmp16) & (tmp16 < 8192), "index out of bounds: 0 <= tmp16 < 8192")
    tmp18 = tl.load(in_ptr5 + (r0_1 + 640*tmp16), r0_mask, other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = 640.0
    tmp22 = (tmp20 / tmp21)
    tmp23 = 1.1920928955078125e-07
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp19 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp11 * tmp27
    tmp29 = tmp8 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(r0_mask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp36 = (tmp35 / tmp21)
    tmp37 = tmp36 + tmp23
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp30 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp29, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp40, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/7b/c7bch54vwgpqnmusyhutz2wswkidpjox7lzftzd2bevjiutv66ub.py
# Topologically Sorted Source Nodes: [x, x_1, getitem_86, y_15, x_45, x_48, x_49, mul_88, getitem_87, mul_89, x_50, rms_norm_33], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
# Source node to ATen node mapping:
#   getitem_86 => select_16
#   getitem_87 => select_17
#   mul_88 => mul_121
#   mul_89 => mul_122
#   rms_norm_33 => add_94, convert_element_type_238, convert_element_type_239, mean_33, mul_123, pow_42, rsqrt_33
#   x => embedding
#   x_1 => add, convert_element_type, convert_element_type_1, mean, mul, pow_1, rsqrt
#   x_45 => add_90
#   x_48 => view_143
#   x_49 => add_92
#   x_50 => add_93
#   y_15 => view_139
# Graph fragment:
#   %arg4_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add_81 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_81]
#   %mm_45 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_45]
#   %mm_47 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_47]
#   %arg5_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg0_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %buf0 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf0]
#   %add_93 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_93]
#   %buf246 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf246]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1), kwargs = {})
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %select_16 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg4_1, 0, 8), kwargs = {})
#   %view_139 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_45, [128, 2048, 640]), kwargs = {})
#   %add_90 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_81, %view_139), kwargs = {})
#   %view_143 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_47, [128, 2048, 640]), kwargs = {})
#   %add_92 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_90, %view_143), kwargs = {})
#   %mul_121 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_16, %add_92), kwargs = {})
#   %select_17 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg5_1, 0, 8), kwargs = {})
#   %mul_122 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_17, %convert_element_type_1), kwargs = {})
#   %add_93 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_121, %mul_122), kwargs = {})
#   %convert_element_type_238 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_93, torch.float32), kwargs = {})
#   %pow_42 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_238, 2), kwargs = {})
#   %mean_33 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_42, [2], True), kwargs = {})
#   %add_94 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_33, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_33 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_94,), kwargs = {})
#   %mul_123 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_238, %rsqrt_33), kwargs = {})
#   %convert_element_type_239 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_123, torch.bfloat16), kwargs = {})
#   return %add_93,%buf246,%convert_element_type_239
triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_15 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (8))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (8))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert((0 <= tmp16) & (tmp16 < 8192), "index out of bounds: 0 <= tmp16 < 8192")
    tmp18 = tl.load(in_ptr5 + (r0_1 + 640*tmp16), r0_mask, other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = 640.0
    tmp22 = (tmp20 / tmp21)
    tmp23 = 1.1920928955078125e-07
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp19 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp11 * tmp27
    tmp29 = tmp8 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(r0_mask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp36 = (tmp35 / tmp21)
    tmp37 = tmp36 + tmp23
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp30 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp29, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp40, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/cl/ccltac6qwcxmnhpjuk5ysfuldje5tw67v7z7eb6t7zxkxyr2k5ry.py
# Topologically Sorted Source Nodes: [x, x_1, getitem_96, y_17, x_51, x_54, x_55, mul_98, getitem_97, mul_99, x_56, rms_norm_37], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
# Source node to ATen node mapping:
#   getitem_96 => select_18
#   getitem_97 => select_19
#   mul_98 => mul_135
#   mul_99 => mul_136
#   rms_norm_37 => add_105, convert_element_type_266, convert_element_type_267, mean_37, mul_137, pow_47, rsqrt_37
#   x => embedding
#   x_1 => add, convert_element_type, convert_element_type_1, mean, mul, pow_1, rsqrt
#   x_51 => add_101
#   x_54 => view_159
#   x_55 => add_103
#   x_56 => add_104
#   y_17 => view_155
# Graph fragment:
#   %arg4_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add_93 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_93]
#   %mm_51 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_51]
#   %mm_53 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_53]
#   %arg5_1 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg0_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %buf0 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf0]
#   %add_104 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_104]
#   %buf275 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf275]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1), kwargs = {})
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %select_18 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg4_1, 0, 9), kwargs = {})
#   %view_155 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_51, [128, 2048, 640]), kwargs = {})
#   %add_101 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_93, %view_155), kwargs = {})
#   %view_159 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_53, [128, 2048, 640]), kwargs = {})
#   %add_103 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_101, %view_159), kwargs = {})
#   %mul_135 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_18, %add_103), kwargs = {})
#   %select_19 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg5_1, 0, 9), kwargs = {})
#   %mul_136 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_19, %convert_element_type_1), kwargs = {})
#   %add_104 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_135, %mul_136), kwargs = {})
#   %convert_element_type_266 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_104, torch.float32), kwargs = {})
#   %pow_47 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_266, 2), kwargs = {})
#   %mean_37 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_47, [2], True), kwargs = {})
#   %add_105 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_37, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_37 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_105,), kwargs = {})
#   %mul_137 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_266, %rsqrt_37), kwargs = {})
#   %convert_element_type_267 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_137, torch.bfloat16), kwargs = {})
#   return %add_104,%buf275,%convert_element_type_267
triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_16 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (9))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (9))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert((0 <= tmp16) & (tmp16 < 8192), "index out of bounds: 0 <= tmp16 < 8192")
    tmp18 = tl.load(in_ptr5 + (r0_1 + 640*tmp16), r0_mask, other=0.0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = 640.0
    tmp22 = (tmp20 / tmp21)
    tmp23 = 1.1920928955078125e-07
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp19 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp11 * tmp27
    tmp29 = tmp8 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(r0_mask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp36 = (tmp35 / tmp21)
    tmp37 = tmp36 + tmp23
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp30 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp29, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp40, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/be/cbeke7bsqdv4bkxayeqvxll5fj4nnj256f7tgg4tyllyimmnk3ni.py
# Topologically Sorted Source Nodes: [y_19, x_57, x_60, x_61, x_62], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   x_57 => add_113
#   x_60 => view_179
#   x_61 => add_115
#   x_62 => add_116, convert_element_type_297, convert_element_type_298, mean_41, mul_151, pow_52, rsqrt_41
#   y_19 => view_175
# Graph fragment:
#   %add_104 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_104]
#   %mm_57 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_57]
#   %mm_59 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_59]
#   %buf306 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf306]
#   %view_175 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_57, [128, 2048, 640]), kwargs = {})
#   %add_113 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_104, %view_175), kwargs = {})
#   %view_179 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_59, [128, 2048, 640]), kwargs = {})
#   %add_115 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_113, %view_179), kwargs = {})
#   %convert_element_type_297 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_115, torch.float32), kwargs = {})
#   %pow_52 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_297, 2), kwargs = {})
#   %mean_41 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_52, [2], True), kwargs = {})
#   %add_116 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_41, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_41 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_116,), kwargs = {})
#   %mul_151 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_297, %rsqrt_41), kwargs = {})
#   %convert_element_type_298 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_151, torch.bfloat16), kwargs = {})
#   return %buf306,%convert_element_type_298
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 1677721600}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = 640.0
    tmp12 = (tmp10 / tmp11)
    tmp13 = 1.1920928955078125e-07
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp5 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp17, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/mx/cmxbcgblgx57dpjlrqs34w2myj3evqt2tobxn7wiglgrvl37aq3h.py
# Topologically Sorted Source Nodes: [logits], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   logits => convert_element_type_299
# Graph fragment:
#   %arg76_1 : Tensor "f32[8192, 640][640, 1]cuda:0" = PlaceHolder[target=arg76_1]
#   %convert_element_type_299 : Tensor "bf16[8192, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg76_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_299
triton_poi_fused__to_copy_18 = async_compile.triton('triton_poi_fused__to_copy_18', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41943040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/2p/c2pvk2qlgcpc3uho7xx3b3jmt3bpc26r2hkghjwg5s6oqqcxnr2i.py
# Topologically Sorted Source Nodes: [view_46, loss, logits, logits_1, truediv, tanh, logits_2, view_45], Original ATen: [aten.view, aten.nll_loss_forward, aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, prims.prepare_softmax_online, aten.sub, aten._log_softmax]
# Source node to ATen node mapping:
#   logits => view_181
#   logits_1 => convert_element_type_302
#   logits_2 => mul_152
#   loss => full_default, full_default_1, gather, log, ne, ne_1, neg_20, squeeze, sub_1, unsqueeze_5, where, where_1
#   tanh => tanh
#   truediv => div
#   view_45 => view_182
#   view_46 => view_183
# Graph fragment:
#   %mm_60 : Tensor "bf16[262144, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_60]
#   %arg77_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg77_1]
#   %getitem_40 : Tensor "f32[262144, 1][1, 262144]cuda:0" = PlaceHolder[target=getitem_40]
#   %getitem_41 : Tensor "f32[262144, 1][1, 262144]cuda:0" = PlaceHolder[target=getitem_41]
#   %view_183 : Tensor "i64[262144][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%arg77_1, [-1]), kwargs = {})
#   %ne_1 : Tensor "b8[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_183, -1), kwargs = {})
#   %view_181 : Tensor "bf16[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_60, [128, 2048, 8192]), kwargs = {})
#   %convert_element_type_302 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_181, torch.float32), kwargs = {})
#   %div : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_302, 15), kwargs = {})
#   %tanh : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_152 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 15), kwargs = {})
#   %view_182 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_152, [-1, 8192]), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%view_182, 1), kwargs = {})
#   %sub_tensor : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_182, %getitem_40), kwargs = {})
#   %log : Tensor "f32[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%getitem_41,), kwargs = {})
#   %sub_1 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log), kwargs = {})
#   %ne : Tensor "b8[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_183, -1), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %view_183, %full_default), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 1), kwargs = {})
#   %gather : Tensor "f32[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_1, 1, %unsqueeze_5), kwargs = {})
#   %squeeze : Tensor "f32[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg_20 : Tensor "f32[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f32[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg_20, %full_default_1), kwargs = {})
#   return %getitem_40,%getitem_41,%where_1
triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_prepare_softmax_online_sub_tanh_view_19 = async_compile.triton('triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_prepare_softmax_online_sub_tanh_view_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'xnumel': 'i64', 'r0_numel': 'i64', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_prepare_softmax_online_sub_tanh_view_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_prepare_softmax_online_sub_tanh_view_19(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp8_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp8_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = 0.06666666666666667
        tmp3 = tmp1 * tmp2
        tmp4 = libdevice.tanh(tmp3)
        tmp5 = 15.0
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])

        _tmp8_max_next, _tmp8_sum_next = triton_helpers.online_softmax_combine(
            _tmp8_max, _tmp8_sum, tmp7, False
        )

        _tmp8_max = tl.where(r0_mask, _tmp8_max_next, _tmp8_max)
        _tmp8_sum = tl.where(r0_mask, _tmp8_sum_next, _tmp8_sum)

    tmp8, tmp9 = triton_helpers.online_softmax_reduce(
        _tmp8_max, _tmp8_sum, 1, False)
    tmp8 = tmp8[:, None]
    tmp9 = tmp9[:, None]
    tmp10 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.full([1, 1], -1, tl.int64)
    tmp12 = tmp10 != tmp11
    tmp13 = tl.full([1, 1], 0, tl.int64)
    tmp14 = tl.where(tmp12, tmp10, tmp13)
    tmp15 = tl.full([XBLOCK, 1], 8192, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tl.device_assert((0 <= tmp18) & (tmp18 < 8192), "index out of bounds: 0 <= tmp18 < 8192")
    tmp20 = tl.load(in_ptr0 + (tmp18 + 8192*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 0.06666666666666667
    tmp23 = tmp21 * tmp22
    tmp24 = libdevice.tanh(tmp23)
    tmp25 = 15.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26 - tmp8
    tmp28 = tl_math.log(tmp9)
    tmp29 = tmp27 - tmp28
    tmp30 = -tmp29
    tmp31 = 0.0
    tmp32 = tl.where(tmp12, tmp30, tmp31)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp32, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1 = args
        args.clear()
        assert_size_stride(arg0_1, (128, 2048), (2048, 1))
        assert_size_stride(arg1_1, (1, 20480, 1, 64), (1310720, 64, 64, 1))
        assert_size_stride(arg2_1, (1, 20480, 1, 64), (1310720, 64, 64, 1))
        assert_size_stride(arg3_1, (8192, 640), (640, 1))
        assert_size_stride(arg4_1, (10, ), (1, ))
        assert_size_stride(arg5_1, (10, ), (1, ))
        assert_size_stride(arg6_1, (640, 640), (640, 1))
        assert_size_stride(arg7_1, (640, 640), (640, 1))
        assert_size_stride(arg8_1, (640, 640), (640, 1))
        assert_size_stride(arg9_1, (640, 640), (640, 1))
        assert_size_stride(arg10_1, (2560, 640), (640, 1))
        assert_size_stride(arg11_1, (640, 2560), (2560, 1))
        assert_size_stride(arg12_1, (8192, 640), (640, 1))
        assert_size_stride(arg13_1, (640, 640), (640, 1))
        assert_size_stride(arg14_1, (640, 640), (640, 1))
        assert_size_stride(arg15_1, (640, 640), (640, 1))
        assert_size_stride(arg16_1, (5, 32), (32, 1))
        assert_size_stride(arg17_1, (640, 640), (640, 1))
        assert_size_stride(arg18_1, (2560, 640), (640, 1))
        assert_size_stride(arg19_1, (640, 2560), (2560, 1))
        assert_size_stride(arg20_1, (640, 640), (640, 1))
        assert_size_stride(arg21_1, (640, 640), (640, 1))
        assert_size_stride(arg22_1, (640, 640), (640, 1))
        assert_size_stride(arg23_1, (640, 640), (640, 1))
        assert_size_stride(arg24_1, (2560, 640), (640, 1))
        assert_size_stride(arg25_1, (640, 2560), (2560, 1))
        assert_size_stride(arg26_1, (8192, 640), (640, 1))
        assert_size_stride(arg27_1, (640, 640), (640, 1))
        assert_size_stride(arg28_1, (640, 640), (640, 1))
        assert_size_stride(arg29_1, (640, 640), (640, 1))
        assert_size_stride(arg30_1, (5, 32), (32, 1))
        assert_size_stride(arg31_1, (640, 640), (640, 1))
        assert_size_stride(arg32_1, (2560, 640), (640, 1))
        assert_size_stride(arg33_1, (640, 2560), (2560, 1))
        assert_size_stride(arg34_1, (640, 640), (640, 1))
        assert_size_stride(arg35_1, (640, 640), (640, 1))
        assert_size_stride(arg36_1, (640, 640), (640, 1))
        assert_size_stride(arg37_1, (640, 640), (640, 1))
        assert_size_stride(arg38_1, (2560, 640), (640, 1))
        assert_size_stride(arg39_1, (640, 2560), (2560, 1))
        assert_size_stride(arg40_1, (8192, 640), (640, 1))
        assert_size_stride(arg41_1, (640, 640), (640, 1))
        assert_size_stride(arg42_1, (640, 640), (640, 1))
        assert_size_stride(arg43_1, (640, 640), (640, 1))
        assert_size_stride(arg44_1, (5, 32), (32, 1))
        assert_size_stride(arg45_1, (640, 640), (640, 1))
        assert_size_stride(arg46_1, (2560, 640), (640, 1))
        assert_size_stride(arg47_1, (640, 2560), (2560, 1))
        assert_size_stride(arg48_1, (640, 640), (640, 1))
        assert_size_stride(arg49_1, (640, 640), (640, 1))
        assert_size_stride(arg50_1, (640, 640), (640, 1))
        assert_size_stride(arg51_1, (640, 640), (640, 1))
        assert_size_stride(arg52_1, (2560, 640), (640, 1))
        assert_size_stride(arg53_1, (640, 2560), (2560, 1))
        assert_size_stride(arg54_1, (8192, 640), (640, 1))
        assert_size_stride(arg55_1, (640, 640), (640, 1))
        assert_size_stride(arg56_1, (640, 640), (640, 1))
        assert_size_stride(arg57_1, (640, 640), (640, 1))
        assert_size_stride(arg58_1, (5, 32), (32, 1))
        assert_size_stride(arg59_1, (640, 640), (640, 1))
        assert_size_stride(arg60_1, (2560, 640), (640, 1))
        assert_size_stride(arg61_1, (640, 2560), (2560, 1))
        assert_size_stride(arg62_1, (640, 640), (640, 1))
        assert_size_stride(arg63_1, (640, 640), (640, 1))
        assert_size_stride(arg64_1, (640, 640), (640, 1))
        assert_size_stride(arg65_1, (640, 640), (640, 1))
        assert_size_stride(arg66_1, (2560, 640), (640, 1))
        assert_size_stride(arg67_1, (640, 2560), (2560, 1))
        assert_size_stride(arg68_1, (8192, 640), (640, 1))
        assert_size_stride(arg69_1, (640, 640), (640, 1))
        assert_size_stride(arg70_1, (640, 640), (640, 1))
        assert_size_stride(arg71_1, (640, 640), (640, 1))
        assert_size_stride(arg72_1, (5, 32), (32, 1))
        assert_size_stride(arg73_1, (640, 640), (640, 1))
        assert_size_stride(arg74_1, (2560, 640), (640, 1))
        assert_size_stride(arg75_1, (640, 2560), (2560, 1))
        assert_size_stride(arg76_1, (8192, 640), (640, 1))
        assert_size_stride(arg77_1, (128, 2048), (2048, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf1 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf3 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_2, x, x_1, mul, getitem_3, mul_1, x_2, rms_norm_1], Original ATen: [aten.select, aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_select_0.run(arg0_1, arg3_1, arg4_1, arg5_1, buf0, buf1, buf3, 262144, 640, stream=stream0)
            buf4 = empty_strided_cuda((640, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg6_1, buf4, 409600, stream=stream0)
            del arg6_1
            buf5 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_1, linear], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (262144, 640), (640, 1), 0), reinterpret_tensor(buf4, (640, 640), (1, 640), 0), out=buf5)
            buf14 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, q, x1, cos, mul_2, x2, sin, mul_3, y1, neg, mul_4, mul_5, y2, q_1, q_2, k_2, linear_2, v, _flash_attn_forward_default], Original ATen: [aten._unsafe_view, aten.view, aten.slice, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf5, arg1_1, arg2_1, buf14, 1310720, 128, stream=stream0)
            buf8 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg7_1, buf8, 409600, stream=stream0)
            del arg7_1
            buf9 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (262144, 640), (640, 1), 0), reinterpret_tensor(buf8, (640, 640), (1, 640), 0), out=buf9)
            buf15 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, q_2, linear_1, k, x1_1, mul_6, x2_1, mul_7, y1_1, neg_1, mul_8, mul_9, y2_1, k_1, k_2, linear_2, v, _flash_attn_forward_default], Original ATen: [aten.slice, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.neg, aten.cat, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf9, arg1_1, arg2_1, buf15, 1310720, 128, stream=stream0)
            buf12 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg8_1, buf12, 409600, stream=stream0)
            del arg8_1
            buf13 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (262144, 640), (640, 1), 0), reinterpret_tensor(buf12, (640, 640), (1, 640), 0), out=buf13)
            del buf3
            # Topologically Sorted Source Nodes: [q_2, k_2, linear_2, v, _flash_attn_forward_default], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf16 = torch.ops.flash_attn_3._flash_attn_forward.default(buf14, buf15, reinterpret_tensor(buf13, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf17 = buf16[0]
            assert_size_stride(buf17, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf17, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf16
            buf21 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [y_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg9_1, buf21, 409600, stream=stream0)
            del arg9_1
            buf22 = reinterpret_tensor(buf15, (262144, 640), (640, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [y, y_1], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf17, (262144, 640), (640, 1), 0), reinterpret_tensor(buf21, (640, 640), (1, 640), 0), out=buf22)
            buf24 = reinterpret_tensor(buf17, (128, 2048, 640), (1310720, 640, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [y_1, x_3, rms_norm_4], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3.run(buf1, buf22, buf24, 262144, 640, stream=stream0)
            buf25 = empty_strided_cuda((2560, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg10_1, buf25, 1638400, stream=stream0)
            del arg10_1
            buf26 = empty_strided_cuda((262144, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_1, x_3, rms_norm_4, x_4], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf24, (262144, 640), (640, 1), 0), reinterpret_tensor(buf25, (640, 2560), (1, 640), 0), out=buf26)
            buf27 = reinterpret_tensor(buf26, (128, 2048, 2560), (5242880, 2560, 1), 0); del buf26  # reuse
            # Topologically Sorted Source Nodes: [x_4, relu, x_5, x_6], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf27, 671088640, stream=stream0)
            buf28 = reinterpret_tensor(buf25, (640, 2560), (2560, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg11_1, buf28, 1638400, stream=stream0)
            del arg11_1
            buf29 = reinterpret_tensor(buf24, (262144, 640), (640, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [x_4, relu, x_5, x_6], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf27, (262144, 2560), (2560, 1), 0), reinterpret_tensor(buf28, (2560, 640), (1, 2560), 0), out=buf29)
            buf30 = buf1; del buf1  # reuse
            buf32 = reinterpret_tensor(buf14, (128, 2048, 640), (1310720, 640, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, getitem_12, y_1, x_3, x_6, x_7, mul_10, getitem_13, mul_11, x_8, rms_norm_5], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_6.run(buf30, arg4_1, buf22, buf29, arg5_1, arg0_1, arg3_1, buf0, buf32, 262144, 640, stream=stream0)
            buf33 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg13_1, buf33, 409600, stream=stream0)
            del arg13_1
            buf34 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_5, linear_6], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf32, (262144, 640), (640, 1), 0), reinterpret_tensor(buf33, (640, 640), (1, 640), 0), out=buf34)
            buf45 = reinterpret_tensor(buf22, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, linear_6, q_3, x1_2, mul_14, x2_2, mul_15, y1_2, neg_2, mul_16, mul_17, y2_2, q_4, q_5, k_5, linear_8, v_1, sigmoid, gate, unsqueeze, ve, ve_1, mul_13, v_2, _flash_attn_forward_default_1], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf34, arg1_1, arg2_1, buf45, 1310720, 128, stream=stream0)
            buf37 = buf33; del buf33  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg14_1, buf37, 409600, stream=stream0)
            del arg14_1
            buf38 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf32, (262144, 640), (640, 1), 0), reinterpret_tensor(buf37, (640, 640), (1, 640), 0), out=buf38)
            buf46 = reinterpret_tensor(buf13, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, q_5, linear_7, k_3, x1_3, mul_18, x2_3, mul_19, y1_3, neg_3, mul_20, mul_21, y2_3, k_4, k_5, linear_8, v_1, sigmoid, gate, unsqueeze, ve, ve_1, mul_13, v_2, _flash_attn_forward_default_1], Original ATen: [aten.slice, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.neg, aten.cat, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf38, arg1_1, arg2_1, buf46, 1310720, 128, stream=stream0)
            buf41 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg15_1, buf41, 409600, stream=stream0)
            del arg15_1
            buf42 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf32, (262144, 640), (640, 1), 0), reinterpret_tensor(buf41, (640, 640), (1, 640), 0), out=buf42)
            buf43 = empty_strided_cuda((5, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg16_1, buf43, 160, stream=stream0)
            del arg16_1
            buf44 = empty_strided_cuda((128, 2048, 5), (10240, 5, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_14, linear_9], Original ATen: [aten.slice, aten._to_copy, aten.t, aten.expand, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf32, (128, 2048, 32), (1310720, 640, 1), 0), reinterpret_tensor(buf43, (128, 32, 5), (0, 1, 32), 0), out=buf44)
            del buf32
            buf47 = reinterpret_tensor(buf42, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [q_5, k_5, linear_8, v_1, sigmoid, gate, unsqueeze, ve, ve_1, mul_13, v_2, _flash_attn_forward_default_1], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__flash_attn_forward__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_sigmoid_unsqueeze_view_8.run(buf47, buf44, arg0_1, arg12_1, 167772160, stream=stream0)
            del arg12_1
            # Topologically Sorted Source Nodes: [q_5, k_5, linear_8, v_1, sigmoid, gate, unsqueeze, ve, ve_1, mul_13, v_2, _flash_attn_forward_default_1], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            buf48 = torch.ops.flash_attn_3._flash_attn_forward.default(buf45, buf46, buf47, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf49 = buf48[0]
            assert_size_stride(buf49, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf49, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf48
            buf53 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [y_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg17_1, buf53, 409600, stream=stream0)
            del arg17_1
            buf54 = reinterpret_tensor(buf47, (262144, 640), (640, 1), 0); del buf47  # reuse
            # Topologically Sorted Source Nodes: [y_2, y_3], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf49, (262144, 640), (640, 1), 0), reinterpret_tensor(buf53, (640, 640), (1, 640), 0), out=buf54)
            buf56 = reinterpret_tensor(buf49, (128, 2048, 640), (1310720, 640, 1), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [y_3, x_9, rms_norm_8], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3.run(buf30, buf54, buf56, 262144, 640, stream=stream0)
            buf57 = reinterpret_tensor(buf28, (2560, 640), (640, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg18_1, buf57, 1638400, stream=stream0)
            del arg18_1
            buf58 = reinterpret_tensor(buf27, (262144, 2560), (2560, 1), 0); del buf27  # reuse
            # Topologically Sorted Source Nodes: [y_3, x_9, rms_norm_8, x_10], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf56, (262144, 640), (640, 1), 0), reinterpret_tensor(buf57, (640, 2560), (1, 640), 0), out=buf58)
            buf59 = reinterpret_tensor(buf58, (128, 2048, 2560), (5242880, 2560, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [x_10, relu_1, x_11, x_12], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf59, 671088640, stream=stream0)
            buf60 = reinterpret_tensor(buf57, (640, 2560), (2560, 1), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg19_1, buf60, 1638400, stream=stream0)
            del arg19_1
            buf61 = reinterpret_tensor(buf56, (262144, 640), (640, 1), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [x_10, relu_1, x_11, x_12], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf59, (262144, 2560), (2560, 1), 0), reinterpret_tensor(buf60, (2560, 640), (1, 2560), 0), out=buf61)
            buf62 = buf30; del buf30  # reuse
            buf64 = reinterpret_tensor(buf46, (128, 2048, 640), (1310720, 640, 1), 0); del buf46  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, getitem_23, y_3, x_9, x_12, x_13, mul_22, getitem_24, mul_23, x_14, rms_norm_9], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_9.run(buf62, arg4_1, buf54, buf61, arg5_1, arg0_1, arg3_1, buf0, buf64, 262144, 640, stream=stream0)
            buf65 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg20_1, buf65, 409600, stream=stream0)
            del arg20_1
            buf66 = buf61; del buf61  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_9, linear_13], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf64, (262144, 640), (640, 1), 0), reinterpret_tensor(buf65, (640, 640), (1, 640), 0), out=buf66)
            buf75 = reinterpret_tensor(buf54, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, linear_13, q_6, x1_4, mul_24, x2_4, mul_25, y1_4, neg_4, mul_26, mul_27, y2_4, q_7, q_8, k_8, linear_15, v_3, _flash_attn_forward_default_2], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf66, arg1_1, arg2_1, buf75, 1310720, 128, stream=stream0)
            buf69 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg21_1, buf69, 409600, stream=stream0)
            del arg21_1
            buf70 = buf66; del buf66  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf64, (262144, 640), (640, 1), 0), reinterpret_tensor(buf69, (640, 640), (1, 640), 0), out=buf70)
            buf76 = buf45; del buf45  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, q_8, linear_14, k_6, x1_5, mul_28, x2_5, mul_29, y1_5, neg_5, mul_30, mul_31, y2_5, k_7, k_8, linear_15, v_3, _flash_attn_forward_default_2], Original ATen: [aten.slice, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.neg, aten.cat, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf70, arg1_1, arg2_1, buf76, 1310720, 128, stream=stream0)
            buf73 = buf69; del buf69  # reuse
            # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg22_1, buf73, 409600, stream=stream0)
            del arg22_1
            buf74 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf64, (262144, 640), (640, 1), 0), reinterpret_tensor(buf73, (640, 640), (1, 640), 0), out=buf74)
            del buf64
            # Topologically Sorted Source Nodes: [q_8, k_8, linear_15, v_3, _flash_attn_forward_default_2], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf77 = torch.ops.flash_attn_3._flash_attn_forward.default(buf75, buf76, reinterpret_tensor(buf74, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf78 = buf77[0]
            assert_size_stride(buf78, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf78, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf77
            buf82 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [y_5], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg23_1, buf82, 409600, stream=stream0)
            del arg23_1
            buf83 = reinterpret_tensor(buf76, (262144, 640), (640, 1), 0); del buf76  # reuse
            # Topologically Sorted Source Nodes: [y_4, y_5], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf78, (262144, 640), (640, 1), 0), reinterpret_tensor(buf82, (640, 640), (1, 640), 0), out=buf83)
            buf85 = reinterpret_tensor(buf78, (128, 2048, 640), (1310720, 640, 1), 0); del buf78  # reuse
            # Topologically Sorted Source Nodes: [y_5, x_15, rms_norm_12], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3.run(buf62, buf83, buf85, 262144, 640, stream=stream0)
            buf86 = reinterpret_tensor(buf60, (2560, 640), (640, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg24_1, buf86, 1638400, stream=stream0)
            del arg24_1
            buf87 = reinterpret_tensor(buf59, (262144, 2560), (2560, 1), 0); del buf59  # reuse
            # Topologically Sorted Source Nodes: [y_5, x_15, rms_norm_12, x_16], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf85, (262144, 640), (640, 1), 0), reinterpret_tensor(buf86, (640, 2560), (1, 640), 0), out=buf87)
            buf88 = reinterpret_tensor(buf87, (128, 2048, 2560), (5242880, 2560, 1), 0); del buf87  # reuse
            # Topologically Sorted Source Nodes: [x_16, relu_2, x_17, x_18], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf88, 671088640, stream=stream0)
            buf89 = reinterpret_tensor(buf86, (640, 2560), (2560, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg25_1, buf89, 1638400, stream=stream0)
            del arg25_1
            buf90 = reinterpret_tensor(buf85, (262144, 640), (640, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [x_16, relu_2, x_17, x_18], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf88, (262144, 2560), (2560, 1), 0), reinterpret_tensor(buf89, (2560, 640), (1, 2560), 0), out=buf90)
            buf91 = buf62; del buf62  # reuse
            buf93 = reinterpret_tensor(buf75, (128, 2048, 640), (1310720, 640, 1), 0); del buf75  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, getitem_33, y_5, x_15, x_18, x_19, mul_32, getitem_34, mul_33, x_20, rms_norm_13], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_10.run(buf91, arg4_1, buf83, buf90, arg5_1, arg0_1, arg3_1, buf0, buf93, 262144, 640, stream=stream0)
            buf94 = buf82; del buf82  # reuse
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg27_1, buf94, 409600, stream=stream0)
            del arg27_1
            buf95 = buf90; del buf90  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_13, linear_19], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf93, (262144, 640), (640, 1), 0), reinterpret_tensor(buf94, (640, 640), (1, 640), 0), out=buf95)
            buf106 = reinterpret_tensor(buf83, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf83  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, linear_19, q_9, x1_6, mul_36, x2_6, mul_37, y1_6, neg_6, mul_38, mul_39, y2_6, q_10, q_11, k_11, linear_21, v_4, sigmoid_1, gate_1, unsqueeze_1, ve_2, ve_3, mul_35, v_5, _flash_attn_forward_default_3], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf95, arg1_1, arg2_1, buf106, 1310720, 128, stream=stream0)
            buf98 = buf94; del buf94  # reuse
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg28_1, buf98, 409600, stream=stream0)
            del arg28_1
            buf99 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf93, (262144, 640), (640, 1), 0), reinterpret_tensor(buf98, (640, 640), (1, 640), 0), out=buf99)
            buf107 = reinterpret_tensor(buf74, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf74  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, q_11, linear_20, k_9, x1_7, mul_40, x2_7, mul_41, y1_7, neg_7, mul_42, mul_43, y2_7, k_10, k_11, linear_21, v_4, sigmoid_1, gate_1, unsqueeze_1, ve_2, ve_3, mul_35, v_5, _flash_attn_forward_default_3], Original ATen: [aten.slice, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.neg, aten.cat, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf99, arg1_1, arg2_1, buf107, 1310720, 128, stream=stream0)
            buf102 = buf98; del buf98  # reuse
            # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg29_1, buf102, 409600, stream=stream0)
            del arg29_1
            buf103 = buf99; del buf99  # reuse
            # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf93, (262144, 640), (640, 1), 0), reinterpret_tensor(buf102, (640, 640), (1, 640), 0), out=buf103)
            buf104 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg30_1, buf104, 160, stream=stream0)
            del arg30_1
            buf105 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [getitem_35, linear_22], Original ATen: [aten.slice, aten._to_copy, aten.t, aten.expand, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf93, (128, 2048, 32), (1310720, 640, 1), 0), reinterpret_tensor(buf104, (128, 32, 5), (0, 1, 32), 0), out=buf105)
            del buf93
            buf108 = reinterpret_tensor(buf103, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [q_11, k_11, linear_21, v_4, sigmoid_1, gate_1, unsqueeze_1, ve_2, ve_3, mul_35, v_5, _flash_attn_forward_default_3], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__flash_attn_forward__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_sigmoid_unsqueeze_view_8.run(buf108, buf105, arg0_1, arg26_1, 167772160, stream=stream0)
            del arg26_1
            # Topologically Sorted Source Nodes: [q_11, k_11, linear_21, v_4, sigmoid_1, gate_1, unsqueeze_1, ve_2, ve_3, mul_35, v_5, _flash_attn_forward_default_3], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            buf109 = torch.ops.flash_attn_3._flash_attn_forward.default(buf106, buf107, buf108, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf110 = buf109[0]
            assert_size_stride(buf110, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf110, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf109
            buf114 = buf102; del buf102  # reuse
            # Topologically Sorted Source Nodes: [y_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg31_1, buf114, 409600, stream=stream0)
            del arg31_1
            buf115 = reinterpret_tensor(buf108, (262144, 640), (640, 1), 0); del buf108  # reuse
            # Topologically Sorted Source Nodes: [y_6, y_7], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf110, (262144, 640), (640, 1), 0), reinterpret_tensor(buf114, (640, 640), (1, 640), 0), out=buf115)
            buf117 = reinterpret_tensor(buf110, (128, 2048, 640), (1310720, 640, 1), 0); del buf110  # reuse
            # Topologically Sorted Source Nodes: [y_7, x_21, rms_norm_16], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3.run(buf91, buf115, buf117, 262144, 640, stream=stream0)
            buf118 = reinterpret_tensor(buf89, (2560, 640), (640, 1), 0); del buf89  # reuse
            # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg32_1, buf118, 1638400, stream=stream0)
            del arg32_1
            buf119 = reinterpret_tensor(buf88, (262144, 2560), (2560, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [y_7, x_21, rms_norm_16, x_22], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf117, (262144, 640), (640, 1), 0), reinterpret_tensor(buf118, (640, 2560), (1, 640), 0), out=buf119)
            buf120 = reinterpret_tensor(buf119, (128, 2048, 2560), (5242880, 2560, 1), 0); del buf119  # reuse
            # Topologically Sorted Source Nodes: [x_22, relu_3, x_23, x_24], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf120, 671088640, stream=stream0)
            buf121 = reinterpret_tensor(buf118, (640, 2560), (2560, 1), 0); del buf118  # reuse
            # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg33_1, buf121, 1638400, stream=stream0)
            del arg33_1
            buf122 = reinterpret_tensor(buf117, (262144, 640), (640, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [x_22, relu_3, x_23, x_24], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf120, (262144, 2560), (2560, 1), 0), reinterpret_tensor(buf121, (2560, 640), (1, 2560), 0), out=buf122)
            buf123 = buf91; del buf91  # reuse
            buf125 = reinterpret_tensor(buf107, (128, 2048, 640), (1310720, 640, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, getitem_44, y_7, x_21, x_24, x_25, mul_44, getitem_45, mul_45, x_26, rms_norm_17], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_11.run(buf123, arg4_1, buf115, buf122, arg5_1, arg0_1, arg3_1, buf0, buf125, 262144, 640, stream=stream0)
            buf126 = buf114; del buf114  # reuse
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg34_1, buf126, 409600, stream=stream0)
            del arg34_1
            buf127 = buf122; del buf122  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_17, linear_26], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf125, (262144, 640), (640, 1), 0), reinterpret_tensor(buf126, (640, 640), (1, 640), 0), out=buf127)
            buf136 = reinterpret_tensor(buf115, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf115  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, linear_26, q_12, x1_8, mul_46, x2_8, mul_47, y1_8, neg_8, mul_48, mul_49, y2_8, q_13, q_14, k_14, linear_28, v_6, _flash_attn_forward_default_4], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf127, arg1_1, arg2_1, buf136, 1310720, 128, stream=stream0)
            buf130 = buf126; del buf126  # reuse
            # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg35_1, buf130, 409600, stream=stream0)
            del arg35_1
            buf131 = buf127; del buf127  # reuse
            # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf125, (262144, 640), (640, 1), 0), reinterpret_tensor(buf130, (640, 640), (1, 640), 0), out=buf131)
            buf137 = buf106; del buf106  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, q_14, linear_27, k_12, x1_9, mul_50, x2_9, mul_51, y1_9, neg_9, mul_52, mul_53, y2_9, k_13, k_14, linear_28, v_6, _flash_attn_forward_default_4], Original ATen: [aten.slice, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.neg, aten.cat, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf131, arg1_1, arg2_1, buf137, 1310720, 128, stream=stream0)
            buf134 = buf130; del buf130  # reuse
            # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg36_1, buf134, 409600, stream=stream0)
            del arg36_1
            buf135 = buf131; del buf131  # reuse
            # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf125, (262144, 640), (640, 1), 0), reinterpret_tensor(buf134, (640, 640), (1, 640), 0), out=buf135)
            del buf125
            # Topologically Sorted Source Nodes: [q_14, k_14, linear_28, v_6, _flash_attn_forward_default_4], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf138 = torch.ops.flash_attn_3._flash_attn_forward.default(buf136, buf137, reinterpret_tensor(buf135, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf139 = buf138[0]
            assert_size_stride(buf139, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf139, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf138
            buf143 = buf134; del buf134  # reuse
            # Topologically Sorted Source Nodes: [y_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg37_1, buf143, 409600, stream=stream0)
            del arg37_1
            buf144 = reinterpret_tensor(buf137, (262144, 640), (640, 1), 0); del buf137  # reuse
            # Topologically Sorted Source Nodes: [y_8, y_9], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf139, (262144, 640), (640, 1), 0), reinterpret_tensor(buf143, (640, 640), (1, 640), 0), out=buf144)
            buf146 = reinterpret_tensor(buf139, (128, 2048, 640), (1310720, 640, 1), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [y_9, x_27, rms_norm_20], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3.run(buf123, buf144, buf146, 262144, 640, stream=stream0)
            buf147 = reinterpret_tensor(buf121, (2560, 640), (640, 1), 0); del buf121  # reuse
            # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg38_1, buf147, 1638400, stream=stream0)
            del arg38_1
            buf148 = reinterpret_tensor(buf120, (262144, 2560), (2560, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [y_9, x_27, rms_norm_20, x_28], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf146, (262144, 640), (640, 1), 0), reinterpret_tensor(buf147, (640, 2560), (1, 640), 0), out=buf148)
            buf149 = reinterpret_tensor(buf148, (128, 2048, 2560), (5242880, 2560, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [x_28, relu_4, x_29, x_30], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf149, 671088640, stream=stream0)
            buf150 = reinterpret_tensor(buf147, (640, 2560), (2560, 1), 0); del buf147  # reuse
            # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg39_1, buf150, 1638400, stream=stream0)
            del arg39_1
            buf151 = reinterpret_tensor(buf146, (262144, 640), (640, 1), 0); del buf146  # reuse
            # Topologically Sorted Source Nodes: [x_28, relu_4, x_29, x_30], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf149, (262144, 2560), (2560, 1), 0), reinterpret_tensor(buf150, (2560, 640), (1, 2560), 0), out=buf151)
            buf152 = buf123; del buf123  # reuse
            buf154 = reinterpret_tensor(buf136, (128, 2048, 640), (1310720, 640, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, getitem_54, y_9, x_27, x_30, x_31, mul_54, getitem_55, mul_55, x_32, rms_norm_21], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_12.run(buf152, arg4_1, buf144, buf151, arg5_1, arg0_1, arg3_1, buf0, buf154, 262144, 640, stream=stream0)
            buf155 = buf143; del buf143  # reuse
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg41_1, buf155, 409600, stream=stream0)
            del arg41_1
            buf156 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_21, linear_32], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf154, (262144, 640), (640, 1), 0), reinterpret_tensor(buf155, (640, 640), (1, 640), 0), out=buf156)
            buf167 = reinterpret_tensor(buf144, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, linear_32, q_15, x1_10, mul_58, x2_10, mul_59, y1_10, neg_10, mul_60, mul_61, y2_10, q_16, q_17, k_17, linear_34, v_7, sigmoid_2, gate_2, unsqueeze_2, ve_4, ve_5, mul_57, v_8, _flash_attn_forward_default_5], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf156, arg1_1, arg2_1, buf167, 1310720, 128, stream=stream0)
            buf159 = buf155; del buf155  # reuse
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg42_1, buf159, 409600, stream=stream0)
            del arg42_1
            buf160 = buf156; del buf156  # reuse
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf154, (262144, 640), (640, 1), 0), reinterpret_tensor(buf159, (640, 640), (1, 640), 0), out=buf160)
            buf168 = reinterpret_tensor(buf135, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf135  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, q_17, linear_33, k_15, x1_11, mul_62, x2_11, mul_63, y1_11, neg_11, mul_64, mul_65, y2_11, k_16, k_17, linear_34, v_7, sigmoid_2, gate_2, unsqueeze_2, ve_4, ve_5, mul_57, v_8, _flash_attn_forward_default_5], Original ATen: [aten.slice, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.neg, aten.cat, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf160, arg1_1, arg2_1, buf168, 1310720, 128, stream=stream0)
            buf163 = buf159; del buf159  # reuse
            # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg43_1, buf163, 409600, stream=stream0)
            del arg43_1
            buf164 = buf160; del buf160  # reuse
            # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf154, (262144, 640), (640, 1), 0), reinterpret_tensor(buf163, (640, 640), (1, 640), 0), out=buf164)
            buf165 = buf104; del buf104  # reuse
            # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg44_1, buf165, 160, stream=stream0)
            del arg44_1
            buf166 = buf105; del buf105  # reuse
            # Topologically Sorted Source Nodes: [getitem_56, linear_35], Original ATen: [aten.slice, aten._to_copy, aten.t, aten.expand, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf154, (128, 2048, 32), (1310720, 640, 1), 0), reinterpret_tensor(buf165, (128, 32, 5), (0, 1, 32), 0), out=buf166)
            del buf154
            buf169 = reinterpret_tensor(buf164, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf164  # reuse
            # Topologically Sorted Source Nodes: [q_17, k_17, linear_34, v_7, sigmoid_2, gate_2, unsqueeze_2, ve_4, ve_5, mul_57, v_8, _flash_attn_forward_default_5], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__flash_attn_forward__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_sigmoid_unsqueeze_view_8.run(buf169, buf166, arg0_1, arg40_1, 167772160, stream=stream0)
            del arg40_1
            # Topologically Sorted Source Nodes: [q_17, k_17, linear_34, v_7, sigmoid_2, gate_2, unsqueeze_2, ve_4, ve_5, mul_57, v_8, _flash_attn_forward_default_5], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            buf170 = torch.ops.flash_attn_3._flash_attn_forward.default(buf167, buf168, buf169, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf171 = buf170[0]
            assert_size_stride(buf171, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf171, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf170
            buf175 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [y_11], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg45_1, buf175, 409600, stream=stream0)
            del arg45_1
            buf176 = reinterpret_tensor(buf169, (262144, 640), (640, 1), 0); del buf169  # reuse
            # Topologically Sorted Source Nodes: [y_10, y_11], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf171, (262144, 640), (640, 1), 0), reinterpret_tensor(buf175, (640, 640), (1, 640), 0), out=buf176)
            buf178 = reinterpret_tensor(buf171, (128, 2048, 640), (1310720, 640, 1), 0); del buf171  # reuse
            # Topologically Sorted Source Nodes: [y_11, x_33, rms_norm_24], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3.run(buf152, buf176, buf178, 262144, 640, stream=stream0)
            buf179 = reinterpret_tensor(buf150, (2560, 640), (640, 1), 0); del buf150  # reuse
            # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg46_1, buf179, 1638400, stream=stream0)
            del arg46_1
            buf180 = reinterpret_tensor(buf149, (262144, 2560), (2560, 1), 0); del buf149  # reuse
            # Topologically Sorted Source Nodes: [y_11, x_33, rms_norm_24, x_34], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf178, (262144, 640), (640, 1), 0), reinterpret_tensor(buf179, (640, 2560), (1, 640), 0), out=buf180)
            buf181 = reinterpret_tensor(buf180, (128, 2048, 2560), (5242880, 2560, 1), 0); del buf180  # reuse
            # Topologically Sorted Source Nodes: [x_34, relu_5, x_35, x_36], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf181, 671088640, stream=stream0)
            buf182 = reinterpret_tensor(buf179, (640, 2560), (2560, 1), 0); del buf179  # reuse
            # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg47_1, buf182, 1638400, stream=stream0)
            del arg47_1
            buf183 = reinterpret_tensor(buf178, (262144, 640), (640, 1), 0); del buf178  # reuse
            # Topologically Sorted Source Nodes: [x_34, relu_5, x_35, x_36], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf181, (262144, 2560), (2560, 1), 0), reinterpret_tensor(buf182, (2560, 640), (1, 2560), 0), out=buf183)
            buf184 = buf152; del buf152  # reuse
            buf186 = reinterpret_tensor(buf168, (128, 2048, 640), (1310720, 640, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, getitem_65, y_11, x_33, x_36, x_37, mul_66, getitem_66, mul_67, x_38, rms_norm_25], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_13.run(buf184, arg4_1, buf176, buf183, arg5_1, arg0_1, arg3_1, buf0, buf186, 262144, 640, stream=stream0)
            buf187 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg48_1, buf187, 409600, stream=stream0)
            del arg48_1
            buf188 = buf183; del buf183  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_25, linear_39], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf186, (262144, 640), (640, 1), 0), reinterpret_tensor(buf187, (640, 640), (1, 640), 0), out=buf188)
            buf197 = reinterpret_tensor(buf176, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf176  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, linear_39, q_18, x1_12, mul_68, x2_12, mul_69, y1_12, neg_12, mul_70, mul_71, y2_12, q_19, q_20, k_20, linear_41, v_9, _flash_attn_forward_default_6], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf188, arg1_1, arg2_1, buf197, 1310720, 128, stream=stream0)
            buf191 = buf187; del buf187  # reuse
            # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg49_1, buf191, 409600, stream=stream0)
            del arg49_1
            buf192 = buf188; del buf188  # reuse
            # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf186, (262144, 640), (640, 1), 0), reinterpret_tensor(buf191, (640, 640), (1, 640), 0), out=buf192)
            buf198 = buf167; del buf167  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, q_20, linear_40, k_18, x1_13, mul_72, x2_13, mul_73, y1_13, neg_13, mul_74, mul_75, y2_13, k_19, k_20, linear_41, v_9, _flash_attn_forward_default_6], Original ATen: [aten.slice, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.neg, aten.cat, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf192, arg1_1, arg2_1, buf198, 1310720, 128, stream=stream0)
            buf195 = buf191; del buf191  # reuse
            # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg50_1, buf195, 409600, stream=stream0)
            del arg50_1
            buf196 = buf192; del buf192  # reuse
            # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf186, (262144, 640), (640, 1), 0), reinterpret_tensor(buf195, (640, 640), (1, 640), 0), out=buf196)
            del buf186
            # Topologically Sorted Source Nodes: [q_20, k_20, linear_41, v_9, _flash_attn_forward_default_6], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf199 = torch.ops.flash_attn_3._flash_attn_forward.default(buf197, buf198, reinterpret_tensor(buf196, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf200 = buf199[0]
            assert_size_stride(buf200, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf200, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf199
            buf204 = buf195; del buf195  # reuse
            # Topologically Sorted Source Nodes: [y_13], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg51_1, buf204, 409600, stream=stream0)
            del arg51_1
            buf205 = reinterpret_tensor(buf198, (262144, 640), (640, 1), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [y_12, y_13], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf200, (262144, 640), (640, 1), 0), reinterpret_tensor(buf204, (640, 640), (1, 640), 0), out=buf205)
            buf207 = reinterpret_tensor(buf200, (128, 2048, 640), (1310720, 640, 1), 0); del buf200  # reuse
            # Topologically Sorted Source Nodes: [y_13, x_39, rms_norm_28], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3.run(buf184, buf205, buf207, 262144, 640, stream=stream0)
            buf208 = reinterpret_tensor(buf182, (2560, 640), (640, 1), 0); del buf182  # reuse
            # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg52_1, buf208, 1638400, stream=stream0)
            del arg52_1
            buf209 = reinterpret_tensor(buf181, (262144, 2560), (2560, 1), 0); del buf181  # reuse
            # Topologically Sorted Source Nodes: [y_13, x_39, rms_norm_28, x_40], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf207, (262144, 640), (640, 1), 0), reinterpret_tensor(buf208, (640, 2560), (1, 640), 0), out=buf209)
            buf210 = reinterpret_tensor(buf209, (128, 2048, 2560), (5242880, 2560, 1), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [x_40, relu_6, x_41, x_42], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf210, 671088640, stream=stream0)
            buf211 = reinterpret_tensor(buf208, (640, 2560), (2560, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg53_1, buf211, 1638400, stream=stream0)
            del arg53_1
            buf212 = reinterpret_tensor(buf207, (262144, 640), (640, 1), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [x_40, relu_6, x_41, x_42], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf210, (262144, 2560), (2560, 1), 0), reinterpret_tensor(buf211, (2560, 640), (1, 2560), 0), out=buf212)
            buf213 = buf184; del buf184  # reuse
            buf215 = reinterpret_tensor(buf197, (128, 2048, 640), (1310720, 640, 1), 0); del buf197  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, getitem_75, y_13, x_39, x_42, x_43, mul_76, getitem_76, mul_77, x_44, rms_norm_29], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_14.run(buf213, arg4_1, buf205, buf212, arg5_1, arg0_1, arg3_1, buf0, buf215, 262144, 640, stream=stream0)
            buf216 = buf204; del buf204  # reuse
            # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg55_1, buf216, 409600, stream=stream0)
            del arg55_1
            buf217 = buf212; del buf212  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_29, linear_45], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf215, (262144, 640), (640, 1), 0), reinterpret_tensor(buf216, (640, 640), (1, 640), 0), out=buf217)
            buf228 = reinterpret_tensor(buf205, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf205  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, linear_45, q_21, x1_14, mul_80, x2_14, mul_81, y1_14, neg_14, mul_82, mul_83, y2_14, q_22, q_23, k_23, linear_47, v_10, sigmoid_3, gate_3, unsqueeze_3, ve_6, ve_7, mul_79, v_11, _flash_attn_forward_default_7], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf217, arg1_1, arg2_1, buf228, 1310720, 128, stream=stream0)
            buf220 = buf216; del buf216  # reuse
            # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg56_1, buf220, 409600, stream=stream0)
            del arg56_1
            buf221 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf215, (262144, 640), (640, 1), 0), reinterpret_tensor(buf220, (640, 640), (1, 640), 0), out=buf221)
            buf229 = reinterpret_tensor(buf196, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf196  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, q_23, linear_46, k_21, x1_15, mul_84, x2_15, mul_85, y1_15, neg_15, mul_86, mul_87, y2_15, k_22, k_23, linear_47, v_10, sigmoid_3, gate_3, unsqueeze_3, ve_6, ve_7, mul_79, v_11, _flash_attn_forward_default_7], Original ATen: [aten.slice, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.neg, aten.cat, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf221, arg1_1, arg2_1, buf229, 1310720, 128, stream=stream0)
            buf224 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg57_1, buf224, 409600, stream=stream0)
            del arg57_1
            buf225 = buf221; del buf221  # reuse
            # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf215, (262144, 640), (640, 1), 0), reinterpret_tensor(buf224, (640, 640), (1, 640), 0), out=buf225)
            buf226 = buf165; del buf165  # reuse
            # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg58_1, buf226, 160, stream=stream0)
            del arg58_1
            buf227 = buf166; del buf166  # reuse
            # Topologically Sorted Source Nodes: [getitem_77, linear_48], Original ATen: [aten.slice, aten._to_copy, aten.t, aten.expand, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf215, (128, 2048, 32), (1310720, 640, 1), 0), reinterpret_tensor(buf226, (128, 32, 5), (0, 1, 32), 0), out=buf227)
            del buf215
            buf230 = reinterpret_tensor(buf225, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf225  # reuse
            # Topologically Sorted Source Nodes: [q_23, k_23, linear_47, v_10, sigmoid_3, gate_3, unsqueeze_3, ve_6, ve_7, mul_79, v_11, _flash_attn_forward_default_7], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__flash_attn_forward__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_sigmoid_unsqueeze_view_8.run(buf230, buf227, arg0_1, arg54_1, 167772160, stream=stream0)
            del arg54_1
            # Topologically Sorted Source Nodes: [q_23, k_23, linear_47, v_10, sigmoid_3, gate_3, unsqueeze_3, ve_6, ve_7, mul_79, v_11, _flash_attn_forward_default_7], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            buf231 = torch.ops.flash_attn_3._flash_attn_forward.default(buf228, buf229, buf230, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf232 = buf231[0]
            assert_size_stride(buf232, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf232, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf231
            buf236 = buf224; del buf224  # reuse
            # Topologically Sorted Source Nodes: [y_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg59_1, buf236, 409600, stream=stream0)
            del arg59_1
            buf237 = reinterpret_tensor(buf230, (262144, 640), (640, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [y_14, y_15], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf232, (262144, 640), (640, 1), 0), reinterpret_tensor(buf236, (640, 640), (1, 640), 0), out=buf237)
            buf239 = reinterpret_tensor(buf232, (128, 2048, 640), (1310720, 640, 1), 0); del buf232  # reuse
            # Topologically Sorted Source Nodes: [y_15, x_45, rms_norm_32], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3.run(buf213, buf237, buf239, 262144, 640, stream=stream0)
            buf240 = reinterpret_tensor(buf211, (2560, 640), (640, 1), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg60_1, buf240, 1638400, stream=stream0)
            del arg60_1
            buf241 = reinterpret_tensor(buf210, (262144, 2560), (2560, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [y_15, x_45, rms_norm_32, x_46], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf239, (262144, 640), (640, 1), 0), reinterpret_tensor(buf240, (640, 2560), (1, 640), 0), out=buf241)
            buf242 = reinterpret_tensor(buf241, (128, 2048, 2560), (5242880, 2560, 1), 0); del buf241  # reuse
            # Topologically Sorted Source Nodes: [x_46, relu_7, x_47, x_48], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf242, 671088640, stream=stream0)
            buf243 = reinterpret_tensor(buf240, (640, 2560), (2560, 1), 0); del buf240  # reuse
            # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg61_1, buf243, 1638400, stream=stream0)
            del arg61_1
            buf244 = reinterpret_tensor(buf239, (262144, 640), (640, 1), 0); del buf239  # reuse
            # Topologically Sorted Source Nodes: [x_46, relu_7, x_47, x_48], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf242, (262144, 2560), (2560, 1), 0), reinterpret_tensor(buf243, (2560, 640), (1, 2560), 0), out=buf244)
            buf245 = buf213; del buf213  # reuse
            buf247 = reinterpret_tensor(buf229, (128, 2048, 640), (1310720, 640, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, getitem_86, y_15, x_45, x_48, x_49, mul_88, getitem_87, mul_89, x_50, rms_norm_33], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_15.run(buf245, arg4_1, buf237, buf244, arg5_1, arg0_1, arg3_1, buf0, buf247, 262144, 640, stream=stream0)
            buf248 = buf236; del buf236  # reuse
            # Topologically Sorted Source Nodes: [linear_52], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg62_1, buf248, 409600, stream=stream0)
            del arg62_1
            buf249 = buf244; del buf244  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_33, linear_52], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf247, (262144, 640), (640, 1), 0), reinterpret_tensor(buf248, (640, 640), (1, 640), 0), out=buf249)
            buf258 = reinterpret_tensor(buf237, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf237  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, linear_52, q_24, x1_16, mul_90, x2_16, mul_91, y1_16, neg_16, mul_92, mul_93, y2_16, q_25, q_26, k_26, linear_54, v_12, _flash_attn_forward_default_8], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf249, arg1_1, arg2_1, buf258, 1310720, 128, stream=stream0)
            buf252 = buf248; del buf248  # reuse
            # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg63_1, buf252, 409600, stream=stream0)
            del arg63_1
            buf253 = buf249; del buf249  # reuse
            # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf247, (262144, 640), (640, 1), 0), reinterpret_tensor(buf252, (640, 640), (1, 640), 0), out=buf253)
            buf259 = buf228; del buf228  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, q_26, linear_53, k_24, x1_17, mul_94, x2_17, mul_95, y1_17, neg_17, mul_96, mul_97, y2_17, k_25, k_26, linear_54, v_12, _flash_attn_forward_default_8], Original ATen: [aten.slice, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.neg, aten.cat, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf253, arg1_1, arg2_1, buf259, 1310720, 128, stream=stream0)
            buf256 = buf252; del buf252  # reuse
            # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg64_1, buf256, 409600, stream=stream0)
            del arg64_1
            buf257 = buf253; del buf253  # reuse
            # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf247, (262144, 640), (640, 1), 0), reinterpret_tensor(buf256, (640, 640), (1, 640), 0), out=buf257)
            del buf247
            # Topologically Sorted Source Nodes: [q_26, k_26, linear_54, v_12, _flash_attn_forward_default_8], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf260 = torch.ops.flash_attn_3._flash_attn_forward.default(buf258, buf259, reinterpret_tensor(buf257, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf261 = buf260[0]
            assert_size_stride(buf261, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf261, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf260
            buf265 = buf256; del buf256  # reuse
            # Topologically Sorted Source Nodes: [y_17], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg65_1, buf265, 409600, stream=stream0)
            del arg65_1
            buf266 = reinterpret_tensor(buf259, (262144, 640), (640, 1), 0); del buf259  # reuse
            # Topologically Sorted Source Nodes: [y_16, y_17], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf261, (262144, 640), (640, 1), 0), reinterpret_tensor(buf265, (640, 640), (1, 640), 0), out=buf266)
            buf268 = reinterpret_tensor(buf261, (128, 2048, 640), (1310720, 640, 1), 0); del buf261  # reuse
            # Topologically Sorted Source Nodes: [y_17, x_51, rms_norm_36], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3.run(buf245, buf266, buf268, 262144, 640, stream=stream0)
            buf269 = reinterpret_tensor(buf243, (2560, 640), (640, 1), 0); del buf243  # reuse
            # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg66_1, buf269, 1638400, stream=stream0)
            del arg66_1
            buf270 = reinterpret_tensor(buf242, (262144, 2560), (2560, 1), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [y_17, x_51, rms_norm_36, x_52], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf268, (262144, 640), (640, 1), 0), reinterpret_tensor(buf269, (640, 2560), (1, 640), 0), out=buf270)
            buf271 = reinterpret_tensor(buf270, (128, 2048, 2560), (5242880, 2560, 1), 0); del buf270  # reuse
            # Topologically Sorted Source Nodes: [x_52, relu_8, x_53, x_54], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf271, 671088640, stream=stream0)
            buf272 = reinterpret_tensor(buf269, (640, 2560), (2560, 1), 0); del buf269  # reuse
            # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg67_1, buf272, 1638400, stream=stream0)
            del arg67_1
            buf273 = reinterpret_tensor(buf268, (262144, 640), (640, 1), 0); del buf268  # reuse
            # Topologically Sorted Source Nodes: [x_52, relu_8, x_53, x_54], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf271, (262144, 2560), (2560, 1), 0), reinterpret_tensor(buf272, (2560, 640), (1, 2560), 0), out=buf273)
            buf274 = buf245; del buf245  # reuse
            buf276 = reinterpret_tensor(buf258, (128, 2048, 640), (1310720, 640, 1), 0); del buf258  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, getitem_96, y_17, x_51, x_54, x_55, mul_98, getitem_97, mul_99, x_56, rms_norm_37], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_select_16.run(buf274, arg4_1, buf266, buf273, arg5_1, arg0_1, arg3_1, buf0, buf276, 262144, 640, stream=stream0)
            del arg3_1
            del arg4_1
            del arg5_1
            del buf0
            buf277 = buf265; del buf265  # reuse
            # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg69_1, buf277, 409600, stream=stream0)
            del arg69_1
            buf278 = buf273; del buf273  # reuse
            # Topologically Sorted Source Nodes: [rms_norm_37, linear_58], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf276, (262144, 640), (640, 1), 0), reinterpret_tensor(buf277, (640, 640), (1, 640), 0), out=buf278)
            buf289 = reinterpret_tensor(buf266, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf266  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, linear_58, q_27, x1_18, mul_102, x2_18, mul_103, y1_18, neg_18, mul_104, mul_105, y2_18, q_28, q_29, k_29, linear_60, v_13, sigmoid_4, gate_4, unsqueeze_4, ve_8, ve_9, mul_101, v_14, _flash_attn_forward_default_9], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf278, arg1_1, arg2_1, buf289, 1310720, 128, stream=stream0)
            buf281 = buf277; del buf277  # reuse
            # Topologically Sorted Source Nodes: [linear_59], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg70_1, buf281, 409600, stream=stream0)
            del arg70_1
            buf282 = buf278; del buf278  # reuse
            # Topologically Sorted Source Nodes: [linear_59], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf276, (262144, 640), (640, 1), 0), reinterpret_tensor(buf281, (640, 640), (1, 640), 0), out=buf282)
            buf290 = reinterpret_tensor(buf257, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf257  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, q_29, linear_59, k_27, x1_19, mul_106, x2_19, mul_107, y1_19, neg_19, mul_108, mul_109, y2_19, k_28, k_29, linear_60, v_13, sigmoid_4, gate_4, unsqueeze_4, ve_8, ve_9, mul_101, v_14, _flash_attn_forward_default_9], Original ATen: [aten.slice, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.neg, aten.cat, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf282, arg1_1, arg2_1, buf290, 1310720, 128, stream=stream0)
            del arg1_1
            del arg2_1
            buf285 = buf281; del buf281  # reuse
            # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg71_1, buf285, 409600, stream=stream0)
            del arg71_1
            buf286 = buf282; del buf282  # reuse
            # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf276, (262144, 640), (640, 1), 0), reinterpret_tensor(buf285, (640, 640), (1, 640), 0), out=buf286)
            buf287 = buf226; del buf226  # reuse
            # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg72_1, buf287, 160, stream=stream0)
            del arg72_1
            buf288 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [getitem_98, linear_61], Original ATen: [aten.slice, aten._to_copy, aten.t, aten.expand, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf276, (128, 2048, 32), (1310720, 640, 1), 0), reinterpret_tensor(buf287, (128, 32, 5), (0, 1, 32), 0), out=buf288)
            del buf276
            del buf287
            buf291 = reinterpret_tensor(buf286, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf286  # reuse
            # Topologically Sorted Source Nodes: [q_29, k_29, linear_60, v_13, sigmoid_4, gate_4, unsqueeze_4, ve_8, ve_9, mul_101, v_14, _flash_attn_forward_default_9], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__flash_attn_forward__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_sigmoid_unsqueeze_view_8.run(buf291, buf288, arg0_1, arg68_1, 167772160, stream=stream0)
            del arg0_1
            del arg68_1
            del buf288
            # Topologically Sorted Source Nodes: [q_29, k_29, linear_60, v_13, sigmoid_4, gate_4, unsqueeze_4, ve_8, ve_9, mul_101, v_14, _flash_attn_forward_default_9], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten._unsafe_view, aten.view, aten.sigmoid, aten.unsqueeze, aten.embedding, flash_attn_3._flash_attn_forward]
            buf292 = torch.ops.flash_attn_3._flash_attn_forward.default(buf289, buf290, buf291, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            del buf289
            del buf290
            buf293 = buf292[0]
            assert_size_stride(buf293, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf293, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf292
            buf297 = buf285; del buf285  # reuse
            # Topologically Sorted Source Nodes: [y_19], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg73_1, buf297, 409600, stream=stream0)
            del arg73_1
            buf298 = reinterpret_tensor(buf291, (262144, 640), (640, 1), 0); del buf291  # reuse
            # Topologically Sorted Source Nodes: [y_18, y_19], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf293, (262144, 640), (640, 1), 0), reinterpret_tensor(buf297, (640, 640), (1, 640), 0), out=buf298)
            del buf297
            buf300 = reinterpret_tensor(buf293, (128, 2048, 640), (1310720, 640, 1), 0); del buf293  # reuse
            # Topologically Sorted Source Nodes: [y_19, x_57, rms_norm_40], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_3.run(buf274, buf298, buf300, 262144, 640, stream=stream0)
            buf301 = reinterpret_tensor(buf272, (2560, 640), (640, 1), 0); del buf272  # reuse
            # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg74_1, buf301, 1638400, stream=stream0)
            del arg74_1
            buf302 = reinterpret_tensor(buf271, (262144, 2560), (2560, 1), 0); del buf271  # reuse
            # Topologically Sorted Source Nodes: [y_19, x_57, rms_norm_40, x_58], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf300, (262144, 640), (640, 1), 0), reinterpret_tensor(buf301, (640, 2560), (1, 640), 0), out=buf302)
            buf303 = reinterpret_tensor(buf302, (128, 2048, 2560), (5242880, 2560, 1), 0); del buf302  # reuse
            # Topologically Sorted Source Nodes: [x_58, relu_9, x_59, x_60], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf303, 671088640, stream=stream0)
            buf304 = reinterpret_tensor(buf301, (640, 2560), (2560, 1), 0); del buf301  # reuse
            # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg75_1, buf304, 1638400, stream=stream0)
            del arg75_1
            buf305 = reinterpret_tensor(buf300, (262144, 640), (640, 1), 0); del buf300  # reuse
            # Topologically Sorted Source Nodes: [x_58, relu_9, x_59, x_60], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf303, (262144, 2560), (2560, 1), 0), reinterpret_tensor(buf304, (2560, 640), (1, 2560), 0), out=buf305)
            del buf303
            del buf304
            buf307 = buf274; del buf274  # reuse
            # Topologically Sorted Source Nodes: [y_19, x_57, x_60, x_61, x_62], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_17.run(buf307, buf298, buf305, 262144, 640, stream=stream0)
            del buf298
            del buf305
            buf308 = empty_strided_cuda((8192, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [logits], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg76_1, buf308, 5242880, stream=stream0)
            del arg76_1
            buf309 = empty_strided_cuda((262144, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_19, x_57, x_60, x_61, x_62, logits], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf307, (262144, 640), (640, 1), 0), reinterpret_tensor(buf308, (640, 8192), (1, 640), 0), out=buf309)
            del buf307
            del buf308
            buf310 = empty_strided_cuda((262144, 1), (1, 262144), torch.float32)
            buf312 = reinterpret_tensor(buf310, (262144, ), (1, ), 0); del buf310  # reuse
            # Topologically Sorted Source Nodes: [view_46, loss, logits, logits_1, truediv, tanh, logits_2, view_45], Original ATen: [aten.view, aten.nll_loss_forward, aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, prims.prepare_softmax_online, aten.sub, aten._log_softmax]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_prepare_softmax_online_sub_tanh_view_19.run(buf312, buf309, arg77_1, 262144, 8192, stream=stream0)
            del arg77_1
            del buf309
        return (buf312, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 20480, 1, 64), (1310720, 64, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, 20480, 1, 64), (1310720, 64, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
