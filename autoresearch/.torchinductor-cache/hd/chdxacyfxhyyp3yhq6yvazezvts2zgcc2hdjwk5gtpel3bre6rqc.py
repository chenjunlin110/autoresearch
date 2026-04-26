# AOT ID: ['0_forward']
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


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/3e/c3eh6rlwfyvixy3pxtyl3ufhohskgjn5ebyrizaft6b343xda5k2.py
# Topologically Sorted Source Nodes: [x, x_1, getitem_2, mul, getitem_3, mul_1, x_2, rms_norm_1], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select]
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
#   %primals_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %primals_4 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %buf1 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf1]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %buf3 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf3]
#   %rsqrt_1 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_4, %primals_1), kwargs = {})
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %select : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 0), kwargs = {})
#   %mul_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %convert_element_type_1), kwargs = {})
#   %select_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 0), kwargs = {})
#   %mul_2 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %convert_element_type_1), kwargs = {})
#   %add_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_2, 2), kwargs = {})
#   %mean_1 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_2 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_1 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_3 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %rsqrt_1), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   return %embedding,%buf1,%rsqrt,%buf3,%rsqrt_1,%convert_element_type_3
triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_select_0 = async_compile.triton('triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_select_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_select_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_select_0(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (0))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp24 = tl.load(in_ptr3 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp1 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 8192), "index out of bounds: 0 <= tmp4 < 8192")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 640*tmp4), r0_mask, other=0.0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp13 = 640.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1.1920928955078125e-07
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp7 * tmp17
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 * tmp22
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp22
    tmp28 = tmp23 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
    tmp33 = tl.where(r0_mask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None].to(tl.float32)
    tmp35 = (tmp34 / tmp13)
    tmp36 = tmp35 + tmp15
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp29 * tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr0 + (r0_1 + 640*x0), tmp6, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp17, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp37, None)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp39, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/o4/co4zyhh64sbjrpxrumyv2zcsvhzvwp6vvwusqo25xajjtdo7csxj.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => convert_element_type_4, permute
# Graph fragment:
#   %primals_7 : Tensor "f32[640, 640][640, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %convert_element_type_4 : Tensor "bf16[640, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_7, torch.bfloat16), kwargs = {})
#   %permute : Tensor "bf16[640, 640][1, 640]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_4, [1, 0]), kwargs = {})
#   return %permute
triton_poi_fused__to_copy_t_1 = async_compile.triton('triton_poi_fused__to_copy_t_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3276800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/bd/cbd4hqt3hzghjnznw5jtkwuecacnzs6vszb522k47fxz6lbwkuyc.py
# Topologically Sorted Source Nodes: [cos, sin, linear, q, x1, x2, mul_2, mul_3, y1, neg, mul_4, mul_5, y2, q_1, q_2], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   cos => slice_1
#   linear => view_1
#   mul_2 => mul_4
#   mul_3 => mul_5
#   mul_4 => mul_6
#   mul_5 => mul_7
#   neg => neg
#   q => view_2
#   q_1 => cat
#   q_2 => add_7, convert_element_type_13, convert_element_type_14, mean_2, mul_12, pow_3, rsqrt_2
#   sin => slice_2
#   x1 => slice_3
#   x2 => slice_4
#   y1 => add_3
#   y2 => add_4
# Graph fragment:
#   %mm : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm]
#   %primals_2 : Tensor "bf16[1, 20480, 1, 64][1310720, 64, 64, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "bf16[1, 20480, 1, 64][1310720, 64, 64, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %cat : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0" = PlaceHolder[target=cat]
#   %buf14 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1310720]cuda:0" = PlaceHolder[target=buf14]
#   %rsqrt_2 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_2]
#   %slice_1 : Tensor "bf16[1, 2048, 1, 64][1310720, 64, 64, 1]cuda:0"[num_users=40] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_2, 1, 0, 2048), kwargs = {})
#   %slice_2 : Tensor "bf16[1, 2048, 1, 64][1310720, 64, 64, 1]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_3, 1, 0, 2048), kwargs = {})
#   %view_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 2048, 640]), kwargs = {})
#   %view_2 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [128, 2048, 5, 128]), kwargs = {})
#   %slice_3 : Tensor "bf16[128, 2048, 5, 64][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_2, 3, 0, 64), kwargs = {})
#   %slice_4 : Tensor "bf16[128, 2048, 5, 64][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_2, 3, 64, 9223372036854775807), kwargs = {})
#   %mul_4 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_3, %slice_1), kwargs = {})
#   %mul_5 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_4, %slice_2), kwargs = {})
#   %add_3 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %neg : Tensor "bf16[1, 2048, 1, 64][131072, 64, 64, 1]cuda:0"[num_users=20] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %mul_6 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_3, %neg), kwargs = {})
#   %mul_7 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_4, %slice_1), kwargs = {})
#   %add_4 : Tensor "bf16[128, 2048, 5, 64][655360, 320, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %cat : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_3, %add_4], 3), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_13, 2), kwargs = {})
#   %mean_2 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [3], True), kwargs = {})
#   %add_7 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_2, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_2 : Tensor "f32[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %mul_12 : Tensor "f32[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_13, %rsqrt_2), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12, torch.bfloat16), kwargs = {})
#   return %cat,%buf14,%rsqrt_2,%convert_element_type_14
triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760, 'r0_': 2617245696}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (r0_3 + 128*x4), tmp27, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp37, None)
    tl.store(out_ptr1 + (r0_3 + 128*x4), tmp39, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/kq/ckqa2yynm6ql4lotccrkxitlgpetutqdwrfi4nv4tl3vu53g5mj3.py
# Topologically Sorted Source Nodes: [x_1, getitem_2, mul, getitem_3, mul_1, x_2, y_2, x_3, rms_norm_4], Original ATen: [aten._to_copy, aten.mul, aten.select, aten.add, aten._unsafe_view, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_2 => select
#   getitem_3 => select_1
#   mul => mul_1
#   mul_1 => mul_2
#   rms_norm_4 => add_10, convert_element_type_20, convert_element_type_21, mean_4, mul_14, pow_5, rsqrt_4
#   x_1 => convert_element_type, convert_element_type_1, mul
#   x_2 => add_1
#   x_3 => add_9
#   y_2 => view_11
# Graph fragment:
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %mm_3 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %add_9 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_9]
#   %buf28 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf28]
#   %rsqrt_4 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_4]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %select : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 0), kwargs = {})
#   %mul_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %convert_element_type_1), kwargs = {})
#   %select_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 0), kwargs = {})
#   %mul_2 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %convert_element_type_1), kwargs = {})
#   %add_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %view_11 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [128, 2048, 640]), kwargs = {})
#   %add_9 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_11), kwargs = {})
#   %convert_element_type_20 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.float32), kwargs = {})
#   %pow_5 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_20, 2), kwargs = {})
#   %mean_4 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [2], True), kwargs = {})
#   %add_10 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_4, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_4 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_14 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, %rsqrt_4), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14, torch.bfloat16), kwargs = {})
#   return %add_9,%buf28,%rsqrt_4,%convert_element_type_21
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_3 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 2013265920}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_3(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp2 * tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp7
    tmp13 = tmp8 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp22 = 640.0
    tmp23 = (tmp21 / tmp22)
    tmp24 = 1.1920928955078125e-07
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp16 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp15, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp26, None)
    tl.store(out_ptr0 + (r0_1 + 640*x0), tmp28, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/2q/c2qgtu4po5cmscpckh2f62btptr7vsckz2cot3scpbdew2zk6tiq.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   x_4 => convert_element_type_22, permute_4
# Graph fragment:
#   %primals_11 : Tensor "f32[2560, 640][640, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %convert_element_type_22 : Tensor "bf16[2560, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_11, torch.bfloat16), kwargs = {})
#   %permute_4 : Tensor "bf16[640, 2560][1, 640]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_22, [1, 0]), kwargs = {})
#   return %permute_4
triton_poi_fused__to_copy_t_4 = async_compile.triton('triton_poi_fused__to_copy_t_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 13107200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/ln/clnkqsm7zhdm27yuhgffd7rwho7mch3vqgjkgovjwqiigueyo7zu.py
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_pow_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4026531840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_pow_relu_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 671088640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp5, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/ct/cctuloc45tyxearolrythcolo2aywpvq2owz3mrksr55sjwsfxea.py
# Topologically Sorted Source Nodes: [x_1, x_6, x_7, getitem_8, mul_10, getitem_9, mul_11, x_8, rms_norm_5], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_8 => select_2
#   getitem_9 => select_3
#   mul_10 => mul_15
#   mul_11 => mul_16
#   rms_norm_5 => add_13, convert_element_type_30, convert_element_type_31, mean_5, mul_17, pow_7, rsqrt_5
#   x_1 => convert_element_type, convert_element_type_1, mul
#   x_6 => view_15
#   x_7 => add_11
#   x_8 => add_12
# Graph fragment:
#   %add_9 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_9]
#   %mm_5 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_11 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_11]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_12 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_12]
#   %buf39 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf39]
#   %rsqrt_5 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_5]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %view_15 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [128, 2048, 640]), kwargs = {})
#   %add_11 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_15), kwargs = {})
#   %select_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 1), kwargs = {})
#   %mul_15 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %add_11), kwargs = {})
#   %select_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 1), kwargs = {})
#   %mul_16 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %convert_element_type_1), kwargs = {})
#   %add_12 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %mul_16), kwargs = {})
#   %convert_element_type_30 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %pow_7 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_30, 2), kwargs = {})
#   %mean_5 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [2], True), kwargs = {})
#   %add_13 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_5, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_5 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_17 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_30, %rsqrt_5), kwargs = {})
#   %convert_element_type_31 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17, torch.bfloat16), kwargs = {})
#   return %add_11,%add_12,%buf39,%rsqrt_5,%convert_element_type_31
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_6 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 3019898880}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (1))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (1))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 * tmp14
    tmp16 = tmp6 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = 640.0
    tmp24 = (tmp22 / tmp23)
    tmp25 = 1.1920928955078125e-07
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp2, r0_mask)
    tl.store(out_ptr0 + (r0_1 + 640*x0), tmp16, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp27, None)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp29, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/st/cstsb2fahvq6gbz4z3xcyihfb3hzaa7actsum4jl4gk5ohmohbou.py
# Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_9 => convert_element_type_41, permute_9
# Graph fragment:
#   %primals_17 : Tensor "f32[5, 32][32, 1]cuda:0" = PlaceHolder[target=primals_17]
#   %convert_element_type_41 : Tensor "bf16[5, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_17, torch.bfloat16), kwargs = {})
#   %permute_9 : Tensor "bf16[32, 5][1, 32]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_41, [1, 0]), kwargs = {})
#   return %permute_9
triton_poi_fused__to_copy_t_7 = async_compile.triton('triton_poi_fused__to_copy_t_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/h3/ch3x2rkqfr3mzfp6ehi3svk5tereux6tmhdxma5ihvxbohs3wqcw.py
# Topologically Sorted Source Nodes: [ve, linear_8, v_1, ve_1, linear_9, sigmoid, gate, unsqueeze, mul_13, v_2], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
# Source node to ATen node mapping:
#   gate => mul_18
#   linear_8 => view_23
#   linear_9 => view_27
#   mul_13 => mul_19
#   sigmoid => sigmoid
#   unsqueeze => unsqueeze
#   v_1 => view_24
#   v_2 => add_14
#   ve => embedding_1
#   ve_1 => view_25
# Graph fragment:
#   %primals_1 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %primals_13 : Tensor "bf16[8192, 640][640, 1]cuda:0" = PlaceHolder[target=primals_13]
#   %mm_8 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %mm_9 : Tensor "bf16[262144, 5][5, 1]cuda:0" = PlaceHolder[target=mm_9]
#   %embedding_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding_1]
#   %embedding_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_13, %primals_1), kwargs = {})
#   %view_23 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [128, 2048, 640]), kwargs = {})
#   %view_24 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_23, [128, 2048, 5, 128]), kwargs = {})
#   %view_25 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%embedding_1, [128, 2048, 5, 128]), kwargs = {})
#   %view_27 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [128, 2048, 5]), kwargs = {})
#   %sigmoid : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_27,), kwargs = {})
#   %mul_18 : Tensor "bf16[128, 2048, 5][10240, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, 2), kwargs = {})
#   %unsqueeze : Tensor "bf16[128, 2048, 5, 1][10240, 5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %mul_19 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %view_25), kwargs = {})
#   %add_14 : Tensor "bf16[128, 2048, 5, 128][1310720, 640, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_24, %mul_19), kwargs = {})
#   return %embedding_1,%add_14
triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_8 = async_compile.triton('triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 167772160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 640
    x0 = (xindex % 640)
    x4 = xindex
    x3 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x4), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.full([XBLOCK], 8192, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 8192), "index out of bounds: 0 <= tmp4 < 8192")
    tmp6 = tl.load(in_ptr1 + (x0 + 640*tmp4), None).to(tl.float32)
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 2.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11 * tmp6
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x4), tmp6, None)
    tl.store(in_out_ptr0 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/dv/cdvdvqp4bsh5hfu7ehwl6uw6fbu6eio7tgwof5xinqssblsmnkmj.py
# Topologically Sorted Source Nodes: [y_5, x_9, rms_norm_8], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   rms_norm_8 => add_22, convert_element_type_51, convert_element_type_52, mean_8, mul_30, pow_10, rsqrt_8
#   x_9 => add_21
#   y_5 => view_30
# Graph fragment:
#   %add_12 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_12]
#   %mm_10 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_10]
#   %add_21 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_21]
#   %buf67 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf67]
#   %rsqrt_8 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_8]
#   %view_30 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [128, 2048, 640]), kwargs = {})
#   %add_21 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %view_30), kwargs = {})
#   %convert_element_type_51 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_21, torch.float32), kwargs = {})
#   %pow_10 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_51, 2), kwargs = {})
#   %mean_8 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_10, [2], True), kwargs = {})
#   %add_22 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_8, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_8 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %mul_30 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_51, %rsqrt_8), kwargs = {})
#   %convert_element_type_52 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_30, torch.bfloat16), kwargs = {})
#   return %add_21,%buf67,%rsqrt_8,%convert_element_type_52
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2097152, 'r0_': 2013265920}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
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
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp13, None)
    tl.store(out_ptr0 + (r0_1 + 640*x0), tmp15, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/tb/ctbcweaessxckd4ww3impmwfasfxysnkhyqs5mnskfpbfy3ql3js.py
# Topologically Sorted Source Nodes: [x_1, x_12, x_13, getitem_15, mul_22, getitem_16, mul_23, x_14, rms_norm_9], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_15 => select_4
#   getitem_16 => select_5
#   mul_22 => mul_31
#   mul_23 => mul_32
#   rms_norm_9 => add_25, convert_element_type_61, convert_element_type_62, mean_9, mul_33, pow_12, rsqrt_9
#   x_1 => convert_element_type, convert_element_type_1, mul
#   x_12 => view_34
#   x_13 => add_23
#   x_14 => add_24
# Graph fragment:
#   %add_21 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_21]
#   %mm_12 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_12]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_23 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_23]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_24 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_24]
#   %buf77 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf77]
#   %rsqrt_9 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_9]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %view_34 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_12, [128, 2048, 640]), kwargs = {})
#   %add_23 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_34), kwargs = {})
#   %select_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 2), kwargs = {})
#   %mul_31 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, %add_23), kwargs = {})
#   %select_5 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 2), kwargs = {})
#   %mul_32 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %convert_element_type_1), kwargs = {})
#   %add_24 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %mul_32), kwargs = {})
#   %convert_element_type_61 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_24, torch.float32), kwargs = {})
#   %pow_12 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_61, 2), kwargs = {})
#   %mean_9 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_12, [2], True), kwargs = {})
#   %add_25 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_9, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_9 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_33 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_61, %rsqrt_9), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_33, torch.bfloat16), kwargs = {})
#   return %add_23,%add_24,%buf77,%rsqrt_9,%convert_element_type_62
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_10 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 3019898880}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_10(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (2))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (2))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 * tmp14
    tmp16 = tmp6 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = 640.0
    tmp24 = (tmp22 / tmp23)
    tmp25 = 1.1920928955078125e-07
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp2, r0_mask)
    tl.store(out_ptr0 + (r0_1 + 640*x0), tmp16, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp27, None)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp29, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/tq/ctqou7etdd3ml5bia356fbjby7q24ec5qrxcvvc5owxjmez5zu2l.py
# Topologically Sorted Source Nodes: [x_1, x_18, x_19, getitem_21, mul_32, getitem_22, mul_33, x_20, rms_norm_13], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_21 => select_6
#   getitem_22 => select_7
#   mul_32 => mul_45
#   mul_33 => mul_46
#   rms_norm_13 => add_36, convert_element_type_89, convert_element_type_90, mean_13, mul_47, pow_17, rsqrt_13
#   x_1 => convert_element_type, convert_element_type_1, mul
#   x_18 => view_50
#   x_19 => add_34
#   x_20 => add_35
# Graph fragment:
#   %add_32 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_32]
#   %mm_18 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_18]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_34 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_34]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_35 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_35]
#   %buf113 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf113]
#   %rsqrt_13 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_13]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %view_50 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_18, [128, 2048, 640]), kwargs = {})
#   %add_34 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_32, %view_50), kwargs = {})
#   %select_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 3), kwargs = {})
#   %mul_45 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_6, %add_34), kwargs = {})
#   %select_7 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 3), kwargs = {})
#   %mul_46 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_7, %convert_element_type_1), kwargs = {})
#   %add_35 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %mul_46), kwargs = {})
#   %convert_element_type_89 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_35, torch.float32), kwargs = {})
#   %pow_17 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_89, 2), kwargs = {})
#   %mean_13 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_17, [2], True), kwargs = {})
#   %add_36 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_13, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_13 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_36,), kwargs = {})
#   %mul_47 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_89, %rsqrt_13), kwargs = {})
#   %convert_element_type_90 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_47, torch.bfloat16), kwargs = {})
#   return %add_34,%add_35,%buf113,%rsqrt_13,%convert_element_type_90
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_11 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_11', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 3019898880}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (3))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (3))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 * tmp14
    tmp16 = tmp6 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = 640.0
    tmp24 = (tmp22 / tmp23)
    tmp25 = 1.1920928955078125e-07
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp2, r0_mask)
    tl.store(out_ptr0 + (r0_1 + 640*x0), tmp16, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp27, None)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp29, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/73/c73l7xub65thph7rqw6s2swi3ro7qmhvdeeopu76klgoxxltzkmu.py
# Topologically Sorted Source Nodes: [x_1, x_24, x_25, getitem_28, mul_44, getitem_29, mul_45, x_26, rms_norm_17], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_28 => select_8
#   getitem_29 => select_9
#   mul_44 => mul_61
#   mul_45 => mul_62
#   rms_norm_17 => add_48, convert_element_type_120, convert_element_type_121, mean_17, mul_63, pow_22, rsqrt_17
#   x_1 => convert_element_type, convert_element_type_1, mul
#   x_24 => view_69
#   x_25 => add_46
#   x_26 => add_47
# Graph fragment:
#   %add_44 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_44]
#   %mm_25 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_25]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_46 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_46]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_47 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_47]
#   %buf151 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf151]
#   %rsqrt_17 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_17]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %view_69 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_25, [128, 2048, 640]), kwargs = {})
#   %add_46 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_44, %view_69), kwargs = {})
#   %select_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 4), kwargs = {})
#   %mul_61 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %add_46), kwargs = {})
#   %select_9 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 4), kwargs = {})
#   %mul_62 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, %convert_element_type_1), kwargs = {})
#   %add_47 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_61, %mul_62), kwargs = {})
#   %convert_element_type_120 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_47, torch.float32), kwargs = {})
#   %pow_22 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_120, 2), kwargs = {})
#   %mean_17 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_22, [2], True), kwargs = {})
#   %add_48 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_17, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_17 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_48,), kwargs = {})
#   %mul_63 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_120, %rsqrt_17), kwargs = {})
#   %convert_element_type_121 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_63, torch.bfloat16), kwargs = {})
#   return %add_46,%add_47,%buf151,%rsqrt_17,%convert_element_type_121
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_12 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 3019898880}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_12(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (4))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (4))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 * tmp14
    tmp16 = tmp6 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = 640.0
    tmp24 = (tmp22 / tmp23)
    tmp25 = 1.1920928955078125e-07
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp2, r0_mask)
    tl.store(out_ptr0 + (r0_1 + 640*x0), tmp16, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp27, None)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp29, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/k5/ck5i7uqqrouidxz77slgqvyb67liuo76v24g5x7guz4in5urbabk.py
# Topologically Sorted Source Nodes: [x_1, x_30, x_31, getitem_34, mul_54, getitem_35, mul_55, x_32, rms_norm_21], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_34 => select_10
#   getitem_35 => select_11
#   mul_54 => mul_75
#   mul_55 => mul_76
#   rms_norm_21 => add_59, convert_element_type_148, convert_element_type_149, mean_21, mul_77, pow_27, rsqrt_21
#   x_1 => convert_element_type, convert_element_type_1, mul
#   x_30 => view_85
#   x_31 => add_57
#   x_32 => add_58
# Graph fragment:
#   %add_55 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_55]
#   %mm_31 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_31]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_57 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_57]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_58 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_58]
#   %buf187 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf187]
#   %rsqrt_21 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_21]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %view_85 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_31, [128, 2048, 640]), kwargs = {})
#   %add_57 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_55, %view_85), kwargs = {})
#   %select_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 5), kwargs = {})
#   %mul_75 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_10, %add_57), kwargs = {})
#   %select_11 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 5), kwargs = {})
#   %mul_76 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_11, %convert_element_type_1), kwargs = {})
#   %add_58 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, %mul_76), kwargs = {})
#   %convert_element_type_148 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_58, torch.float32), kwargs = {})
#   %pow_27 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_148, 2), kwargs = {})
#   %mean_21 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_27, [2], True), kwargs = {})
#   %add_59 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_21, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_21 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_59,), kwargs = {})
#   %mul_77 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_148, %rsqrt_21), kwargs = {})
#   %convert_element_type_149 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_77, torch.bfloat16), kwargs = {})
#   return %add_57,%add_58,%buf187,%rsqrt_21,%convert_element_type_149
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_13 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_13', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 3019898880}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_13(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (5))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (5))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 * tmp14
    tmp16 = tmp6 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = 640.0
    tmp24 = (tmp22 / tmp23)
    tmp25 = 1.1920928955078125e-07
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp2, r0_mask)
    tl.store(out_ptr0 + (r0_1 + 640*x0), tmp16, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp27, None)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp29, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/le/clevehwiktvewdrtxk4kjr4piikzu6njqeombvp4vm2n6mvwwn4y.py
# Topologically Sorted Source Nodes: [x_1, x_36, x_37, getitem_41, mul_66, getitem_42, mul_67, x_38, rms_norm_25], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_41 => select_12
#   getitem_42 => select_13
#   mul_66 => mul_91
#   mul_67 => mul_92
#   rms_norm_25 => add_71, convert_element_type_179, convert_element_type_180, mean_25, mul_93, pow_32, rsqrt_25
#   x_1 => convert_element_type, convert_element_type_1, mul
#   x_36 => view_104
#   x_37 => add_69
#   x_38 => add_70
# Graph fragment:
#   %add_67 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_67]
#   %mm_38 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_38]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_69 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_69]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_70 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_70]
#   %buf225 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf225]
#   %rsqrt_25 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_25]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %view_104 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_38, [128, 2048, 640]), kwargs = {})
#   %add_69 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_67, %view_104), kwargs = {})
#   %select_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 6), kwargs = {})
#   %mul_91 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_12, %add_69), kwargs = {})
#   %select_13 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 6), kwargs = {})
#   %mul_92 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_13, %convert_element_type_1), kwargs = {})
#   %add_70 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %mul_92), kwargs = {})
#   %convert_element_type_179 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_70, torch.float32), kwargs = {})
#   %pow_32 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_179, 2), kwargs = {})
#   %mean_25 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_32, [2], True), kwargs = {})
#   %add_71 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_25, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_25 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_71,), kwargs = {})
#   %mul_93 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_179, %rsqrt_25), kwargs = {})
#   %convert_element_type_180 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_93, torch.bfloat16), kwargs = {})
#   return %add_69,%add_70,%buf225,%rsqrt_25,%convert_element_type_180
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_14 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 3019898880}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_14(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (6))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (6))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 * tmp14
    tmp16 = tmp6 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = 640.0
    tmp24 = (tmp22 / tmp23)
    tmp25 = 1.1920928955078125e-07
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp2, r0_mask)
    tl.store(out_ptr0 + (r0_1 + 640*x0), tmp16, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp27, None)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp29, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/he/cheo6vhzlgzwdjmrf7hq3rg2rzw3ye7qbnaw3bhcztp75nomrf5z.py
# Topologically Sorted Source Nodes: [x_1, x_42, x_43, getitem_47, mul_76, getitem_48, mul_77, x_44, rms_norm_29], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_47 => select_14
#   getitem_48 => select_15
#   mul_76 => mul_105
#   mul_77 => mul_106
#   rms_norm_29 => add_82, convert_element_type_207, convert_element_type_208, mean_29, mul_107, pow_37, rsqrt_29
#   x_1 => convert_element_type, convert_element_type_1, mul
#   x_42 => view_120
#   x_43 => add_80
#   x_44 => add_81
# Graph fragment:
#   %add_78 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_78]
#   %mm_44 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_44]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_80 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_80]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_81 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_81]
#   %buf261 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf261]
#   %rsqrt_29 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_29]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %view_120 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_44, [128, 2048, 640]), kwargs = {})
#   %add_80 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_78, %view_120), kwargs = {})
#   %select_14 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 7), kwargs = {})
#   %mul_105 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_14, %add_80), kwargs = {})
#   %select_15 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 7), kwargs = {})
#   %mul_106 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_15, %convert_element_type_1), kwargs = {})
#   %add_81 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_105, %mul_106), kwargs = {})
#   %convert_element_type_207 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_81, torch.float32), kwargs = {})
#   %pow_37 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_207, 2), kwargs = {})
#   %mean_29 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_37, [2], True), kwargs = {})
#   %add_82 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_29, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_29 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_82,), kwargs = {})
#   %mul_107 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_207, %rsqrt_29), kwargs = {})
#   %convert_element_type_208 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_107, torch.bfloat16), kwargs = {})
#   return %add_80,%add_81,%buf261,%rsqrt_29,%convert_element_type_208
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_15 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 3019898880}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_15(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (7))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (7))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 * tmp14
    tmp16 = tmp6 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = 640.0
    tmp24 = (tmp22 / tmp23)
    tmp25 = 1.1920928955078125e-07
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp2, r0_mask)
    tl.store(out_ptr0 + (r0_1 + 640*x0), tmp16, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp27, None)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp29, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/ve/cveqs36ufb3hvsjvrvp5jbbhnknhyrfmg2cewhsosellkftdtpbb.py
# Topologically Sorted Source Nodes: [x_1, x_48, x_49, getitem_54, mul_88, getitem_55, mul_89, x_50, rms_norm_33], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_54 => select_16
#   getitem_55 => select_17
#   mul_88 => mul_121
#   mul_89 => mul_122
#   rms_norm_33 => add_94, convert_element_type_238, convert_element_type_239, mean_33, mul_123, pow_42, rsqrt_33
#   x_1 => convert_element_type, convert_element_type_1, mul
#   x_48 => view_139
#   x_49 => add_92
#   x_50 => add_93
# Graph fragment:
#   %add_90 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_90]
#   %mm_51 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_51]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_92 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_92]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_93 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_93]
#   %buf299 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf299]
#   %rsqrt_33 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_33]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %view_139 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_51, [128, 2048, 640]), kwargs = {})
#   %add_92 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_90, %view_139), kwargs = {})
#   %select_16 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 8), kwargs = {})
#   %mul_121 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_16, %add_92), kwargs = {})
#   %select_17 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 8), kwargs = {})
#   %mul_122 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_17, %convert_element_type_1), kwargs = {})
#   %add_93 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_121, %mul_122), kwargs = {})
#   %convert_element_type_238 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_93, torch.float32), kwargs = {})
#   %pow_42 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_238, 2), kwargs = {})
#   %mean_33 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_42, [2], True), kwargs = {})
#   %add_94 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_33, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_33 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_94,), kwargs = {})
#   %mul_123 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_238, %rsqrt_33), kwargs = {})
#   %convert_element_type_239 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_123, torch.bfloat16), kwargs = {})
#   return %add_92,%add_93,%buf299,%rsqrt_33,%convert_element_type_239
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_16 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_16', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 3019898880}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_16(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (8))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (8))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 * tmp14
    tmp16 = tmp6 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = 640.0
    tmp24 = (tmp22 / tmp23)
    tmp25 = 1.1920928955078125e-07
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp2, r0_mask)
    tl.store(out_ptr0 + (r0_1 + 640*x0), tmp16, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp27, None)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp29, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/5a/c5arujxobaqoisl3ljrqr4hdzjvsq4nnkpj5p5mnv7knuyq6tqbg.py
# Topologically Sorted Source Nodes: [x_1, x_54, x_55, getitem_60, mul_98, getitem_61, mul_99, x_56, rms_norm_37], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_60 => select_18
#   getitem_61 => select_19
#   mul_98 => mul_135
#   mul_99 => mul_136
#   rms_norm_37 => add_105, convert_element_type_266, convert_element_type_267, mean_37, mul_137, pow_47, rsqrt_37
#   x_1 => convert_element_type, convert_element_type_1, mul
#   x_54 => view_155
#   x_55 => add_103
#   x_56 => add_104
# Graph fragment:
#   %add_101 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_101]
#   %mm_57 : Tensor "bf16[262144, 640][640, 1]cuda:0" = PlaceHolder[target=mm_57]
#   %primals_5 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_5]
#   %add_103 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_103]
#   %primals_6 : Tensor "f32[10][1]cuda:0" = PlaceHolder[target=primals_6]
#   %embedding : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_104 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0" = PlaceHolder[target=add_104]
#   %buf335 : Tensor "f32[128, 2048, 1][2048, 1, 262144]cuda:0" = PlaceHolder[target=buf335]
#   %rsqrt_37 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_37]
#   %convert_element_type : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=11] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %view_155 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_57, [128, 2048, 640]), kwargs = {})
#   %add_103 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_101, %view_155), kwargs = {})
#   %select_18 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_5, 0, 9), kwargs = {})
#   %mul_135 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_18, %add_103), kwargs = {})
#   %select_19 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_6, 0, 9), kwargs = {})
#   %mul_136 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_19, %convert_element_type_1), kwargs = {})
#   %add_104 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_135, %mul_136), kwargs = {})
#   %convert_element_type_266 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_104, torch.float32), kwargs = {})
#   %pow_47 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_266, 2), kwargs = {})
#   %mean_37 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_47, [2], True), kwargs = {})
#   %add_105 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_37, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_37 : Tensor "f32[128, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_105,), kwargs = {})
#   %mul_137 : Tensor "f32[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_266, %rsqrt_37), kwargs = {})
#   %convert_element_type_267 : Tensor "bf16[128, 2048, 640][1310720, 640, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_137, torch.bfloat16), kwargs = {})
#   return %add_103,%add_104,%buf335,%rsqrt_37,%convert_element_type_267
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_17 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 3019898880}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_17(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (9))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (9))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 640*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 * tmp14
    tmp16 = tmp6 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = 640.0
    tmp24 = (tmp22 / tmp23)
    tmp25 = 1.1920928955078125e-07
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 640*x0), tmp2, r0_mask)
    tl.store(out_ptr0 + (r0_1 + 640*x0), tmp16, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp27, None)
    tl.store(out_ptr1 + (r0_1 + 640*x0), tmp29, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/et/cethtttpxzc7ym72rdti4nw5jcckagx7grnpf2s44wwogoqojdoq.py
# Topologically Sorted Source Nodes: [logits], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   logits => convert_element_type_299, permute_65
# Graph fragment:
#   %primals_77 : Tensor "f32[8192, 640][640, 1]cuda:0" = PlaceHolder[target=primals_77]
#   %convert_element_type_299 : Tensor "bf16[8192, 640][640, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_77, torch.bfloat16), kwargs = {})
#   %permute_65 : Tensor "bf16[640, 8192][1, 640]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_299, [1, 0]), kwargs = {})
#   return %permute_65
triton_poi_fused__to_copy_t_18 = async_compile.triton('triton_poi_fused__to_copy_t_18', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41943040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/jh/cjhe57flopuieqg5w3zebj5soidamsnux27dpkpn2gnzgrf2waj6.py
# Topologically Sorted Source Nodes: [logits, logits_1, truediv, tanh, logits_2, view_45, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.view, prims.prepare_softmax_online, aten._log_softmax]
# Source node to ATen node mapping:
#   logits => view_176
#   logits_1 => convert_element_type_302
#   logits_2 => mul_152
#   loss => log
#   tanh => tanh
#   truediv => div
#   view_45 => view_177
# Graph fragment:
#   %mm_65 : Tensor "bf16[262144, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_65]
#   %getitem_39 : Tensor "f32[262144, 1][1, 262144]cuda:0" = PlaceHolder[target=getitem_39]
#   %view_176 : Tensor "bf16[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_65, [128, 2048, 8192]), kwargs = {})
#   %convert_element_type_302 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_176, torch.float32), kwargs = {})
#   %div : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_302, 15), kwargs = {})
#   %tanh : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_152 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 15), kwargs = {})
#   %view_177 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_152, [-1, 8192]), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%view_177, 1), kwargs = {})
#   %log : Tensor "f32[262144, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%getitem_39,), kwargs = {})
#   return %getitem_38,%getitem_39,%log
triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view_19 = async_compile.triton('triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i64', 'r0_numel': 'i64', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4194304, 'r0_': 4294967296}}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view_19(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
    tl.store(out_ptr0 + (x0), tmp8, None)
    tmp10 = tl_math.log(tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/u4/cu42673y7elyqwurszxbngnimptu6pwywm2uxfkdpxscbx4l7cho.py
# Topologically Sorted Source Nodes: [logits, logits_1, truediv, tanh, logits_2, view_45, view_46, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.view, aten.sub, aten._log_softmax, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   logits => view_176
#   logits_1 => convert_element_type_302
#   logits_2 => mul_152
#   loss => full_default, full_default_1, gather, ne, neg_20, squeeze, sub_1, sum_2, sum_3, unsqueeze_5, where, where_1
#   tanh => tanh
#   truediv => div
#   view_45 => view_177
#   view_46 => view_178
# Graph fragment:
#   %primals_78 : Tensor "i64[128, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_78]
#   %mm_65 : Tensor "bf16[262144, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_65]
#   %getitem_38 : Tensor "f32[262144, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_38]
#   %log : Tensor "f32[262144, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %view_176 : Tensor "bf16[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_65, [128, 2048, 8192]), kwargs = {})
#   %convert_element_type_302 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_176, torch.float32), kwargs = {})
#   %div : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_302, 15), kwargs = {})
#   %tanh : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_152 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 15), kwargs = {})
#   %view_177 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_152, [-1, 8192]), kwargs = {})
#   %view_178 : Tensor "i64[262144][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_78, [-1]), kwargs = {})
#   %sub_tensor : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_177, %getitem_38), kwargs = {})
#   %sub_1 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log), kwargs = {})
#   %ne : Tensor "b8[262144][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_178, -1), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %view_178, %full_default), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 1), kwargs = {})
#   %gather : Tensor "f32[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_1, 1, %unsqueeze_5), kwargs = {})
#   %squeeze : Tensor "f32[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg_20 : Tensor "f32[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f32[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg_20, %full_default_1), kwargs = {})
#   %sum_2 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne,), kwargs = {})
#   %sum_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   return %buf381,%buf385
triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_20 = async_compile.triton('triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*fp32', 'xnumel': 'i64', 'r0_numel': 'i64', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    rbase = r0_base
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    _tmp29 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr2 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr3 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], -1, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tmp2.to(tl.int64)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
        tmp7 = tl.full([1, 1], 0, tl.int64)
        tmp8 = tl.where(tmp2, tmp0, tmp7)
        tmp9 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp8 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp8)
        tl.device_assert(((0 <= tmp12) & (tmp12 < 8192)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp12 < 8192")
        tmp14 = tl.load(in_ptr1 + (tmp12 + 8192*r0_1 + 67108864*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = 0.06666666666666667
        tmp17 = tmp15 * tmp16
        tmp18 = libdevice.tanh(tmp17)
        tmp19 = 15.0
        tmp20 = tmp18 * tmp19
        tmp22 = tmp20 - tmp21
        tmp24 = tmp22 - tmp23
        tmp25 = -tmp24
        tmp26 = 0.0
        tmp27 = tl.where(tmp2, tmp25, tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(r0_mask & xmask, tmp30, _tmp29)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp29, xmask)
''', device_str='cuda')


# kernel path: /mnt/weka/home/junlin.chen/workspace/autoresearch/autoresearch/.torchinductor-cache/ht/cht57znltqs42gppqy7fqjcazs3syonuwd7xdm47yf37xgij6uqw.py
# Topologically Sorted Source Nodes: [logits, logits_1, truediv, tanh, logits_2, view_45, view_46, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.view, aten.sub, aten._log_softmax, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   logits => view_176
#   logits_1 => convert_element_type_302
#   logits_2 => mul_152
#   loss => convert_element_type_303, div_1, full_default, full_default_1, gather, ne, neg_20, squeeze, sub_1, sum_2, sum_3, unsqueeze_5, where, where_1
#   tanh => tanh
#   truediv => div
#   view_45 => view_177
#   view_46 => view_178
# Graph fragment:
#   %buf385 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=buf385]
#   %buf381 : Tensor "i64[32][1]cuda:0" = PlaceHolder[target=buf381]
#   %sum_2 : Tensor "i64[][]cuda:0" = PlaceHolder[target=sum_2]
#   %sum_3 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_3]
#   %convert_element_type_303 : Tensor "f32[][]cuda:0" = PlaceHolder[target=convert_element_type_303]
#   %view_176 : Tensor "bf16[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_65, [128, 2048, 8192]), kwargs = {})
#   %convert_element_type_302 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_176, torch.float32), kwargs = {})
#   %div : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_302, 15), kwargs = {})
#   %tanh : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_152 : Tensor "f32[128, 2048, 8192][16777216, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 15), kwargs = {})
#   %view_177 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_152, [-1, 8192]), kwargs = {})
#   %view_178 : Tensor "i64[262144][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_78, [-1]), kwargs = {})
#   %sub_tensor : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_177, %getitem_38), kwargs = {})
#   %sub_1 : Tensor "f32[262144, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log), kwargs = {})
#   %ne : Tensor "b8[262144][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_178, -1), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %view_178, %full_default), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 1), kwargs = {})
#   %gather : Tensor "f32[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_1, 1, %unsqueeze_5), kwargs = {})
#   %squeeze : Tensor "f32[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg_20 : Tensor "f32[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f32[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg_20, %full_default_1), kwargs = {})
#   %sum_2 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne,), kwargs = {})
#   %convert_element_type_303 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %sum_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %div_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type_303), kwargs = {})
#   return %sum_3,%sum_2,%convert_element_type_303,%div_1
triton_per_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_21 = async_compile.triton('triton_per_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 384}}
)
@triton.jit
def triton_per_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_21(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp4 = tl.load(in_ptr1 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None].to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.int64)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = (tmp3 / tmp8)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp9, None)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78 = args
        args.clear()
        assert_size_stride(primals_1, (128, 2048), (2048, 1))
        assert_size_stride(primals_2, (1, 20480, 1, 64), (1310720, 64, 64, 1))
        assert_size_stride(primals_3, (1, 20480, 1, 64), (1310720, 64, 64, 1))
        assert_size_stride(primals_4, (8192, 640), (640, 1))
        assert_size_stride(primals_5, (10, ), (1, ))
        assert_size_stride(primals_6, (10, ), (1, ))
        assert_size_stride(primals_7, (640, 640), (640, 1))
        assert_size_stride(primals_8, (640, 640), (640, 1))
        assert_size_stride(primals_9, (640, 640), (640, 1))
        assert_size_stride(primals_10, (640, 640), (640, 1))
        assert_size_stride(primals_11, (2560, 640), (640, 1))
        assert_size_stride(primals_12, (640, 2560), (2560, 1))
        assert_size_stride(primals_13, (8192, 640), (640, 1))
        assert_size_stride(primals_14, (640, 640), (640, 1))
        assert_size_stride(primals_15, (640, 640), (640, 1))
        assert_size_stride(primals_16, (640, 640), (640, 1))
        assert_size_stride(primals_17, (5, 32), (32, 1))
        assert_size_stride(primals_18, (640, 640), (640, 1))
        assert_size_stride(primals_19, (2560, 640), (640, 1))
        assert_size_stride(primals_20, (640, 2560), (2560, 1))
        assert_size_stride(primals_21, (640, 640), (640, 1))
        assert_size_stride(primals_22, (640, 640), (640, 1))
        assert_size_stride(primals_23, (640, 640), (640, 1))
        assert_size_stride(primals_24, (640, 640), (640, 1))
        assert_size_stride(primals_25, (2560, 640), (640, 1))
        assert_size_stride(primals_26, (640, 2560), (2560, 1))
        assert_size_stride(primals_27, (8192, 640), (640, 1))
        assert_size_stride(primals_28, (640, 640), (640, 1))
        assert_size_stride(primals_29, (640, 640), (640, 1))
        assert_size_stride(primals_30, (640, 640), (640, 1))
        assert_size_stride(primals_31, (5, 32), (32, 1))
        assert_size_stride(primals_32, (640, 640), (640, 1))
        assert_size_stride(primals_33, (2560, 640), (640, 1))
        assert_size_stride(primals_34, (640, 2560), (2560, 1))
        assert_size_stride(primals_35, (640, 640), (640, 1))
        assert_size_stride(primals_36, (640, 640), (640, 1))
        assert_size_stride(primals_37, (640, 640), (640, 1))
        assert_size_stride(primals_38, (640, 640), (640, 1))
        assert_size_stride(primals_39, (2560, 640), (640, 1))
        assert_size_stride(primals_40, (640, 2560), (2560, 1))
        assert_size_stride(primals_41, (8192, 640), (640, 1))
        assert_size_stride(primals_42, (640, 640), (640, 1))
        assert_size_stride(primals_43, (640, 640), (640, 1))
        assert_size_stride(primals_44, (640, 640), (640, 1))
        assert_size_stride(primals_45, (5, 32), (32, 1))
        assert_size_stride(primals_46, (640, 640), (640, 1))
        assert_size_stride(primals_47, (2560, 640), (640, 1))
        assert_size_stride(primals_48, (640, 2560), (2560, 1))
        assert_size_stride(primals_49, (640, 640), (640, 1))
        assert_size_stride(primals_50, (640, 640), (640, 1))
        assert_size_stride(primals_51, (640, 640), (640, 1))
        assert_size_stride(primals_52, (640, 640), (640, 1))
        assert_size_stride(primals_53, (2560, 640), (640, 1))
        assert_size_stride(primals_54, (640, 2560), (2560, 1))
        assert_size_stride(primals_55, (8192, 640), (640, 1))
        assert_size_stride(primals_56, (640, 640), (640, 1))
        assert_size_stride(primals_57, (640, 640), (640, 1))
        assert_size_stride(primals_58, (640, 640), (640, 1))
        assert_size_stride(primals_59, (5, 32), (32, 1))
        assert_size_stride(primals_60, (640, 640), (640, 1))
        assert_size_stride(primals_61, (2560, 640), (640, 1))
        assert_size_stride(primals_62, (640, 2560), (2560, 1))
        assert_size_stride(primals_63, (640, 640), (640, 1))
        assert_size_stride(primals_64, (640, 640), (640, 1))
        assert_size_stride(primals_65, (640, 640), (640, 1))
        assert_size_stride(primals_66, (640, 640), (640, 1))
        assert_size_stride(primals_67, (2560, 640), (640, 1))
        assert_size_stride(primals_68, (640, 2560), (2560, 1))
        assert_size_stride(primals_69, (8192, 640), (640, 1))
        assert_size_stride(primals_70, (640, 640), (640, 1))
        assert_size_stride(primals_71, (640, 640), (640, 1))
        assert_size_stride(primals_72, (640, 640), (640, 1))
        assert_size_stride(primals_73, (5, 32), (32, 1))
        assert_size_stride(primals_74, (640, 640), (640, 1))
        assert_size_stride(primals_75, (2560, 640), (640, 1))
        assert_size_stride(primals_76, (640, 2560), (2560, 1))
        assert_size_stride(primals_77, (8192, 640), (640, 1))
        assert_size_stride(primals_78, (128, 2048), (2048, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf1 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf2 = reinterpret_tensor(buf1, (128, 2048, 1), (2048, 1, 1), 0); del buf1  # reuse
            buf3 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf4 = reinterpret_tensor(buf3, (128, 2048, 1), (2048, 1, 1), 0); del buf3  # reuse
            buf6 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x, x_1, getitem_2, mul, getitem_3, mul_1, x_2, rms_norm_1], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.select]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_select_0.run(buf2, buf4, primals_1, primals_4, primals_5, primals_6, buf0, buf6, 262144, 640, stream=stream0)
            del primals_4
            buf5 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_7, buf5, 409600, stream=stream0)
            del primals_7
            buf7 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, getitem_2, mul, getitem_3, mul_1, x_2, rms_norm_1, linear], Original ATen: [aten._to_copy, aten.mul, aten.select, aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf6, (262144, 640), (640, 1), 0), buf5, out=buf7)
            buf8 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_8, buf8, 409600, stream=stream0)
            del primals_8
            buf9 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, getitem_2, mul, getitem_3, mul_1, x_2, rms_norm_1, linear, linear_1], Original ATen: [aten._to_copy, aten.mul, aten.select, aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf6, (262144, 640), (640, 1), 0), buf8, out=buf9)
            buf10 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_9, buf10, 409600, stream=stream0)
            del primals_9
            buf11 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, getitem_2, mul, getitem_3, mul_1, x_2, rms_norm_1, linear, linear_2], Original ATen: [aten._to_copy, aten.mul, aten.select, aten.add, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf6, (262144, 640), (640, 1), 0), buf10, out=buf11)
            buf12 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            buf14 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf15 = reinterpret_tensor(buf14, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf14  # reuse
            buf16 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, linear, q, x1, x2, mul_2, mul_3, y1, neg, mul_4, mul_5, y2, q_1, q_2], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf15, buf7, primals_2, primals_3, buf12, buf16, 1310720, 128, stream=stream0)
            buf13 = reinterpret_tensor(buf7, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf7  # reuse
            buf17 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf18 = reinterpret_tensor(buf17, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf17  # reuse
            buf19 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, linear_1, k, neg, x1_1, x2_1, mul_6, mul_7, y1_1, mul_8, mul_9, y2_1, k_1, k_2], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.neg, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf18, buf9, primals_2, primals_3, buf13, buf19, 1310720, 128, stream=stream0)
            del buf9
            # Topologically Sorted Source Nodes: [linear_2, v, y], Original ATen: [aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf20 = torch.ops.flash_attn_3._flash_attn_forward.default(buf16, buf19, reinterpret_tensor(buf11, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf21 = buf20[0]
            assert_size_stride(buf21, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf21, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf22 = buf20[1]
            assert_size_stride(buf22, (128, 5, 2048), (10240, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf22, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf20
            buf25 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_2], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_10, buf25, 409600, stream=stream0)
            del primals_10
            buf26 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_1, y_2], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf21, (262144, 640), (640, 1), 0), buf25, out=buf26)
            buf27 = reinterpret_tensor(buf26, (128, 2048, 640), (1310720, 640, 1), 0); del buf26  # reuse
            buf28 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf29 = reinterpret_tensor(buf28, (128, 2048, 1), (2048, 1, 1), 0); del buf28  # reuse
            buf31 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, getitem_2, mul, getitem_3, mul_1, x_2, y_2, x_3, rms_norm_4], Original ATen: [aten._to_copy, aten.mul, aten.select, aten.add, aten._unsafe_view, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_3.run(buf27, buf29, primals_5, buf0, buf2, primals_6, buf31, 262144, 640, stream=stream0)
            buf30 = empty_strided_cuda((640, 2560), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_11, buf30, 1638400, stream=stream0)
            del primals_11
            buf32 = empty_strided_cuda((262144, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_4, x_4], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf31, (262144, 640), (640, 1), 0), buf30, out=buf32)
            buf33 = empty_strided_cuda((2560, 640), (1, 2560), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_12, buf33, 1638400, stream=stream0)
            del primals_12
            buf34 = empty_strided_cuda((128, 2048, 2560), (5242880, 2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_4, relu, x_5, x_6], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf32, buf34, 671088640, stream=stream0)
            buf35 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_4, relu, x_5, x_6], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf34, (262144, 2560), (2560, 1), 0), buf33, out=buf35)
            buf36 = reinterpret_tensor(buf35, (128, 2048, 640), (1310720, 640, 1), 0); del buf35  # reuse
            buf37 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf39 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf40 = reinterpret_tensor(buf39, (128, 2048, 1), (2048, 1, 1), 0); del buf39  # reuse
            buf42 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, x_6, x_7, getitem_8, mul_10, getitem_9, mul_11, x_8, rms_norm_5], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_6.run(buf36, buf40, buf27, primals_5, primals_6, buf0, buf2, buf37, buf42, 262144, 640, stream=stream0)
            buf46 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_16, buf46, 409600, stream=stream0)
            del primals_16
            buf47 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_5, linear_6, linear_8], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf42, (262144, 640), (640, 1), 0), buf46, out=buf47)
            buf48 = empty_strided_cuda((32, 5), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_7.run(primals_17, buf48, 160, stream=stream0)
            del primals_17
            buf49 = empty_strided_cuda((262144, 5), (5, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_10, linear_9], Original ATen: [aten.slice, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf42, (262144, 32), (640, 1), 0), buf48, out=buf49)
            buf38 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf50 = reinterpret_tensor(buf47, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf47  # reuse
            # Topologically Sorted Source Nodes: [ve, linear_8, v_1, ve_1, linear_9, sigmoid, gate, unsqueeze, mul_13, v_2], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_8.run(buf50, primals_1, primals_13, buf49, buf38, 167772160, stream=stream0)
            del primals_13
            buf41 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_14, buf41, 409600, stream=stream0)
            del primals_14
            buf43 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_5, linear_6], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf42, (262144, 640), (640, 1), 0), buf41, out=buf43)
            buf44 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_15, buf44, 409600, stream=stream0)
            del primals_15
            buf45 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_5, linear_6, linear_7], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf42, (262144, 640), (640, 1), 0), buf44, out=buf45)
            buf51 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            buf53 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf54 = reinterpret_tensor(buf53, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf53  # reuse
            buf55 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_6, q_3, x1_2, x2_2, mul_14, mul_15, y1_2, mul_16, mul_17, y2_2, q_4, q_5], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf54, buf43, primals_2, primals_3, buf51, buf55, 1310720, 128, stream=stream0)
            buf52 = reinterpret_tensor(buf43, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf43  # reuse
            buf56 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf57 = reinterpret_tensor(buf56, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf56  # reuse
            buf58 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_7, k_3, x1_3, x2_3, mul_18, mul_19, y1_3, mul_20, mul_21, y2_3, k_4, k_5], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf57, buf45, primals_2, primals_3, buf52, buf58, 1310720, 128, stream=stream0)
            del buf45
            # Topologically Sorted Source Nodes: [y_3], Original ATen: [flash_attn_3._flash_attn_forward]
            buf59 = torch.ops.flash_attn_3._flash_attn_forward.default(buf55, buf58, buf50, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf60 = buf59[0]
            assert_size_stride(buf60, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf60, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf61 = buf59[1]
            assert_size_stride(buf61, (128, 5, 2048), (10240, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf61, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf59
            buf64 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_5], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_18, buf64, 409600, stream=stream0)
            del primals_18
            buf65 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_4, y_5], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf60, (262144, 640), (640, 1), 0), buf64, out=buf65)
            buf66 = reinterpret_tensor(buf65, (128, 2048, 640), (1310720, 640, 1), 0); del buf65  # reuse
            buf67 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf68 = reinterpret_tensor(buf67, (128, 2048, 1), (2048, 1, 1), 0); del buf67  # reuse
            buf70 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_5, x_9, rms_norm_8], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf66, buf68, buf37, buf70, 262144, 640, stream=stream0)
            buf69 = empty_strided_cuda((640, 2560), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_19, buf69, 1638400, stream=stream0)
            del primals_19
            buf71 = empty_strided_cuda((262144, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_8, x_10], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf70, (262144, 640), (640, 1), 0), buf69, out=buf71)
            buf72 = empty_strided_cuda((2560, 640), (1, 2560), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_20, buf72, 1638400, stream=stream0)
            del primals_20
            buf73 = empty_strided_cuda((128, 2048, 2560), (5242880, 2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_10, relu_1, x_11, x_12], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf71, buf73, 671088640, stream=stream0)
            buf74 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_10, relu_1, x_11, x_12], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf73, (262144, 2560), (2560, 1), 0), buf72, out=buf74)
            buf75 = reinterpret_tensor(buf74, (128, 2048, 640), (1310720, 640, 1), 0); del buf74  # reuse
            buf76 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf77 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf78 = reinterpret_tensor(buf77, (128, 2048, 1), (2048, 1, 1), 0); del buf77  # reuse
            buf80 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, x_12, x_13, getitem_15, mul_22, getitem_16, mul_23, x_14, rms_norm_9], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_10.run(buf75, buf78, buf66, primals_5, primals_6, buf0, buf2, buf76, buf80, 262144, 640, stream=stream0)
            buf79 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_21, buf79, 409600, stream=stream0)
            del primals_21
            buf81 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_9, linear_13], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf80, (262144, 640), (640, 1), 0), buf79, out=buf81)
            buf82 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_22, buf82, 409600, stream=stream0)
            del primals_22
            buf83 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_9, linear_13, linear_14], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf80, (262144, 640), (640, 1), 0), buf82, out=buf83)
            buf84 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_23, buf84, 409600, stream=stream0)
            del primals_23
            buf85 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_9, linear_13, linear_15], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf80, (262144, 640), (640, 1), 0), buf84, out=buf85)
            buf86 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            buf88 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf89 = reinterpret_tensor(buf88, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf88  # reuse
            buf90 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_13, q_6, x1_4, x2_4, mul_24, mul_25, y1_4, mul_26, mul_27, y2_4, q_7, q_8], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf89, buf81, primals_2, primals_3, buf86, buf90, 1310720, 128, stream=stream0)
            buf87 = reinterpret_tensor(buf81, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf81  # reuse
            buf91 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf92 = reinterpret_tensor(buf91, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf91  # reuse
            buf93 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_14, k_6, x1_5, x2_5, mul_28, mul_29, y1_5, mul_30, mul_31, y2_5, k_7, k_8], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf92, buf83, primals_2, primals_3, buf87, buf93, 1310720, 128, stream=stream0)
            del buf83
            # Topologically Sorted Source Nodes: [linear_15, v_3, y_6], Original ATen: [aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf94 = torch.ops.flash_attn_3._flash_attn_forward.default(buf90, buf93, reinterpret_tensor(buf85, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf95 = buf94[0]
            assert_size_stride(buf95, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf95, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf96 = buf94[1]
            assert_size_stride(buf96, (128, 5, 2048), (10240, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf96, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf94
            buf99 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_8], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_24, buf99, 409600, stream=stream0)
            del primals_24
            buf100 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_7, y_8], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf95, (262144, 640), (640, 1), 0), buf99, out=buf100)
            buf101 = reinterpret_tensor(buf100, (128, 2048, 640), (1310720, 640, 1), 0); del buf100  # reuse
            buf102 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf103 = reinterpret_tensor(buf102, (128, 2048, 1), (2048, 1, 1), 0); del buf102  # reuse
            buf105 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_8, x_15, rms_norm_12], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf101, buf103, buf76, buf105, 262144, 640, stream=stream0)
            buf104 = empty_strided_cuda((640, 2560), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_25, buf104, 1638400, stream=stream0)
            del primals_25
            buf106 = empty_strided_cuda((262144, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_12, x_16], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf105, (262144, 640), (640, 1), 0), buf104, out=buf106)
            buf107 = empty_strided_cuda((2560, 640), (1, 2560), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_26, buf107, 1638400, stream=stream0)
            del primals_26
            buf108 = empty_strided_cuda((128, 2048, 2560), (5242880, 2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_16, relu_2, x_17, x_18], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf106, buf108, 671088640, stream=stream0)
            buf109 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_16, relu_2, x_17, x_18], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf108, (262144, 2560), (2560, 1), 0), buf107, out=buf109)
            buf110 = reinterpret_tensor(buf109, (128, 2048, 640), (1310720, 640, 1), 0); del buf109  # reuse
            buf111 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf113 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf114 = reinterpret_tensor(buf113, (128, 2048, 1), (2048, 1, 1), 0); del buf113  # reuse
            buf116 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, x_18, x_19, getitem_21, mul_32, getitem_22, mul_33, x_20, rms_norm_13], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_11.run(buf110, buf114, buf101, primals_5, primals_6, buf0, buf2, buf111, buf116, 262144, 640, stream=stream0)
            buf120 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_30, buf120, 409600, stream=stream0)
            del primals_30
            buf121 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_13, linear_19, linear_21], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf116, (262144, 640), (640, 1), 0), buf120, out=buf121)
            buf122 = empty_strided_cuda((32, 5), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_7.run(primals_31, buf122, 160, stream=stream0)
            del primals_31
            buf123 = empty_strided_cuda((262144, 5), (5, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_23, linear_22], Original ATen: [aten.slice, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf116, (262144, 32), (640, 1), 0), buf122, out=buf123)
            buf112 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf124 = reinterpret_tensor(buf121, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf121  # reuse
            # Topologically Sorted Source Nodes: [ve_2, linear_21, v_4, ve_3, linear_22, sigmoid_1, gate_1, unsqueeze_1, mul_35, v_5], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_8.run(buf124, primals_1, primals_27, buf123, buf112, 167772160, stream=stream0)
            del primals_27
            buf115 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_28, buf115, 409600, stream=stream0)
            del primals_28
            buf117 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_13, linear_19], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf116, (262144, 640), (640, 1), 0), buf115, out=buf117)
            buf118 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_29, buf118, 409600, stream=stream0)
            del primals_29
            buf119 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_13, linear_19, linear_20], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf116, (262144, 640), (640, 1), 0), buf118, out=buf119)
            buf125 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            buf127 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf128 = reinterpret_tensor(buf127, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf127  # reuse
            buf129 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_19, q_9, x1_6, x2_6, mul_36, mul_37, y1_6, mul_38, mul_39, y2_6, q_10, q_11], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf128, buf117, primals_2, primals_3, buf125, buf129, 1310720, 128, stream=stream0)
            buf126 = reinterpret_tensor(buf117, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf117  # reuse
            buf130 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf131 = reinterpret_tensor(buf130, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf130  # reuse
            buf132 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_20, k_9, x1_7, x2_7, mul_40, mul_41, y1_7, mul_42, mul_43, y2_7, k_10, k_11], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf131, buf119, primals_2, primals_3, buf126, buf132, 1310720, 128, stream=stream0)
            del buf119
            # Topologically Sorted Source Nodes: [y_9], Original ATen: [flash_attn_3._flash_attn_forward]
            buf133 = torch.ops.flash_attn_3._flash_attn_forward.default(buf129, buf132, buf124, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf134 = buf133[0]
            assert_size_stride(buf134, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf134, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf135 = buf133[1]
            assert_size_stride(buf135, (128, 5, 2048), (10240, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf135, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf133
            buf138 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_11], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_32, buf138, 409600, stream=stream0)
            del primals_32
            buf139 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_10, y_11], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf134, (262144, 640), (640, 1), 0), buf138, out=buf139)
            buf140 = reinterpret_tensor(buf139, (128, 2048, 640), (1310720, 640, 1), 0); del buf139  # reuse
            buf141 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf142 = reinterpret_tensor(buf141, (128, 2048, 1), (2048, 1, 1), 0); del buf141  # reuse
            buf144 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_11, x_21, rms_norm_16], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf140, buf142, buf111, buf144, 262144, 640, stream=stream0)
            buf143 = empty_strided_cuda((640, 2560), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_33, buf143, 1638400, stream=stream0)
            del primals_33
            buf145 = empty_strided_cuda((262144, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_16, x_22], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf144, (262144, 640), (640, 1), 0), buf143, out=buf145)
            buf146 = empty_strided_cuda((2560, 640), (1, 2560), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_34, buf146, 1638400, stream=stream0)
            del primals_34
            buf147 = empty_strided_cuda((128, 2048, 2560), (5242880, 2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_22, relu_3, x_23, x_24], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf145, buf147, 671088640, stream=stream0)
            buf148 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_22, relu_3, x_23, x_24], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf147, (262144, 2560), (2560, 1), 0), buf146, out=buf148)
            buf149 = reinterpret_tensor(buf148, (128, 2048, 640), (1310720, 640, 1), 0); del buf148  # reuse
            buf150 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf151 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf152 = reinterpret_tensor(buf151, (128, 2048, 1), (2048, 1, 1), 0); del buf151  # reuse
            buf154 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, x_24, x_25, getitem_28, mul_44, getitem_29, mul_45, x_26, rms_norm_17], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_12.run(buf149, buf152, buf140, primals_5, primals_6, buf0, buf2, buf150, buf154, 262144, 640, stream=stream0)
            buf153 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_35, buf153, 409600, stream=stream0)
            del primals_35
            buf155 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_17, linear_26], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf154, (262144, 640), (640, 1), 0), buf153, out=buf155)
            buf156 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_36, buf156, 409600, stream=stream0)
            del primals_36
            buf157 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_17, linear_26, linear_27], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf154, (262144, 640), (640, 1), 0), buf156, out=buf157)
            buf158 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_37, buf158, 409600, stream=stream0)
            del primals_37
            buf159 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_17, linear_26, linear_28], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf154, (262144, 640), (640, 1), 0), buf158, out=buf159)
            buf160 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            buf162 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf163 = reinterpret_tensor(buf162, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf162  # reuse
            buf164 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_26, q_12, x1_8, x2_8, mul_46, mul_47, y1_8, mul_48, mul_49, y2_8, q_13, q_14], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf163, buf155, primals_2, primals_3, buf160, buf164, 1310720, 128, stream=stream0)
            buf161 = reinterpret_tensor(buf155, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf155  # reuse
            buf165 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf166 = reinterpret_tensor(buf165, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf165  # reuse
            buf167 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_27, k_12, x1_9, x2_9, mul_50, mul_51, y1_9, mul_52, mul_53, y2_9, k_13, k_14], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf166, buf157, primals_2, primals_3, buf161, buf167, 1310720, 128, stream=stream0)
            del buf157
            # Topologically Sorted Source Nodes: [linear_28, v_6, y_12], Original ATen: [aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf168 = torch.ops.flash_attn_3._flash_attn_forward.default(buf164, buf167, reinterpret_tensor(buf159, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf169 = buf168[0]
            assert_size_stride(buf169, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf169, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf170 = buf168[1]
            assert_size_stride(buf170, (128, 5, 2048), (10240, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf170, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf168
            buf173 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_14], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_38, buf173, 409600, stream=stream0)
            del primals_38
            buf174 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_13, y_14], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf169, (262144, 640), (640, 1), 0), buf173, out=buf174)
            buf175 = reinterpret_tensor(buf174, (128, 2048, 640), (1310720, 640, 1), 0); del buf174  # reuse
            buf176 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf177 = reinterpret_tensor(buf176, (128, 2048, 1), (2048, 1, 1), 0); del buf176  # reuse
            buf179 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_14, x_27, rms_norm_20], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf175, buf177, buf150, buf179, 262144, 640, stream=stream0)
            buf178 = empty_strided_cuda((640, 2560), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_39, buf178, 1638400, stream=stream0)
            del primals_39
            buf180 = empty_strided_cuda((262144, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_20, x_28], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf179, (262144, 640), (640, 1), 0), buf178, out=buf180)
            buf181 = empty_strided_cuda((2560, 640), (1, 2560), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_40, buf181, 1638400, stream=stream0)
            del primals_40
            buf182 = empty_strided_cuda((128, 2048, 2560), (5242880, 2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_28, relu_4, x_29, x_30], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf180, buf182, 671088640, stream=stream0)
            buf183 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_28, relu_4, x_29, x_30], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf182, (262144, 2560), (2560, 1), 0), buf181, out=buf183)
            buf184 = reinterpret_tensor(buf183, (128, 2048, 640), (1310720, 640, 1), 0); del buf183  # reuse
            buf185 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf187 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf188 = reinterpret_tensor(buf187, (128, 2048, 1), (2048, 1, 1), 0); del buf187  # reuse
            buf190 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, x_30, x_31, getitem_34, mul_54, getitem_35, mul_55, x_32, rms_norm_21], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_13.run(buf184, buf188, buf175, primals_5, primals_6, buf0, buf2, buf185, buf190, 262144, 640, stream=stream0)
            buf194 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_44, buf194, 409600, stream=stream0)
            del primals_44
            buf195 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_21, linear_32, linear_34], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf190, (262144, 640), (640, 1), 0), buf194, out=buf195)
            buf196 = empty_strided_cuda((32, 5), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_7.run(primals_45, buf196, 160, stream=stream0)
            del primals_45
            buf197 = empty_strided_cuda((262144, 5), (5, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_36, linear_35], Original ATen: [aten.slice, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf190, (262144, 32), (640, 1), 0), buf196, out=buf197)
            buf186 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf198 = reinterpret_tensor(buf195, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf195  # reuse
            # Topologically Sorted Source Nodes: [ve_4, linear_34, v_7, ve_5, linear_35, sigmoid_2, gate_2, unsqueeze_2, mul_57, v_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_8.run(buf198, primals_1, primals_41, buf197, buf186, 167772160, stream=stream0)
            del primals_41
            buf189 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_42, buf189, 409600, stream=stream0)
            del primals_42
            buf191 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_21, linear_32], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf190, (262144, 640), (640, 1), 0), buf189, out=buf191)
            buf192 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_43, buf192, 409600, stream=stream0)
            del primals_43
            buf193 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_21, linear_32, linear_33], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf190, (262144, 640), (640, 1), 0), buf192, out=buf193)
            buf199 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            buf201 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf202 = reinterpret_tensor(buf201, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf201  # reuse
            buf203 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_32, q_15, x1_10, x2_10, mul_58, mul_59, y1_10, mul_60, mul_61, y2_10, q_16, q_17], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf202, buf191, primals_2, primals_3, buf199, buf203, 1310720, 128, stream=stream0)
            buf200 = reinterpret_tensor(buf191, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf191  # reuse
            buf204 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf205 = reinterpret_tensor(buf204, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf204  # reuse
            buf206 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_33, k_15, x1_11, x2_11, mul_62, mul_63, y1_11, mul_64, mul_65, y2_11, k_16, k_17], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf205, buf193, primals_2, primals_3, buf200, buf206, 1310720, 128, stream=stream0)
            del buf193
            # Topologically Sorted Source Nodes: [y_15], Original ATen: [flash_attn_3._flash_attn_forward]
            buf207 = torch.ops.flash_attn_3._flash_attn_forward.default(buf203, buf206, buf198, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf208 = buf207[0]
            assert_size_stride(buf208, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf208, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf209 = buf207[1]
            assert_size_stride(buf209, (128, 5, 2048), (10240, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf209, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf207
            buf212 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_17], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_46, buf212, 409600, stream=stream0)
            del primals_46
            buf213 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_16, y_17], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf208, (262144, 640), (640, 1), 0), buf212, out=buf213)
            buf214 = reinterpret_tensor(buf213, (128, 2048, 640), (1310720, 640, 1), 0); del buf213  # reuse
            buf215 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf216 = reinterpret_tensor(buf215, (128, 2048, 1), (2048, 1, 1), 0); del buf215  # reuse
            buf218 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_17, x_33, rms_norm_24], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf214, buf216, buf185, buf218, 262144, 640, stream=stream0)
            buf217 = empty_strided_cuda((640, 2560), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_47, buf217, 1638400, stream=stream0)
            del primals_47
            buf219 = empty_strided_cuda((262144, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_24, x_34], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf218, (262144, 640), (640, 1), 0), buf217, out=buf219)
            buf220 = empty_strided_cuda((2560, 640), (1, 2560), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_48, buf220, 1638400, stream=stream0)
            del primals_48
            buf221 = empty_strided_cuda((128, 2048, 2560), (5242880, 2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_34, relu_5, x_35, x_36], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf219, buf221, 671088640, stream=stream0)
            buf222 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_34, relu_5, x_35, x_36], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf221, (262144, 2560), (2560, 1), 0), buf220, out=buf222)
            buf223 = reinterpret_tensor(buf222, (128, 2048, 640), (1310720, 640, 1), 0); del buf222  # reuse
            buf224 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf225 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf226 = reinterpret_tensor(buf225, (128, 2048, 1), (2048, 1, 1), 0); del buf225  # reuse
            buf228 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, x_36, x_37, getitem_41, mul_66, getitem_42, mul_67, x_38, rms_norm_25], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_14.run(buf223, buf226, buf214, primals_5, primals_6, buf0, buf2, buf224, buf228, 262144, 640, stream=stream0)
            buf227 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_49, buf227, 409600, stream=stream0)
            del primals_49
            buf229 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_25, linear_39], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf228, (262144, 640), (640, 1), 0), buf227, out=buf229)
            buf230 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_50, buf230, 409600, stream=stream0)
            del primals_50
            buf231 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_25, linear_39, linear_40], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf228, (262144, 640), (640, 1), 0), buf230, out=buf231)
            buf232 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_51, buf232, 409600, stream=stream0)
            del primals_51
            buf233 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_25, linear_39, linear_41], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf228, (262144, 640), (640, 1), 0), buf232, out=buf233)
            buf234 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            buf236 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf237 = reinterpret_tensor(buf236, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf236  # reuse
            buf238 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_39, q_18, x1_12, x2_12, mul_68, mul_69, y1_12, mul_70, mul_71, y2_12, q_19, q_20], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf237, buf229, primals_2, primals_3, buf234, buf238, 1310720, 128, stream=stream0)
            buf235 = reinterpret_tensor(buf229, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf229  # reuse
            buf239 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf240 = reinterpret_tensor(buf239, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf239  # reuse
            buf241 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_40, k_18, x1_13, x2_13, mul_72, mul_73, y1_13, mul_74, mul_75, y2_13, k_19, k_20], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf240, buf231, primals_2, primals_3, buf235, buf241, 1310720, 128, stream=stream0)
            del buf231
            # Topologically Sorted Source Nodes: [linear_41, v_9, y_18], Original ATen: [aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf242 = torch.ops.flash_attn_3._flash_attn_forward.default(buf238, buf241, reinterpret_tensor(buf233, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf243 = buf242[0]
            assert_size_stride(buf243, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf243, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf244 = buf242[1]
            assert_size_stride(buf244, (128, 5, 2048), (10240, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf244, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf242
            buf247 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_20], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_52, buf247, 409600, stream=stream0)
            del primals_52
            buf248 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_19, y_20], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf243, (262144, 640), (640, 1), 0), buf247, out=buf248)
            buf249 = reinterpret_tensor(buf248, (128, 2048, 640), (1310720, 640, 1), 0); del buf248  # reuse
            buf250 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf251 = reinterpret_tensor(buf250, (128, 2048, 1), (2048, 1, 1), 0); del buf250  # reuse
            buf253 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_20, x_39, rms_norm_28], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf249, buf251, buf224, buf253, 262144, 640, stream=stream0)
            buf252 = empty_strided_cuda((640, 2560), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_53, buf252, 1638400, stream=stream0)
            del primals_53
            buf254 = empty_strided_cuda((262144, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_28, x_40], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf253, (262144, 640), (640, 1), 0), buf252, out=buf254)
            buf255 = empty_strided_cuda((2560, 640), (1, 2560), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_54, buf255, 1638400, stream=stream0)
            del primals_54
            buf256 = empty_strided_cuda((128, 2048, 2560), (5242880, 2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_40, relu_6, x_41, x_42], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf254, buf256, 671088640, stream=stream0)
            buf257 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_40, relu_6, x_41, x_42], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf256, (262144, 2560), (2560, 1), 0), buf255, out=buf257)
            buf258 = reinterpret_tensor(buf257, (128, 2048, 640), (1310720, 640, 1), 0); del buf257  # reuse
            buf259 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf261 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf262 = reinterpret_tensor(buf261, (128, 2048, 1), (2048, 1, 1), 0); del buf261  # reuse
            buf264 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, x_42, x_43, getitem_47, mul_76, getitem_48, mul_77, x_44, rms_norm_29], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_15.run(buf258, buf262, buf249, primals_5, primals_6, buf0, buf2, buf259, buf264, 262144, 640, stream=stream0)
            buf268 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_58, buf268, 409600, stream=stream0)
            del primals_58
            buf269 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_29, linear_45, linear_47], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf264, (262144, 640), (640, 1), 0), buf268, out=buf269)
            buf270 = empty_strided_cuda((32, 5), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_7.run(primals_59, buf270, 160, stream=stream0)
            del primals_59
            buf271 = empty_strided_cuda((262144, 5), (5, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_49, linear_48], Original ATen: [aten.slice, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf264, (262144, 32), (640, 1), 0), buf270, out=buf271)
            buf260 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf272 = reinterpret_tensor(buf269, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf269  # reuse
            # Topologically Sorted Source Nodes: [ve_6, linear_47, v_10, ve_7, linear_48, sigmoid_3, gate_3, unsqueeze_3, mul_79, v_11], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_8.run(buf272, primals_1, primals_55, buf271, buf260, 167772160, stream=stream0)
            del primals_55
            buf263 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_56, buf263, 409600, stream=stream0)
            del primals_56
            buf265 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_29, linear_45], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf264, (262144, 640), (640, 1), 0), buf263, out=buf265)
            buf266 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_57, buf266, 409600, stream=stream0)
            del primals_57
            buf267 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_29, linear_45, linear_46], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf264, (262144, 640), (640, 1), 0), buf266, out=buf267)
            buf273 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            buf275 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf276 = reinterpret_tensor(buf275, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf275  # reuse
            buf277 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_45, q_21, x1_14, x2_14, mul_80, mul_81, y1_14, mul_82, mul_83, y2_14, q_22, q_23], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf276, buf265, primals_2, primals_3, buf273, buf277, 1310720, 128, stream=stream0)
            buf274 = reinterpret_tensor(buf265, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf265  # reuse
            buf278 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf279 = reinterpret_tensor(buf278, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf278  # reuse
            buf280 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_46, k_21, x1_15, x2_15, mul_84, mul_85, y1_15, mul_86, mul_87, y2_15, k_22, k_23], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf279, buf267, primals_2, primals_3, buf274, buf280, 1310720, 128, stream=stream0)
            del buf267
            # Topologically Sorted Source Nodes: [y_21], Original ATen: [flash_attn_3._flash_attn_forward]
            buf281 = torch.ops.flash_attn_3._flash_attn_forward.default(buf277, buf280, buf272, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf282 = buf281[0]
            assert_size_stride(buf282, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf282, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf283 = buf281[1]
            assert_size_stride(buf283, (128, 5, 2048), (10240, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf283, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf281
            buf286 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_23], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_60, buf286, 409600, stream=stream0)
            del primals_60
            buf287 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_22, y_23], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf282, (262144, 640), (640, 1), 0), buf286, out=buf287)
            buf288 = reinterpret_tensor(buf287, (128, 2048, 640), (1310720, 640, 1), 0); del buf287  # reuse
            buf289 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf290 = reinterpret_tensor(buf289, (128, 2048, 1), (2048, 1, 1), 0); del buf289  # reuse
            buf292 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_23, x_45, rms_norm_32], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf288, buf290, buf259, buf292, 262144, 640, stream=stream0)
            buf291 = empty_strided_cuda((640, 2560), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_61, buf291, 1638400, stream=stream0)
            del primals_61
            buf293 = empty_strided_cuda((262144, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_32, x_46], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf292, (262144, 640), (640, 1), 0), buf291, out=buf293)
            buf294 = empty_strided_cuda((2560, 640), (1, 2560), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_62, buf294, 1638400, stream=stream0)
            del primals_62
            buf295 = empty_strided_cuda((128, 2048, 2560), (5242880, 2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_46, relu_7, x_47, x_48], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf293, buf295, 671088640, stream=stream0)
            buf296 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_46, relu_7, x_47, x_48], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf295, (262144, 2560), (2560, 1), 0), buf294, out=buf296)
            buf297 = reinterpret_tensor(buf296, (128, 2048, 640), (1310720, 640, 1), 0); del buf296  # reuse
            buf298 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf299 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf300 = reinterpret_tensor(buf299, (128, 2048, 1), (2048, 1, 1), 0); del buf299  # reuse
            buf302 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, x_48, x_49, getitem_54, mul_88, getitem_55, mul_89, x_50, rms_norm_33], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_16.run(buf297, buf300, buf288, primals_5, primals_6, buf0, buf2, buf298, buf302, 262144, 640, stream=stream0)
            buf301 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_52], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_63, buf301, 409600, stream=stream0)
            del primals_63
            buf303 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_33, linear_52], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf302, (262144, 640), (640, 1), 0), buf301, out=buf303)
            buf304 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_64, buf304, 409600, stream=stream0)
            del primals_64
            buf305 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_33, linear_52, linear_53], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf302, (262144, 640), (640, 1), 0), buf304, out=buf305)
            buf306 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_65, buf306, 409600, stream=stream0)
            del primals_65
            buf307 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_33, linear_52, linear_54], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf302, (262144, 640), (640, 1), 0), buf306, out=buf307)
            buf308 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            buf310 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf311 = reinterpret_tensor(buf310, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf310  # reuse
            buf312 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_52, q_24, x1_16, x2_16, mul_90, mul_91, y1_16, mul_92, mul_93, y2_16, q_25, q_26], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf311, buf303, primals_2, primals_3, buf308, buf312, 1310720, 128, stream=stream0)
            buf309 = reinterpret_tensor(buf303, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf303  # reuse
            buf313 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf314 = reinterpret_tensor(buf313, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf313  # reuse
            buf315 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_53, k_24, x1_17, x2_17, mul_94, mul_95, y1_17, mul_96, mul_97, y2_17, k_25, k_26], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf314, buf305, primals_2, primals_3, buf309, buf315, 1310720, 128, stream=stream0)
            del buf305
            # Topologically Sorted Source Nodes: [linear_54, v_12, y_24], Original ATen: [aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf316 = torch.ops.flash_attn_3._flash_attn_forward.default(buf312, buf315, reinterpret_tensor(buf307, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 1024, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf317 = buf316[0]
            assert_size_stride(buf317, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf317, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf318 = buf316[1]
            assert_size_stride(buf318, (128, 5, 2048), (10240, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf318, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf316
            buf321 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_26], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_66, buf321, 409600, stream=stream0)
            del primals_66
            buf322 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_25, y_26], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf317, (262144, 640), (640, 1), 0), buf321, out=buf322)
            buf323 = reinterpret_tensor(buf322, (128, 2048, 640), (1310720, 640, 1), 0); del buf322  # reuse
            buf324 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf325 = reinterpret_tensor(buf324, (128, 2048, 1), (2048, 1, 1), 0); del buf324  # reuse
            buf327 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_26, x_51, rms_norm_36], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf323, buf325, buf298, buf327, 262144, 640, stream=stream0)
            buf326 = empty_strided_cuda((640, 2560), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_67, buf326, 1638400, stream=stream0)
            del primals_67
            buf328 = empty_strided_cuda((262144, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_36, x_52], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf327, (262144, 640), (640, 1), 0), buf326, out=buf328)
            buf329 = empty_strided_cuda((2560, 640), (1, 2560), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_68, buf329, 1638400, stream=stream0)
            del primals_68
            buf330 = empty_strided_cuda((128, 2048, 2560), (5242880, 2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_52, relu_8, x_53, x_54], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf328, buf330, 671088640, stream=stream0)
            buf331 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_52, relu_8, x_53, x_54], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf330, (262144, 2560), (2560, 1), 0), buf329, out=buf331)
            buf332 = reinterpret_tensor(buf331, (128, 2048, 640), (1310720, 640, 1), 0); del buf331  # reuse
            buf333 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf335 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf336 = reinterpret_tensor(buf335, (128, 2048, 1), (2048, 1, 1), 0); del buf335  # reuse
            buf338 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, x_54, x_55, getitem_60, mul_98, getitem_61, mul_99, x_56, rms_norm_37], Original ATen: [aten._to_copy, aten.mul, aten._unsafe_view, aten.add, aten.select, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_17.run(buf332, buf336, buf323, primals_5, primals_6, buf0, buf2, buf333, buf338, 262144, 640, stream=stream0)
            buf342 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_72, buf342, 409600, stream=stream0)
            del primals_72
            buf343 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_37, linear_58, linear_60], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf338, (262144, 640), (640, 1), 0), buf342, out=buf343)
            buf344 = empty_strided_cuda((32, 5), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_7.run(primals_73, buf344, 160, stream=stream0)
            del primals_73
            buf345 = empty_strided_cuda((262144, 5), (5, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_62, linear_61], Original ATen: [aten.slice, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf338, (262144, 32), (640, 1), 0), buf344, out=buf345)
            buf334 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            buf346 = reinterpret_tensor(buf343, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf343  # reuse
            # Topologically Sorted Source Nodes: [ve_8, linear_60, v_13, ve_9, linear_61, sigmoid_4, gate_4, unsqueeze_4, mul_101, v_14], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_8.run(buf346, primals_1, primals_69, buf345, buf334, 167772160, stream=stream0)
            del primals_69
            buf337 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_70, buf337, 409600, stream=stream0)
            del primals_70
            buf339 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_37, linear_58], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf338, (262144, 640), (640, 1), 0), buf337, out=buf339)
            buf340 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_59], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_71, buf340, 409600, stream=stream0)
            del primals_71
            buf341 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_37, linear_58, linear_59], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf338, (262144, 640), (640, 1), 0), buf340, out=buf341)
            buf347 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            buf349 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf350 = reinterpret_tensor(buf349, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf349  # reuse
            buf351 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_58, q_27, x1_18, x2_18, mul_102, mul_103, y1_18, mul_104, mul_105, y2_18, q_28, q_29], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf350, buf339, primals_2, primals_3, buf347, buf351, 1310720, 128, stream=stream0)
            buf348 = reinterpret_tensor(buf339, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0); del buf339  # reuse
            buf352 = empty_strided_cuda((128, 2048, 5, 1), (10240, 5, 1, 1310720), torch.float32)
            buf353 = reinterpret_tensor(buf352, (128, 2048, 5, 1), (10240, 5, 1, 1), 0); del buf352  # reuse
            buf354 = empty_strided_cuda((128, 2048, 5, 128), (1310720, 640, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_59, k_27, x1_19, x2_19, mul_106, mul_107, y1_19, mul_108, mul_109, y2_19, k_28, k_29], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_2.run(buf353, buf341, primals_2, primals_3, buf348, buf354, 1310720, 128, stream=stream0)
            del buf341
            # Topologically Sorted Source Nodes: [y_27], Original ATen: [flash_attn_3._flash_attn_forward]
            buf355 = torch.ops.flash_attn_3._flash_attn_forward.default(buf351, buf354, buf346, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf356 = buf355[0]
            assert_size_stride(buf356, (128, 2048, 5, 128), (1310720, 640, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf356, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf357 = buf355[1]
            assert_size_stride(buf357, (128, 5, 2048), (10240, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf357, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf355
            buf360 = empty_strided_cuda((640, 640), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_29], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_74, buf360, 409600, stream=stream0)
            del primals_74
            buf361 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_28, y_29], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf356, (262144, 640), (640, 1), 0), buf360, out=buf361)
            buf362 = reinterpret_tensor(buf361, (128, 2048, 640), (1310720, 640, 1), 0); del buf361  # reuse
            buf363 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf364 = reinterpret_tensor(buf363, (128, 2048, 1), (2048, 1, 1), 0); del buf363  # reuse
            buf366 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_29, x_57, rms_norm_40], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf362, buf364, buf333, buf366, 262144, 640, stream=stream0)
            buf365 = empty_strided_cuda((640, 2560), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_75, buf365, 1638400, stream=stream0)
            del primals_75
            buf367 = empty_strided_cuda((262144, 2560), (2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_40, x_58], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf366, (262144, 640), (640, 1), 0), buf365, out=buf367)
            buf368 = empty_strided_cuda((2560, 640), (1, 2560), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_4.run(primals_76, buf368, 1638400, stream=stream0)
            del primals_76
            buf369 = empty_strided_cuda((128, 2048, 2560), (5242880, 2560, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_58, relu_9, x_59, x_60], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_pow_relu_5.run(buf367, buf369, 671088640, stream=stream0)
            buf370 = empty_strided_cuda((262144, 640), (640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_58, relu_9, x_59, x_60], Original ATen: [aten._unsafe_view, aten.relu, aten._to_copy, aten.pow, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf369, (262144, 2560), (2560, 1), 0), buf368, out=buf370)
            buf371 = reinterpret_tensor(buf370, (128, 2048, 640), (1310720, 640, 1), 0); del buf370  # reuse
            buf372 = empty_strided_cuda((128, 2048, 1), (2048, 1, 262144), torch.float32)
            buf373 = reinterpret_tensor(buf372, (128, 2048, 1), (2048, 1, 1), 0); del buf372  # reuse
            buf375 = empty_strided_cuda((128, 2048, 640), (1310720, 640, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_60, x_61, x_62], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf371, buf373, buf362, buf375, 262144, 640, stream=stream0)
            buf374 = empty_strided_cuda((640, 8192), (1, 640), torch.bfloat16)
            # Topologically Sorted Source Nodes: [logits], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_18.run(primals_77, buf374, 5242880, stream=stream0)
            del primals_77
            buf376 = empty_strided_cuda((262144, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_62, logits], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf375, (262144, 640), (640, 1), 0), buf374, out=buf376)
            buf377 = empty_strided_cuda((262144, 1), (1, 1), torch.float32)
            buf378 = empty_strided_cuda((262144, 1), (1, 262144), torch.float32)
            buf379 = reinterpret_tensor(buf378, (262144, 1), (1, 1), 0); del buf378  # reuse
            # Topologically Sorted Source Nodes: [logits, logits_1, truediv, tanh, logits_2, view_45, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.view, prims.prepare_softmax_online, aten._log_softmax]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view_19.run(buf379, buf376, buf377, 262144, 8192, stream=stream0)
            buf381 = empty_strided_cuda((32, ), (1, ), torch.int64)
            buf385 = empty_strided_cuda((32, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [logits, logits_1, truediv, tanh, logits_2, view_45, view_46, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.view, aten.sub, aten._log_softmax, aten.nll_loss_forward]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_20.run(primals_78, buf376, buf377, buf379, buf381, buf385, 32, 8192, stream=stream0)
            buf386 = empty_strided_cuda((), (), torch.float32)
            buf383 = empty_strided_cuda((), (), torch.float32)
            buf387 = buf386; del buf386  # reuse
            # Topologically Sorted Source Nodes: [logits, logits_1, truediv, tanh, logits_2, view_45, view_46, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.view, aten.sub, aten._log_softmax, aten.nll_loss_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_21.run(buf387, buf385, buf381, buf383, 1, 32, stream=stream0)
            del buf381
            del buf385
        return (buf387, primals_1, primals_2, primals_3, primals_5, primals_6, primals_78, buf0, buf2, buf4, reinterpret_tensor(buf6, (262144, 640), (640, 1), 0), reinterpret_tensor(buf11, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), buf12, buf13, buf15, buf16, buf18, buf19, buf21, buf22, buf27, buf29, reinterpret_tensor(buf31, (262144, 640), (640, 1), 0), buf32, reinterpret_tensor(buf34, (262144, 2560), (2560, 1), 0), buf36, buf37, buf38, buf40, reinterpret_tensor(buf42, (262144, 640), (640, 1), 0), reinterpret_tensor(buf42, (262144, 32), (640, 1), 0), buf49, buf50, buf51, buf52, buf54, buf55, buf57, buf58, buf60, buf61, buf66, buf68, reinterpret_tensor(buf70, (262144, 640), (640, 1), 0), buf71, reinterpret_tensor(buf73, (262144, 2560), (2560, 1), 0), buf75, buf76, buf78, reinterpret_tensor(buf80, (262144, 640), (640, 1), 0), reinterpret_tensor(buf85, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), buf86, buf87, buf89, buf90, buf92, buf93, buf95, buf96, buf101, buf103, reinterpret_tensor(buf105, (262144, 640), (640, 1), 0), buf106, reinterpret_tensor(buf108, (262144, 2560), (2560, 1), 0), buf110, buf111, buf112, buf114, reinterpret_tensor(buf116, (262144, 640), (640, 1), 0), reinterpret_tensor(buf116, (262144, 32), (640, 1), 0), buf123, buf124, buf125, buf126, buf128, buf129, buf131, buf132, buf134, buf135, buf140, buf142, reinterpret_tensor(buf144, (262144, 640), (640, 1), 0), buf145, reinterpret_tensor(buf147, (262144, 2560), (2560, 1), 0), buf149, buf150, buf152, reinterpret_tensor(buf154, (262144, 640), (640, 1), 0), reinterpret_tensor(buf159, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), buf160, buf161, buf163, buf164, buf166, buf167, buf169, buf170, buf175, buf177, reinterpret_tensor(buf179, (262144, 640), (640, 1), 0), buf180, reinterpret_tensor(buf182, (262144, 2560), (2560, 1), 0), buf184, buf185, buf186, buf188, reinterpret_tensor(buf190, (262144, 640), (640, 1), 0), reinterpret_tensor(buf190, (262144, 32), (640, 1), 0), buf197, buf198, buf199, buf200, buf202, buf203, buf205, buf206, buf208, buf209, buf214, buf216, reinterpret_tensor(buf218, (262144, 640), (640, 1), 0), buf219, reinterpret_tensor(buf221, (262144, 2560), (2560, 1), 0), buf223, buf224, buf226, reinterpret_tensor(buf228, (262144, 640), (640, 1), 0), reinterpret_tensor(buf233, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), buf234, buf235, buf237, buf238, buf240, buf241, buf243, buf244, buf249, buf251, reinterpret_tensor(buf253, (262144, 640), (640, 1), 0), buf254, reinterpret_tensor(buf256, (262144, 2560), (2560, 1), 0), buf258, buf259, buf260, buf262, reinterpret_tensor(buf264, (262144, 640), (640, 1), 0), reinterpret_tensor(buf264, (262144, 32), (640, 1), 0), buf271, buf272, buf273, buf274, buf276, buf277, buf279, buf280, buf282, buf283, buf288, buf290, reinterpret_tensor(buf292, (262144, 640), (640, 1), 0), buf293, reinterpret_tensor(buf295, (262144, 2560), (2560, 1), 0), buf297, buf298, buf300, reinterpret_tensor(buf302, (262144, 640), (640, 1), 0), reinterpret_tensor(buf307, (128, 2048, 5, 128), (1310720, 640, 128, 1), 0), buf308, buf309, buf311, buf312, buf314, buf315, buf317, buf318, buf323, buf325, reinterpret_tensor(buf327, (262144, 640), (640, 1), 0), buf328, reinterpret_tensor(buf330, (262144, 2560), (2560, 1), 0), buf332, buf333, buf334, buf336, reinterpret_tensor(buf338, (262144, 640), (640, 1), 0), reinterpret_tensor(buf338, (262144, 32), (640, 1), 0), buf345, buf346, buf347, buf348, buf350, buf351, buf353, buf354, buf356, buf357, buf362, buf364, reinterpret_tensor(buf366, (262144, 640), (640, 1), 0), buf367, reinterpret_tensor(buf369, (262144, 2560), (2560, 1), 0), buf371, buf373, reinterpret_tensor(buf375, (262144, 640), (640, 1), 0), buf376, buf377, buf379, buf383, reinterpret_tensor(buf374, (8192, 640), (640, 1), 0), reinterpret_tensor(buf368, (640, 2560), (2560, 1), 0), reinterpret_tensor(buf365, (2560, 640), (640, 1), 0), reinterpret_tensor(buf360, (640, 640), (640, 1), 0), reinterpret_tensor(buf344, (5, 32), (32, 1), 0), reinterpret_tensor(buf342, (640, 640), (640, 1), 0), reinterpret_tensor(buf340, (640, 640), (640, 1), 0), reinterpret_tensor(buf337, (640, 640), (640, 1), 0), reinterpret_tensor(buf329, (640, 2560), (2560, 1), 0), reinterpret_tensor(buf326, (2560, 640), (640, 1), 0), reinterpret_tensor(buf321, (640, 640), (640, 1), 0), reinterpret_tensor(buf306, (640, 640), (640, 1), 0), reinterpret_tensor(buf304, (640, 640), (640, 1), 0), reinterpret_tensor(buf301, (640, 640), (640, 1), 0), reinterpret_tensor(buf294, (640, 2560), (2560, 1), 0), reinterpret_tensor(buf291, (2560, 640), (640, 1), 0), reinterpret_tensor(buf286, (640, 640), (640, 1), 0), reinterpret_tensor(buf270, (5, 32), (32, 1), 0), reinterpret_tensor(buf268, (640, 640), (640, 1), 0), reinterpret_tensor(buf266, (640, 640), (640, 1), 0), reinterpret_tensor(buf263, (640, 640), (640, 1), 0), reinterpret_tensor(buf255, (640, 2560), (2560, 1), 0), reinterpret_tensor(buf252, (2560, 640), (640, 1), 0), reinterpret_tensor(buf247, (640, 640), (640, 1), 0), reinterpret_tensor(buf232, (640, 640), (640, 1), 0), reinterpret_tensor(buf230, (640, 640), (640, 1), 0), reinterpret_tensor(buf227, (640, 640), (640, 1), 0), reinterpret_tensor(buf220, (640, 2560), (2560, 1), 0), reinterpret_tensor(buf217, (2560, 640), (640, 1), 0), reinterpret_tensor(buf212, (640, 640), (640, 1), 0), reinterpret_tensor(buf196, (5, 32), (32, 1), 0), reinterpret_tensor(buf194, (640, 640), (640, 1), 0), reinterpret_tensor(buf192, (640, 640), (640, 1), 0), reinterpret_tensor(buf189, (640, 640), (640, 1), 0), reinterpret_tensor(buf181, (640, 2560), (2560, 1), 0), reinterpret_tensor(buf178, (2560, 640), (640, 1), 0), reinterpret_tensor(buf173, (640, 640), (640, 1), 0), reinterpret_tensor(buf158, (640, 640), (640, 1), 0), reinterpret_tensor(buf156, (640, 640), (640, 1), 0), reinterpret_tensor(buf153, (640, 640), (640, 1), 0), reinterpret_tensor(buf146, (640, 2560), (2560, 1), 0), reinterpret_tensor(buf143, (2560, 640), (640, 1), 0), reinterpret_tensor(buf138, (640, 640), (640, 1), 0), reinterpret_tensor(buf122, (5, 32), (32, 1), 0), reinterpret_tensor(buf120, (640, 640), (640, 1), 0), reinterpret_tensor(buf118, (640, 640), (640, 1), 0), reinterpret_tensor(buf115, (640, 640), (640, 1), 0), reinterpret_tensor(buf107, (640, 2560), (2560, 1), 0), reinterpret_tensor(buf104, (2560, 640), (640, 1), 0), reinterpret_tensor(buf99, (640, 640), (640, 1), 0), reinterpret_tensor(buf84, (640, 640), (640, 1), 0), reinterpret_tensor(buf82, (640, 640), (640, 1), 0), reinterpret_tensor(buf79, (640, 640), (640, 1), 0), reinterpret_tensor(buf72, (640, 2560), (2560, 1), 0), reinterpret_tensor(buf69, (2560, 640), (640, 1), 0), reinterpret_tensor(buf64, (640, 640), (640, 1), 0), reinterpret_tensor(buf48, (5, 32), (32, 1), 0), reinterpret_tensor(buf46, (640, 640), (640, 1), 0), reinterpret_tensor(buf44, (640, 640), (640, 1), 0), reinterpret_tensor(buf41, (640, 640), (640, 1), 0), reinterpret_tensor(buf33, (640, 2560), (2560, 1), 0), reinterpret_tensor(buf30, (2560, 640), (640, 1), 0), reinterpret_tensor(buf25, (640, 640), (640, 1), 0), reinterpret_tensor(buf10, (640, 640), (640, 1), 0), reinterpret_tensor(buf8, (640, 640), (640, 1), 0), reinterpret_tensor(buf5, (640, 640), (640, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((1, 20480, 1, 64), (1310720, 64, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((1, 20480, 1, 64), (1310720, 64, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_14 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_28 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_42 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_56 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_70 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((5, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((640, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((2560, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((640, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
