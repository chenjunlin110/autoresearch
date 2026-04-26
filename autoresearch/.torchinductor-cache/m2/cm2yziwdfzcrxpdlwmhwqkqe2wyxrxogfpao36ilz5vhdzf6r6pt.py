
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': 'fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_mul_rsqrt_rsub_sqrt_sub_11', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_mul_rsqrt_rsub_sqrt_sub_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 32)
    x2 = xindex // 128
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
