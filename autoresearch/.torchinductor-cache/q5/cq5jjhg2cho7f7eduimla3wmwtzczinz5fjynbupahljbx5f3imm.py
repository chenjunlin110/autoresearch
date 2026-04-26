
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_rsub_sqrt_sum_13', 'mutated_arg_names': ['in_ptr2', 'out_ptr4'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 393216}}
)
@triton.jit
def triton_red_fused__to_copy_add_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_rsub_sqrt_sum_13(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp7 = in_ptr1
    _tmp28 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 512.0
        tmp2 = (tmp0 / tmp1)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp6 = tmp2 * tmp1
        tmp8 = tmp7.to(tl.float32)
        tmp9 = 1.0
        tmp10 = tmp9 - tmp8
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl_math.abs(tmp11)
        tmp13 = 0.5
        tmp14 = tmp12 >= tmp13
        tmp15 = tmp11 - tmp9
        tmp16 = tl.where(tmp14, tmp15, tmp11)
        tmp18 = tmp2 - tmp17
        tmp19 = tmp16 * tmp18
        tmp20 = tl.where(tmp14, tmp2, tmp17)
        tmp21 = tmp19 + tmp20
        tmp22 = 1e-10
        tmp23 = triton_helpers.maximum(tmp21, tmp22)
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp24 * tmp24
        tmp26 = tmp6 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(r0_mask & xmask, tmp29, _tmp28)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp39 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp42 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tmp7.to(tl.float32)
        tmp31 = 1.0
        tmp32 = tmp31 - tmp30
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tl_math.abs(tmp33)
        tmp35 = 0.5
        tmp36 = tmp34 >= tmp35
        tmp37 = tmp33 - tmp31
        tmp38 = tl.where(tmp36, tmp37, tmp33)
        tmp40 = 512.0
        tmp41 = (tmp39 / tmp40)
        tmp43 = tmp41 - tmp42
        tmp44 = tmp38 * tmp43
        tmp45 = tl.where(tmp36, tmp41, tmp42)
        tmp46 = tmp44 + tmp45
        tmp47 = 1e-10
        tmp48 = triton_helpers.maximum(tmp46, tmp47)
        tmp49 = libdevice.rsqrt(tmp48)
        tmp50 = tmp4 * tmp40
        tmp51 = libdevice.sqrt(tmp50)
        tmp52 = libdevice.sqrt(tmp28)
        tmp53 = triton_helpers.maximum(tmp52, tmp47)
        tmp54 = (tmp51 / tmp53)
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr2 + (r0_1 + 2048*x0), tmp55, r0_mask & xmask)
        tl.store(out_ptr4 + (r0_1 + 2048*x0), tmp46, r0_mask & xmask)
