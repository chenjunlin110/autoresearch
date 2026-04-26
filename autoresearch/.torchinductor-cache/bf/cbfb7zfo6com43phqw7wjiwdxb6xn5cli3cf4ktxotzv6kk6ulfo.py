
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': 'fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_lerp_linalg_vector_norm_rsub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4160, 'r0_': 131073280}}
)
@triton.jit
def triton_red_fused__to_copy_lerp_linalg_vector_norm_rsub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 520
    r0_numel = 31508
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 13)
    x1 = xindex // 13
    _tmp31 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 31508*x0
        tmp1 = tl.full([1, 1], 409600, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = in_ptr0
        tmp4 = tl_math.abs(tmp3)
        tmp5 = 0.5
        tmp6 = tmp4 >= tmp5
        tmp7 = 1.0
        tmp8 = tmp3 - tmp7
        tmp9 = tl.where(tmp6, tmp8, tmp3)
        tmp10 = tmp7 - tmp3
        tmp11 = tl_math.abs(tmp10)
        tmp12 = tmp11 >= tmp5
        tmp13 = tmp10 - tmp7
        tmp14 = tl.where(tmp12, tmp13, tmp10)
        tmp15 = tl.load(in_ptr1 + (409600*x1 + (((r0_2 + 31508*x0) % 409600))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr2 + (409600*x1 + (((r0_2 + 31508*x0) % 409600))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp14 * tmp17
        tmp19 = tl.where(tmp12, tmp15, tmp16)
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20 - tmp15
        tmp22 = tmp9 * tmp21
        tmp23 = tl.where(tmp6, tmp20, tmp15)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 * tmp26
        tmp28 = tl.full(tmp27.shape, 0, tmp27.dtype)
        tmp29 = tl.where(tmp2, tmp27, tmp28)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(r0_mask & xmask, tmp32, _tmp31)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp31, xmask)
