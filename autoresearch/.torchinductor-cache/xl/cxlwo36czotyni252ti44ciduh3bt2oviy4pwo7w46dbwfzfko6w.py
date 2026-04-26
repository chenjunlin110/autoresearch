
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
