
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
