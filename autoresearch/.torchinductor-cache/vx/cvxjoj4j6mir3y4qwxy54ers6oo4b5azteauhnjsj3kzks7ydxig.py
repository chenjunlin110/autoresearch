
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
