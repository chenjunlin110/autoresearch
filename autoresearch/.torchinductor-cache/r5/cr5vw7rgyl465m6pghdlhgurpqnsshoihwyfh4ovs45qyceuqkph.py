
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
