
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*bf16', 'in_ptr12': '*i64', 'in_ptr13': '*bf16', 'in_ptr14': '*bf16', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_select_sigmoid_sum_unsqueeze_view_30', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr7', 'out_ptr8'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 18, 'num_reduction': 5, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_select_sigmoid_sum_unsqueeze_view_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr3, out_ptr4, out_ptr6, out_ptr7, out_ptr8, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp18 = tl.load(in_ptr5 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp27 = tl.load(in_ptr7 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp35 = tl.load(in_ptr8 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp36 = tl.load(in_ptr9 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp37 = tl.load(in_ptr3 + (2))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, R0_BLOCK])
    tmp42 = tl.load(in_ptr10 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp43 = tl.load(in_ptr3 + (1))
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK, R0_BLOCK])
    tmp67 = tl.load(in_ptr11 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp73 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr13 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp82 = tl.load(in_ptr14 + (4*x0 + (r0_1 // 128)), None, eviction_policy='evict_last').to(tl.float32)
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
    tmp26 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp28 = 0.001953125
    tmp29 = tmp16 * tmp28
    tmp30 = tmp29 * tmp26
    tmp31 = tmp22 - tmp30
    tmp32 = tmp31 * tmp15
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp27 + tmp33
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp36 * tmp39
    tmp41 = tmp35 + tmp40
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp42 * tmp45
    tmp47 = tmp41 + tmp46
    tmp48 = tmp34 * tmp11
    tmp49 = tmp47 + tmp48
    tmp50 = tmp34 * tmp2
    tmp51 = tmp49 + tmp50
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp42 * tmp7
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, R0_BLOCK])
    tmp57 = tl.sum(tmp55, 1)[:, None].to(tl.float32)
    tmp58 = tmp34 * tmp7
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK])
    tmp62 = tl.sum(tmp60, 1)[:, None].to(tl.float32)
    tmp63 = tmp6 * tmp52
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, R0_BLOCK])
    tmp66 = tl.sum(tmp64, 1)[:, None].to(tl.float32)
    tmp68 = tmp42 * tmp67
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK, R0_BLOCK])
    tmp72 = tl.sum(tmp70, 1)[:, None].to(tl.float32)
    tmp74 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp75 = tmp73 + tmp74
    tmp76 = tmp73 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp73)
    tl.device_assert((0 <= tmp77) & (tmp77 < 8192), "index out of bounds: 0 <= tmp77 < 8192")
    tmp79 = tl.full([1, 1], -1, tl.int64)
    tmp80 = tmp73 == tmp79
    tmp83 = tl.sigmoid(tmp82)
    tmp84 = 2.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp81 * tmp85
    tmp87 = tmp86.to(tl.float32)
    tmp88 = 0.0
    tmp89 = tl.where(tmp80, tmp88, tmp87)
    tmp90 = tmp6 * tmp28
    tmp91 = tmp90 * tmp66
    tmp92 = tmp52 - tmp91
    tmp93 = tmp92 * tmp5
    tmp94 = tl.where(tmp80, tmp88, tmp93)
    tl.atomic_add(out_ptr7 + (tl.broadcast_to(r0_1 + 512*tmp77, [XBLOCK, R0_BLOCK])), tmp89, None, sem='relaxed')
    tl.atomic_add(out_ptr8 + (tl.broadcast_to(r0_1 + 512*tmp77, [XBLOCK, R0_BLOCK])), tmp94, None, sem='relaxed')
    tl.store(out_ptr3 + (x0), tmp57, None)
    tl.store(out_ptr4 + (x0), tmp62, None)
    tl.store(out_ptr6 + (x0), tmp72, None)
