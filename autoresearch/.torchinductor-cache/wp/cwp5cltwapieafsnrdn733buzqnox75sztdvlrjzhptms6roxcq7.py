
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1048576, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 30, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8388608, 'r0_': 6442450944}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_slice_backward_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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
    r0_1 = r0_index
    x0 = xindex
    x3 = ((xindex // 4) % 2048)
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp20 = r0_1
    tmp21 = tl.full([1, 1], 64, tl.int64)
    tmp22 = tmp20 >= tmp21
    tmp23 = tl.load(in_ptr5 + (r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.load(in_ptr3 + (r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), tmp22, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = 0.0078125
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30 * tmp19
    tmp32 = tmp24 - tmp31
    tmp33 = tmp32 * tmp27
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.load(in_ptr6 + ((-64) + r0_1 + 64*x3), tmp22, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tl.load(in_ptr5 + ((-64) + r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.load(in_ptr3 + ((-64) + r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp40 * tmp27
    tmp42 = tmp41 * tmp29
    tmp43 = tmp42 * tmp19
    tmp44 = tmp38 - tmp43
    tmp45 = tmp44 * tmp27
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tl.load(in_ptr7 + ((-64) + r0_1 + 64*x3), tmp22, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp48 = tmp46 * tmp47
    tmp49 = tmp36 + tmp48
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp22, tmp49, tmp50)
    tmp52 = 0.0
    tmp53 = tl.where(tmp22, tmp51, tmp52)
    tmp54 = tmp20 < tmp21
    tmp55 = tl.load(in_ptr5 + (64 + r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tl.load(in_ptr3 + (64 + r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), tmp54, eviction_policy='evict_last', other=0.0)
    tmp60 = tmp58 * tmp59
    tmp61 = 0.0078125
    tmp62 = tmp60 * tmp61
    tmp63 = tmp62 * tmp19
    tmp64 = tmp56 - tmp63
    tmp65 = tmp64 * tmp59
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tl.load(in_ptr7 + (r0_1 + 64*x3), tmp54, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp68 = -tmp67
    tmp69 = tmp66 * tmp68
    tmp70 = tl.load(in_ptr5 + (r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tl.load(in_ptr3 + (r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp73 * tmp59
    tmp75 = tmp74 * tmp61
    tmp76 = tmp75 * tmp19
    tmp77 = tmp71 - tmp76
    tmp78 = tmp77 * tmp59
    tmp79 = tmp78.to(tl.float32)
    tmp80 = tl.load(in_ptr6 + (r0_1 + 64*x3), tmp54, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp81 = tmp79 * tmp80
    tmp82 = tmp69 + tmp81
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp54, tmp82, tmp83)
    tmp85 = tl.where(tmp54, tmp84, tmp52)
    tmp86 = tmp53 + tmp85
    tmp87 = tl.load(in_ptr2 + (r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp88 = tmp87.to(tl.float32)
    tmp89 = tl.load(in_ptr0 + (r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp90 = tmp89.to(tl.float32)
    tmp91 = tl.load(in_ptr1 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), tmp22, eviction_policy='evict_last', other=0.0)
    tmp92 = tmp90 * tmp91
    tmp93 = tmp92 * tmp29
    tmp94 = tmp93 * tmp9
    tmp95 = tmp88 - tmp94
    tmp96 = tmp95 * tmp91
    tmp97 = tmp96.to(tl.float32)
    tmp98 = tmp97 * tmp35
    tmp99 = tl.load(in_ptr2 + ((-64) + r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp100 = tmp99.to(tl.float32)
    tmp101 = tl.load(in_ptr0 + ((-64) + r0_1 + 128*x0), tmp22, other=0.0).to(tl.float32)
    tmp102 = tmp101.to(tl.float32)
    tmp103 = tmp102 * tmp91
    tmp104 = tmp103 * tmp29
    tmp105 = tmp104 * tmp9
    tmp106 = tmp100 - tmp105
    tmp107 = tmp106 * tmp91
    tmp108 = tmp107.to(tl.float32)
    tmp109 = tmp108 * tmp47
    tmp110 = tmp98 + tmp109
    tmp111 = tl.full(tmp110.shape, 0.0, tmp110.dtype)
    tmp112 = tl.where(tmp22, tmp110, tmp111)
    tmp113 = tl.where(tmp22, tmp112, tmp52)
    tmp114 = tl.load(in_ptr2 + (64 + r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp115 = tmp114.to(tl.float32)
    tmp116 = tl.load(in_ptr0 + (64 + r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp117 = tmp116.to(tl.float32)
    tmp118 = tl.load(in_ptr1 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), tmp54, eviction_policy='evict_last', other=0.0)
    tmp119 = tmp117 * tmp118
    tmp120 = tmp119 * tmp61
    tmp121 = tmp120 * tmp9
    tmp122 = tmp115 - tmp121
    tmp123 = tmp122 * tmp118
    tmp124 = tmp123.to(tl.float32)
    tmp125 = tmp124 * tmp68
    tmp126 = tl.load(in_ptr2 + (r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp127 = tmp126.to(tl.float32)
    tmp128 = tl.load(in_ptr0 + (r0_1 + 128*x0), tmp54, other=0.0).to(tl.float32)
    tmp129 = tmp128.to(tl.float32)
    tmp130 = tmp129 * tmp118
    tmp131 = tmp130 * tmp61
    tmp132 = tmp131 * tmp9
    tmp133 = tmp127 - tmp132
    tmp134 = tmp133 * tmp118
    tmp135 = tmp134.to(tl.float32)
    tmp136 = tmp135 * tmp80
    tmp137 = tmp125 + tmp136
    tmp138 = tl.full(tmp137.shape, 0.0, tmp137.dtype)
    tmp139 = tl.where(tmp54, tmp137, tmp138)
    tmp140 = tl.where(tmp54, tmp139, tmp52)
    tmp141 = tmp113 + tmp140
    tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp86, None)
    tl.store(in_out_ptr1 + (r0_1 + 128*x0), tmp141, None)
