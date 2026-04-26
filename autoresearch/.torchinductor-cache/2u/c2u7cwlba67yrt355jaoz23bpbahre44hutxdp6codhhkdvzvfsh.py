
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': 'fp32', 'in_ptr3': '*fp32', 'out_ptr3': '*bf16', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_rsub_sqrt_sum_10', 'mutated_arg_names': ['in_ptr3', 'out_ptr5'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 12, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 5760}}
)
@triton.jit
def triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_rsub_sqrt_sum_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 5
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (32 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (32 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (64 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr1 + (64 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr0 + (96 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp23 = tl.load(in_ptr1 + (96 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr0 + (128 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr1 + (128 + r0_1 + 160*x0), xmask, other=0.0).to(tl.float32)
    tmp42 = in_ptr2
    tmp52 = tl.load(in_ptr3 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp1 = 2.3465413258596377
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp5
    tmp8 = tmp7 * tmp1
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp11
    tmp13 = tmp6 + tmp12
    tmp15 = tmp14 * tmp1
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp18
    tmp20 = tmp13 + tmp19
    tmp22 = tmp21 * tmp1
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp25
    tmp27 = tmp20 + tmp26
    tmp29 = tmp28 * tmp1
    tmp31 = tmp29 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 * tmp32
    tmp34 = tmp27 + tmp33
    tmp35 = 5.0
    tmp36 = (tmp34 / tmp35)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp39 = tl.where(xmask, tmp37, 0)
    tmp40 = tl.sum(tmp39, 1)[:, None].to(tl.float32)
    tmp41 = tmp36 * tmp35
    tmp43 = tmp42.to(tl.float32)
    tmp44 = 1.0
    tmp45 = tmp44 - tmp43
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tl_math.abs(tmp46)
    tmp48 = 0.5
    tmp49 = tmp47 >= tmp48
    tmp50 = tmp46 - tmp44
    tmp51 = tl.where(tmp49, tmp50, tmp46)
    tmp53 = tmp36 - tmp52
    tmp54 = tmp51 * tmp53
    tmp55 = tl.where(tmp49, tmp36, tmp52)
    tmp56 = tmp54 + tmp55
    tmp57 = 1e-10
    tmp58 = triton_helpers.maximum(tmp56, tmp57)
    tmp59 = libdevice.rsqrt(tmp58)
    tmp60 = tmp59 * tmp59
    tmp61 = tmp41 * tmp60
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK, R0_BLOCK])
    tmp64 = tl.where(xmask, tmp62, 0)
    tmp65 = tl.sum(tmp64, 1)[:, None].to(tl.float32)
    tmp66 = tmp40 * tmp35
    tmp67 = libdevice.sqrt(tmp66)
    tmp68 = libdevice.sqrt(tmp65)
    tmp69 = triton_helpers.maximum(tmp68, tmp57)
    tmp70 = (tmp67 / tmp69)
    tmp71 = tmp59 * tmp70
    tmp72 = tmp71.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 32*x0), tmp72, xmask)
    tl.store(out_ptr5 + (r0_1 + 32*x0), tmp56, xmask)
