
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': 'fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_mul_rsub_0', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr5'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 14336}}
)
@triton.jit
def triton_per_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_mul_rsub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
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
    tmp0 = in_ptr0
    tmp12 = tl.load(in_ptr1 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp13 = tl.load(in_ptr2 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 0.5
    tmp3 = tmp1 >= tmp2
    tmp4 = 1.0
    tmp5 = tmp0 - tmp4
    tmp6 = tl.where(tmp3, tmp5, tmp0)
    tmp7 = tmp4 - tmp0
    tmp8 = tl_math.abs(tmp7)
    tmp9 = tmp8 >= tmp2
    tmp10 = tmp7 - tmp4
    tmp11 = tl.where(tmp9, tmp10, tmp7)
    tmp14 = tmp12 - tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp9, tmp12, tmp13)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17 - tmp12
    tmp19 = tmp6 * tmp18
    tmp20 = tl.where(tmp3, tmp17, tmp12)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = 1.02
    tmp32 = tmp30 * tmp31
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = (tmp22 / tmp34)
    tl.store(out_ptr1 + (r0_1 + 128*x0), tmp35, xmask)
    tl.store(out_ptr4 + (r0_1 + 128*x0), tmp21, xmask)
    tl.store(out_ptr5 + (r0_1 + 128*x0), tmp17, xmask)
