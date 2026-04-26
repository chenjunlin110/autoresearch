
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': 'fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': 'fp32', 'in_ptr4': '*fp32', 'in_ptr5': 'fp32', 'in_ptr6': 'fp32', 'in_ptr7': '*fp32', 'in_ptr8': 'fp32', 'in_ptr9': 'fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0', 'mutated_arg_names': ['in_ptr2', 'in_ptr4', 'in_ptr7', 'out_ptr4', 'out_ptr5', 'out_ptr6'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 167772160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr4, out_ptr5, out_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = in_ptr0
    tmp8 = tl.load(in_ptr1 + (x0), None)
    tmp9 = tl.load(in_ptr2 + (x0), None)
    tmp14 = in_ptr3
    tmp21 = tl.load(in_ptr4 + (x0), None)
    tmp26 = in_ptr5
    tmp31 = in_ptr6
    tmp34 = tl.load(in_ptr7 + (x0), None)
    tmp35 = in_ptr8
    tmp36 = in_ptr9
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp3 = tl_math.abs(tmp2)
    tmp4 = 0.5
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp2 - tmp1
    tmp7 = tl.where(tmp5, tmp6, tmp2)
    tmp10 = tmp8 - tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tl.where(tmp5, tmp8, tmp9)
    tmp13 = tmp11 + tmp12
    tmp15 = tmp1 - tmp14
    tmp16 = tl_math.abs(tmp15)
    tmp17 = tmp16 >= tmp4
    tmp18 = tmp15 - tmp1
    tmp19 = tl.where(tmp17, tmp18, tmp15)
    tmp20 = tmp8 * tmp8
    tmp22 = tmp20 - tmp21
    tmp23 = tmp19 * tmp22
    tmp24 = tl.where(tmp17, tmp20, tmp21)
    tmp25 = tmp23 + tmp24
    tmp27 = libdevice.pow(tmp14, tmp26)
    tmp28 = tmp1 - tmp27
    tmp29 = (tmp25 / tmp28)
    tmp30 = libdevice.sqrt(tmp29)
    tmp32 = tmp30 + tmp31
    tmp33 = (tmp13 / tmp32)
    tmp37 = tmp35 * tmp36
    tmp38 = tmp1 - tmp37
    tmp39 = tmp34 * tmp38
    tmp40 = libdevice.pow(tmp0, tmp26)
    tmp41 = tmp1 - tmp40
    tmp42 = (tmp35 / tmp41)
    tmp43 = -tmp42
    tmp44 = tmp33 * tmp43
    tmp45 = tmp39 + tmp44
    tl.store(out_ptr4 + (x0), tmp13, None)
    tl.store(out_ptr5 + (x0), tmp25, None)
    tl.store(out_ptr6 + (x0), tmp45, None)
