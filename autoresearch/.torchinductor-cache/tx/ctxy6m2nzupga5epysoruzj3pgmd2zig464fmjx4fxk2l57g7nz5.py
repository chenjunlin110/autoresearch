
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': 'fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': 'fp32', 'in_ptr4': '*bf16', 'in_ptr5': 'fp32', 'in_ptr6': 'fp32', 'in_ptr7': '*bf16', 'in_ptr8': 'fp32', 'in_ptr9': 'fp32', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'out_ptr6': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0', 'mutated_arg_names': ['in_ptr2', 'in_ptr4', 'in_ptr7', 'out_ptr4', 'out_ptr5', 'out_ptr6'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 83886080}},
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
    tmp8 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp17 = in_ptr3
    tmp25 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp32 = in_ptr5
    tmp38 = in_ptr6
    tmp42 = tl.load(in_ptr7 + (x0), None).to(tl.float32)
    tmp43 = in_ptr8
    tmp44 = in_ptr9
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp3 = tl_math.abs(tmp2)
    tmp4 = 0.5
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp2 - tmp1
    tmp7 = tl.where(tmp5, tmp6, tmp2)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 - tmp11
    tmp13 = tmp7 * tmp12
    tmp14 = tl.where(tmp5, tmp9, tmp11)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp1 - tmp17
    tmp19 = tl_math.abs(tmp18)
    tmp20 = tmp19 >= tmp4
    tmp21 = tmp18 - tmp1
    tmp22 = tl.where(tmp20, tmp21, tmp18)
    tmp23 = tmp8 * tmp8
    tmp24 = tmp23.to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 - tmp26
    tmp28 = tmp22 * tmp27
    tmp29 = tl.where(tmp20, tmp24, tmp26)
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp33 = libdevice.pow(tmp17, tmp32)
    tmp34 = tmp1 - tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = (tmp31 / tmp35)
    tmp37 = libdevice.sqrt(tmp36)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp37 + tmp39
    tmp41 = (tmp16 / tmp40)
    tmp45 = tmp43 * tmp44
    tmp46 = tmp1 - tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp42 * tmp47
    tmp49 = libdevice.pow(tmp0, tmp32)
    tmp50 = tmp1 - tmp49
    tmp51 = (tmp43 / tmp50)
    tmp52 = -tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp41 * tmp53
    tmp55 = tmp48 + tmp54
    tl.store(out_ptr4 + (x0), tmp16, None)
    tl.store(out_ptr5 + (x0), tmp31, None)
    tl.store(out_ptr6 + (x0), tmp55, None)
