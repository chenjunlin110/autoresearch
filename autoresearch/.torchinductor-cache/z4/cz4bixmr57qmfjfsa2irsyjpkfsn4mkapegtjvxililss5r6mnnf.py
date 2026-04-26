
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_select_backward_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_select_backward_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp3 = tl.load(in_ptr0 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp9 = tl.load(in_ptr1 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp15 = tl.load(in_ptr2 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp27 = tl.load(in_ptr4 + (0))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp33 = tl.load(in_ptr5 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp39 = tl.load(in_ptr6 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp45 = tl.load(in_ptr7 + (0))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 7, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = 0.0
    tmp6 = tl.where(tmp2, tmp4, tmp5)
    tmp7 = tl.full([1], 6, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp11 = tl.where(tmp8, tmp10, tmp5)
    tmp12 = tmp6 + tmp11
    tmp13 = tl.full([1], 5, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp17 = tl.where(tmp14, tmp16, tmp5)
    tmp18 = tmp12 + tmp17
    tmp19 = tl.full([1], 4, tl.int32)
    tmp20 = tmp0 == tmp19
    tmp23 = tl.where(tmp20, tmp22, tmp5)
    tmp24 = tmp18 + tmp23
    tmp25 = tl.full([1], 3, tl.int32)
    tmp26 = tmp0 == tmp25
    tmp29 = tl.where(tmp26, tmp28, tmp5)
    tmp30 = tmp24 + tmp29
    tmp31 = tl.full([1], 2, tl.int32)
    tmp32 = tmp0 == tmp31
    tmp35 = tl.where(tmp32, tmp34, tmp5)
    tmp36 = tmp30 + tmp35
    tmp37 = tl.full([1], 1, tl.int32)
    tmp38 = tmp0 == tmp37
    tmp41 = tl.where(tmp38, tmp40, tmp5)
    tmp42 = tmp36 + tmp41
    tmp43 = tl.full([1], 0, tl.int32)
    tmp44 = tmp0 == tmp43
    tmp47 = tl.where(tmp44, tmp46, tmp5)
    tmp48 = tmp42 + tmp47
    tl.store(out_ptr0 + (x0), tmp48, xmask)
