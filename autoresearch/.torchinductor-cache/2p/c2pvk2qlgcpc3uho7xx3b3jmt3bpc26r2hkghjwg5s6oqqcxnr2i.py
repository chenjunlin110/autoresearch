
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'xnumel': 'i64', 'r0_numel': 'i64', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_prepare_softmax_online_sub_tanh_view_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_prepare_softmax_online_sub_tanh_view_19(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    rbase = r0_base
    x0 = xindex
    _tmp8_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp8_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = 0.06666666666666667
        tmp3 = tmp1 * tmp2
        tmp4 = libdevice.tanh(tmp3)
        tmp5 = 15.0
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])

        _tmp8_max_next, _tmp8_sum_next = triton_helpers.online_softmax_combine(
            _tmp8_max, _tmp8_sum, tmp7, False
        )

        _tmp8_max = tl.where(r0_mask, _tmp8_max_next, _tmp8_max)
        _tmp8_sum = tl.where(r0_mask, _tmp8_sum_next, _tmp8_sum)

    tmp8, tmp9 = triton_helpers.online_softmax_reduce(
        _tmp8_max, _tmp8_sum, 1, False)
    tmp8 = tmp8[:, None]
    tmp9 = tmp9[:, None]
    tmp10 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.full([1, 1], -1, tl.int64)
    tmp12 = tmp10 != tmp11
    tmp13 = tl.full([1, 1], 0, tl.int64)
    tmp14 = tl.where(tmp12, tmp10, tmp13)
    tmp15 = tl.full([XBLOCK, 1], 8192, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tl.device_assert((0 <= tmp18) & (tmp18 < 8192), "index out of bounds: 0 <= tmp18 < 8192")
    tmp20 = tl.load(in_ptr0 + (tmp18 + 8192*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 0.06666666666666667
    tmp23 = tmp21 * tmp22
    tmp24 = libdevice.tanh(tmp23)
    tmp25 = 15.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26 - tmp8
    tmp28 = tl_math.log(tmp9)
    tmp29 = tmp27 - tmp28
    tmp30 = -tmp29
    tmp31 = 0.0
    tmp32 = tl.where(tmp12, tmp30, tmp31)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp32, None)
