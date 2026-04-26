
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2097152, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1310720
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
    x3 = xindex // 5
    x2 = (xindex % 5)
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp26 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([XBLOCK, R0_BLOCK], 8192, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 8192), "index out of bounds: 0 <= tmp11 < 8192")
    tmp13 = tl.full([1, 1], -1, tl.int64)
    tmp14 = tmp7 == tmp13
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = 2.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp0 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 0.0
    tmp22 = tl.where(tmp14, tmp21, tmp20)
    tmp23 = tmp6.to(tl.float32)
    tmp24 = tmp23 * tmp17
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tl.sigmoid(tmp26)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = 1.0
    tmp30 = tmp29 - tmp28
    tmp31 = tmp28 * tmp30
    tmp32 = tmp25 * tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.atomic_add(out_ptr1 + (r0_1 + 128*x2 + 640*tmp11), tmp22, None, sem='relaxed')
    tl.store(out_ptr2 + (x0), tmp33, None)
