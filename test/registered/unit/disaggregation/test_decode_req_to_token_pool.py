from sglang.srt.disaggregation.decode import (
    DecodeReqToTokenPool,
    HybridMambaDecodeReqToTokenPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _init_decode_pool(pool):
    DecodeReqToTokenPool.__init__(
        pool,
        size=1,
        max_context_len=4,
        device="cpu",
        enable_memory_saver=False,
        pre_alloc_size=1,
    )
    return pool


def test_decode_pool_reports_physical_capacity():
    pool = _init_decode_pool(DecodeReqToTokenPool.__new__(DecodeReqToTokenPool))

    assert pool.schedulable_token_capacity(17) == 17


def test_hybrid_decode_pool_initializes_aux_cache_contract():
    pool = _init_decode_pool(
        HybridMambaDecodeReqToTokenPool.__new__(HybridMambaDecodeReqToTokenPool)
    )

    assert pool.schedulable_token_capacity(17) == 17
